"""Export Clearwell pipeline results into static files for the web map in frontend/.

Run after any pipeline re-run:

    python export_web_data.py

Reads only (never writes) the pipeline files in outputs/ and results/, and writes:

    frontend/data/villages.geojson    village proxy shapes joined with the risk ranking
    frontend/data/districts.geojson   study district outlines
    frontend/data/flood_*.png         flood overlays in EPSG:3857 (max extent + each pass)
    frontend/data/villages.csv        ranked list for download
    frontend/data/meta.json           event, scoring, accuracy, validation, case check, limitations
    outputs/web_alignment_check.png   village outlines over a flood PNG (visual check)

Every number the site shows comes from these files. If an input is missing, the
matching field is left out and the site shows "not available".
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "frontend" / "data"
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"

SIMPLIFY_VILLAGE_M = 20.0
SIMPLIFY_DISTRICT_M = 100.0
COORD_DECIMALS = 5
OVERLAY_TARGET_M = 80.0        # ground metres per overlay pixel (approx)
OVERLAY_RGB = (33, 102, 172)   # flooded pixel colour
OVERLAY_ALPHA = 200            # 0..255; the site also has an opacity slider
MAX_PNG_BYTES = 1_500_000

# Columns of results/village_risk_ranked.csv this script reads.
COMPONENTS = ["exposure", "duration", "exposed_people", "well_reliance", "low_elevation"]

WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print("WARNING:", msg)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def clean(obj):
    """Make an object strict-JSON safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def read_json(path: Path):
    if not path.exists():
        warn(f"missing {path.relative_to(ROOT)}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warn(f"could not parse {path.relative_to(ROOT)}: {exc}")
        return None


def get(d, *keys):
    """Nested dict lookup that returns None on any missing key."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def read_config_constants(names: list[str]) -> dict:
    """Read literal constants from config.py without importing it (no side effects)."""
    path = ROOT / "config.py"
    out: dict = {}
    if not path.exists():
        warn("missing config.py")
        return out
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                try:
                    out[name] = ast.literal_eval(node.value)
                except Exception:  # noqa: BLE001
                    pass
    return out


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_dirty_results() -> bool | None:
    """True if results/ has uncommitted changes (so the commit hash may not match)."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", "results"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def round_coords(geom, nd=COORD_DECIMALS):
    import shapely
    return shapely.transform(geom, lambda xy: np.round(xy, nd))


def fix_geom(geom):
    import shapely
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
        # keep polygonal parts only
        if geom.geom_type == "GeometryCollection":
            polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            geom = shapely.unary_union(polys) if polys else geom
    return geom


def write_geojson(gdf, path: Path) -> None:
    feats = []
    for _, row in gdf.iterrows():
        props = {k: clean(row[k]) for k in gdf.columns if k != "geometry"}
        geom = row.geometry.__geo_interface__ if row.geometry is not None else None
        feats.append({"type": "Feature", "properties": props, "geometry": geom})
    fc = {"type": "FeatureCollection", "features": feats}
    path.write_text(json.dumps(clean(fc), separators=(",", ":")), encoding="utf-8")


# ---------------------------------------------------------------------------
# villages
# ---------------------------------------------------------------------------

def read_ranked_csv(path: Path):
    """Read the ranked CSV, keeping the '# ...' header comment lines."""
    comments = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                comments.append(line[1:].strip())
            else:
                break
    df = pd.read_csv(path, comment="#")
    return df, comments


def unnamed_label(unit_id: str) -> str:
    m = re.search(r"(\d+)$", str(unit_id))
    return f"Unnamed village (OSM {m.group(1)})" if m else f"Unnamed village ({unit_id})"


def export_villages(meta: dict) -> tuple | None:
    import geopandas as gpd

    gpkg = OUTPUTS / "villages.gpkg"
    csv = RESULTS / "village_risk_ranked.csv"
    if not gpkg.exists() or not csv.exists():
        warn("villages.gpkg or village_risk_ranked.csv missing: no village layer")
        return None

    shapes = gpd.read_file(gpkg)
    ranked, comments = read_ranked_csv(csv)
    meta["ranking_file_header"] = comments

    # Column mapping: output property -> source column in village_risk_ranked.csv.
    mapping = {
        "id": "unit_id",
        "name": "name",
        "district": "district",
        "rank": "rank",
        "score": "score",
        "pct_flooded": "flooded_frac (x100)",
        "people_est": "exposed_pop (rounded to 10)",
        "duration_days": "max_duration_days",
    }
    for c in COMPONENTS:
        mapping[f"c_{c}"] = f"c_{c} (x100)"
    meta["column_mapping"] = mapping

    need = ["unit_id", "rank", "score"]
    for col in need:
        if col not in ranked.columns:
            warn(f"village_risk_ranked.csv has no '{col}' column: no village layer")
            return None

    shapes = shapes.rename(columns={"name": "gpkg_name", "district": "gpkg_district"})
    g = shapes.merge(ranked, on="unit_id", how="inner")
    if len(g) < len(ranked):
        warn(f"{len(ranked) - len(g)} ranked units have no shape in villages.gpkg")
    if len(g) < len(shapes):
        warn(f"{len(shapes) - len(g)} shapes in villages.gpkg have no ranking row")

    def pick_name(r):
        for col in ("name", "gpkg_name"):
            v = r.get(col)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return unnamed_label(r["unit_id"])

    def pick_district(r):
        for col in ("district", "gpkg_district"):
            v = r.get(col)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    out = gpd.GeoDataFrame(geometry=g.geometry, crs=shapes.crs)
    out["id"] = g["unit_id"].astype(str)
    out["name"] = g.apply(pick_name, axis=1)
    out["district"] = g.apply(pick_district, axis=1)
    out["rank"] = g["rank"].astype(int)
    out["score"] = g["score"].round(1)

    def opt(col, fn):
        if col in g.columns:
            return g[col].map(lambda v: None if pd.isna(v) else fn(v))
        warn(f"village_risk_ranked.csv has no '{col}' column: omitted")
        return None

    for prop, col, fn in [
        ("pct_flooded", "flooded_frac", lambda v: round(float(v) * 100, 1)),
        ("people_est", "exposed_pop", lambda v: int(round(float(v) / 10.0) * 10)),
        ("duration_days", "max_duration_days", lambda v: int(round(float(v)))),
    ] + [(f"c_{c}", f"c_{c}", lambda v: int(round(float(v) * 100))) for c in COMPONENTS]:
        s = opt(col, fn)
        if s is not None:
            out[prop] = s

    out = out.sort_values("rank").reset_index(drop=True)

    # Centroids in the projected CRS, before simplification.
    proj = out if out.crs and out.crs.is_projected else out.to_crs(3857)
    cent = gpd.GeoSeries(proj.geometry.centroid, crs=proj.crs).to_crs(4326)
    out["centroid_lat"] = cent.y.round(COORD_DECIMALS).values
    out["centroid_lon"] = cent.x.round(COORD_DECIMALS).values

    # Simplify in metres, then reproject and round.
    simp = proj.geometry.simplify(SIMPLIFY_VILLAGE_M, preserve_topology=True)
    geo = gpd.GeoSeries(simp, crs=proj.crs).to_crs(4326)
    out = gpd.GeoDataFrame(out.drop(columns="geometry"),
                           geometry=[fix_geom(round_coords(x)) for x in geo], crs=4326)

    write_geojson(out, OUT / "villages.geojson")

    # CSV for download
    csv_cols = {
        "rank": "rank", "name": "village", "district": "district", "score": "risk_score",
        "pct_flooded": "percent_area_flooded", "people_est": "people_in_flooded_area_estimate",
        "duration_days": "flood_duration_days_estimate",
        "centroid_lat": "centroid_lat", "centroid_lon": "centroid_lon",
    }
    keep = [c for c in csv_cols if c in out.columns]
    out[keep].rename(columns=csv_cols).to_csv(OUT / "villages.csv", index=False, encoding="utf-8")
    meta["csv_columns"] = {csv_cols[c]: c for c in keep}

    counts = out["district"].value_counts().sort_index()
    meta["village_counts"] = {
        "total": int(len(out)),
        "by_district": {k: int(v) for k, v in counts.items()},
        "unnamed": int(out["name"].str.startswith("Unnamed village").sum()),
    }
    if "duration_days" in out.columns:
        meta["duration_values_days"] = sorted({int(v) for v in out["duration_days"].dropna()})
    return out, proj


# ---------------------------------------------------------------------------
# districts
# ---------------------------------------------------------------------------

def export_districts(meta: dict):
    import geopandas as gpd
    path = OUTPUTS / "aoi_districts.gpkg"
    if not path.exists():
        warn("outputs/aoi_districts.gpkg missing: no district layer")
        return None
    d = gpd.read_file(path)
    proj = d if d.crs and d.crs.is_projected else d.to_crs(3857)
    area = (proj.geometry.area / 1e6).round(1)
    simp = proj.geometry.simplify(SIMPLIFY_DISTRICT_M, preserve_topology=True)
    geo = gpd.GeoSeries(simp, crs=proj.crs).to_crs(4326)
    name_col = "district" if "district" in d.columns else d.columns[0]
    out = gpd.GeoDataFrame({"district": d[name_col].astype(str), "area_km2": area.values},
                           geometry=[fix_geom(round_coords(x)) for x in geo], crs=4326)
    write_geojson(out, OUT / "districts.geojson")
    b = out.total_bounds
    meta["study_bounds"] = [[round(b[1], 5), round(b[0], 5)], [round(b[3], 5), round(b[2], 5)]]
    return out


# ---------------------------------------------------------------------------
# flood overlays
# ---------------------------------------------------------------------------

def flood_png(src_path: Path, out_name: str) -> dict | None:
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.warp import Resampling, reproject, transform_bounds

    with rasterio.open(src_path) as src:
        data = src.read(1)
        # Source: 1 = flood, 0 = not flood, 255 = no data. Recode to 2 = flood,
        # 1 = anything else, so 0 can serve as an unambiguous nodata value for the
        # warp (with 0 as a real value GDAL fills most of the image with 1).
        flooded = np.where(data == 1, 2, 1).astype(np.uint8)
        del data
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        lat_c = (south + north) / 2.0
        # Web Mercator metres per ground metre at this latitude.
        merc_scale = 1.0 / math.cos(math.radians(lat_c))
        native_ground = abs(src.transform.a)
        factor = max(1, int(round(OVERLAY_TARGET_M / native_ground)))
        res = native_ground * merc_scale
        mx0, my0, mx1, my1 = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        width = int(math.ceil((mx1 - mx0) / res / factor)) * factor
        height = int(math.ceil((my1 - my0) / res / factor)) * factor
        dst_transform = rasterio.Affine(res, 0, mx0, 0, -res, my1)
        dst = np.zeros((height, width), dtype=np.uint8)
        reproject(flooded, dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs="EPSG:3857",
                  resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        del flooded

    # Max-aggregate so thin flooded features survive; 1 = flooded, 0 = transparent.
    dst = (dst == 2).astype(np.uint8)
    agg = dst.reshape(height // factor, factor, width // factor, factor).max(axis=(1, 3))
    del dst

    agg = np.ascontiguousarray(agg)
    img = Image.frombytes("P", (agg.shape[1], agg.shape[0]), agg.tobytes())
    img.putpalette([0, 0, 0, *OVERLAY_RGB])
    out_path = OUT / out_name
    img.save(out_path, format="PNG", optimize=True,
             transparency=bytes([0, OVERLAY_ALPHA]))
    size = out_path.stat().st_size
    if size > MAX_PNG_BYTES:
        warn(f"{out_name} is {size} bytes, over the {MAX_PNG_BYTES} byte budget")

    # Bounds of the aggregated image in lat/lon.
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    ex = mx0 + agg.shape[1] * res * factor
    ey = my1 - agg.shape[0] * res * factor
    lon0, lat0 = t.transform(mx0, ey)
    lon1, lat1 = t.transform(ex, my1)
    return {
        "file": out_name,
        "bounds": [[round(lat0, 6), round(lon0, 6)], [round(lat1, 6), round(lon1, 6)]],
        "width": int(agg.shape[1]), "height": int(agg.shape[0]),
        "pixel_ground_m": round(native_ground * factor, 1),
        "flooded_pixels": int(agg.sum()),
        "bytes": size,
    }


def export_overlays(meta: dict, s1: dict | None) -> list:
    layers = []
    ext = OUTPUTS / "flood_extent.tif"
    if ext.exists():
        info = flood_png(ext, "flood_max.png")
        if info:
            info.update({"id": "max", "label": "Maximum extent (all passes)"})
            layers.append(info)
    else:
        warn("outputs/flood_extent.tif missing: no maximum extent overlay")

    orbit = get(s1, "aoi", "orbit") or {}
    track = None
    if orbit:
        parts = [orbit.get("orbit_state"), f"relative orbit {orbit['relative_orbit']}"
                 if orbit.get("relative_orbit") is not None else None]
        track = ", ".join(p for p in parts if p) or None

    pass_dir = OUTPUTS / "flood_by_pass"
    for tif in sorted(pass_dir.glob("*.tif")) if pass_dir.exists() else []:
        date = tif.stem
        info = flood_png(tif, f"flood_{date}.png")
        if info:
            info.update({"id": date, "label": f"Pass {date}", "date": date})
            if track:
                info["track"] = track
            layers.append(info)
    meta["flood_layers"] = layers
    return layers


def alignment_check(villages_proj, layer: dict) -> None:
    """Draw village outlines (in EPSG:3857) over a flood PNG to check alignment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    (lat0, lon0), (lat1, lon1) = layer["bounds"]
    x0, y0 = t.transform(lon0, lat0)
    x1, y1 = t.transform(lon1, lat1)
    img = np.array(Image.open(OUT / layer["file"]).convert("RGBA"))
    v = villages_proj.to_crs(3857)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=110)
    for ax, zoom in zip(axes, [False, True]):
        ax.set_facecolor("#f4f1ea")
        ax.imshow(img, extent=(x0, x1, y0, y1), origin="upper", interpolation="nearest")
        v.boundary.plot(ax=ax, color="#d95f02", linewidth=0.8 if zoom else 0.5)
        if zoom:
            top = v.sort_values("rank").iloc[0]
            cx, cy = top.geometry.centroid.x, top.geometry.centroid.y
            r = 6000
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_title(f"Zoom on rank 1 ({top['name']}), 12 km window")
        else:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_title(f"Village outlines over {layer['file']} (EPSG:3857)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out = OUTPUTS / "web_alignment_check.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------

def model1_block(m1: dict | None) -> dict | None:
    if not m1:
        return None
    groups = {}
    for key, label in [("test_all", "Test tiles, all countries"),
                       ("test_india", "Test tiles, India only"),
                       ("bolivia", "Bolivia holdout (country not seen in training)")]:
        g = get(m1, "groups", key)
        if not g:
            continue
        groups[key] = {
            "label": label,
            "n_tiles": g.get("n_tiles"),
            "unet": {k: get(g, "unet", k) for k in ("iou", "precision", "recall")},
            "otsu": {k: get(g, "otsu", k) for k in ("iou", "precision", "recall")},
        }
    return {"source": "results/model1_metrics.json", "groups": groups,
            "generated_at": m1.get("generated_at"),
            "iou_definition": m1.get("iou_definition")} if groups else None


def optical_block(sc: dict | None) -> dict | None:
    if not sc:
        return None
    s2 = sc.get("sentinel2_optical") or {}
    jrc = sc.get("jrc_gsw_normal_water") or {}
    out = {"source": "results/sanity_check.json"}
    if s2:
        out["sentinel2"] = {
            "status": s2.get("status"),
            "flood_pass": s2.get("flood_pass"),
            "s2_date": s2.get("s2_date"),
            "days_apart": s2.get("days_apart"),
            "clear_fraction": s2.get("clear_fraction"),
            "metrics": {k: get(s2, "metrics_model_vs_s2", k)
                        for k in ("iou", "precision", "recall", "agreement")},
        }
    if jrc:
        out["jrc_normal_water"] = {
            "status": jrc.get("status"),
            "occurrence_threshold": jrc.get("occurrence_threshold"),
            "metrics": {k: get(jrc, "metrics_normal_water_vs_gsw", k)
                        for k in ("iou", "precision", "recall", "agreement")},
        }
    return out


def validation_block(val: dict | None, risk_meta: dict | None) -> dict | None:
    if not val:
        return None
    dres = val.get("district_results") or []
    computable = [r for r in dres if r.get("rho") is not None and not (
        isinstance(r.get("rho"), float) and math.isnan(r["rho"]))]
    notes = sorted({r.get("note") for r in dres if r.get("note")})
    mapped = val.get("mapped_districts") or []
    week = [{"method": r.get("method"), "n_rows": r.get("n_rows"),
             "n_positive": r.get("n_positive"), "auc": r.get("auc")}
            for r in (val.get("district_week_results") or [])]

    # Plain statement built only from fields in validation.json.
    s = []
    if val.get("status") == "DONE":
        s.append("The ranking was compared with reported disease outbreaks (IDSP), but the "
                 "comparison is too small to show whether the score works.")
    else:
        s.append("The comparison with reported disease outbreaks (IDSP) has not been completed "
                 f"(status: {val.get('status')}).")
    if mapped:
        s.append(f"Only {len(mapped)} {'district was' if len(mapped) == 1 else 'districts were'} "
                 f"mapped ({', '.join(mapped)}).")
    if dres and not computable:
        s.append("The district-level correlation between the score and outbreaks could not be "
                 "computed" + (f" ({notes[0]})." if notes else "."))
    if week:
        n_rows = week[0].get("n_rows")
        n_pos = week[0].get("n_positive")
        s.append(f"A small district-week check used {n_rows} district-weeks, of which {n_pos} "
                 "had an outbreak in the following weeks. With this few cases the result is "
                 "not evidence that the score predicts outbreaks.")
    s.append("The ranking has not been checked against outbreaks or well tests at village level.")

    stale = None
    vt, rt = val.get("generated_at"), get(risk_meta, "generated_at")
    if vt and rt and vt < rt:
        stale = ("The validation was run before the latest version of the ranking "
                 f"(validation {vt[:16]}, ranking {rt[:16]}), so its numbers refer to an "
                 "earlier scoring.")
        s.append(stale)

    return {
        "source": "results/validation.json",
        "status": val.get("status"),
        "statement": " ".join(s),
        "mapped_districts": mapped,
        "outcome_window": val.get("outcome_window"),
        "district_correlation_computable": bool(computable),
        "district_week": week,
        "learned_model": get(val, "learned_model", "status"),
        "report_limitations": val.get("limitations") or [],
        "generated_at": vt,
        "stale_vs_ranking": stale,
    }


CASE_HEAD_RE = re.compile(r"^### (?P<name>.+?) \((?P<district>[^;]+); (?P<cases>\d+) cases from "
                          r"(?P<onset>\d{4}-\d{2}-\d{2})\)\s*$", re.M)
CASE_RANK_RE = re.compile(r"As a \d+th unit: risk score \*\*(?P<score>[\d.]+)\*\*, rank "
                          r"\*\*(?P<rank>\d+) / (?P<n>\d+)\*\*")
CASE_POINT_RE = re.compile(r"^- Point [-\d.]+, [-\d.]+ \((?P<src>[^)]*)\)", re.M)
CASE_NOCOORD_RE = re.compile(r"\*\*Not checked:\*\* no coordinates")
CASE_GEN_RE = re.compile(r"Generated by risk_checks\.py on (\d{4}-\d{2}-\d{2} \d{2}:\d{2})")


def case_check_block(meta: dict, risk_meta: dict | None) -> dict | None:
    """Plain case-check line built only from results/risk_checks.md (section 1).

    Every pattern must match; anything unexpected skips the block with a warning
    instead of guessing.
    """
    path = RESULTS / "risk_checks.md"
    if not path.exists():
        warn("missing results/risk_checks.md: no case check line")
        return None
    text = path.read_text(encoding="utf-8")
    sec = re.search(r"^## 1\. Case check.*?(?=^## 2\.|\Z)", text, re.S | re.M)
    gen = CASE_GEN_RE.search(text)
    heads = list(CASE_HEAD_RE.finditer(sec.group(0))) if sec else []
    if not sec or not gen or not heads:
        warn("could not parse the case check in results/risk_checks.md: no case check line")
        return None
    rt = get(risk_meta, "generated_at")
    if rt and gen.group(1) < rt[:16].replace("T", " "):
        warn("results/risk_checks.md is older than the ranking: no case check line")
        return None

    body = sec.group(0)
    points = []
    for i, h in enumerate(heads):
        chunk = body[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(body)]
        p = {"name": h["name"], "district": h["district"], "cases": int(h["cases"]),
             "onset": h["onset"]}
        r, nc = CASE_RANK_RE.search(chunk), CASE_NOCOORD_RE.search(chunk)
        if r and not nc:
            pt = CASE_POINT_RE.search(chunk)
            p.update(located=True, rank=int(r["rank"]), n_units=int(r["n"]), score=float(r["score"]),
                     approximate=bool(pt and "approximate" in pt["src"].lower()))
        elif nc and not r:
            p["located"] = False
        else:
            warn(f"case check for {h['name']} not in the expected format: no case check line")
            return None
        points.append(p)

    located = [p for p in points if p["located"]]
    missing = [p for p in points if not p["located"]]
    pass_dates = sorted(l["date"] for l in meta.get("flood_layers", []) if l.get("date"))
    last_pass = dt.date.fromisoformat(pass_dates[-1]) if pass_dates else None

    s = []
    if len(located) == 1:
        p = located[0]
        s.append(f"Case check: {p['name']}, the one reported outbreak village we could locate, "
                 f"ranked about {p['rank']} of {p['n_units']}.")
        detail = "one approximate point" if p["approximate"] else "one point"
        if last_pass:
            days = (dt.date.fromisoformat(p["onset"]) - last_pass).days
            if days >= 0:
                detail += f", with the outbreak recorded {days} days after the last radar pass"
        s.append(f"That is {detail}, so it is a case check, not a statistical test.")
    elif located:
        s.append("Case check: " + "; ".join(f"{p['name']} ranked about {p['rank']} of {p['n_units']}"
                                           for p in located) + ". This is a case check, not a statistical test.")
    else:
        s.append("Case check: no reported outbreak village could be located yet.")
    if missing:
        names = [p["name"] for p in missing]
        s.append(f"{' and '.join(names)} {'has' if len(names) == 1 else 'have'} no coordinates yet.")

    return {"source": "results/risk_checks.md", "generated_at": gen.group(1),
            "statement": " ".join(s), "points": points}


def status_warnings() -> list[str]:
    path = RESULTS / "STATUS.md"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    m = re.search(r"## Warnings\s*\n(.*?)(\n## |\Z)", text, re.S)
    if not m:
        return []
    return [re.sub(r"\*\*", "", ln[2:]).strip() for ln in m.group(1).splitlines()
            if ln.startswith("- ")]


def build_limitations(meta: dict, flood_summary: dict | None) -> list[str]:
    L = []
    passes = [l for l in meta.get("flood_layers", []) if l.get("date")]
    if passes:
        dates = ", ".join(p["date"] for p in passes)
        L.append(f"Few satellite passes: the flood was seen on only {len(passes)} radar "
                 f"{'pass' if len(passes) == 1 else 'passes'} ({dates}). Water that came and went between passes is missed.")
    vc = meta.get("village_counts") or {}
    if vc:
        per = ", ".join(f"{k} {v}" for k, v in vc.get("by_district", {}).items())
        L.append(f"Only {vc.get('total')} villages are included ({per}): the ones mapped as "
                 "village or hamlet points in OpenStreetMap. Villages missing from "
                 "OpenStreetMap are not on this map.")
        if vc.get("unnamed"):
            L.append(f"{vc['unnamed']} of these villages have no name in OpenStreetMap and are "
                     "shown by their OpenStreetMap id.")
    ev = meta.get("event") or {}
    if ev.get("flood_window"):
        L.append(f"One flood event only ({ev['flood_window'][0]} to {ev['flood_window'][1]}). "
                 "The ranking says nothing about other years or other districts.")
    rev = get(flood_summary, "revisit_interval_days")
    dv = meta.get("duration_values_days")
    if rev:
        txt = (f"Flood duration is a rough estimate. It counts the days between flooded passes "
               f"plus one revisit interval ({rev} days)")
        if dv:
            txt += f", so it can only take a few values (here: {', '.join(str(x) for x in dv)} days)"
        L.append(txt + ".")
    L.append("No outbreak validation: the score has not been shown to predict disease "
             "outbreaks or contaminated wells.")
    L.append("Village shapes are not official boundaries. A real village can be split or "
             "merged.")
    L.append("Flooding is measured against water seen in the same weeks of earlier years "
             "(normal water). Areas that flood every monsoon may be undercounted.")
    L.append("People in the flooded area is modelled from WorldPop population estimates, not "
             "counted.")
    return L


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = read_config_constants(["DISTRICTS", "FLOOD_START", "FLOOD_END", "STATE_NAME",
                                 "FLOOD_YEAR", "NORMAL_YEARS", "RISK_WEIGHTS",
                                 "WELL_RELIANCE_SCORING", "VILLAGE_MAX_RADIUS_M"])
    s1 = read_json(RESULTS / "export_s1_summary.json")
    fsum = read_json(RESULTS / "flood_map_summary.json")
    risk_meta = read_json(RESULTS / "risk_meta.json")
    m1 = read_json(RESULTS / "model1_metrics.json")
    sc = read_json(RESULTS / "sanity_check.json")
    val = read_json(RESULTS / "validation.json")

    meta: dict = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "results_uncommitted_changes": git_dirty_results(),
    }

    window = get(s1, "aoi", "flood_window") or (
        [cfg["FLOOD_START"], cfg["FLOOD_END"]] if "FLOOD_START" in cfg and "FLOOD_END" in cfg else None)
    meta["event"] = clean({
        "state": cfg.get("STATE_NAME"),
        "districts": get(s1, "aoi", "districts") or cfg.get("DISTRICTS"),
        "flood_window": window,
        "normal_years": cfg.get("NORMAL_YEARS"),
        "normal_dates": get(s1, "aoi", "normal_dates") or get(fsum, "normal_dates"),
        "satellite": "Sentinel-1 radar" + (f" ({get(s1, 'aoi', 'collection')})"
                                           if get(s1, "aoi", "collection") else ""),
        "district_boundary_source": get(s1, "aoi", "district_source"),
        "revisit_interval_days": get(fsum, "revisit_interval_days"),
        "flooded_km2_by_district": {k: round(v["max_extent_flooded_km2"], 1)
                                    for k, v in (get(fsum, "per_district") or {}).items()
                                    if v.get("max_extent_flooded_km2") is not None} or None,
    })

    print("villages ...")
    vres = export_villages(meta)
    print("districts ...")
    export_districts(meta)
    print("flood overlays ...")
    layers = export_overlays(meta, s1)
    meta["passes"] = [{"date": l["date"], **({"track": l["track"]} if l.get("track") else {})}
                      for l in layers if l.get("date")]

    if vres is not None and layers:
        alignment_check(vres[1].assign(name=vres[0]["name"].values, rank=vres[0]["rank"].values),
                        layers[0])

    weights = get(risk_meta, "weights_used") or cfg.get("RISK_WEIGHTS")
    meta["scoring"] = clean({
        "weights": weights,
        "components_used": list((get(risk_meta, "components") or {}).keys()) or None,
        "component_columns": get(risk_meta, "components"),
        "components_dropped": get(risk_meta, "dropped"),
        "well_reliance_scoring": cfg.get("WELL_RELIANCE_SCORING"),
        "unit_source": get(risk_meta, "unit_source"),
        "village_radius_m": cfg.get("VILLAGE_MAX_RADIUS_M"),
        "n_units": get(risk_meta, "n_units"),
        "generated_at": get(risk_meta, "generated_at"),
    })
    meta["model1"] = clean(model1_block(m1))
    meta["optical_check"] = clean(optical_block(sc))
    meta["validation"] = clean(validation_block(val, risk_meta))
    meta["case_check"] = clean(case_check_block(meta, risk_meta))
    meta["pipeline_warnings"] = status_warnings()
    meta["limitations"] = build_limitations(meta, fsum)
    meta["export_warnings"] = WARNINGS
    meta = {k: v for k, v in clean(meta).items() if v is not None}

    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nColumn mapping (site property <- village_risk_ranked.csv column):")
    for k, v in (meta.get("column_mapping") or {}).items():
        print(f"  {k:15s} <- {v}")
    print("\nFiles in frontend/data:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1024:9.1f} KB")
    if WARNINGS:
        print(f"\n{len(WARNINGS)} warning(s); see meta.json export_warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
