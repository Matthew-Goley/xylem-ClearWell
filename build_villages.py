"""
build_villages.py - MODEL 2: village proxies and their feature table.

Census 2011 village boundaries are not freely downloadable, so this script
builds VILLAGE PROXIES:
  * OSM place=village / place=hamlet points -> Voronoi cells, each clipped to
    its district and to VILLAGE_MAX_RADIUS_M around its point, or
  * if OSM gives fewer than MIN_OSM_POINTS points: a 1 km grid over populated
    areas (or the whole district if no population data).

Features per village proxy (zonal statistics on the Model 1 outputs plus
open data): flooded_frac, max/mean_duration_days, first_flood_date,
per-pass columns (flooded_frac_YYYYMMDD, cum_flooded_frac_YYYYMMDD,
max_duration_so_far_YYYYMMDD, exposed_pop_YYYYMMDD), pop and exposed_pop
(WorldPop), buildings and buildings_flooded (Google Open Buildings v3),
elev_mean_m and elev_pctile (Copernicus DEM), pct_shallow_well_handpump
(Census 2011, district level, only if you supply the table).

Outputs: outputs/village_features.csv, outputs/villages.gpkg,
outputs/village_features_meta.json, results/village_feature_summary.md

Run alone:  python build_villages.py [--smoke]
"""

import gzip
import hashlib
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import common
import config

STAGES_IN_THIS_FILE = ["villages"]
NODATA_U8 = 255
NODATA_U16 = 65535


def datecol(prefix, day):
    """Column name for a per-pass feature, e.g. flooded_frac_20200701."""
    return f"{prefix}_{day.replace('-', '')}"


def bbox_key(bbox):
    return hashlib.md5(",".join(f"{v:.4f}" for v in bbox).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Village proxy units
# ---------------------------------------------------------------------------
def fetch_osm_places(paths, bbox_lonlat, logger):
    """OSM place=village/hamlet as a list of dicts (cached in data/cache/)."""
    cache = paths.cache / f"osm_places_{bbox_key(bbox_lonlat)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    w, s, e, n = bbox_lonlat
    query = (f'[out:json][timeout:300];nwr["place"~"^(village|hamlet)$"]({s},{w},{n},{e});'
             "out center tags;")
    session = common.http_session()
    last_err = None
    for url in config.OVERPASS_URLS:
        try:
            def _post(url=url):
                resp = session.post(url, data={"data": query}, timeout=360)
                resp.raise_for_status()
                return resp.json()
            data = common.retry(_post, what=f"Overpass {url}", logger=logger, wait_s=20)
            places = []
            for el in data.get("elements", []):
                lat = el.get("lat", el.get("center", {}).get("lat"))
                lon = el.get("lon", el.get("center", {}).get("lon"))
                if lat is None or lon is None:
                    continue
                tags = el.get("tags", {})
                places.append({"osm_id": f"{el['type']}/{el['id']}", "lat": lat, "lon": lon,
                               "name": tags.get("name:en") or tags.get("name") or "",
                               "place": tags.get("place", "")})
            cache.write_text(json.dumps(places))
            logger.info(f"OSM: {len(places)} village/hamlet places from {url}")
            return places
        except Exception as err:  # noqa: BLE001
            last_err = err
            logger.warning(f"Overpass {url} failed: {err}")
    raise RuntimeError(f"all Overpass servers failed: {last_err}")


def voronoi_units(points, districts_utm, max_radius):
    """Voronoi cells around points, clipped to the point's district and a radius.

    points: GeoDataFrame (UTM) with osm_id, name, place. Returns a GeoDataFrame
    with unit_id, name, place, district, geometry.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram
    pts = gpd.sjoin(points, districts_utm[["district", "geometry"]], predicate="within", how="inner")
    pts = pts.drop(columns="index_right")
    pts = pts[~pts.geometry.apply(lambda g: (round(g.x, 1), round(g.y, 1))).duplicated()]
    pts = pts.reset_index(drop=True)
    if len(pts) < 2:
        raise ValueError("need at least 2 points for Voronoi cells")
    envelope = districts_utm.geometry.union_all().buffer(max_radius * 2).envelope
    cells = voronoi_diagram(MultiPoint(list(pts.geometry)), envelope=envelope)
    cells = gpd.GeoDataFrame(geometry=list(cells.geoms), crs=points.crs)
    joined = gpd.sjoin(pts, cells, predicate="within", how="left")
    district_geom = dict(zip(districts_utm["district"], districts_utm.geometry))
    rows = []
    for _, row in joined.iterrows():
        if pd.isna(row["index_right"]):
            continue
        cell = cells.geometry.iloc[int(row["index_right"])]
        geom = cell.intersection(district_geom[row["district"]]).intersection(
            row.geometry.buffer(max_radius))
        if geom.is_empty or geom.area == 0:
            continue
        rows.append({"unit_id": f"osm_{row['osm_id'].replace('/', '_')}", "name": row["name"],
                     "place": row["place"], "district": row["district"], "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=points.crs)


def grid_units(districts_utm, cell_m, populated_xy=None):
    """Square grid cells clipped to each district. If populated_xy (x, y arrays)
    is given, only cells containing at least one populated point are kept."""
    import geopandas as gpd
    from shapely.geometry import box
    rows = []
    for d in districts_utm.itertuples():
        xmin, ymin, xmax, ymax = d.geometry.bounds
        x0, y0 = np.floor(xmin / cell_m) * cell_m, np.floor(ymin / cell_m) * cell_m
        occupied = None
        if populated_xy is not None:
            cols = np.floor((populated_xy[0] - x0) / cell_m).astype(int)
            rows_ = np.floor((populated_xy[1] - y0) / cell_m).astype(int)
            occupied = set(zip(rows_.tolist(), cols.tolist()))
        n_rows = int(np.ceil((ymax - y0) / cell_m))
        n_cols = int(np.ceil((xmax - x0) / cell_m))
        for r in range(n_rows):
            for c in range(n_cols):
                if occupied is not None and (r, c) not in occupied:
                    continue
                cell = box(x0 + c * cell_m, y0 + r * cell_m, x0 + (c + 1) * cell_m, y0 + (r + 1) * cell_m)
                if not cell.intersects(d.geometry):
                    continue
                geom = cell.intersection(d.geometry)
                if geom.area < 0.05 * cell_m * cell_m:
                    continue
                rows.append({"unit_id": f"grid_{common.clean_name(d.district).replace(' ', '-')}_{r}_{c}",
                             "name": "", "place": "grid", "district": d.district, "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=districts_utm.crs)


# ---------------------------------------------------------------------------
# Population (WorldPop)
# ---------------------------------------------------------------------------
def get_worldpop(paths, logger):
    """Path to a WorldPop 2020 100 m GeoTIFF for India, or None. Source string too."""
    manual = paths.manual / "worldpop.tif"
    if manual.exists():
        return manual, "data/manual/worldpop.tif (supplied by hand)"
    folder = paths.cache / "worldpop"
    existing = sorted(folder.glob("*.tif")) if folder.exists() else []
    if existing:
        return existing[0], f"cached {existing[0].name}"
    urls = []
    session = common.http_session()
    for alias in config.WORLDPOP_ALIASES:
        try:
            resp = session.get(config.WORLDPOP_API.format(alias=alias), timeout=60)
            resp.raise_for_status()
            entries = resp.json().get("data", [])
            entries = sorted(entries, key=lambda e: str(e.get("popyear", "")) == "2020", reverse=True)
            for e in entries:
                urls += [u for u in e.get("files", []) if u.lower().endswith(".tif")]
            logger.info(f"WorldPop API '{alias}': {len(entries)} entries")
        except Exception as err:  # noqa: BLE001
            logger.warning(f"WorldPop API '{alias}' failed: {err}")
    urls += config.WORLDPOP_DIRECT_URLS
    for url in urls:
        try:
            dest = folder / url.split("/")[-1]
            common.download_file(url, dest, logger, timeout=120)
            return dest, url
        except Exception as err:  # noqa: BLE001
            logger.warning(f"WorldPop download {url} failed: {err}")
    return None, None


def population_points(worldpop_path, aoi):
    """WorldPop pixel centres inside the AOI as UTM x, y and population arrays."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window, from_bounds
    crs, _, _, _ = common.aoi_grid(aoi)
    with rasterio.open(worldpop_path) as src:
        win = from_bounds(*aoi["bbox_lonlat"], transform=src.transform).round_offsets().round_lengths()
        # Keep only the part of the window that lies inside the raster.
        win = win.intersection(Window(0, 0, src.width, src.height))
        pop = src.read(1, window=win).astype(np.float64)
        wt = src.window_transform(win)
        nodata = src.nodata
    rows, cols = np.nonzero(np.isfinite(pop) & (pop > 0) & (pop != nodata if nodata is not None else True))
    lon, lat = rasterio.transform.xy(wt, rows, cols)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_utm.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y), pop[rows, cols]


def points_to_pixels(x, y, transform, width, height):
    """UTM coordinates -> (row, col) on the grid and a mask of points inside it."""
    row, col = common.xy_to_rowcol(transform, x, y)
    row, col = np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)
    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    return row, col, inside


def neighbourhood_fraction(mask, factor):
    """Share of True pixels in each factor x factor block (padded at the edges)."""
    h, w = mask.shape
    ph, pw = (-h) % factor, (-w) % factor
    padded = np.pad(mask.astype(np.float32), ((0, ph), (0, pw)))
    return common.block_reduce(padded, factor, "mean")


# ---------------------------------------------------------------------------
# Buildings (Google Open Buildings v3)
# ---------------------------------------------------------------------------
def s2_tokens_for_bbox(bbox_lonlat, level=4):
    import s2sphere
    w, s, e, n = bbox_lonlat
    rect = s2sphere.LatLngRect.from_point_pair(s2sphere.LatLng.from_degrees(s, w),
                                               s2sphere.LatLng.from_degrees(n, e))
    coverer = s2sphere.RegionCoverer()
    coverer.min_level = coverer.max_level = level
    return sorted({c.to_token() for c in coverer.get_covering(rect)})


def get_buildings(paths, aoi, logger, smoke):
    """DataFrame of building centroids (lon, lat) with confidence >= threshold, or (None, reason)."""
    bbox = aoi["bbox_lonlat"]
    filtered = paths.cache / "open_buildings" / f"aoi_{bbox_key(bbox)}.csv.gz"
    if filtered.exists():
        return pd.read_csv(filtered), "cached"
    tokens = s2_tokens_for_bbox(bbox)
    kind = config.OPEN_BUILDINGS_KIND
    urls = {t: config.OPEN_BUILDINGS_URL.format(kind=kind, token=t) for t in tokens}
    session = common.http_session()
    total = 0
    for t, url in urls.items():
        resp = session.head(url, timeout=60)
        if resp.status_code == 404:
            logger.info(f"Open Buildings: no file for S2 cell {t}")
            urls[t] = None
            continue
        resp.raise_for_status()
        total += int(resp.headers.get("Content-Length", 0))
    urls = {t: u for t, u in urls.items() if u}
    logger.info(f"Open Buildings: S2 cells {list(urls)}, {total / 1e9:.2f} GB to read")
    if total > config.OPEN_BUILDINGS_MAX_GB * 1e9:
        return None, f"files total {total / 1e9:.1f} GB > OPEN_BUILDINGS_MAX_GB"
    w, s, e, n = bbox
    parts = []
    for t, url in urls.items():
        dest = paths.cache / "open_buildings" / f"{t}_{kind}.csv.gz"
        if smoke and not dest.exists():
            return None, (f"smoke mode does not download the {total / 1e9:.1f} GB building file "
                          "(run the full stage, or place it in data/cache/open_buildings/)")
        common.download_file(url, dest, logger, timeout=120)
        with gzip.open(dest, "rt") as fh:
            for chunk in pd.read_csv(fh, usecols=["latitude", "longitude", "confidence"],
                                     chunksize=2_000_000):
                keep = ((chunk["confidence"] >= config.OPEN_BUILDINGS_MIN_CONFIDENCE)
                        & chunk["longitude"].between(w, e) & chunk["latitude"].between(s, n))
                parts.append(chunk[keep])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["latitude", "longitude", "confidence"])
    df.to_csv(filtered, index=False)
    logger.info(f"Open Buildings: {len(df)} buildings with confidence >= "
                f"{config.OPEN_BUILDINGS_MIN_CONFIDENCE} in the AOI")
    return df, ", ".join(urls.values())


# ---------------------------------------------------------------------------
# Census 2011 drinking water (manual table)
# ---------------------------------------------------------------------------
def census_well_reliance(paths, districts, logger):
    """{district: pct_shallow_well_handpump} from data/manual/census_water.csv, or None.

    Never guessed: if the file is missing, a template is written and the
    feature is left out of the index.
    """
    path = paths.manual / config.CENSUS_WATER_FILE
    if not path.exists():
        template = paths.manual / config.CENSUS_WATER_TEMPLATE
        pd.DataFrame({"district": districts, "pct_shallow_well_handpump": [""] * len(districts)}
                     ).to_csv(template, index=False)
        logger.warning(f"{path} not found. Wrote {template}: fill it from Census 2011 houselisting "
                       "tables (main source of drinking water: hand pump + tube well + covered "
                       "and uncovered well, % of households), save it as "
                       f"{config.CENSUS_WATER_FILE}. Until then the feature is EXCLUDED.")
        return None
    table = pd.read_csv(path)
    table.columns = [c.strip().lower() for c in table.columns]
    if not {"district", "pct_shallow_well_handpump"} <= set(table.columns):
        logger.warning(f"{path} needs columns district, pct_shallow_well_handpump; feature excluded")
        return None
    table = table.dropna(subset=["pct_shallow_well_handpump"])
    matches = common.match_names(table["district"].tolist(), districts)
    out = {}
    for m, value in zip(matches, table["pct_shallow_well_handpump"]):
        if m["match"] is not None:
            out[m["match"]] = float(value)
    missing = [d for d in districts if d not in out]
    if missing:
        logger.warning(f"census table has no value for {missing}; those units get NaN")
    return out or None


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------
def unit_index_raster(units, transform, width, height):
    """int32 raster: 0 = no unit, i + 1 = units.iloc[i]."""
    from rasterio import features
    shapes = ((geom, i + 1) for i, geom in enumerate(units.geometry))
    return features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=0,
                              dtype="int32")


def count_per_unit(unit_idx, mask, n_units):
    """Number of True pixels of mask inside each unit (array of length n_units)."""
    return np.bincount(unit_idx[mask], minlength=n_units + 1)[1:].astype(np.float64)


def max_per_unit(values, unit_idx, n_units):
    from scipy import ndimage
    if n_units == 0:
        return np.zeros(0)
    return np.asarray(ndimage.maximum(values, labels=unit_idx, index=np.arange(1, n_units + 1)),
                      dtype=np.float64)


def safe_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def flood_features(paths, aoi, unit_idx, n_units, pop_pixels=None, bld_pixels=None):
    """Flood features per unit, including per-pass columns.

    pop_pixels / bld_pixels: optional (row, col, weight, unit) arrays for
    WorldPop centres and buildings, used for exposed people per pass.
    """
    import rasterio

    from flood_map import iterate_flood_passes
    feats = {}
    px_per_unit = count_per_unit(unit_idx, unit_idx > 0, n_units)
    feats["n_pixels"] = px_per_unit
    factor = max(1, int(round(100 / aoi["grid"]["pixel_size_m"])))  # ~100 m neighbourhood
    tracker = None
    for day, flood, tracker in iterate_flood_passes(paths, aoi):
        observed = count_per_unit(unit_idx, flood != NODATA_U8, n_units)
        feats[datecol("flooded_frac", day)] = safe_div(count_per_unit(unit_idx, flood == 1, n_units),
                                                       observed)
        cum_obs = count_per_unit(unit_idx, tracker.ever_valid, n_units)
        feats[datecol("cum_flooded_frac", day)] = safe_div(
            count_per_unit(unit_idx, tracker.ever_flooded, n_units), cum_obs)
        feats[datecol("max_duration_so_far", day)] = max_per_unit(tracker.longest, unit_idx, n_units)
        if pop_pixels is not None:
            frac = neighbourhood_fraction(tracker.ever_flooded, factor)
            r, c, w, u = pop_pixels
            feats[datecol("exposed_pop", day)] = np.bincount(
                u, weights=w * frac[r // factor, c // factor], minlength=n_units + 1)[1:]
        if bld_pixels is not None:
            r, c, w, u = bld_pixels
            feats[datecol("buildings_flooded", day)] = np.bincount(
                u, weights=tracker.ever_flooded[r, c].astype(float), minlength=n_units + 1)[1:]
    if tracker is None:
        raise RuntimeError("no flood_by_pass rasters found; run stage flood_map first")
    last = sorted(aoi["flood_dates"])[-1]
    # Final values = state after the last pass (identical to flood_extent.tif / flood_duration.tif).
    feats["observed_frac"] = safe_div(count_per_unit(unit_idx, tracker.ever_valid, n_units), px_per_unit)
    feats["flooded_frac"] = feats[datecol("cum_flooded_frac", last)]
    feats["max_duration_days"] = feats[datecol("max_duration_so_far", last)]
    with rasterio.open(paths.outputs / "flood_duration.tif") as src:
        duration = src.read(1)
    flooded = tracker.ever_flooded
    dur_sum = np.bincount(unit_idx[flooded], weights=duration[flooded].astype(float),
                          minlength=n_units + 1)[1:]
    feats["mean_duration_days"] = np.nan_to_num(safe_div(dur_sum, count_per_unit(unit_idx, flooded, n_units)))
    if pop_pixels is not None:
        feats["exposed_pop"] = feats[datecol("exposed_pop", last)]
    if bld_pixels is not None:
        feats["buildings_flooded"] = feats[datecol("buildings_flooded", last)]
    return pd.DataFrame(feats)


def first_flood_dates(df, dates):
    """First pass date where the unit is at least FIRST_FLOOD_MIN_FRAC flooded ('' if never)."""
    out = []
    for _, row in df.iterrows():
        first = ""
        for day in sorted(dates):
            v = row.get(datecol("flooded_frac", day))
            if v is not None and np.isfinite(v) and v >= config.FIRST_FLOOD_MIN_FRAC:
                first = day
                break
        out.append(first)
    return out


def summary_markdown(df, meta, smoke):
    lines = ["# Village proxy feature summary", "",
             "Generated by build_villages.py. Units are VILLAGE PROXIES (OSM Voronoi cells or a "
             "1 km grid), not census villages." + (" SMOKE RUN." if smoke else ""), "",
             f"- unit source: {meta['unit_source']}", f"- number of village proxies: {len(df)}", ""]
    lines += ["## Units per district", "", "| District | Village proxies | Mean area (km2) |",
              "|---|---|---|"]
    for d, g in df.groupby("district"):
        lines.append(f"| {d} | {len(g)} | {g['area_km2'].mean():.2f} |")
    lines += ["", "## Feature availability", "", "| Feature | Status | Source / reason |", "|---|---|---|"]
    for name, info in meta["features"].items():
        lines.append(f"| {name} | {'available' if info['available'] else 'UNAVAILABLE'} | "
                     f"{str(info.get('source', '')).replace('|', '/')} |")
    core = ["flooded_frac", "max_duration_days", "mean_duration_days", "first_flood_date", "pop",
            "exposed_pop", "buildings", "buildings_flooded", "elev_mean_m", "elev_pctile",
            "pct_shallow_well_handpump"]
    lines += ["", "## Missing values per feature", "", "| Feature | Missing (share of units) |", "|---|---|"]
    for c in core:
        if c in df.columns:
            miss = df[c].replace("", np.nan).isna().mean()
            lines.append(f"| {c} | {miss:.3f} |")
        else:
            lines.append(f"| {c} | column absent (feature unavailable) |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stage villages
# ---------------------------------------------------------------------------
def run_villages(smoke=False, logger=None, **kwargs):
    import geopandas as gpd
    import rasterio
    from pyproj import Transformer

    paths = config.get_paths(smoke)
    aoi = common.load_aoi(paths)
    crs, transform, width, height = common.aoi_grid(aoi)
    px_km2 = common.pixel_area_km2(aoi)
    districts_utm = common.districts_for_aoi(paths, aoi)
    warnings = []
    meta = {"features": {}, "generated_at": datetime.now().isoformat(), "smoke": smoke}

    # 1) population first (also used to find populated areas for the grid fallback)
    pop_xy = None
    wp_path, wp_source = get_worldpop(paths, logger)
    if wp_path is not None:
        try:
            px_x, px_y, pop_vals = population_points(wp_path, aoi)
            pop_xy = (px_x, px_y, pop_vals)
            meta["features"]["population"] = {"available": True, "source": f"WorldPop: {wp_source}"}
        except Exception as err:  # noqa: BLE001
            meta["features"]["population"] = {"available": False, "source": f"WorldPop read failed: {err}"}
    else:
        meta["features"]["population"] = {"available": False, "source": "WorldPop could not be downloaded"}
    if not meta["features"]["population"]["available"]:
        warnings.append("population unavailable: " + meta["features"]["population"]["source"])

    # 2) village proxy units
    units = None
    try:
        places = fetch_osm_places(paths, aoi["bbox_lonlat"], logger)
        pts = gpd.GeoDataFrame(places, geometry=gpd.points_from_xy(
            [p["lon"] for p in places], [p["lat"] for p in places]), crs="EPSG:4326").to_crs(crs)
        inside = pts[pts.within(districts_utm.geometry.union_all())]
        logger.info(f"OSM points inside the chosen districts: {len(inside)}")
        if len(inside) >= config.MIN_OSM_POINTS:
            units = voronoi_units(inside, districts_utm, config.VILLAGE_MAX_RADIUS_M)
            meta["unit_source"] = (f"OSM place=village/hamlet Voronoi cells clipped to district and "
                                   f"{config.VILLAGE_MAX_RADIUS_M} m radius ({len(inside)} points)")
        else:
            logger.warning(f"Only {len(inside)} OSM points (< {config.MIN_OSM_POINTS}): grid fallback")
    except Exception as err:  # noqa: BLE001
        logger.warning(f"OSM places unavailable ({err}): grid fallback")
    if units is None:
        populated = (pop_xy[0], pop_xy[1]) if pop_xy is not None else None
        units = grid_units(districts_utm, config.GRID_CELL_M, populated)
        meta["unit_source"] = (f"{config.GRID_CELL_M} m grid over "
                               + ("populated cells (WorldPop > 0)" if populated is not None
                                  else "the whole district (no population data)"))
        warnings.append("village proxies are a grid fallback, not OSM villages")
    units = units.reset_index(drop=True)
    n = len(units)
    if n == 0:
        return common.result("FAILED", "no village proxy units could be built", warnings)
    logger.info(f"Village proxies: {n} ({meta['unit_source']})")

    unit_idx = unit_index_raster(units, transform, width, height)
    pop_pixels = bld_pixels = None
    if pop_xy is not None:
        r, c, inside = points_to_pixels(pop_xy[0], pop_xy[1], transform, width, height)
        u = unit_idx[r[inside], c[inside]]
        pop_pixels = (r[inside], c[inside], pop_xy[2][inside], u)

    # 3) buildings
    try:
        bld, bsrc = get_buildings(paths, aoi, logger, smoke)
    except Exception as err:  # noqa: BLE001
        bld, bsrc = None, f"download/read failed: {err}"
    if bld is not None and len(bld):
        to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        bx, by = to_utm.transform(bld["longitude"].values, bld["latitude"].values)
        r, c, inside = points_to_pixels(bx, by, transform, width, height)
        bld_pixels = (r[inside], c[inside], np.ones(int(inside.sum())), unit_idx[r[inside], c[inside]])
        meta["features"]["buildings"] = {"available": True, "source": f"Google Open Buildings v3 ({bsrc})"}
    else:
        meta["features"]["buildings"] = {"available": False, "source": bsrc or "no buildings in AOI"}
        warnings.append(f"buildings unavailable: {meta['features']['buildings']['source']}")

    # 4) flood features (per pass and final)
    df = flood_features(paths, aoi, unit_idx, n, pop_pixels, bld_pixels)
    df.insert(0, "unit_id", units["unit_id"].values)
    df.insert(1, "name", units["name"].fillna("").values)
    df.insert(2, "district", units["district"].values)
    df.insert(3, "place", units["place"].values)
    df["area_km2"] = units.geometry.area.values / 1e6
    df["flooded_area_km2"] = df["flooded_frac"] * df["n_pixels"] * px_km2
    cent = units.geometry.centroid
    df["x_utm"], df["y_utm"] = cent.x.values, cent.y.values
    lonlat = gpd.GeoSeries(cent, crs=crs).to_crs("EPSG:4326")
    df["lon"], df["lat"] = lonlat.x.values, lonlat.y.values
    df["first_flood_date"] = first_flood_dates(df, aoi["flood_dates"])
    if pop_pixels is not None:
        df["pop"] = np.bincount(pop_pixels[3], weights=pop_pixels[2], minlength=n + 1)[1:]
    if bld_pixels is not None:
        df["buildings"] = np.bincount(bld_pixels[3], minlength=n + 1)[1:].astype(float)

    # 5) elevation
    dem_path = common.dem_on_grid(paths, aoi, logger)
    if dem_path is not None:
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
        ok = np.isfinite(dem) & (unit_idx > 0)
        sums = np.bincount(unit_idx[ok], weights=dem[ok], minlength=n + 1)[1:]
        df["elev_mean_m"] = safe_div(sums, count_per_unit(unit_idx, ok, n))
        df["elev_pctile"] = df.groupby("district")["elev_mean_m"].rank(pct=True)
        meta["features"]["elevation"] = {"available": True, "source": "Copernicus DEM GLO-30"}
        del dem
    else:
        meta["features"]["elevation"] = {"available": False, "source": "DEM could not be downloaded"}
        warnings.append("elevation unavailable")

    # 6) shallow-well reliance (district level, manual table)
    wells = census_well_reliance(paths, sorted(df["district"].unique()), logger)
    if wells:
        df["pct_shallow_well_handpump"] = df["district"].map(wells)
        meta["features"]["well_reliance"] = {"available": True,
                                             "source": f"data/manual/{config.CENSUS_WATER_FILE}"}
    else:
        meta["features"]["well_reliance"] = {
            "available": False, "source": f"Census 2011 table not supplied (fill data/manual/"
                                          f"{config.CENSUS_WATER_TEMPLATE})"}
        warnings.append("shallow-well reliance unavailable: excluded from the index")

    meta["flood_dates"] = sorted(aoi["flood_dates"])
    meta["districts"] = sorted(df["district"].unique().tolist())
    meta["n_units"] = n
    df.to_csv(paths.outputs / "village_features.csv", index=False)
    units_out = units[["unit_id", "name", "district", "place", "geometry"]].copy()
    units_out.to_file(paths.outputs / "villages.gpkg", driver="GPKG")
    with open(paths.outputs / "village_features_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    (paths.results / "village_feature_summary.md").write_text(summary_markdown(df, meta, smoke),
                                                              encoding="utf-8")
    logger.info(f"Wrote {paths.outputs / 'village_features.csv'} ({n} village proxies)")
    return common.result("OK", "", warnings, {"n_units": n})


if __name__ == "__main__":
    import run_all
    sys.exit(run_all.main(default_stages=STAGES_IN_THIS_FILE))
