"""
export_s1.py - MODEL 1 phase 4: Sentinel-1 radar for Assam from Microsoft
Planetary Computer (no account and no Earth Engine needed).

What it does:
  1. Gets district boundaries (geoBoundaries, then GADM 4.1, then a bbox).
  2. Searches sentinel-1-rtc (IW mode, VV+VH) for the flood window and the
     same calendar dates in normal years.
  3. Picks ONE orbit direction + relative orbit and (if DISTRICTS = "auto")
     2 or 3 adjacent districts with the most fully covered dates.
  4. For each date: mosaics the scenes, converts linear power to dB and
     writes data/assam/{flood,normal}/YYYY-MM-DD.tif (VV dB, VH dB, float32,
     LZW, 10 m, UTM). Written in chunks so RAM stays small. Existing files are
     skipped, network errors are retried.
  5. Checks: no all-NaN bands, normal years less "watery" than the flood
     window, VV/VH distribution vs the training numbers (domain shift), and a
     quick-look PNG.

Run alone:  python export_s1.py [--smoke] [--force] [--extra-normal]
"""

import itertools
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

import common
import config

STAGES_IN_THIS_FILE = ["export_s1"]


# ---------------------------------------------------------------------------
# Dates and geometry helpers
# ---------------------------------------------------------------------------
def window_for_year(year):
    """Flood-window calendar dates moved to another year: (start, end) date strings."""
    start = date.fromisoformat(config.FLOOD_START).replace(year=year)
    end = date.fromisoformat(config.FLOOD_END).replace(year=year)
    return start.isoformat(), end.isoformat()


def normal_years(extra_normal=False):
    years = list(config.NORMAL_YEARS)
    if extra_normal:
        years += [y for y in config.EXTRA_NORMAL_YEARS if y not in years]
    return years


def to_utm_func(epsg):
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return lambda geom: _transform_geom(geom, transformer)


def _transform_geom(geom, transformer):
    """Reproject a shapely geometry with a pyproj Transformer."""
    import shapely

    def _xy(coords):  # coords is an (N, 2) array of x, y
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])
    return shapely.transform(geom, _xy)


def item_solar_date(item):
    """Local solar date of a STAC item (UTC time shifted by longitude / 15 hours)."""
    dt = item.datetime
    if dt is None:
        dt = datetime.fromisoformat(item.properties["start_datetime"].replace("Z", "+00:00"))
    lon = (item.bbox[0] + item.bbox[2]) / 2 if item.bbox else 0.0
    return (dt + timedelta(hours=lon / 15.0)).date().isoformat()


def orbit_key(item):
    p = item.properties
    return (str(p.get("sat:orbit_state", "unknown")), int(p.get("sat:relative_orbit", -1)))


def is_iw_dualpol(item):
    p = item.properties
    pols = {str(x).upper() for x in p.get("sar:polarizations", [])}
    return p.get("sar:instrument_mode", "IW") == "IW" and {"VV", "VH"} <= pols


# ---------------------------------------------------------------------------
# Regions (districts) and coverage
# ---------------------------------------------------------------------------
def candidate_regions(paths, logger, smoke):
    """Regions to consider, as {name: shapely geometry in lon/lat}, plus their source."""
    from shapely.geometry import box
    gdf, source = common.get_districts(paths, logger)
    if smoke:
        aoi = box(*config.SMOKE_BBOX)
        if gdf is not None:
            hits = gdf[gdf.intersects(aoi)]
            if len(hits):
                regions = {row.district: row.geometry.intersection(aoi) for row in hits.itertuples()}
                logger.info(f"Smoke AOI lies in district(s): {list(regions)}")
                return regions, source, True
        return {"AOI": aoi}, "smoke bbox", True
    if gdf is None:
        return {"AOI": box(*config.AOI_BBOX_FALLBACK)}, "config bbox (approximate)", False
    names = config.CANDIDATE_DISTRICTS if config.DISTRICTS == "auto" else config.DISTRICTS
    chosen, matches = common.select_districts(gdf, names)
    for m in matches:
        logger.info(f"District name match: {m['query']!r} -> {m['match']!r} "
                    f"({m['method']}, score {m['score']:.0f})")
    missing = [m["query"] for m in matches if m["match"] is None]
    if missing:
        logger.warning(f"Districts not found in {source}: {missing}")
    regions = {row.district: row.geometry for row in chosen.itertuples()}
    return regions, source, True


def coverage_by_orbit_and_date(items, regions_utm, to_utm):
    """{orbit key: {solar date: {region: covered fraction}}} plus the items per key/date."""
    from shapely import union_all
    from shapely.geometry import shape
    groups = defaultdict(list)
    for item in items:
        groups[(orbit_key(item), item_solar_date(item))].append(item)
    coverage = defaultdict(dict)
    for (okey, day), its in groups.items():
        footprint = to_utm(union_all([shape(i.geometry) for i in its]))
        coverage[okey][day] = {name: float(footprint.intersection(g).area / g.area)
                               for name, g in regions_utm.items()}
    return coverage, groups


def covered_dates(coverage_for_orbit, group, start, end):
    """Dates in [start, end] where every region of the group is covered enough."""
    out = []
    for day, cov in sorted(coverage_for_orbit.items()):
        if start <= day <= end and all(cov.get(r, 0.0) >= config.MIN_AOI_COVERAGE for r in group):
            out.append(day)
    return out


def connected_groups(regions_utm, max_size, allow_single=False):
    """All groups of 2..max_size regions that form one connected (touching) block."""
    names = sorted(regions_utm)
    touching = {(a, b): regions_utm[a].distance(regions_utm[b]) < 200.0
                for a in names for b in names if a != b}
    groups = []
    min_size = 1 if allow_single or len(names) == 1 else 2
    for size in range(min_size, min(max_size, len(names)) + 1):
        for combo in itertools.combinations(names, size):
            seen, todo = {combo[0]}, [combo[0]]
            while todo:  # flood fill over the touching graph
                cur = todo.pop()
                for other in combo:
                    if other not in seen and touching.get((cur, other)):
                        seen.add(other)
                        todo.append(other)
            if len(seen) == len(combo):
                groups.append(list(combo))
    return groups


def choose_orbit_and_group(coverage, windows, groups, logger):
    """Pick the orbit + region group with the most fully covered flood dates.

    Score (compared in order): every normal year has at least one date,
    number of flood dates, number of districts, total normal dates.
    """
    table = []
    for okey, cov in coverage.items():
        for group in groups:
            flood = covered_dates(cov, group, *windows["flood"])
            normal = {y: covered_dates(cov, group, *w) for y, w in windows["normal"].items()}
            score = (all(len(v) > 0 for v in normal.values()), len(flood), len(group),
                     sum(len(v) for v in normal.values()))
            table.append({"orbit_state": okey[0], "relative_orbit": okey[1], "districts": group,
                          "n_flood_dates": len(flood),
                          "n_normal_dates": {str(y): len(v) for y, v in normal.items()},
                          "score": score, "flood_dates": flood,
                          "normal_dates": {str(y): v for y, v in normal.items()}})
    if not table:
        raise RuntimeError("No Sentinel-1 scenes found for the AOI in the flood window")
    table.sort(key=lambda r: r["score"], reverse=True)
    for row in table[:10]:
        logger.info(f"  orbit {row['orbit_state']:>10} rel {row['relative_orbit']:3d} | "
                    f"{'+'.join(row['districts']):<40} | flood dates {row['n_flood_dates']} | "
                    f"normal dates {row['n_normal_dates']}")
    best = table[0]
    logger.info(f"CHOSEN: orbit {best['orbit_state']} relative orbit {best['relative_orbit']}, "
                f"districts {best['districts']}")
    return best, table


# ---------------------------------------------------------------------------
# Writing one date to a GeoTIFF
# ---------------------------------------------------------------------------
def power_to_db(x):
    """Linear power -> dB. Zero, negative or non-finite values become NaN."""
    x = np.asarray(x, dtype=np.float32)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = 10.0 * np.log10(x[ok])
    return out


def _tif_profile(crs, transform, width, height):
    return {"driver": "GTiff", "width": width, "height": height, "count": 2,
            "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
            "compress": "lzw", "predictor": 3, "tiled": True, "blockxsize": 512,
            "blockysize": 512, "BIGTIFF": "IF_SAFER"}


def write_rtc_date(items, crs, transform, width, height, out_path, aoi_mask=None):
    """Mosaic RTC items of one date onto the grid and write VV/VH dB in chunks.

    aoi_mask (bool H x W, optional): pixels outside it are written as NaN, and
    chunks completely outside it are not downloaded at all (saves disk and time).
    """
    import odc.stac
    import rasterio
    from odc.geo.geobox import GeoBox
    from rasterio.windows import Window
    gbox = GeoBox((height, width), transform, crs)
    ds = odc.stac.load(items, bands=["vv", "vh"], geobox=gbox, groupby="solar_day",
                       chunks={"x": config.EXPORT_CHUNK, "y": config.EXPORT_CHUNK},
                       resampling="bilinear", dtype="float32", nodata=np.nan)
    ds = ds.isel(time=0)
    tmp = out_path.with_name(out_path.stem + ".tmp.tif")
    with rasterio.open(tmp, "w", **_tif_profile(crs, transform, width, height)) as dst:
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")
        step = config.EXPORT_CHUNK
        for r0 in range(0, height, step):
            for c0 in range(0, width, step):
                h, w = min(step, height - r0), min(step, width - c0)
                inside = None if aoi_mask is None else aoi_mask[r0:r0 + h, c0:c0 + w]
                if inside is not None and not inside.any():
                    block = np.full((2, h, w), np.nan, dtype=np.float32)  # skip the download
                else:
                    sub = ds.isel(y=slice(r0, r0 + h), x=slice(c0, c0 + w)).compute()
                    block = np.stack([power_to_db(sub["vv"].values), power_to_db(sub["vh"].values)])
                    if inside is not None:
                        block[:, ~inside] = np.nan
                dst.write(block, window=Window(c0, r0, w, h))
    os.replace(tmp, out_path)


def calibration_lut(xml_text, shape):
    """Sentinel-1 GRD sigmaNought calibration LUT interpolated to every pixel.

    sigma0 (linear) = DN^2 / LUT^2. The XML gives the LUT on a sparse grid of
    (line, pixel) points; we interpolate linearly along pixels, then lines.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    vectors = root.findall(".//calibrationVector")
    lines = np.array([int(v.find("line").text) for v in vectors], dtype=np.float64)
    height, width = shape
    cols = np.arange(width, dtype=np.float64)
    per_line = np.stack([
        np.interp(cols, np.array(v.find("pixel").text.split(), dtype=np.float64),
                  np.array(v.find("sigmaNought").text.split(), dtype=np.float64))
        for v in vectors])
    rows = np.arange(height, dtype=np.float64)
    i1 = np.clip(np.searchsorted(lines, rows, side="right"), 1, len(lines) - 1)
    i0 = i1 - 1
    t = np.clip((rows - lines[i0]) / np.maximum(lines[i1] - lines[i0], 1), 0, 1)[:, None]
    return (per_line[i0] * (1 - t) + per_line[i1] * t).astype(np.float32)


def write_grd_date(items, crs, transform, width, height, out_path, logger, aoi_mask=None):
    """FALLBACK: GRD items -> approximate sigma0 dB on the grid.

    Applies the sigmaNought calibration LUT and warps with the scene GCPs.
    NOT done (documented in README): thermal noise removal, terrain
    flattening, border noise removal. Expect a larger domain shift.
    """
    import warnings as py_warnings  # (named so it is not confused with our warning lists)

    import rasterio
    import rasterio.errors
    from rasterio.warp import Resampling, reproject
    session = common.http_session()
    bands = []
    for band in ["vv", "vh"]:
        dst = np.full((height, width), np.nan, dtype=np.float32)
        for item in items:
            cal_key = f"schema-calibration-{band}"
            if band not in item.assets or cal_key not in item.assets:
                raise RuntimeError(f"GRD item {item.id} lacks asset {band} or {cal_key}")
            xml_text = session.get(item.assets[cal_key].href, timeout=120).text
            with rasterio.open(item.assets[band].href) as src:
                dn = src.read(1).astype(np.float32)
                gcps, gcp_crs = src.gcps
            lut = calibration_lut(xml_text, dn.shape)
            sigma0 = np.where(dn > 0, dn ** 2 / np.maximum(lut, 1e-6) ** 2, np.nan).astype(np.float32)
            del dn, lut
            with py_warnings.catch_warnings():  # the array has GCPs, not a transform: expected
                py_warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
                reproject(sigma0, dst, gcps=gcps, src_crs=gcp_crs, dst_transform=transform,
                          dst_crs=crs, resampling=Resampling.bilinear, src_nodata=np.nan,
                          dst_nodata=np.nan, init_dest_nodata=False)
            del sigma0
        if aoi_mask is not None:
            dst[~aoi_mask] = np.nan
        bands.append(power_to_db(dst))
    common.write_tif(out_path, np.stack(bands), crs, transform, nodata=np.nan,
                     dtype="float32", band_names=["VV_dB", "VH_dB"])
    logger.warning(f"{out_path.name}: written from GRD with approximate calibration "
                   "(no thermal noise removal, no terrain flattening)")


def matches_grid(path, crs, transform, width, height):
    """True if an existing GeoTIFF is on exactly this grid (so it can be reused)."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            return (src.width == width and src.height == height and src.count == 2
                    and src.crs.to_string() == str(crs)
                    and np.allclose(list(src.transform)[:6], list(transform)[:6]))
    except Exception:  # noqa: BLE001 - unreadable file: not reusable
        return False


def pass_stats(path):
    """Per-band count/sum/sum of squares of clipped dB and the water-ish fraction.

    Read in chunks so big rasters fit in memory. Cached next to the file.
    """
    import rasterio
    from rasterio.windows import Window
    cache = path.with_name(path.stem + ".stats.json")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        with open(cache) as f:
            return json.load(f)
    count = np.zeros(2)
    total = np.zeros(2)
    total_sq = np.zeros(2)
    waterish = 0
    pixels = 0
    with rasterio.open(path) as src:
        step = config.EXPORT_CHUNK
        pixels = src.width * src.height
        for r0 in range(0, src.height, step):
            for c0 in range(0, src.width, step):
                win = Window(c0, r0, min(step, src.width - c0), min(step, src.height - r0))
                block = np.clip(src.read(window=win).astype(np.float64),
                                config.CLIP_DB_MIN, config.CLIP_DB_MAX)
                for b in range(2):
                    v = block[b][np.isfinite(block[b])]
                    count[b] += v.size
                    total[b] += v.sum()
                    total_sq[b] += (v ** 2).sum()
                vv = block[0]
                waterish += int((np.isfinite(vv) & (vv < config.WATERISH_VV_DB)).sum())
    stats = {"count": count.tolist(), "sum": total.tolist(), "sum_sq": total_sq.tolist(),
             "pixels": pixels, "valid_pixels": int(count[0]),
             "valid_fraction": float(count[0] / pixels) if pixels else 0.0,
             "waterish_fraction": float(waterish / count[0]) if count[0] else float("nan")}
    with open(cache, "w") as f:
        json.dump(stats, f)
    return stats


def pooled_mean_std(stat_list):
    """Combine per-pass sums into one mean/std per band."""
    if not stat_list:
        return None
    c = np.sum([s["count"] for s in stat_list], axis=0)
    t = np.sum([s["sum"] for s in stat_list], axis=0)
    q = np.sum([s["sum_sq"] for s in stat_list], axis=0)
    mean = t / np.maximum(c, 1)
    std = np.sqrt(np.maximum(q / np.maximum(c, 1) - mean ** 2, 0))
    return {"vv_mean": float(mean[0]), "vv_std": float(std[0]), "vh_mean": float(mean[1]),
            "vh_std": float(std[1]), "n_pixels": int(c[0]), "n_passes": len(stat_list)}


def quicklook(flood_path, normal_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.enums import Resampling
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, path, title in [(axes[0], flood_path, "flood window"), (axes[1], normal_path, "normal year")]:
        if path is None:
            ax.set_title(f"{title}: no pass")
            ax.axis("off")
            continue
        with rasterio.open(path) as src:
            factor = max(1, int(np.ceil(max(src.width, src.height) / 1200)))
            vv = src.read(1, out_shape=(src.height // factor or 1, src.width // factor or 1),
                          resampling=Resampling.average)
        im = ax.imshow(vv, cmap="gray", vmin=-25, vmax=0)
        ax.set_title(f"{title}: {path.stem} (VV dB, rivers should look dark)", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage export_s1
# ---------------------------------------------------------------------------
def search_s1(catalog, collection, geom_lonlat, start, end, logger):
    from shapely.geometry import mapping

    def _search():
        search = catalog.search(collections=[collection], intersects=mapping(geom_lonlat),
                                datetime=f"{start}/{end}")
        return list(search.items())
    items = common.retry(_search, what=f"STAC search {collection} {start}/{end}", logger=logger)
    kept = [i for i in items if is_iw_dualpol(i)]
    logger.info(f"{collection} {start}..{end}: {len(items)} items, {len(kept)} IW with VV+VH")
    return kept


def build_grid(region_union_utm, epsg):
    """Pixel grid (10 m, north up) covering the chosen districts plus a buffer."""
    from affine import Affine
    px = config.PIXEL_SIZE_M
    xmin, ymin, xmax, ymax = region_union_utm.buffer(config.AOI_BUFFER_M).bounds
    xmin, ymin = np.floor(xmin / px) * px, np.floor(ymin / px) * px
    xmax, ymax = np.ceil(xmax / px) * px, np.ceil(ymax / px) * px
    width, height = int(round((xmax - xmin) / px)), int(round((ymax - ymin) / px))
    transform = Affine(px, 0.0, xmin, 0.0, -px, ymax)
    return {"crs": f"EPSG:{epsg}", "transform": list(transform)[:6], "width": width,
            "height": height, "pixel_size_m": px}


def run_export(smoke=False, logger=None, extra_normal=False, **kwargs):
    import geopandas as gpd
    import planetary_computer
    from shapely import union_all

    paths = config.get_paths(smoke)
    warnings = []
    regions, region_source, have_boundaries = candidate_regions(paths, logger, smoke)
    union_lonlat = union_all(list(regions.values()))
    c = union_lonlat.centroid
    epsg = common.utm_epsg(c.x, c.y)
    to_utm = to_utm_func(epsg)
    regions_utm = {k: to_utm(v) for k, v in regions.items()}
    search_geom = union_lonlat.envelope

    windows = {"flood": (config.FLOOD_START, config.FLOOD_END),
               "normal": {y: window_for_year(y) for y in normal_years(extra_normal)}}
    catalog = common.open_planetary_computer()

    collection = config.S1_COLLECTION
    try:
        flood_items = search_s1(catalog, collection, search_geom, *windows["flood"], logger)
        if not flood_items:
            raise RuntimeError("no RTC items in the flood window")
    except Exception as err:  # noqa: BLE001
        common.loud_warning(logger, f"{collection} search failed ({err}). Trying "
                                    f"{config.S1_FALLBACK_COLLECTION} (needs extra calibration)")
        collection = config.S1_FALLBACK_COLLECTION
        flood_items = search_s1(catalog, collection, search_geom, *windows["flood"], logger)
        warnings.append("used GRD fallback with approximate calibration (see README)")
    normal_items = []
    for s, e in windows["normal"].values():
        normal_items += search_s1(catalog, collection, search_geom, s, e, logger)

    coverage, groups_items = coverage_by_orbit_and_date(flood_items + normal_items, regions_utm, to_utm)
    if smoke or config.DISTRICTS != "auto" or not have_boundaries:
        candidate_groups = [sorted(regions_utm)]
    else:
        candidate_groups = connected_groups(regions_utm, config.MAX_DISTRICTS)
        if not candidate_groups:
            candidate_groups = [[d] for d in sorted(regions_utm)]
    logger.info(f"Comparing {len(coverage)} orbits x {len(candidate_groups)} district groups")
    best, table = choose_orbit_and_group(coverage, windows, candidate_groups, logger)
    okey = (best["orbit_state"], best["relative_orbit"])

    flood_dates = best["flood_dates"]
    normal_dates_by_year = best["normal_dates"]
    if smoke:
        flood_dates = flood_dates[:config.SMOKE_FLOOD_DATES]
        flat = [(y, d) for y in sorted(normal_dates_by_year) for d in normal_dates_by_year[y][:1]]
        flat += [(y, d) for y in sorted(normal_dates_by_year) for d in normal_dates_by_year[y][1:]]
        normal_dates_by_year = defaultdict(list)
        for y, d in flat[:config.SMOKE_NORMAL_DATES]:
            normal_dates_by_year[y].append(d)
        normal_dates_by_year = dict(normal_dates_by_year)
    normal_dates = sorted(d for v in normal_dates_by_year.values() for d in v)
    if len(flood_dates) < 2:
        warnings.append(f"only {len(flood_dates)} flood-window date(s) fully cover the AOI")
    for y, v in normal_dates_by_year.items():
        if not v:
            warnings.append(f"normal year {y} has no fully covered date")
    if not flood_dates or not normal_dates:
        return common.result("FAILED", "not enough fully covered dates (flood "
                                       f"{len(flood_dates)}, normal {len(normal_dates)})", warnings)

    chosen_regions = {k: regions_utm[k] for k in best["districts"]}
    grid = build_grid(union_all(list(chosen_regions.values())), epsg)
    logger.info(f"Grid: {grid['crs']} {grid['width']} x {grid['height']} px at "
                f"{grid['pixel_size_m']} m ({grid['width'] * grid['height'] / 1e6:.1f} Mpx per band)")
    bbox_lonlat = list(union_all([regions[k] for k in best["districts"]]).buffer(0.01).bounds)
    aoi = {"smoke": smoke, "districts": best["districts"], "district_source": region_source,
           "collection": collection,
           "orbit": {"orbit_state": okey[0], "relative_orbit": okey[1]},
           "flood_window": list(windows["flood"]),
           "normal_windows": {str(y): list(w) for y, w in windows["normal"].items()},
           "flood_dates": flood_dates, "normal_dates": normal_dates,
           "normal_dates_by_year": normal_dates_by_year, "grid": grid,
           "bbox_lonlat": bbox_lonlat, "created_at": datetime.now().isoformat()}
    with open(paths.aoi, "w") as f:
        json.dump(aoi, f, indent=2)
    gpd.GeoDataFrame({"district": list(chosen_regions)}, geometry=list(chosen_regions.values()),
                     crs=f"EPSG:{epsg}").to_file(paths.outputs / "aoi_districts.gpkg", driver="GPKG")

    crs, transform, width, height = common.aoi_grid(aoi)
    # Only pixels inside the chosen districts (plus a buffer) are kept; the rest
    # of the bounding box is NaN, which costs almost no disk space.
    from rasterio import features
    buffered = union_all(list(chosen_regions.values())).buffer(config.AOI_BUFFER_M)
    aoi_mask = features.rasterize([(buffered, 1)], out_shape=(height, width), transform=transform,
                                  fill=0, dtype="uint8").astype(bool)
    n_aoi_px = int(aoi_mask.sum())
    logger.info(f"AOI covers {n_aoi_px / 1e6:.1f} Mpx of the {width * height / 1e6:.1f} Mpx grid")
    written = {"flood": [], "normal": []}
    for kind, dates in [("flood", flood_dates), ("normal", normal_dates)]:
        for day in dates:
            out_path = paths.assam / kind / f"{day}.tif"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            items = groups_items[(okey, day)]
            if out_path.exists() and not matches_grid(out_path, crs, transform, width, height):
                logger.warning(f"{out_path} is on a different grid (districts or settings changed): "
                               "downloading it again")
                out_path.unlink()
            if not out_path.exists():
                logger.info(f"Exporting {kind} pass {day} ({len(items)} scene(s)) -> {out_path}")

                def _write(items=items, out_path=out_path):
                    signed = [planetary_computer.sign(i) for i in items]  # fresh tokens each try
                    if collection == config.S1_COLLECTION:
                        write_rtc_date(signed, crs, transform, width, height, out_path, aoi_mask)
                    else:
                        write_grd_date(signed, crs, transform, width, height, out_path, logger,
                                       aoi_mask)
                common.retry(_write, what=f"export {day}", logger=logger, wait_s=15)
            else:
                logger.info(f"{out_path} exists, skipping download")
            stats = pass_stats(out_path)
            if min(stats["count"]) == 0:
                logger.error(f"{out_path} has an all-NaN band; deleting it")
                out_path.unlink()
                warnings.append(f"{kind} pass {day} was all NaN and was removed")
                continue
            stats["aoi_coverage"] = stats["valid_pixels"] / max(n_aoi_px, 1)
            logger.info(f"  {day}: radar covers {stats['aoi_coverage']:.3f} of the AOI, water-ish "
                        f"(VV < {config.WATERISH_VV_DB} dB) {stats['waterish_fraction']:.3f}")
            written[kind].append({"date": day, "path": str(out_path.relative_to(paths.root)),
                                  **{k: stats[k] for k in ("aoi_coverage", "waterish_fraction")},
                                  "_stats": stats})

    # Normal years must look less flooded than the flood window.
    flood_water = np.nanmean([p["waterish_fraction"] for p in written["flood"]]) if written["flood"] else np.nan
    year_water = {}
    for y, days in normal_dates_by_year.items():
        vals = [p["waterish_fraction"] for p in written["normal"] if p["date"] in days]
        year_water[str(y)] = float(np.nanmean(vals)) if vals else float("nan")
        if vals and np.isfinite(flood_water) and year_water[str(y)] >= config.NORMAL_FLOOD_RATIO_WARN * flood_water:
            msg = (f"NORMAL YEAR {y} LOOKS FLOODED: water-ish fraction {year_water[str(y)]:.3f} vs "
                   f"flood window {flood_water:.3f}. The normal-water mask may include flood water.")
            common.loud_warning(logger, msg)
            warnings.append(msg)
    logger.info(f"Water-ish fraction: flood window {flood_water:.3f}, normal years {year_water}")

    # Domain shift: Assam dB distribution vs the Sen1Floods11 training numbers.
    shift = {"note": ("Training: Sen1Floods11 S1Hand (Earth Engine GRD, sigma0 dB). Assam: "
                      f"{collection} ({'gamma0 RTC' if collection == config.S1_COLLECTION else 'GRD approx sigma0'}) "
                      "converted to dB. Both clipped to the same range. Flood-window imagery has "
                      "more water, so its mean is expected to be lower even without a sensor shift.")}
    try:
        train_stats = config.load_norm_stats(paths)
        shift["train"] = {k: train_stats[k] for k in ("vv_mean", "vv_std", "vh_mean", "vh_std")}
    except FileNotFoundError:
        shift["train"] = None
        warnings.append("norm_stats.json missing: domain shift not compared with training")
    shift["assam_all"] = pooled_mean_std([p["_stats"] for k in written for p in written[k]])
    shift["assam_flood"] = pooled_mean_std([p["_stats"] for p in written["flood"]])
    shift["assam_normal"] = pooled_mean_std([p["_stats"] for p in written["normal"]])
    if shift["train"] and shift["assam_normal"]:
        shift["normal_minus_train"] = {k: shift["assam_normal"][k] - shift["train"][k]
                                       for k in ("vv_mean", "vv_std", "vh_mean", "vh_std")}
        logger.info(f"Domain shift (normal-year Assam minus training): {shift['normal_minus_train']}")
    with open(paths.results / "domain_shift.json", "w") as f:
        json.dump(shift, f, indent=2)

    quicklook(paths.assam / "flood" / f"{written['flood'][0]['date']}.tif" if written["flood"] else None,
              paths.assam / "normal" / f"{written['normal'][0]['date']}.tif" if written["normal"] else None,
              paths.results / "assam_quicklook.png")

    for k in written:
        for p in written[k]:
            p.pop("_stats")
    summary = {"aoi": {k: v for k, v in aoi.items() if k != "grid"}, "grid": grid,
               "passes": written, "waterish_flood": flood_water, "waterish_normal": year_water,
               "orbit_table": [{k: v for k, v in r.items() if k != "score"} for r in table[:20]],
               "warnings": warnings}
    with open(paths.results / "export_s1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if not written["flood"] or not written["normal"]:
        return common.result("FAILED", "no valid flood or normal pass was written", warnings)
    # The final dates are what was actually written.
    aoi["flood_dates"] = [p["date"] for p in written["flood"]]
    aoi["normal_dates"] = [p["date"] for p in written["normal"]]
    with open(paths.aoi, "w") as f:
        json.dump(aoi, f, indent=2)
    return common.result("OK", "", warnings, {"flood_passes": len(written["flood"]),
                                              "normal_passes": len(written["normal"])})


if __name__ == "__main__":
    import run_all
    sys.exit(run_all.main(default_stages=STAGES_IN_THIS_FILE))
