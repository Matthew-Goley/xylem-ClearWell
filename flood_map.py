"""
flood_map.py - MODEL 1 phases 5 and 6: Assam flood maps from the trained U-Net.

Stage flood_map (phase 5):
  1. Run the U-Net on every Assam pass (512 x 512 tiles with 64 px overlap,
     overlapping probabilities are averaged, big rasters are done in blocks).
  2. Normal water = water in at least half of the normal-year passes.
  3. Flood in a pass = water in that pass AND NOT normal water, then small
     blobs (< 0.5 ha) are removed.
  4. Duration = longest run of consecutive flooded passes, in days. A run
     from pass i to pass j lasts (date_j - date_i) + one revisit interval, so
     a single flooded pass counts as one revisit interval. This is an
     estimate with a resolution of one revisit interval.
  5. Writes outputs/flood_extent.tif, flood_by_pass/*.tif, flood_duration.tif,
     normal_water.tif, the per-district time series and overview figures, and
     a check that flooded pixels sit closer to rivers / lower than random.

Stage sanity_check (phase 6): Sentinel-2 MNDWI comparison (if a clear scene
exists), JRC Global Surface Water comparison of the normal-water mask, and a
side-by-side figure for visual inspection.

Run alone:  python flood_map.py [--smoke] [--stage flood_map|sanity_check]
"""

import csv
import json
import shutil
import sys
from datetime import date, datetime, timedelta

import numpy as np

import common
import config

STAGES_IN_THIS_FILE = ["flood_map", "sanity_check"]
NODATA_U8 = 255
NODATA_U16 = 65535


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------
def tile_starts(length, tile, stride):
    """Start offsets so tiles of size `tile` cover [0, length), last one flush with the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def predict_array(model, x, device, use_amp, tile=None, overlap=None, batch=None, tta=None):
    """Water probability (H, W) for a normalized (2, H, W) array of any size.

    The array is cut into tile x tile pieces overlapping by `overlap` pixels;
    where pieces overlap, their probabilities are averaged.
    """
    import torch
    from train import predict_logits
    tile = tile or config.INFER_TILE
    overlap = config.INFER_OVERLAP if overlap is None else overlap
    batch = batch or config.INFER_BATCH
    tta = config.TTA_HFLIP if tta is None else tta
    _, h, w = x.shape
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:  # small inputs: pad with 0 (the training mean after normalizing)
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    _, hp, wp = x.shape
    stride = tile - overlap
    prob_sum = np.zeros((hp, wp), dtype=np.float32)
    weight = np.zeros((hp, wp), dtype=np.float32)
    coords = [(r, c) for r in tile_starts(hp, tile, stride) for c in tile_starts(wp, tile, stride)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = torch.from_numpy(np.stack([x[:, r:r + tile, c:c + tile] for r, c in chunk]))
        probs = torch.sigmoid(predict_logits(model, tiles, device, use_amp))
        if tta:
            flipped = torch.sigmoid(predict_logits(model, tiles.flip(-1), device, use_amp)).flip(-1)
            probs = (probs + flipped) / 2
        probs = probs.cpu().numpy()
        for (r, c), p in zip(chunk, probs):
            prob_sum[r:r + tile, c:c + tile] += p
            weight[r:r + tile, c:c + tile] += 1
    return (prob_sum / weight)[:h, :w]


def predict_raster(model, src_path, dst_path, stats, device, use_amp, logger=None):
    """Water mask (uint8: 1 water, 0 not water, 255 no radar data) for one pass.

    Processed in blocks of INFER_BLOCK pixels, each read with an extra margin of
    INFER_OVERLAP pixels that is cropped off again, so block edges get context.
    """
    import rasterio
    from rasterio.windows import Window
    from train import preprocess_radar
    with rasterio.open(src_path) as src:
        height, width = src.height, src.width
        crs, transform = src.crs, src.transform
        out = np.full((height, width), NODATA_U8, dtype=np.uint8)
        block, margin = config.INFER_BLOCK, config.INFER_OVERLAP
        for r0 in range(0, height, block):
            for c0 in range(0, width, block):
                r1, c1 = min(r0 + block, height), min(c0 + block, width)
                rr0, cc0 = max(0, r0 - margin), max(0, c0 - margin)
                rr1, cc1 = min(height, r1 + margin), min(width, c1 + margin)
                radar = src.read(window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0)).astype(np.float32)
                valid = np.isfinite(radar).all(axis=0)
                if not valid.any():
                    continue
                prob = predict_array(model, preprocess_radar(radar, stats), device, use_amp)
                water = (prob >= config.WATER_THRESHOLD).astype(np.uint8)
                water[~valid] = NODATA_U8
                out[r0:r1, c0:c1] = water[r0 - rr0:r0 - rr0 + (r1 - r0), c0 - cc0:c0 - cc0 + (c1 - c0)]
    common.write_tif(dst_path, out, crs, transform, nodata=NODATA_U8, dtype="uint8")
    if logger:
        valid_px = out != NODATA_U8
        frac = float((out == 1).sum() / max(valid_px.sum(), 1))
        logger.info(f"  {src_path.name}: water fraction {frac:.3f} of valid pixels")
    return dst_path


# ---------------------------------------------------------------------------
# Normal water, flood, cleanup, duration
# ---------------------------------------------------------------------------
def normal_water_mask(water_masks, min_frac=None):
    """uint8 mask: 1 = water in >= min_frac of the passes where the pixel was seen,
    0 = not normal water, 255 = never seen in any normal pass."""
    min_frac = config.NORMAL_WATER_MIN_FRAC if min_frac is None else min_frac
    water_count = np.zeros(water_masks[0].shape, dtype=np.int16)
    valid_count = np.zeros(water_masks[0].shape, dtype=np.int16)
    for m in water_masks:
        water_count += (m == 1)
        valid_count += (m != NODATA_U8)
    out = np.zeros(water_masks[0].shape, dtype=np.uint8)
    seen = valid_count > 0
    out[seen & (water_count >= min_frac * valid_count)] = 1
    out[~seen] = NODATA_U8
    return out


def remove_small_blobs(mask, min_pixels):
    """Remove connected blobs (8-connected) smaller than min_pixels. Returns (mask, n removed)."""
    from scipy import ndimage
    if min_pixels <= 1 or not mask.any():
        return mask.copy(), 0
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    small = sizes < min_pixels
    small[0] = False  # background
    remove = small[labels]
    cleaned = mask & ~remove
    return cleaned, int(remove.sum())


def flood_mask_for_pass(water, normal, min_pixels):
    """Flood (uint8 1/0/255) = water AND NOT normal water, then small blobs removed."""
    valid = (water != NODATA_U8) & (normal != NODATA_U8)
    flood = (water == 1) & (normal == 0) & valid
    flood, removed = remove_small_blobs(flood, min_pixels)
    out = flood.astype(np.uint8)
    out[~valid] = NODATA_U8
    return out, removed


def min_blob_pixels(pixel_size_m):
    """Hectares -> pixel count (1 ha = 10,000 m2)."""
    return int(np.ceil(config.MIN_FLOOD_BLOB_HA * 10000 / (pixel_size_m ** 2)))


def revisit_interval_days(dates):
    """Typical days between passes (median of the gaps); 12 if only one pass."""
    days = sorted(date.fromisoformat(d) for d in dates)
    if len(days) < 2:
        return 12
    gaps = [(b - a).days for a, b in zip(days[:-1], days[1:])]
    return int(np.median(gaps))


class FloodDurationTracker:
    """Keeps, for every pixel, the longest run of consecutive flooded passes (in days).

    Call update() once per flood pass, in date order. A pixel without data in a
    pass is treated as not flooded in that pass (it breaks the run).
    """

    def __init__(self, shape, interval_days, first_day):
        self.interval = int(interval_days)
        self.first_day = date.fromisoformat(first_day)
        self.run_start = np.full(shape, -1, dtype=np.int16)  # day number where the current run began
        self.longest = np.zeros(shape, dtype=np.int16)       # days
        self.ever_flooded = np.zeros(shape, dtype=bool)
        self.ever_valid = np.zeros(shape, dtype=bool)

    def update(self, flood_u8, day):
        day_n = (date.fromisoformat(day) - self.first_day).days
        flooded = flood_u8 == 1
        valid = flood_u8 != NODATA_U8
        starting = flooded & (self.run_start < 0)
        self.run_start[starting] = day_n
        current = np.where(flooded, day_n - self.run_start + self.interval, 0).astype(np.int16)
        np.maximum(self.longest, current, out=self.longest)
        self.run_start[~flooded] = -1
        self.ever_flooded |= flooded
        self.ever_valid |= valid

    def extent_u8(self):
        out = self.ever_flooded.astype(np.uint8)
        out[~self.ever_valid] = NODATA_U8
        return out

    def duration_u16(self):
        out = self.longest.astype(np.uint16)
        out[~self.ever_valid] = NODATA_U16
        return out


def iterate_flood_passes(paths, aoi):
    """Yield (date, flood mask uint8, tracker) for each flood pass in date order.

    Used by this stage and by build_villages.py (per-pass ranking), so both
    compute duration-so-far the same way.
    """
    import rasterio
    dates = sorted(aoi["flood_dates"])
    tracker = None
    for day in dates:
        path = paths.outputs / "flood_by_pass" / f"{day}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            flood = src.read(1)
        if tracker is None:
            tracker = FloodDurationTracker(flood.shape, revisit_interval_days(dates), dates[0])
        tracker.update(flood, day)
        yield day, flood, tracker


# ---------------------------------------------------------------------------
# Checks and figures
# ---------------------------------------------------------------------------
def placement_test(values, flood, land, n_perm, rng, lower_is_plausible=True):
    """Compare mean `values` at flooded pixels against random land pixels.

    values: 2D array (e.g. distance to normal water, or elevation); flood and
    land: boolean masks (flood is a subset of land). Returns observed mean,
    random mean, permutation p-value (one-sided: flooded lower than random).
    """
    ok = land & np.isfinite(values)
    f = flood & ok
    n = int(f.sum())
    if n == 0 or ok.sum() <= n:
        return None
    pool = values[ok]
    observed = float(values[f].mean())
    random_means = np.array([pool[rng.choice(pool.size, n, replace=False)].mean()
                             for _ in range(n_perm)])
    if lower_is_plausible:
        p = (1 + int((random_means <= observed).sum())) / (n_perm + 1)
    else:
        p = (1 + int((random_means >= observed).sum())) / (n_perm + 1)
    return {"flooded_mean": observed, "random_land_mean": float(random_means.mean()),
            "all_land_mean": float(pool.mean()), "n_flooded_cells": n, "p_value": p,
            "flooded_lower_than_random": bool(observed < random_means.mean())}


def plausibility_checks(extent, normal, dem, pixel_size_m, logger):
    """Are flooded pixels closer to normal water (and lower) than random land pixels?"""
    from scipy import ndimage
    f = config.PLAUSIBILITY_DOWNSAMPLE
    rng = np.random.default_rng(config.SEED)
    normal_ds = common.block_reduce(normal == 1, f, "max")
    flood_ds = common.block_reduce(extent == 1, f, "max") & ~normal_ds
    valid_ds = common.block_reduce((extent != NODATA_U8) & (normal != NODATA_U8), f, "max")
    land_ds = valid_ds & ~normal_ds
    out = {"grid_m": pixel_size_m * f, "n_permutations": config.PLAUSIBILITY_PERMUTATIONS}
    if normal_ds.any():
        dist = ndimage.distance_transform_edt(~normal_ds) * pixel_size_m * f
        out["distance_to_normal_water_m"] = placement_test(
            dist, flood_ds, land_ds, config.PLAUSIBILITY_PERMUTATIONS, rng)
    else:
        out["distance_to_normal_water_m"] = None
    if dem is not None:
        dem_ds = common.block_reduce(dem, f, "nanmean")
        out["elevation_m"] = placement_test(dem_ds, flood_ds, land_ds,
                                            config.PLAUSIBILITY_PERMUTATIONS, rng)
    else:
        out["elevation_m"] = None
    for key in ("distance_to_normal_water_m", "elevation_m"):
        r = out[key]
        if r:
            logger.info(f"Plausibility ({key}): flooded mean {r['flooded_mean']:.1f} vs random land "
                        f"{r['random_land_mean']:.1f} (p = {r['p_value']:.3f}, "
                        f"flooded lower: {r['flooded_lower_than_random']})")
        else:
            logger.info(f"Plausibility ({key}): not computable")
    return out


def district_timeseries(paths, aoi, district_idx, district_names, logger):
    """Flooded km2 per district for every pass -> CSV + PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    km2 = common.pixel_area_km2(aoi)
    rows = []
    n = len(district_names)
    for day in sorted(aoi["flood_dates"]):
        path = paths.outputs / "flood_by_pass" / f"{day}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            flood = src.read(1)
        flooded = np.bincount(district_idx[flood == 1], minlength=n + 1)
        valid = np.bincount(district_idx[flood != NODATA_U8], minlength=n + 1)
        for i, name in enumerate(district_names, start=1):
            rows.append({"date": day, "district": name, "flooded_km2": round(flooded[i] * km2, 3),
                         "observed_km2": round(valid[i] * km2, 3)})
    with open(paths.results / "flood_timeseries.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "district", "flooded_km2", "observed_km2"])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in district_names:
        pts = [(r["date"], r["flooded_km2"]) for r in rows if r["district"] == name]
        ax.plot([date.fromisoformat(d) for d, _ in pts], [v for _, v in pts], marker="o", label=name)
    ax.set_ylabel("flooded area (km2, excl. normal water)")
    ax.set_title("Flooded area per Sentinel-1 pass")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(paths.results / "flood_timeseries.png", dpi=100)
    plt.close(fig)
    return rows


def overview_maps(paths, extent, duration, districts_utm, transform, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    h, w = extent.shape
    f = max(1, int(np.ceil(max(h, w) / 1500)))
    ext_ds = common.block_reduce((extent == 1).astype(np.uint8), f, "max")
    nod_ds = common.block_reduce((extent == NODATA_U8).astype(np.uint8), f, "min")
    dur = np.where(duration == NODATA_U16, 0, duration)
    dur_ds = common.block_reduce(dur, f, "max").astype(float)
    dur_ds[dur_ds == 0] = np.nan
    xmin, ymin, xmax, ymax = common.grid_bounds(transform, ext_ds.shape[1] * f, ext_ds.shape[0] * f)
    extent_xy = [xmin, xmax, ymin, ymax]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    img = np.where(nod_ds == 1, 0, ext_ds + 1)
    axes[0].imshow(img, cmap=ListedColormap(["#bdbdbd", "#f4f1ea", "#d7301f"]), vmin=0, vmax=2,
                   extent=extent_xy, interpolation="nearest")
    axes[0].set_title("Maximum flood extent (red), grey = no data")
    im = axes[1].imshow(dur_ds, cmap="viridis", extent=extent_xy, interpolation="nearest")
    axes[1].set_title("Flood duration estimate (days)")
    fig.colorbar(im, ax=axes[1], fraction=0.035)
    for ax in axes:
        districts_utm.boundary.plot(ax=ax, color="black", linewidth=0.8)
        for row in districts_utm.itertuples():
            p = row.geometry.representative_point()
            ax.annotate(row.district, (p.x, p.y), fontsize=8, ha="center")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def read_u8(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1)


# ---------------------------------------------------------------------------
# Stage flood_map
# ---------------------------------------------------------------------------
def run_flood_map(smoke=False, logger=None, **kwargs):
    import rasterio
    import torch
    from train import load_trained_model

    paths = config.get_paths(smoke)
    aoi = common.load_aoi(paths)
    crs, transform, width, height = common.aoi_grid(aoi)
    px = aoi["grid"]["pixel_size_m"]
    warnings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.USE_AMP and device.type == "cuda"
    model, ckpt = load_trained_model(paths, device, logger)
    stats = ckpt["norm_stats"]
    if ckpt.get("smoke") and not smoke:
        warnings.append("the checkpoint came from a SMOKE training run")

    # Water masks are cached; throw them away if the model or settings changed.
    water_dir = paths.outputs / "water_by_pass"
    model_info = {"checkpoint_saved_at": ckpt.get("saved_at"), "threshold": config.WATER_THRESHOLD,
                  "tta": config.TTA_HFLIP, "tile": config.INFER_TILE, "overlap": config.INFER_OVERLAP}
    info_file = water_dir / "model_info.json"
    if water_dir.exists() and (not info_file.exists() or json.loads(info_file.read_text()) != model_info):
        logger.info("Model or inference settings changed: recomputing all water masks")
        shutil.rmtree(water_dir)
    water_dir.mkdir(parents=True, exist_ok=True)
    info_file.write_text(json.dumps(model_info, indent=2))

    water_paths = {"flood": {}, "normal": {}}
    for kind in ("flood", "normal"):
        for day in sorted(aoi[f"{kind}_dates"]):
            src = paths.assam / kind / f"{day}.tif"
            if not src.exists():
                warnings.append(f"{kind} pass {day} missing from {src.parent}")
                continue
            dst = water_dir / kind / f"{day}.tif"
            if not dst.exists():
                logger.info(f"Predicting water for {kind} pass {day}")
                predict_raster(model, src, dst, stats, device, use_amp, logger)
            water_paths[kind][day] = dst
    if not water_paths["flood"] or not water_paths["normal"]:
        return common.result("FAILED", "need at least one flood pass and one normal pass", warnings)

    normal = normal_water_mask([read_u8(p) for p in water_paths["normal"].values()])
    common.write_tif(paths.outputs / "normal_water.tif", normal, crs, transform,
                     nodata=NODATA_U8, dtype="uint8")
    unseen = float((normal == NODATA_U8).mean())
    logger.info(f"Normal water: {float((normal == 1).mean()):.3f} of the grid; "
                f"{unseen:.3f} never seen in a normal pass (treated as no data)")
    if unseen > 0.05:
        warnings.append(f"{unseen:.1%} of the grid has no normal-year observation")

    min_px = min_blob_pixels(px)
    removed_total = 0
    for day, wpath in sorted(water_paths["flood"].items()):
        flood, removed = flood_mask_for_pass(read_u8(wpath), normal, min_px)
        removed_total += removed
        logger.info(f"Flood {day}: {int((flood == 1).sum())} flooded px after removing {removed} px "
                    f"in blobs < {config.MIN_FLOOD_BLOB_HA} ha ({min_px} px)")
        common.write_tif(paths.outputs / "flood_by_pass" / f"{day}.tif", flood, crs, transform,
                         nodata=NODATA_U8, dtype="uint8")

    tracker = None
    for _, _, tracker in iterate_flood_passes(paths, aoi):
        pass
    extent, duration = tracker.extent_u8(), tracker.duration_u16()
    common.write_tif(paths.outputs / "flood_extent.tif", extent, crs, transform,
                     nodata=NODATA_U8, dtype="uint8")
    common.write_tif(paths.outputs / "flood_duration.tif", duration, crs, transform,
                     nodata=NODATA_U16, dtype="uint16")
    interval = tracker.interval
    logger.info(f"Revisit interval used for duration: {interval} days")

    districts_utm = common.districts_for_aoi(paths, aoi)
    district_idx = common.rasterize_districts(districts_utm, transform, width, height)
    names = districts_utm["district"].tolist()
    district_timeseries(paths, aoi, district_idx, names, logger)
    km2 = common.pixel_area_km2(aoi)
    per_district = {}
    for i, name in enumerate(names, start=1):
        inside = district_idx == i
        per_district[name] = {
            "max_extent_flooded_km2": float(((extent == 1) & inside).sum() * km2),
            "observed_km2": float(((extent != NODATA_U8) & inside).sum() * km2),
            "normal_water_km2": float(((normal == 1) & inside).sum() * km2),
            "max_duration_days": int(duration[inside & (duration != NODATA_U16)].max())
            if (inside & (duration != NODATA_U16)).any() else 0}
        logger.info(f"{name}: max extent {per_district[name]['max_extent_flooded_km2']:.1f} km2 flooded "
                    f"(excluding normal water)")
    overview_maps(paths, extent, duration, districts_utm, transform,
                  paths.results / "assam_flood_maps.png")

    dem_path = common.dem_on_grid(paths, aoi, logger)
    dem = None
    if dem_path is not None:
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
    plaus = plausibility_checks(extent, normal, dem, px, logger)
    d = plaus.get("distance_to_normal_water_m")
    if d and not d["flooded_lower_than_random"]:
        warnings.append("flooded pixels are NOT closer to normal water than random land pixels")

    summary = {"smoke": smoke, "flood_dates": sorted(water_paths["flood"]),
               "normal_dates": sorted(water_paths["normal"]), "revisit_interval_days": interval,
               "duration_note": "duration = (last - first date of the longest run of consecutive "
                                "flooded passes) + one revisit interval; resolution one interval",
               "min_blob_pixels": min_px, "pixels_removed_by_cleanup": removed_total,
               "normal_water_fraction": float((normal == 1).mean()),
               "unseen_in_normal_fraction": unseen, "per_district": per_district,
               "plausibility": plaus, "threshold": config.WATER_THRESHOLD,
               "checkpoint": model_info, "generated_at": datetime.now().isoformat()}
    with open(paths.results / "flood_map_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return common.result("OK", "", warnings, {"districts": names})


# ---------------------------------------------------------------------------
# Stage sanity_check
# ---------------------------------------------------------------------------
def compare_masks(pred, ref, valid):
    """IoU / precision / recall / agreement of boolean pred vs ref on valid pixels."""
    p, r = pred & valid, ref & valid
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    n = int(valid.sum())
    agree = int(((pred == ref) & valid).sum())
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "agreement": agree / n if n else float("nan"), "n_pixels": n}


def coarse_geobox(aoi, factor):
    from odc.geo.geobox import GeoBox
    crs, transform, width, height = common.aoi_grid(aoi)
    return GeoBox((height // factor, width // factor), common.scaled_transform(transform, factor), crs)


def optical_check(paths, aoi, logger):
    """Sentinel-2 MNDWI water vs model water for a flood pass, on clear pixels."""
    import odc.stac
    import planetary_computer
    catalog = common.open_planetary_computer()
    candidates = []
    for day in aoi["flood_dates"]:
        d = date.fromisoformat(day)
        window = f"{(d - timedelta(days=config.S2_MAX_DAYS)).isoformat()}/" \
                 f"{(d + timedelta(days=config.S2_MAX_DAYS)).isoformat()}"
        items = common.retry(lambda: list(catalog.search(
            collections=[config.S2_COLLECTION], bbox=aoi["bbox_lonlat"], datetime=window,
            query={"eo:cloud_cover": {"lt": config.S2_MAX_CLOUD}}).items()),
            what="Sentinel-2 search", logger=logger)
        for it in items:
            gap = abs((it.datetime.date() - d).days)
            candidates.append((gap, it.properties.get("eo:cloud_cover", 100), day, it))
    if not candidates:
        return {"status": "NOT POSSIBLE",
                "reason": f"no Sentinel-2 L2A scene within {config.S2_MAX_DAYS} days of any flood "
                          f"pass with cloud cover < {config.S2_MAX_CLOUD}% (normal in the monsoon)"}
    candidates.sort(key=lambda c: (c[0], c[1]))
    gap, _, flood_day, first = candidates[0]
    s2_day = first.datetime.date()
    items = [c[3] for c in candidates if c[2] == flood_day and c[3].datetime.date() == s2_day]
    items = [planetary_computer.sign(i) for i in items]
    factor = 2  # compare on a 20 m grid (B11 is 20 m)
    gbox = coarse_geobox(aoi, factor)
    ds = odc.stac.load(items, bands=["B03", "B11", "SCL"], geobox=gbox, groupby="solar_day",
                       resampling={"SCL": "nearest", "*": "average"},
                       chunks={"x": 2048, "y": 2048}).isel(time=0).compute()
    green = ds["B03"].values.astype(np.float32)
    swir = ds["B11"].values.astype(np.float32)
    baseline = str(items[0].properties.get("s2:processing_baseline", "00.00"))
    if baseline >= "04.00":  # newer products carry a +1000 offset
        green, swir = green - 1000, swir - 1000
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - swir) / (green + swir)
    scl = ds["SCL"].values
    clear = np.isin(scl, [4, 5, 6, 7]) & np.isfinite(mndwi) & (green > 0)
    water_full = read_u8(paths.outputs / "water_by_pass" / "flood" / f"{flood_day}.tif")
    h, w = clear.shape
    model_water = common.block_reduce((water_full == 1).astype(np.float32), factor)[:h, :w] >= 0.5
    model_valid = common.block_reduce((water_full != NODATA_U8).astype(np.uint8), factor, "min")[:h, :w] == 1
    valid = clear & model_valid
    clear_frac = float(valid.sum() / max(model_valid.sum(), 1))
    if clear_frac < config.S2_MIN_CLEAR_FRACTION:
        return {"status": "NOT POSSIBLE", "reason": f"best scene {s2_day} (for pass {flood_day}) "
                                                    f"is only {clear_frac:.1%} clear over the AOI"}
    metrics = compare_masks(model_water, mndwi > 0, valid)
    return {"status": "DONE", "flood_pass": flood_day, "s2_date": s2_day.isoformat(),
            "days_apart": gap, "clear_fraction": clear_frac, "s2_items": [i.id for i in items],
            "metrics_model_vs_s2": metrics,
            "note": "Model water = U-Net water for that pass (flood + normal water). "
                    "Reference = MNDWI > 0 on cloud-free pixels (SCL 4-7). Days apart matter: "
                    "flood water can recede or rise between the two dates."}


def jrc_occurrence_on_grid(aoi, factor, logger):
    """JRC GSW occurrence (percent) on a coarse version of the flood grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    gbox = coarse_geobox(aoi, factor)
    try:
        import odc.stac
        catalog = common.open_planetary_computer()
        items = list(catalog.search(collections=[config.JRC_COLLECTION],
                                    bbox=aoi["bbox_lonlat"]).items())
        if not items:
            raise RuntimeError("no jrc-gsw items")
        ds = odc.stac.load(items, bands=["occurrence"], geobox=gbox, resampling="nearest")
        occ = ds["occurrence"].max(dim="time").values.astype(np.float32)
        return occ, "Planetary Computer jrc-gsw"
    except Exception as err:  # noqa: BLE001
        logger.info(f"jrc-gsw on Planetary Computer failed ({err}); trying the GCS tiles")
    occ = np.full(gbox.shape, 255, dtype=np.uint8)
    lon_min, lat_min, lon_max, lat_max = aoi["bbox_lonlat"]
    n = 0
    for lon in range(int(np.floor(lon_min / 10) * 10), int(np.floor(lon_max / 10) * 10) + 1, 10):
        for lat in range(int(np.ceil(lat_min / 10) * 10), int(np.ceil(lat_max / 10) * 10) + 1, 10):
            name_lon = f"{abs(lon)}{'E' if lon >= 0 else 'W'}"
            name_lat = f"{abs(lat)}{'N' if lat >= 0 else 'S'}"
            url = config.JRC_GCS_URL.format(lon=name_lon, lat=name_lat)
            try:
                with rasterio.open("/vsicurl/" + url) as src:
                    reproject(rasterio.band(src, 1), occ, dst_transform=gbox.transform,
                              dst_crs=str(gbox.crs), resampling=Resampling.nearest,
                              src_nodata=255, dst_nodata=255, init_dest_nodata=False)
                n += 1
            except Exception as err:  # noqa: BLE001
                logger.warning(f"GSW tile {url} not read: {err}")
    if n == 0:
        raise RuntimeError("no JRC GSW tile could be read")
    occ = occ.astype(np.float32)
    occ[occ == 255] = np.nan
    return occ, "JRC GSW v1.4 2021 tiles on Google Cloud Storage"


def jrc_check(paths, aoi, logger):
    factor = 3  # compare on a 30 m grid (GSW is 30 m)
    occ, source = jrc_occurrence_on_grid(aoi, factor, logger)
    normal = read_u8(paths.outputs / "normal_water.tif")
    h, w = occ.shape
    ours = common.block_reduce((normal == 1).astype(np.float32), factor)[:h, :w] >= 0.5
    ours_valid = common.block_reduce((normal != NODATA_U8).astype(np.uint8), factor, "min")[:h, :w] == 1
    valid = ours_valid & np.isfinite(occ)
    gsw = np.nan_to_num(occ) >= config.JRC_OCCURRENCE_MIN
    metrics = compare_masks(ours, gsw, valid)
    return {"status": "DONE", "source": source, "occurrence_threshold": config.JRC_OCCURRENCE_MIN,
            "metrics_normal_water_vs_gsw": metrics,
            "note": "GSW occurrence = percent of 1984-2021 observations that were water. "
                    "Our normal water comes from only a few monsoon-season passes, so some "
                    "disagreement (seasonal water, paddy) is expected."}


def visual_figure(paths, aoi, out_png):
    """Radar / model water / flood for the most flooded pass: overview + zoomed crop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap
    best_day, best_n = None, -1
    for day in aoi["flood_dates"]:
        p = paths.outputs / "flood_by_pass" / f"{day}.tif"
        if p.exists():
            n = int((read_u8(p) == 1).sum())
            if n > best_n:
                best_day, best_n = day, n
    with rasterio.open(paths.assam / "flood" / f"{best_day}.tif") as src:
        vv = src.read(1)
    water = read_u8(paths.outputs / "water_by_pass" / "flood" / f"{best_day}.tif")
    flood = read_u8(paths.outputs / "flood_by_pass" / f"{best_day}.tif")
    normal = read_u8(paths.outputs / "normal_water.tif")
    classes = np.zeros(flood.shape, dtype=np.uint8)  # 0 no data, 1 land, 2 normal water, 3 flood
    classes[flood != NODATA_U8] = 1
    classes[normal == 1] = 2
    classes[flood == 1] = 3
    cmap4 = ListedColormap(["#bdbdbd", "#f4f1ea", "#2b8cbe", "#d7301f"])
    h, w = vv.shape
    f = max(1, int(np.ceil(max(h, w) / 1200)))
    # zoom: the 1000 x 1000 px window with the most flood
    z = min(1000, h, w)
    fl = (flood == 1).astype(np.float32)
    zf = max(1, z // 4)
    coarse = common.block_reduce(fl, zf, "sum") if min(h, w) >= zf else fl
    r, c = np.unravel_index(np.argmax(coarse), coarse.shape)
    r0 = int(min(max(r * zf - z // 2, 0), h - z))
    c0 = int(min(max(c * zf - z // 2, 0), w - z))
    sl = (slice(r0, r0 + z), slice(c0, c0 + z))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (cut, step, label) in enumerate([((slice(None), slice(None)), f, "whole AOI"),
                                              (sl, 1, "zoom (full 10 m resolution)")]):
        axes[row, 0].imshow(vv[cut][::step, ::step], cmap="gray", vmin=-25, vmax=0)
        axes[row, 0].set_title(f"{best_day} VV dB ({label})", fontsize=9)
        axes[row, 1].imshow(np.where(water[cut] == NODATA_U8, 0, water[cut] + 1)[::step, ::step],
                            cmap=ListedColormap(["#bdbdbd", "#f4f1ea", "#2b8cbe"]), vmin=0, vmax=2,
                            interpolation="nearest")
        axes[row, 1].set_title("U-Net water (blue)", fontsize=9)
        axes[row, 2].imshow(classes[cut][::step, ::step], cmap=cmap4, vmin=0, vmax=3,
                            interpolation="nearest")
        axes[row, 2].set_title("flood (red), normal water (blue), no data (grey)", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    return best_day


def run_sanity_check(smoke=False, logger=None, **kwargs):
    paths = config.get_paths(smoke)
    aoi = common.load_aoi(paths)
    report = {"smoke": smoke, "generated_at": datetime.now().isoformat()}
    for name, func in [("sentinel2_optical", optical_check), ("jrc_gsw_normal_water", jrc_check)]:
        try:
            report[name] = func(paths, aoi, logger)
        except Exception as err:  # noqa: BLE001
            short, tb = common.format_exception(err)
            logger.warning(f"{name} check failed: {short}")
            report[name] = {"status": "FAILED", "reason": short}
        logger.info(f"{name}: {report[name].get('status')} {report[name].get('reason', '')}")
    try:
        day = visual_figure(paths, aoi, paths.results / "assam_sanity_visual.png")
        report["visual_figure"] = {"status": "DONE", "pass": day,
                                   "file": "results/assam_sanity_visual.png"}
    except Exception as err:  # noqa: BLE001
        report["visual_figure"] = {"status": "FAILED", "reason": common.format_exception(err)[0]}
    with open(paths.results / "sanity_check.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    def fmt(m):
        return ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items())
    lines = ["# Sanity checks for the Assam flood maps", "",
             "Generated by flood_map.py (stage sanity_check). "
             + ("SMOKE RUN: numbers are not meaningful." if smoke else ""), ""]
    s2 = report["sentinel2_optical"]
    lines += ["## 1. Independent optical check (Sentinel-2 MNDWI)", "", f"Status: **{s2['status']}**"]
    if s2["status"] == "DONE":
        lines += [f"- flood pass {s2['flood_pass']} vs Sentinel-2 {s2['s2_date']} "
                  f"({s2['days_apart']} days apart), clear fraction {s2['clear_fraction']:.3f}",
                  f"- model water vs MNDWI water: {fmt(s2['metrics_model_vs_s2'])}", f"- {s2['note']}"]
    else:
        lines.append(f"- reason: {s2.get('reason')}")
    j = report["jrc_gsw_normal_water"]
    lines += ["", "## 2. Normal-water mask vs JRC Global Surface Water", "", f"Status: **{j['status']}**"]
    if j["status"] == "DONE":
        lines += [f"- source: {j['source']}, GSW occurrence >= {j['occurrence_threshold']}%",
                  f"- {fmt(j['metrics_normal_water_vs_gsw'])}", f"- {j['note']}"]
    else:
        lines.append(f"- reason: {j.get('reason')}")
    v = report["visual_figure"]
    lines += ["", "## 3. Visual figure", "", f"Status: **{v['status']}**"]
    lines.append(f"- {v.get('file', '')} (pass {v.get('pass', '')}): open it next to the radar "
                 "and check that red flood areas sit next to rivers and low ground."
                 if v["status"] == "DONE" else f"- reason: {v.get('reason')}")
    lines += ["", "## Not done", "",
              "- No comparison with official government flood maps (cannot be automated). "
              "Nothing here claims agreement with them."]
    (paths.results / "sanity_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["visual_figure"]["status"] != "DONE":
        return common.result("FAILED", "visual figure could not be produced")
    warnings = [f"{k}: {report[k]['status']} ({report[k].get('reason', '')})"
                for k in ("sentinel2_optical", "jrc_gsw_normal_water") if report[k]["status"] != "DONE"]
    return common.result("OK", "", warnings)


if __name__ == "__main__":
    import run_all
    sys.exit(run_all.main(default_stages=STAGES_IN_THIS_FILE))
