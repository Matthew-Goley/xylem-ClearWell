"""
Otsu-on-VV baseline vs the U-Net, for the Assam flood pass 2019-08-10.

Read-only on existing data and results. The only file written is
results/otsu_vs_unet_assam.md.

Steps:
  1. Read VV dB (band 1) of data/assam/flood/2019-08-10.tif, clip it like the
     model input (config.CLIP_DB_MIN..MAX), compute one Otsu threshold over all
     valid pixels of the AOI. Otsu water = VV < threshold.
  2. Rebuild the Sentinel-2 reference exactly as the sanity_check stage did:
     the same S2 items (ids read from results/sanity_check.json), same 20 m
     grid, MNDWI > 0 as water, clear = SCL 4-7, and the same U-Net validity mask.
  3. Bring Otsu water to the 20 m grid the same way the U-Net mask is (block
     mean >= 0.5) and score both with flood_map.compare_masks on identical pixels.
  4. Recompute the U-Net numbers as a check that the reference really is the same.

Run: python compare_otsu_assam.py
"""
import json
from datetime import datetime

import numpy as np
import rasterio

import common
import config
import flood_map
from flood_map import NODATA_U8, coarse_geobox, compare_masks, read_u8

FLOOD_DAY = "2019-08-10"
FACTOR = 2  # 20 m comparison grid, as in flood_map.optical_check
UNET_REPORTED = {"iou": 0.402, "precision": 0.582, "recall": 0.565}


def otsu_threshold(values, bins=2048):
    """Otsu threshold of a 1D array (histogram based, maximises between-class variance)."""
    lo, hi = float(values.min()), float(values.max())
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    hist = hist.astype(np.float64)
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    k = int(np.argmax(between))
    return float(edges[k + 1])  # water = value < threshold


def load_s2_reference(paths, aoi, item_ids):
    """MNDWI and clear mask on the 20 m grid, same code path as optical_check."""
    import odc.stac
    import planetary_computer
    catalog = common.open_planetary_computer()
    items = common.retry(lambda: list(catalog.search(collections=[config.S2_COLLECTION],
                                                     ids=item_ids).items()),
                         what="Sentinel-2 search by id")
    found = sorted(i.id for i in items)
    if found != sorted(item_ids):
        raise RuntimeError(f"expected S2 items {sorted(item_ids)}, got {found}")
    # keep the order recorded by sanity_check (it affects mosaic order)
    by_id = {i.id: i for i in items}
    items = [planetary_computer.sign(by_id[i]) for i in item_ids]
    gbox = coarse_geobox(aoi, FACTOR)
    ds = odc.stac.load(items, bands=["B03", "B11", "SCL"], geobox=gbox, groupby="solar_day",
                       resampling={"SCL": "nearest", "*": "average"},
                       chunks={"x": 2048, "y": 2048}).isel(time=0).compute()
    green = ds["B03"].values.astype(np.float32)
    swir = ds["B11"].values.astype(np.float32)
    baseline = str(items[0].properties.get("s2:processing_baseline", "00.00"))
    if baseline >= "04.00":
        green, swir = green - 1000, swir - 1000
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - swir) / (green + swir)
    clear = np.isin(ds["SCL"].values, [4, 5, 6, 7]) & np.isfinite(mndwi) & (green > 0)
    return mndwi, clear


def fmt(x):
    return f"{x:.3f}"


def main():
    paths = config.get_paths(False)
    aoi = common.load_aoi(paths)
    sanity = json.loads((paths.results / "sanity_check.json").read_text())["sentinel2_optical"]
    assert sanity["flood_pass"] == FLOOD_DAY, sanity["flood_pass"]
    s2_ids = sanity["s2_items"]

    # 1. Otsu on VV dB
    with rasterio.open(paths.data / "assam" / "flood" / f"{FLOOD_DAY}.tif") as src:
        assert (src.descriptions[0] or "VV_dB").upper().startswith("VV"), src.descriptions
        vv = src.read(1).astype(np.float32)
    vv_ok = np.isfinite(vv)
    vv_c = np.clip(vv, config.CLIP_DB_MIN, config.CLIP_DB_MAX)
    thr = otsu_threshold(vv_c[vv_ok])
    otsu_full = vv_ok & (vv_c < thr)
    print(f"Otsu threshold on VV: {thr:.2f} dB over {int(vv_ok.sum()):,} valid 10 m pixels")

    # 2. reference + masks
    mndwi, clear = load_s2_reference(paths, aoi, s2_ids)
    h, w = clear.shape
    water_full = read_u8(paths.outputs / "water_by_pass" / "flood" / f"{FLOOD_DAY}.tif")
    unet = common.block_reduce((water_full == 1).astype(np.float32), FACTOR)[:h, :w] >= 0.5
    unet_valid = common.block_reduce((water_full != NODATA_U8).astype(np.uint8), FACTOR, "min")[:h, :w] == 1
    otsu = common.block_reduce(otsu_full.astype(np.float32), FACTOR)[:h, :w] >= 0.5
    otsu_valid = common.block_reduce(vv_ok.astype(np.uint8), FACTOR, "min")[:h, :w] == 1
    valid = clear & unet_valid  # identical to sanity_check
    ref = mndwi > 0
    otsu_gap = int((valid & ~otsu_valid).sum())

    # 3./4. metrics
    m_unet = compare_masks(unet, ref, valid)
    m_otsu = compare_masks(otsu, ref, valid)
    for k, v in UNET_REPORTED.items():
        if abs(m_unet[k] - v) > 0.002:
            raise RuntimeError(f"U-Net {k} recomputed as {m_unet[k]:.4f}, reported {v}: "
                               "reference does not match sanity_check")
    if m_unet["n_pixels"] != sanity["metrics_model_vs_s2"]["n_pixels"]:
        raise RuntimeError("valid pixel count differs from sanity_check")

    counts = {
        "ref": int((ref & valid).sum()),
        "unet": int((unet & valid).sum()),
        "otsu": int((otsu & valid).sum()),
        "unet_aoi": int((unet & unet_valid).sum()),
        "otsu_aoi": int((otsu & otsu_valid).sum()),
        "unet_aoi_10m": int((water_full == 1).sum()),
        "otsu_aoi_10m": int(otsu_full.sum()),
    }
    both = int((unet & otsu & valid).sum())
    print(json.dumps({"unet": m_unet, "otsu": m_otsu, "counts": counts}, indent=2))

    n = m_unet["n_pixels"]
    pct = lambda c: f"{c / n:.1%}"  # noqa: E731
    lines = [
        "# Otsu (VV) vs U-Net on Assam, flood pass 2019-08-10", "",
        f"Generated by compare_otsu_assam.py on {datetime.now():%Y-%m-%d %H:%M}. "
        "One scene only; read nothing general into it.", "",
        "## Setup", "",
        f"- Radar: `data/assam/flood/{FLOOD_DAY}.tif`, VV dB clipped to "
        f"[{config.CLIP_DB_MIN}, {config.CLIP_DB_MAX}] like the model input.",
        f"- Otsu: one global threshold over all {int(vv_ok.sum()):,} valid 10 m pixels of the AOI: "
        f"**{thr:.2f} dB**. Water = VV < threshold.",
        f"- Reference: Sentinel-2 MNDWI > 0 on {sanity['s2_date']} ({sanity['days_apart']} days after the "
        "radar pass), same 6 S2 items as `results/sanity_check.json`.",
        "- Pixels scored: cloud-free (SCL 4-7) and valid in the U-Net map, on the 20 m grid, the exact "
        f"mask the sanity_check stage used ({n:,} pixels). Both 10 m masks are reduced to 20 m the same "
        "way (block mean >= 0.5).",
        f"- Check: recomputing the U-Net scores on this mask gives IoU {m_unet['iou']:.3f}, precision "
        f"{m_unet['precision']:.3f}, recall {m_unet['recall']:.3f}, matching the reported numbers.",
        f"- Scored pixels where VV is missing (Otsu forced to dry): {otsu_gap:,}.", "",
        "## Scores against MNDWI", "",
        "| Method | IoU | Precision | Recall | Agreement |",
        "|---|---|---|---|---|",
        f"| U-Net | {fmt(m_unet['iou'])} | {fmt(m_unet['precision'])} | {fmt(m_unet['recall'])} | "
        f"{fmt(m_unet['agreement'])} |",
        f"| Otsu on VV | {fmt(m_otsu['iou'])} | {fmt(m_otsu['precision'])} | {fmt(m_otsu['recall'])} | "
        f"{fmt(m_otsu['agreement'])} |", "",
        "## Water pixel counts", "",
        "| | On the scored pixels (20 m) | Whole AOI (20 m) | Whole AOI (10 m) |",
        "|---|---|---|---|",
        f"| MNDWI reference | {counts['ref']:,} ({pct(counts['ref'])}) | n/a | n/a |",
        f"| U-Net | {counts['unet']:,} ({pct(counts['unet'])}) | {counts['unet_aoi']:,} | "
        f"{counts['unet_aoi_10m']:,} |",
        f"| Otsu on VV | {counts['otsu']:,} ({pct(counts['otsu'])}) | {counts['otsu_aoi']:,} | "
        f"{counts['otsu_aoi_10m']:,} |", "",
        f"On the scored pixels, {both:,} are water in both U-Net and Otsu.", "",
        "## Caveats", "",
        "- One radar pass, one optical scene, 2 days apart; water can rise or recede in between.",
        "- MNDWI > 0 is itself an imperfect reference (flooded vegetation, turbid water, shadows).",
        "- Only the ~56% of the AOI that was cloud-free is scored; cloudy areas may differ.",
        "- Otsu here is a single global threshold, the simplest form; tiled or split-based Otsu "
        "variants could score differently.",
    ]
    out = paths.results / "otsu_vs_unet_assam.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
