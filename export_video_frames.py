"""
export_video_frames.py - clean 1920x1080 frames for the video, plus GeoTIFFs for the map.

Writes to results_final/:
  01_radar_2019-07-23_whole.png       VV radar, flood pass, whole area
  02_radar_2018-07-22_normal_whole.png VV radar, normal year, whole area (same framing as 01)
  03_zoom_radar.png                   zoom close-up: VV radar
  04_zoom_unet_water.png              zoom close-up: U-Net water
  05_zoom_flood_vs_normal.png         zoom close-up: flood vs normal water
  06_zoom_flood_buildings.png         zoom close-up: flood map + Open Buildings (or WorldPop)
  flood_extent.tif, flood_<day>.tif, normal_water.tif   masks for the top-20 map
  README.md                           colours, sources, zoom bounds

No titles, axes, legends or colorbars. 01/02 share one framing and 03-06 share another, so
they can be cross-faded in the video. The zoom is centred on the same spot as the zoom row in
results/assam_sanity_visual.png (the most flooded 1000 x 1000 px window), widened to 16:9 at
full 10 m resolution (one radar pixel = one video pixel).

Needs the full-run files in data/ and outputs/ (build machine). Run:
    python export_video_frames.py
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import common
import config

W, H = 1920, 1080
NODATA_U8 = 255
FLOOD_DAY = "2019-07-23"
NORMAL_DAY = "2018-07-22"   # 2018 pass closest in calendar date to the flood pass
VV_MIN, VV_MAX = -25.0, 0.0  # same stretch as assam_sanity_visual.png

# Same colours as the pipeline figures.
GREY, LAND, WATER, FLOOD = "#bdbdbd", "#f4f1ea", "#2b8cbe", "#d7301f"
BLDG_DRY, BLDG_FLOODED = "#3a3a3a", "#ffd60a"
POP_COLOR = "#6a3d9a"


def rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def palette_image(classes, colors):
    lut = np.array([rgb(c) for c in colors], dtype=np.uint8)
    return lut[classes]


def radar_rgb(vv):
    """VV dB -> grey RGB, no data black."""
    g = np.clip((vv - VV_MIN) / (VV_MAX - VV_MIN), 0, 1)
    g = np.where(np.isfinite(vv), g * 255, 0).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def save_png(arr, path):
    from PIL import Image
    img = arr if hasattr(arr, "save") else Image.fromarray(arr)
    if img.size != (W, H):
        raise RuntimeError(f"{path.name}: {img.size} is not {W}x{H}")
    img.save(path, optimize=True)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Whole area: fit the full grid inside 1920x1080, letterbox in black
# ---------------------------------------------------------------------------
def whole_area_frame(radar_path, height, width):
    import rasterio
    from rasterio.enums import Resampling
    s = min(W / width, H / height)
    ow, oh = max(1, round(width * s)), max(1, round(height * s))
    with rasterio.open(radar_path) as src:
        resampling = Resampling.average if s < 1 else Resampling.bilinear
        vv = src.read(1, out_shape=(oh, ow), resampling=resampling, masked=True)
    vv = np.ma.filled(vv.astype(np.float32), np.nan)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    r0, c0 = (H - oh) // 2, (W - ow) // 2
    frame[r0:r0 + oh, c0:c0 + ow] = radar_rgb(vv)
    return frame


# ---------------------------------------------------------------------------
# Zoom window
# ---------------------------------------------------------------------------
def zoom_center(flood):
    """Centre of the zoom in flood_map.visual_figure (same maths, same result)."""
    h, w = flood.shape
    z = min(1000, h, w)
    fl = (flood == 1).astype(np.float32)
    zf = max(1, z // 4)
    coarse = common.block_reduce(fl, zf, "sum") if min(h, w) >= zf else fl
    r, c = np.unravel_index(np.argmax(coarse), coarse.shape)
    r0 = int(min(max(r * zf - z // 2, 0), h - z))
    c0 = int(min(max(c * zf - z // 2, 0), w - z))
    return r0 + z // 2, c0 + z // 2


def zoom_window(center, height, width):
    """16:9 window at native resolution, shifted to stay inside the grid."""
    zh, zw = min(H, height), min(W, width)
    if zw / zh > W / H:      # grid too short for 16:9 at native size: trim width
        zw = round(zh * W / H)
    else:
        zh = round(zw * H / W)
    r0 = int(min(max(center[0] - zh // 2, 0), height - zh))
    c0 = int(min(max(center[1] - zw // 2, 0), width - zw))
    return r0, c0, zh, zw


def to_frame(arr, nearest):
    """Resize a window array (h, w, 3) to exactly 1920x1080 if it is not already."""
    from PIL import Image
    img = Image.fromarray(arr)
    if img.size != (W, H):
        img = img.resize((W, H), Image.NEAREST if nearest else Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Buildings / population overlay
# ---------------------------------------------------------------------------
def load_buildings(paths, aoi):
    import pandas as pd

    from build_villages import bbox_key
    f = paths.cache / "open_buildings" / f"aoi_{bbox_key(aoi['bbox_lonlat'])}.csv.gz"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df["longitude"].to_numpy(), df["latitude"].to_numpy()


def draw_overlay(base_img, flood_win, win, transform, paths, aoi, crs):
    """Draw buildings (or WorldPop cells) on the flood map. Returns (image, source note)."""
    from PIL import ImageDraw
    from pyproj import Transformer
    r0, c0, zh, zw = win
    sx, sy = W / zw, H / zh
    img = base_img.convert("RGB")
    draw = ImageDraw.Draw(img)

    bld = load_buildings(paths, aoi)
    if bld is not None:
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*bld)
        row, col = common.xy_to_rowcol(transform, x, y)
        row, col = row - r0, col - c0
        inside = (row >= 0) & (row < zh) & (col >= 0) & (col < zw)
        row, col = row[inside], col[inside]
        flooded = flood_win[row.astype(int), col.astype(int)] == 1
        rad = 2.0 * max(sx, sy)
        # dry first, flooded on top
        for sel, color in ((~flooded, BLDG_DRY), (flooded, BLDG_FLOODED)):
            for rr, cc in zip(row[sel] * sy, col[sel] * sx):
                draw.ellipse([cc - rad, rr - rad, cc + rad, rr + rad], fill=color)
        note = (f"Google Open Buildings v3 centroids (confidence >= "
                f"{config.OPEN_BUILDINGS_MIN_CONFIDENCE}): {int(inside.sum())} in view, "
                f"{int(flooded.sum())} on flooded pixels (yellow), rest dark grey")
        return img, note

    from build_villages import get_worldpop, population_points

    class _Quiet:
        def info(self, *a, **k): pass
        warning = info
    wp_path, _ = get_worldpop(paths, aoi, _Quiet())
    if wp_path is None:
        return img, "no buildings cache and no WorldPop raster found: overlay is empty"
    x, y, pop = population_points(wp_path, aoi)
    row, col = common.xy_to_rowcol(transform, x, y)
    row, col = row - r0, col - c0
    inside = (row >= 0) & (row < zh) & (col >= 0) & (col < zw) & (pop >= 1)
    half = 50 / aoi["grid"]["pixel_size_m"]  # WorldPop cells are 100 m
    over = img.copy()
    od = ImageDraw.Draw(over)
    for rr, cc in zip(row[inside], col[inside]):
        od.rectangle([(cc - half) * sx, (rr - half) * sy, (cc + half) * sx, (rr + half) * sy],
                     fill=POP_COLOR)
    from PIL import Image
    img = Image.blend(img, over, 0.55)
    return img, f"WorldPop 2020 100 m cells with >= 1 person (purple, 55% opacity): {int(inside.sum())}"


# ---------------------------------------------------------------------------
def main():
    import rasterio
    from rasterio.windows import Window

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(config.ROOT / "results_final"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = config.get_paths(False)
    aoi = common.load_aoi(paths)
    crs, transform, width, height = common.aoi_grid(aoi)
    flood_radar = paths.assam / "flood" / f"{FLOOD_DAY}.tif"
    normal_radar = paths.assam / "normal" / f"{NORMAL_DAY}.tif"
    water_p = paths.outputs / "water_by_pass" / "flood" / f"{FLOOD_DAY}.tif"
    flood_p = paths.outputs / "flood_by_pass" / f"{FLOOD_DAY}.tif"
    normal_p = paths.outputs / "normal_water.tif"
    extent_p = paths.outputs / "flood_extent.tif"
    missing = [p for p in (flood_radar, normal_radar, water_p, flood_p, normal_p, extent_p)
               if not p.exists()]
    if missing:
        raise SystemExit("Missing inputs (run the full pipeline first):\n  "
                         + "\n  ".join(str(p) for p in missing))

    print("Whole-area radar frames")
    save_png(whole_area_frame(flood_radar, height, width), out / f"01_radar_{FLOOD_DAY}_whole.png")
    save_png(whole_area_frame(normal_radar, height, width),
             out / f"02_radar_{NORMAL_DAY}_normal_whole.png")

    print("Zoom frames")
    with rasterio.open(flood_p) as src:
        flood_full = src.read(1)
    win = zoom_window(zoom_center(flood_full), height, width)
    r0, c0, zh, zw = win
    rw = Window(c0, r0, zw, zh)
    flood = flood_full[r0:r0 + zh, c0:c0 + zw]
    del flood_full
    with rasterio.open(flood_radar) as src:
        vv = src.read(1, window=rw)
    with rasterio.open(water_p) as src:
        water = src.read(1, window=rw)
    with rasterio.open(normal_p) as src:
        normal = src.read(1, window=rw)

    save_png(to_frame(radar_rgb(vv), nearest=False), out / "03_zoom_radar.png")
    water_cls = np.where(water == NODATA_U8, 0, water + 1).astype(np.uint8)
    save_png(to_frame(palette_image(water_cls, [GREY, LAND, WATER]), nearest=True),
             out / "04_zoom_unet_water.png")
    classes = np.zeros(flood.shape, dtype=np.uint8)  # same classes as assam_sanity_visual.png
    classes[flood != NODATA_U8] = 1
    classes[normal == 1] = 2
    classes[flood == 1] = 3
    flood_img = to_frame(palette_image(classes, [GREY, LAND, WATER, FLOOD]), nearest=True)
    save_png(flood_img, out / "05_zoom_flood_vs_normal.png")
    overlay, overlay_note = draw_overlay(flood_img, flood, win, transform, paths, aoi, crs)
    save_png(overlay, out / "06_zoom_flood_buildings.png")

    print("GeoTIFFs")
    for src, name in ((extent_p, "flood_extent.tif"), (flood_p, f"flood_{FLOOD_DAY}.tif"),
                      (normal_p, "normal_water.tif")):
        shutil.copy2(src, out / name)
        print(f"  wrote {out / name}")

    from rasterio.transform import xy
    from pyproj import Transformer
    xs, ys = xy(transform, [r0, r0 + zh], [c0, c0 + zw], offset="ul")
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    px = aoi["grid"]["pixel_size_m"]
    info = {
        "flood_day": FLOOD_DAY, "normal_day": NORMAL_DAY, "crs": crs, "pixel_size_m": px,
        "grid_width_px": width, "grid_height_px": height,
        "zoom_window_px": {"row": r0, "col": c0, "height": zh, "width": zw},
        "zoom_bbox_lonlat": [min(lon), min(lat), max(lon), max(lat)],
        "zoom_size_km": [zw * px / 1000, zh * px / 1000],
        "overlay": overlay_note,
    }
    (out / "frames_info.json").write_text(json.dumps(info, indent=2))
    write_readme(out, info)
    print("Done.")


def write_readme(out, info):
    z = info["zoom_bbox_lonlat"]
    text = f"""# results_final: video frames and map layers

Generated by `python export_video_frames.py`. Do not edit by hand. All PNGs are exactly
{W}x{H}, with no titles, axes, legends or colorbars.

| File | What it shows |
|---|---|
| `01_radar_{info['flood_day']}_whole.png` | Sentinel-1 VV radar, flood pass {info['flood_day']}, whole area. Dark = smooth surface (water). |
| `02_radar_{info['normal_day']}_normal_whole.png` | VV radar, normal year {info['normal_day']} (2018 pass closest in calendar date), same framing as 01 so the two cross-fade. |
| `03_zoom_radar.png` | Zoom close-up, VV radar. |
| `04_zoom_unet_water.png` | Zoom close-up, U-Net water. |
| `05_zoom_flood_vs_normal.png` | Zoom close-up, flood vs normal water. |
| `06_zoom_flood_buildings.png` | Zoom close-up, flood map with buildings on top. |
| `flood_extent.tif` | Maximum flood extent over both 2019 passes (the layer the ranking uses). |
| `flood_{info['flood_day']}.tif` | Flood on the {info['flood_day']} pass only. |
| `normal_water.tif` | Normal (non-flood) water from the 2018 and 2021 passes. |

Frames 03-06 are the same window, pixel for pixel.

## Colours

- Radar (01-03): grey stretch from {VV_MIN:.0f} dB (black) to {VV_MAX:.0f} dB (white). Black bars and no-data are black.
- U-Net water (04): blue `{WATER}` water, cream `{LAND}` land, grey `{GREY}` no data.
- Flood vs normal (05, 06): red `{FLOOD}` flood water, blue `{WATER}` normal water, cream `{LAND}` land, grey `{GREY}` no data.
- Overlay (06): {info['overlay']}.

## Zoom area

- Centred on the zoom in `results/assam_sanity_visual.png` (the most flooded 1000 x 1000 px window),
  widened to 16:9 at the native {info['pixel_size_m']:.0f} m pixel size: {info['zoom_size_km'][0]:.1f} x {info['zoom_size_km'][1]:.1f} km.
- Lon/lat box: {z[0]:.5f}, {z[1]:.5f} to {z[2]:.5f}, {z[3]:.5f}.

## GeoTIFFs (for the top-20 map)

- CRS `{info['crs']}` ({info['pixel_size_m']:.0f} m pixels), uint8, LZW-compressed.
- Values: `1` = flood (or normal water in `normal_water.tif`), `0` = not, `255` = no data (set as nodata).
- In QGIS: Symbology > Paletted/Unique values, make 0 transparent and colour 1. QGIS reprojects
  on the fly, so the UTM CRS is fine next to lon/lat village points from `results/village_risk_ranked.csv`.
- `.tif` is gitignored, so these files are not in the repo; send them directly.
"""
    (out / "README.md").write_text(text)
    print(f"  wrote {out / 'README.md'}")


if __name__ == "__main__":
    main()
