"""End-to-end test of the Assam part of the pipeline on SYNTHETIC rasters.

flood_map -> villages -> risk -> validate -> summary, all inside tmp_path.
Every network call (model weights, OSM, WorldPop, buildings, DEM, boundaries)
is replaced by synthetic local data. Nothing here touches results/ in the repo.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import common
import config

CRS = "EPSG:32646"
X0, Y0, PX = 500000.0, 3000000.0, 10.0
H, W = 400, 600
FLOOD_DATES = ["2020-07-01", "2020-07-07", "2020-07-13", "2020-07-25"]
NORMAL_DATES = ["2018-07-02", "2021-07-03"]
STATS = {"vv_mean": -10.0, "vv_std": 4.0, "vh_mean": -17.0, "vh_std": 5.0}


class PixelModel(torch.nn.Module):
    """Stand-in for the trained U-Net: dark VV = water."""

    def forward(self, x):
        return -3.0 * x[:, :1] - 3.0


def _transform():
    from affine import Affine
    return Affine(PX, 0, X0, 0, -PX, Y0)


def _bbox_lonlat():
    from pyproj import Transformer
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lons, lats = t.transform([X0, X0 + W * PX, X0, X0 + W * PX], [Y0, Y0, Y0 - H * PX, Y0 - H * PX])
    return [min(lons) - 0.01, min(lats) - 0.01, max(lons) + 0.01, max(lats) + 0.01]


def _make_inputs(paths):
    import geopandas as gpd
    from shapely.geometry import box
    rng = np.random.default_rng(0)
    tr = _transform()
    aoi = {"smoke": False, "districts": ["DistA", "DistB"], "district_source": "synthetic",
           "collection": "synthetic", "orbit": {"orbit_state": "descending", "relative_orbit": 1},
           "flood_dates": FLOOD_DATES, "normal_dates": NORMAL_DATES,
           "grid": {"crs": CRS, "transform": list(tr)[:6], "width": W, "height": H, "pixel_size_m": PX},
           "bbox_lonlat": _bbox_lonlat()}
    paths.aoi.write_text(json.dumps(aoi))
    xmid = X0 + W * PX / 2
    gpd.GeoDataFrame({"district": ["DistA", "DistB"]},
                     geometry=[box(X0, Y0 - H * PX, xmid, Y0), box(xmid, Y0 - H * PX, X0 + W * PX, Y0)],
                     crs=CRS).to_file(paths.outputs / "aoi_districts.gpkg", driver="GPKG")
    # Radar: land -8 dB, a permanent river (rows 300-319), a flood (rows 50-149, cols 50-249)
    # present in the first three flood passes only. Pass 2 has a no-data strip.
    for kind, dates in [("flood", FLOOD_DATES), ("normal", NORMAL_DATES)]:
        for k, d in enumerate(dates):
            vv = -8.0 + rng.normal(0, 0.5, (H, W))
            vv[300:320, :] = -22.0
            if kind == "flood" and k < 3:
                vv[50:150, 50:250] = -21.0
            vv[0:3, 400:403] = -21.0  # tiny blob, must be removed by the cleanup
            radar = np.stack([vv, vv - 7]).astype(np.float32)
            if kind == "flood" and k == 1:
                radar[:, :, 590:] = np.nan
            common.write_tif(paths.assam / kind / f"{d}.tif", radar, CRS, tr, nodata=np.nan)
    return aoi


def _fake_worldpop(paths):
    import rasterio
    from rasterio.transform import from_origin
    w, s, e, n = _bbox_lonlat()
    res = 0.001
    width, height = int((e - w) / res), int((n - s) / res)
    pop = np.full((height, width), 5.0, dtype=np.float32)
    path = paths.cache / "worldpop" / "fake_pop.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(w, n, res, res), nodata=-99999) as dst:
        dst.write(pop, 1)
    return path


@pytest.fixture
def pipeline(tmp_paths, logger, monkeypatch):
    from pyproj import Transformer

    import build_villages
    import train
    aoi = _make_inputs(tmp_paths)
    ckpt = {"norm_stats": STATS, "saved_at": "synthetic", "smoke": False, "epoch": 0, "val_iou": 0.0,
            "encoder_weights": None}
    monkeypatch.setattr(train, "load_trained_model", lambda paths, device, logger=None: (PixelModel(), ckpt))

    def fake_dem(paths, aoi_, logger_):
        out = paths.outputs / "dem_on_grid.tif"
        dem = np.full((H, W), 90.0, dtype=np.float32)
        dem[40:160, 40:260] = 80.0  # flooded area is low ground
        common.write_tif(out, dem, CRS, _transform(), nodata=np.nan)
        return out
    monkeypatch.setattr(common, "dem_on_grid", fake_dem)

    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    rng = np.random.default_rng(1)
    xs = rng.uniform(X0 + 50, X0 + W * PX - 50, 120)
    ys = rng.uniform(Y0 - H * PX + 50, Y0 - 50, 120)
    lons, lats = to_ll.transform(xs, ys)
    places = [{"osm_id": f"node/{i}", "lat": la, "lon": lo, "name": f"Village {i}", "place": "village"}
              for i, (lo, la) in enumerate(zip(lons, lats))]
    monkeypatch.setattr(build_villages, "fetch_osm_places", lambda *a, **k: places)
    wp = _fake_worldpop(tmp_paths)
    monkeypatch.setattr(build_villages, "get_worldpop", lambda paths, aoi_, logger: (wp, "synthetic"))
    bx, by = rng.uniform(X0, X0 + W * PX, 500), rng.uniform(Y0 - H * PX, Y0, 500)
    blon, blat = to_ll.transform(bx, by)
    bld = pd.DataFrame({"longitude": blon, "latitude": blat, "confidence": 0.9})
    monkeypatch.setattr(build_villages, "get_buildings", lambda *a, **k: (bld, "synthetic"))
    pd.DataFrame({"district": ["DistA", "DistB"], "pct_shallow_well_handpump": [90.0, 60.0]}).to_csv(
        tmp_paths.manual / "census_water.csv", index=False)

    import geopandas as gpd
    from shapely.geometry import box
    lon0, lat0, lon1, lat1 = _bbox_lonlat()
    fake_gdf = gpd.GeoDataFrame({"district": ["DistA", "DistB", "Jorhat"]},
                                geometry=[box(lon0, lat0, lon1, lat1)] * 3, crs="EPSG:4326")
    monkeypatch.setattr(common, "get_districts", lambda paths, logger, state=None: (fake_gdf, "synthetic"))
    return tmp_paths, aoi, logger


def test_full_synthetic_pipeline(pipeline, monkeypatch):
    import rasterio
    from synthetic_idsp import make_fake_idsp_csv

    import build_villages
    import flood_map
    import risk_score
    import summary
    import validate_risk

    paths, aoi, logger = pipeline
    monkeypatch.setattr(config, "INFER_TILE", 256)

    res = flood_map.run_flood_map(smoke=False, logger=logger)
    assert res["status"] == "OK", res
    with rasterio.open(paths.outputs / "flood_extent.tif") as src:
        extent = src.read(1)
    with rasterio.open(paths.outputs / "flood_duration.tif") as src:
        duration = src.read(1)
    assert extent[60:140, 60:240].mean() > 0.99             # planted flood found
    assert (extent[300:320, :] == 0).all()                  # river is normal water, not flood
    assert extent[0:3, 400:403].max() == 0                  # tiny blob removed
    assert duration[100, 100] == 18                         # 07-01..07-13 + 6 day interval
    summ = json.loads((paths.results / "flood_map_summary.json").read_text())
    assert summ["plausibility"]["elevation_m"]["flooded_lower_than_random"]
    ts = pd.read_csv(paths.results / "flood_timeseries.csv")
    assert set(ts["district"]) == {"DistA", "DistB"}

    # Sanity checks: the two network checks are stubbed, the figure and report are real.
    monkeypatch.setattr(flood_map, "optical_check",
                        lambda *a: {"status": "NOT POSSIBLE", "reason": "stubbed in test"})

    def offline_jrc_check(*args):
        raise RuntimeError("offline")
    monkeypatch.setattr(flood_map, "jrc_check", offline_jrc_check)
    res = flood_map.run_sanity_check(smoke=False, logger=logger)
    assert res["status"] == "OK", res
    report = (paths.results / "sanity_check.md").read_text()
    assert "NOT POSSIBLE" in report and "FAILED" in report and "official government" in report
    assert (paths.results / "assam_sanity_visual.png").exists()

    res = build_villages.run_villages(smoke=False, logger=logger)
    assert res["status"] == "OK", res
    feats = pd.read_csv(paths.outputs / "village_features.csv")
    assert len(feats) > 50 and feats["unit_id"].is_unique
    assert feats["flooded_frac"].max() > 0.5 and feats["pop"].sum() > 0
    assert (feats["exposed_pop"] <= feats["pop"] + 1e-6).all()
    assert set(feats["pct_shallow_well_handpump"]) == {90.0, 60.0}

    res = risk_score.run_risk(smoke=False, logger=logger)
    assert res["status"] == "OK", res
    assert not any("differs" in w for w in res["warnings"])
    ranked = pd.read_csv(paths.results / "village_risk_ranked.csv", comment="#")
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["flooded_frac"].iloc[:5].min() > 0         # top units are flooded ones
    header = (paths.results / "village_risk_ranked.csv").read_text().splitlines()[1]
    assert header.startswith("# components used:")
    assert (paths.results / "top20_map.png").exists()
    evo = pd.read_csv(paths.results / "ranking_evolution.csv")
    assert len(evo) == len(FLOOD_DATES)

    # Validation without the IDSP file -> NOT RUN, with a reason, and nothing invented.
    res = validate_risk.run_validate(smoke=False, logger=logger)
    assert res["status"] == "NOT RUN" and "not found" in res["reason"]
    # With a (fake, test-only) IDSP file the code path runs end to end.
    make_fake_idsp_csv(paths.manual / config.IDSP_FILE, ["DistA", "DistB", "Jorhat"])
    res = validate_risk.run_validate(smoke=False, logger=logger)
    assert res["status"] == "OK", res
    report = json.loads((paths.results / "validation.json").read_text())
    assert report["mapped_districts"] == ["DistA", "DistB"]
    assert all(np.isnan(r["rho"]) for r in report["district_results"])  # n = 2: not computable
    assert "very low" in (paths.results / "validation.md").read_text()

    res = summary.run_summary(smoke=False, logger=logger)
    text = (paths.results / "SUMMARY.md").read_text()
    assert "Top 10 village proxies" in text and "Pitch numbers" in text
