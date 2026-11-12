"""Tests for flood_map.py (Model 1 phases 5-6) using synthetic arrays only."""
import numpy as np
import pytest
import torch

import common
import config
import flood_map


class PixelModel(torch.nn.Module):
    """Stand-in for the U-Net: water logit from VV only, no spatial context."""

    def forward(self, x):
        return -3.0 * x[:, :1] - 3.0


def test_tile_starts_cover_everything():
    assert flood_map.tile_starts(100, 512, 448) == [0]
    starts = flood_map.tile_starts(1000, 512, 448)
    assert starts[0] == 0 and starts[-1] == 1000 - 512
    covered = np.zeros(1000, bool)
    for s in starts:
        covered[s:s + 512] = True
    assert covered.all()


@pytest.mark.parametrize("shape", [(300, 200), (700, 900)])
def test_tiled_prediction_matches_whole_image(shape):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2,) + shape).astype(np.float32)
    prob = flood_map.predict_array(PixelModel(), x, torch.device("cpu"), False,
                                   tile=256, overlap=64, batch=4, tta=False)
    expected = 1 / (1 + np.exp(-(-3.0 * x[0] - 3.0)))
    assert prob.shape == shape
    np.testing.assert_allclose(prob, expected, atol=1e-5)
    prob_tta = flood_map.predict_array(PixelModel(), x, torch.device("cpu"), False,
                                       tile=256, overlap=64, batch=4, tta=True)
    np.testing.assert_allclose(prob_tta, expected, atol=1e-5)


def test_predict_raster_blocks_and_nodata(tmp_path, monkeypatch):
    import rasterio
    from rasterio.transform import from_origin
    monkeypatch.setattr(config, "INFER_BLOCK", 300)
    monkeypatch.setattr(config, "INFER_TILE", 256)
    rng = np.random.default_rng(3)
    vv = np.where(rng.random((520, 610)) < 0.3, -22.0, -8.0).astype(np.float32)
    radar = np.stack([vv, vv - 7])
    radar[:, 10:20, 10:20] = np.nan
    src = tmp_path / "pass.tif"
    common.write_tif(src, radar, "EPSG:32646", from_origin(500000, 3000000, 10, 10), nodata=np.nan)
    stats = {"vv_mean": -10.0, "vv_std": 4.0, "vh_mean": -17.0, "vh_std": 5.0}
    dst = flood_map.predict_raster(PixelModel(), src, tmp_path / "water.tif", stats,
                                   torch.device("cpu"), False)
    with rasterio.open(dst) as ds:
        water = ds.read(1)
    assert (water[10:20, 10:20] == 255).all()
    ok = np.ones_like(water, bool)
    ok[10:20, 10:20] = False
    assert np.array_equal(water[ok] == 1, vv[ok] < -15)


def test_normal_water_and_flood_mask():
    a = np.array([[1, 1, 0, 255]], dtype=np.uint8)
    b = np.array([[1, 0, 0, 255]], dtype=np.uint8)
    c = np.array([[0, 0, 0, 1]], dtype=np.uint8)
    normal = flood_map.normal_water_mask([a, b, c], min_frac=0.5)
    assert normal.tolist() == [[1, 0, 0, 1]]
    normal2 = flood_map.normal_water_mask([a, b], min_frac=0.5)
    assert normal2.tolist() == [[1, 1, 0, 255]]
    water = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    flood, removed = flood_map.flood_mask_for_pass(water, normal2, min_pixels=1)
    assert flood.tolist() == [[0, 0, 1, 255]] and removed == 0


def test_remove_small_blobs():
    mask = np.zeros((20, 20), bool)
    mask[1:3, 1:3] = True        # 4 px blob
    mask[10:18, 10:18] = True    # 64 px blob
    cleaned, removed = flood_map.remove_small_blobs(mask, 50)
    assert removed == 4
    assert not cleaned[1:3, 1:3].any() and cleaned[10:18, 10:18].all()
    assert flood_map.min_blob_pixels(10) == 50  # 0.5 ha at 10 m pixels


def test_duration_tracker_with_irregular_dates():
    dates = ["2020-07-01", "2020-07-07", "2020-07-13", "2020-07-25"]
    assert flood_map.revisit_interval_days(dates) == 6
    t = flood_map.FloodDurationTracker((1, 4), 6, dates[0])
    passes = [[1, 1, 0, 255], [1, 0, 0, 255], [1, 1, 1, 255], [0, 1, 1, 255]]
    for d, p in zip(dates, passes):
        t.update(np.array([p], dtype=np.uint8), d)
    # px0: flooded 07-01..07-13 -> 12 + 6 = 18 days; px1: 07-13..07-25 -> 12 + 6 = 18;
    # px2: 07-13..07-25 -> 18; px3: never observed -> no data
    assert t.duration_u16().tolist() == [[18, 18, 18, 65535]]
    assert t.extent_u8().tolist() == [[1, 1, 1, 255]]
    single = flood_map.FloodDurationTracker((1, 1), 12, "2020-07-01")
    single.update(np.array([[1]], dtype=np.uint8), "2020-07-01")
    assert single.duration_u16()[0, 0] == 12  # one pass = one revisit interval


def test_placement_test_detects_clustered_flooding():
    rng = np.random.default_rng(0)
    dist = np.tile(np.arange(100, dtype=float), (100, 1))
    land = np.ones_like(dist, bool)
    flood = np.zeros_like(land)
    flood[:, :10] = True
    r = flood_map.placement_test(dist, flood, land, 100, rng)
    assert r["flooded_lower_than_random"] and r["p_value"] < 0.05
    assert flood_map.placement_test(dist, np.zeros_like(land), land, 10, rng) is None


def test_compare_masks():
    pred = np.array([1, 1, 0, 0], bool)
    ref = np.array([1, 0, 1, 0], bool)
    m = flood_map.compare_masks(pred, ref, np.ones(4, bool))
    assert m["iou"] == pytest.approx(1 / 3) and m["agreement"] == pytest.approx(0.5)
