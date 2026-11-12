"""Tests for train.py (Model 1 phases 1-3) using synthetic arrays only."""
import numpy as np
import pytest
import torch

import train

STATS = {"vv_mean": -10.0, "vv_std": 4.0, "vh_mean": -17.0, "vh_std": 5.0}


def test_preprocess_replaces_nan_inf_and_clips():
    radar = np.array([[[np.nan, np.inf, -100.0, 50.0]], [[-17.0, -np.inf, -17.0, -17.0]]],
                     dtype=np.float32)
    x = train.preprocess_radar(radar, STATS)
    assert x.shape == radar.shape
    assert np.isfinite(x).all()
    assert x[0, 0, 0] == 0.0 and x[0, 0, 1] == 0.0          # NaN / inf -> 0 (the mean)
    assert x[0, 0, 2] == pytest.approx((-50.0 + 10.0) / 4.0)  # clipped at -50 dB
    assert x[0, 0, 3] == pytest.approx((5.0 + 10.0) / 4.0)    # clipped at +5 dB
    assert x[1, 0, 0] == pytest.approx(0.0)                   # VH at its mean


def test_masked_loss_ignores_no_data_pixels():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    labels = torch.randint(0, 2, (2, 8, 8))
    labels[:, :4, :] = -1
    loss_a = train.masked_bce_dice_loss(logits, labels)
    logits_b = logits.clone()
    logits_b[:, :, :4, :] = 100.0  # change only the no-data region
    loss_b = train.masked_bce_dice_loss(logits_b, labels)
    assert torch.isclose(loss_a, loss_b)
    all_nodata = torch.full((1, 8, 8), -1)
    assert train.masked_bce_dice_loss(logits[:1], all_nodata).item() == 0.0


def test_confusion_counts_and_iou():
    labels = np.array([[1, 1, 0, -1], [0, 0, 1, -1]])
    pred = np.array([[1, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    c = train.confusion_counts(pred, labels)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    m = train.metrics_from_counts(c)
    assert m["iou"] == pytest.approx(2 / 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert np.isnan(train.metrics_from_counts({"tp": 0, "fp": 0, "fn": 0})["iou"])


def test_augment_keeps_radar_and_label_aligned():
    rng = np.random.default_rng(1)
    label = rng.integers(-1, 2, size=(16, 16)).astype(np.int16)
    radar = np.stack([label.astype(np.float32), label.astype(np.float32) * 2])
    for _ in range(20):
        r, lab = train.augment(radar, label, 8, rng)
        assert r.shape == (2, 8, 8) and lab.shape == (8, 8)
        assert np.array_equal(r[0], lab.astype(np.float32))
        assert np.array_equal(r[1], 2 * lab.astype(np.float32))


def test_smoke_subset_is_deterministic_and_has_india():
    ids = [f"{c}_{i}" for c in ["Ghana", "India", "USA", "Spain"] for i in range(5)]
    a = train.smoke_subset(ids, 6)
    assert a == train.smoke_subset(list(reversed(ids)), 6)
    assert a[0].startswith("India") and len(a) == 6


def _write_tile(paths, tid, radar, label, otsu):
    import rasterio
    from rasterio.transform import from_origin
    tr = from_origin(90.0, 25.0, 0.0001, 0.0001)
    for folder, arr, dtype in [("S1Hand", radar, "float32"), ("LabelHand", label[None], "int16"),
                               ("S1OtsuLabelHand", otsu[None], "int16")]:
        path = train.tile_file(paths, folder, tid)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                           count=arr.shape[0], dtype=dtype, transform=tr, crs="EPSG:4326") as dst:
            dst.write(arr.astype(dtype))


def test_dataset_norm_stats_and_shapes(tmp_paths, logger):
    rng = np.random.default_rng(0)
    ids = ["India_1", "Ghana_2"]
    for tid in ids:
        label = rng.integers(-1, 2, size=(32, 32)).astype(np.int16)
        vv = np.where(label == 1, -20.0, -8.0) + rng.normal(0, 1, (32, 32))
        vh = vv - 7
        radar = np.stack([vv, vh]).astype(np.float32)
        radar[0, 0, 0] = np.nan
        _write_tile(tmp_paths, tid, radar, label, label)
    stats = train.compute_norm_stats(tmp_paths, ids, logger)
    assert np.isfinite(stats["vv_mean"]) and stats["vv_std"] > 0
    assert stats["n_valid_pixels"][0] == 2 * 32 * 32 - 2  # the two NaN pixels are ignored
    ds = train.Sen1Floods11Dataset(tmp_paths, ids, stats, augment_data=True, crop_size=16)
    radar, label, name = ds[0]
    assert radar.shape == (2, 16, 16) and radar.dtype == torch.float32
    assert label.shape == (16, 16) and name == "India_1"
    assert set(np.unique(label.numpy())) <= {-1, 0, 1}
    ds_full = train.Sen1Floods11Dataset(tmp_paths, ids, stats)
    assert ds_full[1][0].shape == (2, 32, 32)
    check = train.data_check(tmp_paths, ids, logger)
    assert check["water_darker_than_nonwater"] is True


def test_training_verdict():
    rows = [{"train_loss": 1.0 - 0.05 * i, "val_iou": min(0.3 + 0.05 * i, 0.6)} for i in range(12)]
    ok, text = train.training_verdict(rows, early_stopped=True)
    assert ok is True and "loss falls: True" in text
    flat = [{"train_loss": 1.0, "val_iou": 0.3} for _ in range(5)]
    assert train.training_verdict(flat, early_stopped=False)[0] is False
    assert train.training_verdict(rows[:1], early_stopped=False)[0] is None


def test_unet_builds_without_pretrained_weights():
    model, used = train.build_model(None)
    assert used is None
    with torch.no_grad():
        out = model(torch.zeros(1, 2, 64, 64))
    assert out.shape == (1, 1, 64, 64)
