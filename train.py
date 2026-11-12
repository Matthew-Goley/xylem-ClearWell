"""
train.py - MODEL 1, phases 1 to 3: water / flood detection U-Net on Sen1Floods11.

Stages in this file (see run_all.py):
  get_sen1floods11  download hand-labeled data, compute normalization numbers,
                    save a data check plot
  train             train the U-Net with early stopping on validation IoU
  test              evaluate the best checkpoint ONCE on the test split and the
                    Bolivia holdout, next to the provided Otsu baseline

Run alone:
    python train.py --smoke             # all three stages, tiny
    python train.py --stage train       # one stage
Labels: -1 = no data, 0 = not water, 1 = water. Radar bands: VV, VH in dB.
"""

import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import common
import config

STAGES_IN_THIS_FILE = ["get_sen1floods11", "train", "test"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def set_seed(seed=config.SEED):
    """Fix random seeds for python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tile_country(tile_id):
    """'Sri-Lanka_12345' -> 'Sri-Lanka'."""
    return tile_id.split("_")[0]


def tile_file(paths, folder, tile_id):
    return paths.sen1floods11 / folder / f"{tile_id}_{folder}.tif"


def read_split(paths, split):
    """Tile ids listed in a split CSV (no header, first column = S1Hand file name)."""
    path = paths.sen1floods11 / "splits" / config.S1F11_SPLIT_FILES[split]
    ids = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                ids.append(row[0].strip().replace("_S1Hand.tif", ""))
    return ids


def smoke_subset(tile_ids, n):
    """Pick n tiles deterministically, spread across countries, India first."""
    by_country = {}
    for tid in sorted(tile_ids):
        by_country.setdefault(tile_country(tid), []).append(tid)
    countries = sorted(by_country, key=lambda c: (c != "India", c))
    picked = []
    while len(picked) < min(n, len(tile_ids)):
        for c in countries:
            if by_country[c] and len(picked) < n:
                picked.append(by_country[c].pop(0))
    return picked


def get_split_ids(paths, split, smoke):
    ids = read_split(paths, split)
    return smoke_subset(ids, config.SMOKE_TILES[split]) if smoke else ids


# ---------------------------------------------------------------------------
# Reading tiles and preprocessing radar
# ---------------------------------------------------------------------------
def read_radar(paths, tile_id):
    import rasterio
    with rasterio.open(tile_file(paths, "S1Hand", tile_id)) as src:
        return src.read().astype(np.float32)  # (2, H, W): VV, VH in dB


def read_label(paths, tile_id, folder="LabelHand"):
    import rasterio
    with rasterio.open(tile_file(paths, folder, tile_id)) as src:
        return src.read(1).astype(np.int16)  # (H, W): -1, 0, 1


def clean_and_clip(radar_db):
    """Replace inf with NaN and clip dB values to the configured range (NaN stays NaN)."""
    x = np.array(radar_db, dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    return np.clip(x, config.CLIP_DB_MIN, config.CLIP_DB_MAX)


def preprocess_radar(radar_db, stats):
    """dB radar (2, H, W) -> normalized float32 array for the U-Net.

    Steps: inf -> NaN, clip to [CLIP_DB_MIN, CLIP_DB_MAX], subtract the training
    mean and divide by the training std, then missing (NaN) pixels -> 0, which
    is the training average after normalization. The SAME function is used for
    Sen1Floods11 and for the Assam imagery.
    """
    x = clean_and_clip(radar_db)
    mean = np.array([stats["vv_mean"], stats["vh_mean"]], dtype=np.float32)[:, None, None]
    std = np.array([stats["vv_std"], stats["vh_std"]], dtype=np.float32)[:, None, None]
    x = (x - mean) / std
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def augment(radar, label, crop_size, rng):
    """Random crop, flips and 90 degree rotation, applied identically to radar and label."""
    h, w = label.shape
    if crop_size and crop_size < h and crop_size < w:
        r = int(rng.integers(0, h - crop_size + 1))
        c = int(rng.integers(0, w - crop_size + 1))
        radar = radar[:, r:r + crop_size, c:c + crop_size]
        label = label[r:r + crop_size, c:c + crop_size]
    if rng.random() < 0.5:
        radar, label = radar[:, :, ::-1], label[:, ::-1]
    if rng.random() < 0.5:
        radar, label = radar[:, ::-1, :], label[::-1, :]
    k = int(rng.integers(0, 4))
    radar = np.rot90(radar, k, axes=(1, 2))
    label = np.rot90(label, k)
    return np.ascontiguousarray(radar), np.ascontiguousarray(label)


class Sen1Floods11Dataset(Dataset):
    """Returns (2 x H x W float radar tensor, H x W label tensor with -1 kept, tile name)."""

    def __init__(self, paths, tile_ids, stats, augment_data=False, crop_size=None, preload=True):
        self.paths = paths
        self.tile_ids = list(tile_ids)
        self.stats = stats
        self.augment_data = augment_data
        self.crop_size = crop_size
        self.rng = None  # created lazily so each DataLoader worker gets its own
        self.cache = {}
        if preload:  # the hand-labeled set is small, so keep it in memory
            for tid in self.tile_ids:
                self.cache[tid] = self._load(tid)

    def _load(self, tile_id):
        radar = preprocess_radar(read_radar(self.paths, tile_id), self.stats)
        label = read_label(self.paths, tile_id)
        return radar, label

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index):
        tid = self.tile_ids[index]
        radar, label = self.cache[tid] if tid in self.cache else self._load(tid)
        if self.augment_data:
            if self.rng is None:
                self.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
            radar, label = augment(radar, label, self.crop_size, self.rng)
        return torch.from_numpy(radar.copy()), torch.from_numpy(label.astype(np.int64)), tid


# ---------------------------------------------------------------------------
# Loss and metrics (label -1 pixels are ignored everywhere)
# ---------------------------------------------------------------------------
def masked_bce_dice_loss(logits, labels):
    """BCE-with-logits + Dice, using only pixels with label 0 or 1.

    logits: (B, 1, H, W) or (B, H, W); labels: (B, H, W) with values -1/0/1.
    """
    if logits.dim() == 4:
        logits = logits[:, 0]
    logits = logits.float()
    valid = labels >= 0
    if valid.sum() == 0:
        return logits.sum() * 0.0  # keeps the graph, contributes nothing
    target = (labels == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
    probs = torch.sigmoid(logits[valid])
    t = target[valid]
    intersection = (probs * t).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (probs.sum() + t.sum() + 1.0)
    return bce + dice


def confusion_counts(pred_water, labels):
    """True/false positive/negative counts for the water class over valid pixels.

    Works with numpy arrays or torch tensors. pred_water is boolean.
    """
    valid = labels >= 0
    truth = labels == 1
    tp = int((pred_water & truth & valid).sum())
    fp = int((pred_water & ~truth & valid).sum())
    fn = int((~pred_water & truth & valid).sum())
    tn = int((~pred_water & ~truth & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def add_counts(total, new):
    for k in ("tp", "fp", "fn", "tn"):
        total[k] = total.get(k, 0) + new[k]
    return total


def metrics_from_counts(c):
    """Water IoU, precision and recall from summed counts (NaN if undefined)."""
    tp, fp, fn = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"iou": iou, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(encoder_weights, logger=None):
    """U-Net with a ResNet-34 encoder, 2 input bands, 1 output (water logit).

    Returns (model, weights actually used). If the pretrained weights cannot be
    downloaded, falls back to random init and says so loudly.
    """
    import segmentation_models_pytorch as smp
    if encoder_weights:
        try:
            model = smp.Unet(config.ENCODER, encoder_weights=encoder_weights,
                             in_channels=2, classes=1)
            return model, encoder_weights
        except Exception as err:  # noqa: BLE001 - usually a blocked download
            if logger:
                common.loud_warning(
                    logger, f"Could not load '{encoder_weights}' encoder weights ({err}).\n"
                            "FALLING BACK TO encoder_weights=None (random init). Expect a "
                            "lower IoU than with ImageNet weights.")
    model = smp.Unet(config.ENCODER, encoder_weights=None, in_channels=2, classes=1)
    return model, None


def label_image(label):
    """Map labels -1/0/1 to 0/1/2 for plotting with LABEL_CMAP (grey/sand/blue)."""
    return np.clip(np.asarray(label).astype(int), -1, 1) + 1


def label_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(["#bdbdbd", "#efe6cf", "#1f5fa8"])  # no data, land, water


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_logits(model, radar_batch, device, use_amp):
    """Run the model on a (B, 2, H, W) tensor and return float32 logits (B, H, W)."""
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                         enabled=use_amp):
        logits = model(radar_batch.to(device, non_blocking=True))
    return logits[:, 0].float()


# ---------------------------------------------------------------------------
# Stage get_sen1floods11
# ---------------------------------------------------------------------------
def _gcs_bucket():
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    return client, client.bucket(config.S1F11_BUCKET)


def list_bucket_layout(client, bucket, logger):
    """List the bucket so the folder names are confirmed, not assumed."""
    it = client.list_blobs(bucket, prefix=config.S1F11_HAND_PREFIX, delimiter="/")
    list(it)  # the folder list is only filled after reading the pages
    folders = sorted(p.rstrip("/").split("/")[-1] for p in it.prefixes)
    splits = sorted(b.name.split("/")[-1]
                    for b in client.list_blobs(bucket, prefix=config.S1F11_SPLITS_PREFIX))
    logger.info(f"Bucket gs://{config.S1F11_BUCKET}/{config.S1F11_HAND_PREFIX} folders: {folders}")
    logger.info(f"Split files under {config.S1F11_SPLITS_PREFIX}: {splits}")
    missing_folders = [f for f in config.S1F11_FOLDERS if f not in folders]
    missing_splits = [s for s in config.S1F11_SPLIT_FILES.values() if s not in splits]
    if missing_folders or missing_splits:
        raise RuntimeError(f"Bucket layout differs from config: missing folders "
                           f"{missing_folders}, missing splits {missing_splits}")
    return folders, splits


def _download_blobs(blobs_and_dest, logger, what):
    """Download (blob, local path) pairs in parallel, skipping files already present."""
    todo = [(b, d) for b, d in blobs_and_dest
            if not (d.exists() and (b.size is None or d.stat().st_size == b.size))]
    logger.info(f"{what}: {len(blobs_and_dest) - len(todo)} already present, "
                f"{len(todo)} to download")
    if not todo:
        return 0

    def _one(pair):
        blob, dest = pair
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        common.retry(blob.download_to_filename, str(tmp), what=f"download {blob.name}",
                     logger=logger)
        os.replace(tmp, dest)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(_one, todo):
            done += 1
            if done % 50 == 0 or done == len(todo):
                logger.info(f"{what}: {done}/{len(todo)} downloaded")
    return done


def download_sen1floods11(paths, logger, smoke):
    client, bucket = _gcs_bucket()
    list_bucket_layout(client, bucket, logger)
    # 1) split CSVs
    pairs = []
    for split_file in config.S1F11_SPLIT_FILES.values():
        blob = bucket.blob(config.S1F11_SPLITS_PREFIX + split_file)
        blob.reload()  # gets the size so we can skip complete files
        pairs.append((blob, paths.sen1floods11 / "splits" / split_file))
    _download_blobs(pairs, logger, "split csv files")
    # 2) tiles
    if smoke:
        wanted = set()
        for split in config.S1F11_SPLIT_FILES:
            wanted.update(get_split_ids(paths, split, smoke=True))
        pairs = []
        for folder in config.S1F11_FOLDERS:
            for tid in sorted(wanted):
                blob = bucket.blob(f"{config.S1F11_HAND_PREFIX}{folder}/{tid}_{folder}.tif")
                pairs.append((blob, tile_file(paths, folder, tid)))
        _download_blobs(pairs, logger, "smoke tiles")
    else:
        for folder in config.S1F11_FOLDERS:
            blobs = [b for b in client.list_blobs(bucket, prefix=f"{config.S1F11_HAND_PREFIX}{folder}/")
                     if b.name.endswith(".tif")]
            pairs = [(b, paths.sen1floods11 / folder / b.name.split("/")[-1]) for b in blobs]
            _download_blobs(pairs, logger, folder)


def compute_norm_stats(paths, tile_ids, logger):
    """Mean and std of clipped VV and VH dB over TRAIN tiles only, ignoring NaN."""
    sums = np.zeros(2)
    sq_sums = np.zeros(2)
    counts = np.zeros(2)
    for tid in tile_ids:
        x = clean_and_clip(read_radar(paths, tid))
        for b in range(2):
            v = x[b][np.isfinite(x[b])].astype(np.float64)
            sums[b] += v.sum()
            sq_sums[b] += (v ** 2).sum()
            counts[b] += v.size
    mean = sums / counts
    std = np.sqrt(sq_sums / counts - mean ** 2)
    stats = {
        "vv_mean": float(mean[0]), "vv_std": float(std[0]),
        "vh_mean": float(mean[1]), "vh_std": float(std[1]),
        "clip_db": [config.CLIP_DB_MIN, config.CLIP_DB_MAX],
        "n_tiles": len(tile_ids), "n_valid_pixels": [int(counts[0]), int(counts[1])],
        "split": "train", "source": "Sen1Floods11 v1.1 HandLabeled S1Hand",
        "smoke": paths.smoke,
        "generated_by": "train.py compute_norm_stats", "generated_at": datetime.now().isoformat(),
    }
    with open(paths.norm_stats, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Normalization numbers (train tiles only, n={len(tile_ids)}): "
                f"VV {stats['vv_mean']:.3f} +/- {stats['vv_std']:.3f} dB, "
                f"VH {stats['vh_mean']:.3f} +/- {stats['vh_std']:.3f} dB -> {paths.norm_stats}")
    return stats


def data_check(paths, train_ids, logger):
    """Plot 4 tiles (incl. India) and check that water is darker than land in VV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    water_sum = land_sum = 0.0
    water_n = land_n = 0
    water_frac = {}
    for tid in train_ids:
        vv = clean_and_clip(read_radar(paths, tid))[0]
        label = read_label(paths, tid)
        ok = np.isfinite(vv)
        w = ok & (label == 1)
        land = ok & (label == 0)
        water_sum += float(vv[w].sum())
        water_n += int(w.sum())
        land_sum += float(vv[land].sum())
        land_n += int(land.sum())
        valid = (label >= 0).sum()
        water_frac[tid] = float(w.sum() / valid) if valid else 0.0
    water_mean = water_sum / water_n if water_n else float("nan")
    land_mean = land_sum / land_n if land_n else float("nan")
    darker = bool(water_mean < land_mean)
    logger.info(f"Data check: mean VV of water pixels = {water_mean:.2f} dB, "
                f"non-water pixels = {land_mean:.2f} dB -> water darker: {darker}")

    # Pick tiles to plot: one India tile with some water, then 3 others (seeded).
    rng = random.Random(config.SEED)
    india = [t for t in train_ids if tile_country(t) == "India" and water_frac[t] > 0.05]
    india = india or [t for t in train_ids if tile_country(t) == "India"]
    others = [t for t in train_ids if tile_country(t) != "India" and water_frac[t] > 0.05]
    others = others or [t for t in train_ids if tile_country(t) != "India"]
    chosen = (rng.sample(india, 1) if india else []) + rng.sample(others, min(3, len(others)))
    if not india:
        logger.warning("No India tile available for the data check plot")

    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3.2 * len(chosen)), squeeze=False)
    for row, tid in enumerate(chosen):
        x = clean_and_clip(read_radar(paths, tid))
        label = read_label(paths, tid)
        axes[row, 0].imshow(x[0], cmap="gray", vmin=-25, vmax=0)
        axes[row, 0].set_title(f"{tid}  VV dB")
        axes[row, 1].imshow(x[1], cmap="gray", vmin=-30, vmax=-5)
        axes[row, 1].set_title("VH dB")
        axes[row, 2].imshow(label_image(label), cmap=label_cmap(), vmin=0, vmax=2,
                            interpolation="nearest")
        axes[row, 2].set_title("label (blue water, sand land, grey no data)", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Sen1Floods11 data check: water mean VV {water_mean:.1f} dB vs "
                 f"non-water {land_mean:.1f} dB", fontsize=11)
    fig.tight_layout()
    out = paths.results / "model1_data_check.png"
    fig.savefig(out, dpi=70)
    plt.close(fig)
    check = {"water_mean_vv_db": water_mean, "nonwater_mean_vv_db": land_mean,
             "water_darker_than_nonwater": darker, "n_water_pixels": water_n,
             "n_nonwater_pixels": land_n, "n_tiles": len(train_ids), "plotted_tiles": chosen}
    with open(paths.results / "model1_data_check.json", "w") as f:
        json.dump(check, f, indent=2)
    return check


def run_get_data(smoke=False, logger=None, **kwargs):
    paths = config.get_paths(smoke)
    download_sen1floods11(paths, logger, smoke)
    warnings = []
    counts = {s: len(get_split_ids(paths, s, smoke)) for s in config.S1F11_SPLIT_FILES}
    logger.info(f"Tiles per split ({'smoke subset' if smoke else 'full'}): {counts}")
    # Every listed tile must exist locally.
    missing = [t for s in counts for t in get_split_ids(paths, s, smoke)
               for folder in config.S1F11_FOLDERS if not tile_file(paths, folder, t).exists()]
    if missing:
        return common.result("FAILED", f"{len(missing)} tile files missing after download, "
                                       f"for example {missing[:3]}")
    train_ids = get_split_ids(paths, "train", smoke)
    compute_norm_stats(paths, train_ids, logger)
    check = data_check(paths, train_ids, logger)
    if not check["water_darker_than_nonwater"]:
        warnings.append("water pixels are NOT darker than non-water in VV: check the data")
    return common.result("OK", "", warnings, {"tiles": counts})


# ---------------------------------------------------------------------------
# Stage train
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, use_amp):
    """Average loss and water confusion counts over a loader (full tiles)."""
    model.eval()
    counts = {}
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for radar, label, _ in loader:
            label = label.to(device)
            logits = predict_logits(model, radar, device, use_amp)
            loss_sum += float(masked_bce_dice_loss(logits, label)) * radar.size(0)
            n += radar.size(0)
            add_counts(counts, confusion_counts(torch.sigmoid(logits) >= config.WATER_THRESHOLD,
                                                label))
    return loss_sum / max(n, 1), counts


def training_verdict(rows, early_stopped):
    """Check that loss falls and val IoU rises then levels off. Returns (ok, text)."""
    if len(rows) < 3:
        return None, (f"only {len(rows)} epoch(s): too few to judge the training curve "
                      "(expected for a smoke run)")
    losses = [r["train_loss"] for r in rows]
    ious = [r["val_iou"] for r in rows]
    loss_falls = np.mean(losses[-3:]) < 0.9 * losses[0]
    iou_rises = max(ious) > ious[0] + 0.01
    last = ious[-5:]
    levels_off = early_stopped or (max(last) - min(last) < 0.02)
    ok = bool(loss_falls and iou_rises and levels_off)
    text = (f"loss falls: {loss_falls} (first {losses[0]:.3f}, last-3 mean "
            f"{np.mean(losses[-3:]):.3f}); val IoU rises: {iou_rises} (first {ious[0]:.3f}, "
            f"best {max(ious):.3f}); levels off: {levels_off} "
            f"(early stop {early_stopped}, last-5 range {max(last) - min(last):.3f})")
    return ok, text


def plot_training_curves(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [r["train_loss"] for r in rows], label="train loss")
    ax1.plot(epochs, [r["val_loss"] for r in rows], label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE + Dice loss")
    ax1.legend()
    ax2.plot(epochs, [r["val_iou"] for r in rows], color="tab:green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation water IoU")
    ax2.set_ylim(0, 1)
    fig.suptitle("Model 1 training curves (U-Net on Sen1Floods11)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _fit(paths, logger, smoke, batch_size, device):
    """One full training run at a given batch size. Returns a summary dict."""
    set_seed()
    stats = config.load_norm_stats(paths)
    train_ds = Sen1Floods11Dataset(paths, get_split_ids(paths, "train", smoke), stats,
                                   augment_data=True, crop_size=config.CROP_SIZE)
    valid_ds = Sen1Floods11Dataset(paths, get_split_ids(paths, "valid", smoke), stats)
    epochs = 1 if smoke else config.MAX_EPOCHS
    use_amp = config.USE_AMP and device.type == "cuda"
    workers = config.NUM_WORKERS
    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    loader_kw = {"num_workers": workers, "pin_memory": device.type == "cuda",
                 "persistent_workers": workers > 0}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=len(train_ds) > batch_size, generator=generator,
                              **loader_kw)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **loader_kw)

    model, weights_used = build_model(config.ENCODER_WEIGHTS, logger)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(f"Training on {device} | batch {batch_size} | AMP {use_amp} | workers {workers} "
                f"| train tiles {len(train_ds)} | valid tiles {len(valid_ds)} | max epochs "
                f"{epochs} | encoder weights {weights_used}")

    log_path = paths.outputs / "train_log.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_iou", "val_precision", "val_recall",
              "lr", "seconds"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    rows, best_iou, best_epoch, bad_epochs, early_stopped = [], -1.0, 0, 0, False
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for radar, label, _ in train_loader:
            radar = radar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(radar)
            loss = masked_bce_dice_loss(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * radar.size(0)
            n += radar.size(0)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        val_loss, counts = evaluate(model, valid_loader, device, use_amp)
        m = metrics_from_counts(counts)
        row = {"epoch": epoch, "train_loss": loss_sum / max(n, 1), "val_loss": val_loss,
               "val_iou": m["iou"], "val_precision": m["precision"], "val_recall": m["recall"],
               "lr": lr, "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        logger.info(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} | val loss "
                    f"{val_loss:.4f} | val IoU {m['iou']:.4f} | {row['seconds']} s")
        if m["iou"] > best_iou:
            best_iou, best_epoch, bad_epochs = m["iou"], epoch, 0
            ckpt = {"model_state": model.state_dict(), "encoder": config.ENCODER,
                    "encoder_weights": weights_used, "epoch": epoch, "val_iou": m["iou"],
                    "norm_stats": stats, "clip_db": [config.CLIP_DB_MIN, config.CLIP_DB_MAX],
                    "batch_size": batch_size, "smoke": smoke,
                    "saved_at": datetime.now().isoformat()}
            tmp = paths.checkpoint.with_name("unet.tmp.pt")
            torch.save(ckpt, tmp)
            os.replace(tmp, paths.checkpoint)
            logger.info(f"  new best val IoU {best_iou:.4f}, checkpoint saved to {paths.checkpoint}")
        else:
            bad_epochs += 1
            if bad_epochs >= config.EARLY_STOP_PATIENCE:
                early_stopped = True
                logger.info(f"Early stopping: no val IoU improvement for {bad_epochs} epochs")
                break

    plot_training_curves(rows, paths.results / "model1_training_curves.png")
    ok, text = training_verdict(rows, early_stopped)
    logger.info(f"Training verdict: {'OK' if ok else ('UNCLEAR' if ok is None else 'NOT OK')}: {text}")
    return {"best_val_iou": best_iou, "best_epoch": best_epoch, "epochs_run": len(rows),
            "early_stopped": early_stopped, "encoder_weights_used": weights_used,
            "batch_size": batch_size, "device": str(device), "amp": use_amp,
            "verdict_ok": ok, "verdict": text, "smoke": smoke,
            "learning_rate": config.LEARNING_RATE, "optimizer": "AdamW + cosine schedule",
            "train_tiles": len(train_ds), "valid_tiles": len(valid_ds)}


def _is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def run_train(smoke=False, logger=None, **kwargs):
    paths = config.get_paths(smoke)
    device = get_device()
    if device.type == "cpu" and not smoke:
        common.loud_warning(logger, "Full training on the CPU will be very slow (hours). "
                                    "This is meant for the GPU desktop.")
    batch_size = config.BATCH_SIZE
    while True:
        try:
            summary = _fit(paths, logger, smoke, batch_size, device)
            break
        except Exception as err:  # noqa: BLE001
            if device.type == "cuda" and _is_oom(err) and batch_size > config.BATCH_SIZE_OOM_FALLBACK:
                common.loud_warning(logger, f"CUDA out of memory at batch {batch_size}; "
                                            f"restarting with batch {config.BATCH_SIZE_OOM_FALLBACK}")
                torch.cuda.empty_cache()
                batch_size = config.BATCH_SIZE_OOM_FALLBACK
                continue
            raise
    with open(paths.results / "model1_train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    warnings = []
    if summary["encoder_weights_used"] is None:
        warnings.append("ImageNet encoder weights could not be downloaded; trained from random init")
    if summary["verdict_ok"] is False:
        warnings.append(f"training curve check failed: {summary['verdict']}")
    if device.type == "cpu":
        warnings.append("trained on CPU")
    return common.result("OK", "", warnings, summary)


# ---------------------------------------------------------------------------
# Stage test (run ONCE)
# ---------------------------------------------------------------------------
def load_trained_model(paths, device, logger=None):
    """Load outputs/unet.pt. Returns (model, checkpoint dict)."""
    if not paths.checkpoint.exists():
        raise FileNotFoundError(f"{paths.checkpoint} not found. Run stage train first.")
    ckpt = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    model, _ = build_model(None)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    if logger:
        logger.info(f"Loaded checkpoint {paths.checkpoint} (epoch {ckpt['epoch']}, val IoU "
                    f"{ckpt['val_iou']:.4f}, encoder weights {ckpt['encoder_weights']})")
    return model, ckpt


def predict_tiles(model, paths, tile_ids, stats, device, use_amp, batch_size=4):
    """Water probability (H, W) for each Sen1Floods11 tile id."""
    out = {}
    for i in range(0, len(tile_ids), batch_size):
        ids = tile_ids[i:i + batch_size]
        batch = torch.from_numpy(np.stack([preprocess_radar(read_radar(paths, t), stats)
                                           for t in ids]))
        probs = torch.sigmoid(predict_logits(model, batch, device, use_amp)).cpu().numpy()
        for t, p in zip(ids, probs):
            out[t] = p
    return out


def plot_examples(paths, tile_ids, probs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(tile_ids), 4, figsize=(12, 3 * len(tile_ids)), squeeze=False)
    for row, tid in enumerate(tile_ids):
        vv = clean_and_clip(read_radar(paths, tid))[0]
        label = read_label(paths, tid)
        otsu = read_label(paths, tid, "S1OtsuLabelHand")
        pred = (probs[tid] >= config.WATER_THRESHOLD).astype(float)
        cmap = label_cmap()
        panels = [(vv, "gray", -25, 0, f"{tid} VV dB"),
                  (label_image(label), cmap, 0, 2, "truth (grey = no data)"),
                  (label_image(otsu), cmap, 0, 2, "Otsu baseline"),
                  (label_image(pred), cmap, 0, 2, "U-Net")]
        for col, (img, cmap, vmin, vmax, title) in enumerate(panels):
            axes[row, col].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def run_test(smoke=False, logger=None, **kwargs):
    paths = config.get_paths(smoke)
    device = get_device()
    use_amp = config.USE_AMP and device.type == "cuda"
    model, ckpt = load_trained_model(paths, device, logger)
    stats = ckpt["norm_stats"]
    try:
        saved = config.load_norm_stats(paths)
        if abs(saved["vv_mean"] - stats["vv_mean"]) > 1e-6 or abs(saved["vh_mean"] - stats["vh_mean"]) > 1e-6:
            logger.warning("norm_stats.json differs from the numbers stored in the checkpoint; "
                           "using the checkpoint numbers")
    except FileNotFoundError:
        pass
    logger.info("Evaluating on the TEST split and Bolivia holdout. This is the only time the "
                "test data are used; do not tune anything on these numbers.")
    test_ids = get_split_ids(paths, "test", smoke)
    bolivia_ids = get_split_ids(paths, "bolivia", smoke)
    groups = {"test_all": test_ids,
              "test_india": [t for t in test_ids if t.startswith("India")],
              "bolivia": bolivia_ids}
    probs = predict_tiles(model, paths, sorted(set(test_ids + bolivia_ids)), stats, device, use_amp)

    warnings, metrics = [], {}
    for group, ids in groups.items():
        unet_c, otsu_c, unet_tile_iou, otsu_tile_iou, n_valid = {}, {}, [], [], 0
        for tid in ids:
            label = read_label(paths, tid)
            otsu = read_label(paths, tid, "S1OtsuLabelHand")
            # Same valid-pixel mask for both methods: hand label known AND Otsu defined.
            label_masked = np.where((otsu == 0) | (otsu == 1), label, -1)
            n_valid += int((label_masked >= 0).sum())
            cu = confusion_counts(probs[tid] >= config.WATER_THRESHOLD, label_masked)
            co = confusion_counts(otsu == 1, label_masked)
            add_counts(unet_c, cu)
            add_counts(otsu_c, co)
            unet_tile_iou.append(metrics_from_counts(cu)["iou"])
            otsu_tile_iou.append(metrics_from_counts(co)["iou"])
        mu, mo = metrics_from_counts(unet_c), metrics_from_counts(otsu_c)
        mu["mean_tile_iou"] = float(np.nanmean(unet_tile_iou)) if ids else float("nan")
        mo["mean_tile_iou"] = float(np.nanmean(otsu_tile_iou)) if ids else float("nan")
        metrics[group] = {"n_tiles": len(ids), "n_valid_pixels": n_valid,
                          "unet": mu, "otsu": mo, "unet_counts": unet_c, "otsu_counts": otsu_c}
        logger.info(f"{group:<11} n={len(ids):3d} | U-Net IoU {mu['iou']:.4f} P {mu['precision']:.4f} "
                    f"R {mu['recall']:.4f} | Otsu IoU {mo['iou']:.4f} P {mo['precision']:.4f} "
                    f"R {mo['recall']:.4f}")
    if metrics["test_india"]["n_tiles"] < 10:
        msg = (f"India-only test group has only {metrics['test_india']['n_tiles']} tiles "
               "(< 10): its IoU is very uncertain")
        logger.warning(msg)
        warnings.append(msg)

    out = {"groups": metrics, "threshold": config.WATER_THRESHOLD,
           "iou_definition": "water IoU pooled over all valid pixels of all tiles in the group; "
                             "mean_tile_iou is the average of per-tile IoUs",
           "valid_mask": "hand label in {0,1} and Otsu label in {0,1}",
           "checkpoint": {"epoch": ckpt["epoch"], "val_iou": ckpt["val_iou"],
                          "encoder_weights": ckpt["encoder_weights"], "smoke": ckpt.get("smoke")},
           "device": str(device), "smoke": smoke, "warnings": warnings,
           "generated_at": datetime.now().isoformat()}
    with open(paths.results / "model1_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    def fmt(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    lines = [("| Group | n tiles | U-Net IoU | Otsu IoU | U-Net precision | U-Net recall | "
             "Otsu precision | Otsu recall |"), "|---|---|---|---|---|---|---|---|"]
    names = {"test_all": "Test (all)", "test_india": "Test (India only)",
             "bolivia": "Bolivia holdout"}
    for g, m in metrics.items():
        lines.append(f"| {names[g]} | {m['n_tiles']} | {fmt(m['unet']['iou'])} | "
                     f"{fmt(m['otsu']['iou'])} | {fmt(m['unet']['precision'])} | "
                     f"{fmt(m['unet']['recall'])} | {fmt(m['otsu']['precision'])} | "
                     f"{fmt(m['otsu']['recall'])} |")
    header = "SMOKE RUN: numbers are not meaningful.\n\n" if smoke else ""
    (paths.results / "model1_table.md").write_text(
        header + "Water IoU on held-out tiles, generated by train.py (stage test).\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")

    # Example figure: 6 test tiles, at least 2 from India, chosen with a fixed seed.
    rng = random.Random(config.SEED)
    india = groups["test_india"]
    other = [t for t in test_ids if t not in india]
    n_india = min(2, len(india))
    examples = rng.sample(india, n_india) + rng.sample(other, min(6 - n_india, len(other)))
    plot_examples(paths, examples, probs, paths.results / "model1_examples.png")
    return common.result("OK", "", warnings, {g: m["unet"]["iou"] for g, m in metrics.items()})


if __name__ == "__main__":
    import run_all
    sys.exit(run_all.main(default_stages=STAGES_IN_THIS_FILE))
