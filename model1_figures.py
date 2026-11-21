"""
model1_figures.py - Model 1 result figures and results/model1_RESULTS.md.

No training and no new aggregate scoring. Every number written to a figure or
to the results file is read from:
  results/model1_eval_protocol.json   (pooled metrics, SHA256, scoring count)
  results/model1_metrics.json         (selected checkpoint epoch, device)
  results/model1_metrics_run1.json    (earlier epoch-7 checkpoint numbers)
  outputs/train_log.csv               (training curves)
  outputs/logs/test.log, outputs/logs/sweep.log (which checkpoint each test run used, timing)

Figures 3 and 4 need per-tile values, which no saved file holds. They are
recomputed with the same code path as eval_protocol.py (train.load_trained_model,
train.predict_tiles, train.read_label, eval_protocol.counts / ratio, Mask A), and
the per-tile mean is checked against model1_eval_protocol.json before plotting.

Writes:
  results/figures/fig1_iou_by_group.png
  results/figures/fig2_training_curves.png
  results/figures/fig3_examples.png
  results/figures/fig4_tile_iou_vs_water_fraction.png
  results/model1_RESULTS.md

Run:  python model1_figures.py
"""

import csv
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import config  # noqa: E402
import eval_protocol  # noqa: E402
import train  # noqa: E402

DPI = 200
UNET_COLOR = "#1f5fa8"
OTSU_COLOR = "#9e9e9e"
GROUPS = ["test_all", "test_india", "bolivia"]
GROUP_LABEL = {"test_all": "All test", "test_india": "India test subset",
               "bolivia": "Bolivia holdout"}

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def f3(v):
    return "n/a" if v is None else f"{v:.3f}"


def fig1(ep, out):
    a = {g: ep["groups"][g]["A"] for g in GROUPS}
    labels = [f"{GROUP_LABEL[g]}\n(n={a[g]['n_tiles']} tiles)" for g in GROUPS]
    unet = [a[g]["unet"]["pooled_water_iou"] for g in GROUPS]
    otsu = [a[g]["otsu"]["pooled_water_iou"] for g in GROUPS]
    x = np.arange(len(GROUPS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bu = ax.bar(x - w / 2, unet, w, color=UNET_COLOR, label="U-Net")
    bo = ax.bar(x + w / 2, otsu, w, color=OTSU_COLOR, label="Otsu threshold (baseline)")
    for bars in (bu, bo):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pooled water IoU")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig2(rows, selected_epoch, out):
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    viou = [float(r["val_iou"]) for r in rows]
    sel_iou = viou[ep.index(selected_epoch)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.plot(ep, loss, color="#333333", lw=2)
    a1.set_ylabel("Training loss\n(BCE + Dice, unitless)")
    a2.plot(ep, viou, color=UNET_COLOR, lw=2)
    a2.set_ylabel("Validation pooled\nwater IoU")
    a2.set_xlabel("Epoch")
    for ax in (a1, a2):
        ax.axvline(selected_epoch, color="#c0392b", lw=1.5, ls="--")
    a1.text(selected_epoch + 0.6, max(loss), f"selected epoch {selected_epoch}",
            color="#c0392b", va="top", fontsize=13)
    a2.text(selected_epoch + 0.6, min(viou), f"pooled water IoU {sel_iou:.3f}",
            color="#c0392b", va="bottom", fontsize=13)
    a2.set_xlim(min(ep), max(ep))
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def per_tile_values(paths, ep):
    """Per-tile U-Net and Otsu water IoU on the test split, Mask A, same code path as eval_protocol.py."""
    device = train.get_device()
    if device.type != "cuda":
        sys.exit("STOP: no CUDA GPU available.")
    use_amp = config.USE_AMP and device.type == "cuda"
    model, ckpt = train.load_trained_model(paths, device)
    stats = config.load_norm_stats(paths)
    if ckpt["epoch"] != ep["checkpoint_epoch"]:
        sys.exit("STOP: checkpoint epoch differs from model1_eval_protocol.json.")
    if eval_protocol.sha256_of(paths.checkpoint) != ep["checkpoint_sha256"]:
        sys.exit("STOP: checkpoint SHA256 differs from model1_eval_protocol.json.")
    test_ids = train.get_split_ids(paths, "test", smoke=False)
    with torch.no_grad():
        probs = train.predict_tiles(model, paths, test_ids, stats, device, use_amp)
    tiles = {}
    for tid in test_ids:
        label = train.read_label(paths, tid)
        otsu = train.read_label(paths, tid, "S1OtsuLabelHand")
        valid = (label >= 0) & ((otsu == 0) | (otsu == 1))  # Mask A
        truth = label == 1
        pred = probs[tid] >= config.WATER_THRESHOLD
        cu = eval_protocol.counts(pred, truth, valid)
        co = eval_protocol.counts(otsu == 1, truth, valid)
        n_valid, n_water = int(valid.sum()), int((truth & valid).sum())
        tiles[tid] = {
            "unet_iou": eval_protocol.ratio(cu["tp"], cu["tp"] + cu["fp"] + cu["fn"]),
            "otsu_iou": eval_protocol.ratio(co["tp"], co["tp"] + co["fp"] + co["fn"]),
            "n_water": n_water,
            "water_fraction": n_water / n_valid if n_valid else float("nan"),
            "n_valid": n_valid,
            "pred": pred, "label": label, "otsu": otsu,
        }
    # Consistency check against the saved protocol file.
    with_water = [t["unet_iou"] for t in tiles.values() if t["n_water"] > 0]
    saved = ep["groups"]["test_all"]["A"]["unet"]["mean_tile_iou_tiles_with_water"]
    # GPU inference with AMP is not bit-exact between runs, so a few borderline pixels can
    # flip; allow a small tolerance and print the actual difference.
    diff = abs(float(np.mean(with_water)) - saved)
    print(f"Per-tile mean check (test tiles with water): recomputed {np.mean(with_water):.8f}, "
          f"saved {saved:.8f}, abs diff {diff:.2e}")
    if diff > 1e-4:
        sys.exit("STOP: per-tile mean does not match the saved file within 1e-4.")
    if len(with_water) != ep["groups"]["test_all"]["A"]["unet"]["n_tiles_with_water"]:
        sys.exit("STOP: number of test tiles with water does not match the saved file.")
    return tiles


def select_tiles(tiles):
    """Fixed rule: best, median, worst U-Net per-tile IoU among test tiles with water,
    then the India test tile (with water) closest to the India median, skipping any
    tile already chosen. Ties broken by tile id."""
    pool = sorted((t for t in tiles if tiles[t]["n_water"] > 0),
                  key=lambda t: (tiles[t]["unet_iou"], t))
    chosen = [("best", pool[-1]), ("median", pool[(len(pool) - 1) // 2]), ("worst", pool[0])]
    india = [t for t in pool if t.startswith("India")]
    india_median = float(np.median([tiles[t]["unet_iou"] for t in india]))
    taken = {t for _, t in chosen}
    for t in sorted(india, key=lambda t: (abs(tiles[t]["unet_iou"] - india_median), t)):
        if t not in taken:
            chosen.append(("India, closest to India median", t))
            break
    return chosen, len(pool), len(india), india_median


def fig3(paths, tiles, chosen, n_pool, n_india, out):
    vv = {t: train.clean_and_clip(train.read_radar(paths, t))[0] for _, t in chosen}
    allvv = np.concatenate([v[np.isfinite(v)].ravel() for v in vv.values()])
    lo, hi = np.percentile(allvv, [2, 98])
    cmap = train.label_cmap()
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.6 * len(chosen) + 1.2))
    cols = ["Sentinel-1 VV", "Hand label", "U-Net prediction", "Otsu"]
    for c, name in enumerate(cols):
        axes[0, c].set_title(name, fontsize=15)
    im = None
    for r, (role, t) in enumerate(chosen):
        d = tiles[t]
        im = axes[r, 0].imshow(vv[t], cmap="gray", vmin=lo, vmax=hi)
        axes[r, 1].imshow(train.label_image(d["label"]), cmap=cmap, vmin=0, vmax=2,
                          interpolation="nearest")
        axes[r, 2].imshow(d["pred"].astype(int) + 1, cmap=cmap, vmin=0, vmax=2,
                          interpolation="nearest")
        axes[r, 3].imshow(train.label_image(d["otsu"]), cmap=cmap, vmin=0, vmax=2,
                          interpolation="nearest")
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].text(0, -0.06, f"{role}: {t}", transform=axes[r, 0].transAxes,
                        ha="left", va="top", fontsize=13)
        axes[r, 2].text(0.5, -0.06, f"U-Net per-tile water IoU {d['unet_iou']:.3f}",
                        transform=axes[r, 2].transAxes, ha="center", va="top", fontsize=13)
        axes[r, 3].text(0.5, -0.06, f"Otsu per-tile water IoU {f3(d['otsu_iou'])}",
                        transform=axes[r, 3].transAxes, ha="center", va="top", fontsize=13)
    cax = fig.add_axes([0.04, 0.075, 0.2, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("VV backscatter (dB)", fontsize=13)
    handles = [Patch(color=c, label=n) for c, n in
               zip(cmap.colors, ["no data", "not water", "water"])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.62, 0.045), ncol=3,
               frameon=False, fontsize=13)
    fig.text(0.5, 0.012,
             f"Rows chosen by rule: best, median and worst U-Net per-tile water IoU among the "
             f"{n_pool} test tiles with at least one water pixel, then the India test tile "
             f"closest to the India median ({n_india} India tiles with water).",
             ha="center", va="bottom", fontsize=12, wrap=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12, hspace=0.25, wspace=0.05)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig4(tiles, pooled, out):
    pts = [(d["water_fraction"], d["unet_iou"], t.startswith("India"))
           for t, d in tiles.items() if d["n_water"] > 0]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    india = np.array([p[2] for p in pts])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x[~india], y[~india], s=40, color=UNET_COLOR, alpha=0.8, label="other test tiles")
    ax.scatter(x[india], y[india], s=55, color="#e67e22", marker="D", alpha=0.9,
               label="India test tiles")
    ax.axhline(pooled, color="#555555", lw=1.2, ls="--")
    ax.text(x.min(), pooled + 0.015, f"pooled water IoU, all test tiles: {pooled:.3f}",
            fontsize=12, color="#555555", va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Water fraction of valid label pixels (log scale)")
    ax.set_ylabel("U-Net per-tile water IoU")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def test_log_runs(paths):
    """(epoch, timestamp) for every FULL test-stage run in outputs/logs/test.log."""
    text = (paths.logs / "test.log").read_text(encoding="utf-8", errors="replace").splitlines()
    runs, in_full = [], False
    for line in text:
        if "===== Stage test (FULL) started =====" in line:
            in_full = True
        m = re.search(r"Loaded checkpoint .*\(epoch (\d+),", line)
        if in_full and m:
            runs.append((int(m.group(1)), line[:19]))
            in_full = False
    return runs


def results_md(ep, mt, run1, paths, out):
    a = {g: ep["groups"][g]["A"] for g in GROUPS}
    b = {g: ep["groups"][g]["B"] for g in GROUPS}
    log_runs = test_log_runs(paths)
    sweep_start = (paths.logs / "sweep.log").read_text(encoding="utf-8").splitlines()[0][:19]
    by_epoch = {}
    for e, _ in log_runs:
        by_epoch[e] = by_epoch.get(e, 0) + 1
    by_epoch[ep["checkpoint_epoch"]] = by_epoch.get(ep["checkpoint_epoch"], 0) + 1
    total = sum(by_epoch.values())
    if total != ep["test_split_scorings"]["total"]:
        sys.exit("STOP: scoring count from test.log does not match model1_eval_protocol.json.")
    sel, early = mt["checkpoint"]["epoch"], run1["checkpoint"]["epoch"]

    same_b = all(f3(a[g][m][k]) == f3(b[g][m][k]) for g in GROUPS for m in ("unet", "otsu")
                 for k in ("pooled_water_iou", "precision", "recall"))
    lines = [
        "# Model 1 results: water detection on Sen1Floods11",
        "",
        f"We measured pooled water IoU (water pixels only, TP, FP and FN summed over all valid "
        f"pixels of a group) of the U-Net checkpoint from epoch {sel} against the Otsu threshold "
        f"baseline on Sen1Floods11 hand-labeled tiles: the {a['test_all']['n_tiles']}-tile test "
        f"split, its {a['test_india']['n_tiles']}-tile India subset and the "
        f"{a['bolivia']['n_tiles']}-tile Bolivia holdout, water threshold {ep['threshold']} on the "
        f"predicted probability.",
        "",
        "| Group | Tiles | U-Net pooled water IoU | U-Net precision | U-Net recall | "
        "Otsu pooled water IoU | Otsu precision | Otsu recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in GROUPS:
        u, o = a[g]["unet"], a[g]["otsu"]
        lines.append(f"| {GROUP_LABEL[g]} | {a[g]['n_tiles']} | {f3(u['pooled_water_iou'])} | "
                     f"{f3(u['precision'])} | {f3(u['recall'])} | {f3(o['pooled_water_iou'])} | "
                     f"{f3(o['precision'])} | {f3(o['recall'])} |")
    lines += [
        "",
        "Valid pixels: hand label is 0 or 1 and the Otsu label is defined. "
        + ("Using the hand label alone gives the same values to 3 decimals."
           if same_b else "Using the hand label alone changes some values; see "
                          "results/model1_eval_protocol.md."),
        "",
        "The same U-Net predictions under three definitions, for comparison with other papers:",
        "",
        "| Group | Pooled water IoU | 2-class mean IoU (water, not water) | Per-tile mean water IoU |",
        "|---|---|---|---|",
    ]
    for g in GROUPS:
        u = a[g]["unet"]
        lines.append(f"| {GROUP_LABEL[g]} | {f3(u['pooled_water_iou'])} | "
                     f"{f3(u['pooled_mean_iou_2class'])} | {f3(u['mean_tile_iou_all_tiles'])} |")
    und = a["test_all"]["unet"]["n_tiles_iou_undefined"]
    lines += [
        "",
        f"Per-tile mean leaves out tiles where the IoU is 0/0 (no water in the label and none "
        f"predicted: {und} of {a['test_all']['n_tiles']} test tiles). It is low because tiles with "
        f"little water score near 0 (results/figures/fig4_tile_iou_vs_water_fraction.png).",
        "",
        "## Caveats",
        "",
        f"- The test split has been scored {total} times in total: "
        + ", ".join(f"{n} on the epoch-{e} checkpoint" for e, n in sorted(by_epoch.items()))
        + f". The epoch-{early} run (results/model1_metrics_run1.json) gave pooled water IoU "
        f"{f3(run1['groups']['test_all']['unet']['iou'])} on all test tiles, "
        f"{f3(run1['groups']['test_india']['unet']['iou'])} on the India subset and "
        f"{f3(run1['groups']['bolivia']['unet']['iou'])} on Bolivia. It ran at "
        f"{log_runs[0][1]} (outputs/logs/test.log), before the learning-rate sweep started at "
        f"{sweep_start} (outputs/logs/sweep.log). Figures 3 and 4 recomputed per-tile "
        f"predictions from the epoch-{sel} checkpoint for plotting only.",
    ]
    for g in ("test_india", "bolivia"):
        u, o = a[g]["unet"]["pooled_water_iou"], a[g]["otsu"]["pooled_water_iou"]
        if o > u:
            lines.append(f"- Otsu scores higher than the U-Net on the {GROUP_LABEL[g]} "
                         f"(n={a[g]['n_tiles']}): pooled water IoU {f3(o)} vs {f3(u)}.")
    ua, oa = a["test_all"]["unet"]["pooled_water_iou"], a["test_all"]["otsu"]["pooled_water_iou"]
    lines += [
        f"- The U-Net scores higher only on all test tiles pooled: {f3(ua)} vs {f3(oa)}.",
        "- Hyperparameters (learning rates, schedule, checkpoint epoch) were chosen on the "
        "validation split only (results/model1_sweep.md, outputs/train_log.csv).",
        f"- Checkpoint: {ep['checkpoint_path']}, SHA256 {ep['checkpoint_sha256']}.",
        "",
    ]
    text = "\n".join(lines)
    if chr(0x2014) in text or chr(0x2013) in text:  # em dash, en dash
        sys.exit("STOP: dash character found in results text.")
    out.write_text(text, encoding="utf-8")
    return text


def main():
    paths = config.get_paths(smoke=False)
    figdir = paths.results / "figures"
    figdir.mkdir(exist_ok=True)
    ep = load_json(paths.results / "model1_eval_protocol.json")
    mt = load_json(paths.results / "model1_metrics.json")
    run1 = load_json(paths.results / "model1_metrics_run1.json")
    with open(paths.outputs / "train_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    selected = mt["checkpoint"]["epoch"]
    best_row = max(rows, key=lambda r: float(r["val_iou"]))
    if int(best_row["epoch"]) != selected or abs(float(best_row["val_iou"]) - mt["checkpoint"]["val_iou"]) > 1e-9:
        sys.exit("STOP: outputs/train_log.csv best epoch does not match the saved checkpoint.")

    fig1(ep, figdir / "fig1_iou_by_group.png")
    fig2(rows, selected, figdir / "fig2_training_curves.png")
    tiles = per_tile_values(paths, ep)
    chosen, n_pool, n_india, india_median = select_tiles(tiles)
    fig3(paths, tiles, chosen, n_pool, n_india, figdir / "fig3_examples.png")
    fig4(tiles, ep["groups"]["test_all"]["A"]["unet"]["pooled_water_iou"],
         figdir / "fig4_tile_iou_vs_water_fraction.png")
    print("Figure 3 rows:")
    for role, t in chosen:
        print(f"  {role:<32} {t:<16} U-Net per-tile water IoU {tiles[t]['unet_iou']:.4f}  "
              f"water fraction {tiles[t]['water_fraction']:.4f}")
    print("Fig 4 outliers (water fraction > 0.9 and per-tile IoU < 0.1):")
    for t, d in sorted(tiles.items()):
        if d["n_water"] > 0 and d["water_fraction"] > 0.9 and d["unet_iou"] < 0.1:
            print(f"  {t}: water fraction {d['water_fraction']:.4f}, valid pixels {d['n_valid']}, "
                  f"water pixels {d['n_water']}, U-Net IoU {d['unet_iou']:.4f}, "
                  f"Otsu IoU {f3(d['otsu_iou'])}, U-Net water pixels predicted in mask "
                  f"{int((d['pred'] & (d['label'] >= 0)).sum())}")
    print(f"  India median per-tile water IoU {india_median:.4f} over {n_india} tiles")
    print()
    print(results_md(ep, mt, run1, paths, paths.results / "model1_RESULTS.md"))


if __name__ == "__main__":
    main()
