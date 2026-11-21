"""
eval_protocol.py - re-score the existing Model 1 checkpoint under two pixel masks.

No training, no tuning. Reuses train.py's model loading (load_trained_model),
data reading (get_split_ids, read_label) and inference (predict_tiles, which
applies the same preprocess_radar normalization the test stage uses).

  Mask A (current protocol, as in train.py run_test): hand label >= 0 AND Otsu label in {0, 1}
  Mask B (standard): hand label >= 0 only. On Mask B, Otsu pixels whose Otsu label is
         undefined (not 0/1) are counted as predicted NOT water.

Step 1 reproduces the saved pooled water IoU under Mask A and stops (writing
nothing) if any group differs from the expected value by more than 0.001.

Writes (new files only, refuses to overwrite):
  results/model1_eval_protocol.json
  results/model1_eval_protocol.md

Run:  python eval_protocol.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

import config
import train

THRESHOLD = config.WATER_THRESHOLD  # 0.5, unchanged
EXPECTED_MASK_A_UNET_IOU = {"test_all": 0.6626, "test_india": 0.6632, "bolivia": 0.6611}
TOLERANCE = 0.001
GROUP_NAMES = {"test_all": "Test (all)", "test_india": "Test (India only)",
               "bolivia": "Bolivia holdout"}


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_info():
    def run(*args):
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "working_tree_clean_except_untracked": run("status", "--porcelain",
                                                           "--untracked-files=no") == ""}
    except Exception as err:  # noqa: BLE001
        return {"commit": None, "error": str(err)}


def count_full_test_runs(paths):
    """Count FULL test-stage runs recorded in outputs/logs/test.log (smoke runs are separate)."""
    log = paths.logs / "test.log"
    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    return text.count("===== Stage test (FULL) started =====")


def counts(pred, truth, valid):
    tp = int((pred & truth & valid).sum())
    fp = int((pred & ~truth & valid).sum())
    fn = int((~pred & truth & valid).sum())
    tn = int((~pred & ~truth & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ratio(a, b):
    return a / b if b > 0 else float("nan")


def nan_to_none(x):
    """Recursively replace NaN floats with None so the JSON is valid (null = not computable)."""
    if isinstance(x, dict):
        return {k: nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [nan_to_none(v) for v in x]
    return None if isinstance(x, float) and np.isnan(x) else x


def summarize(tile_counts, tile_has_water):
    """Metrics for one method in one group from per-tile confusion counts."""
    tot = {k: sum(c[k] for c in tile_counts) for k in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = tot["tp"], tot["fp"], tot["fn"], tot["tn"]
    water_iou = ratio(tp, tp + fp + fn)
    nonwater_iou = ratio(tn, tn + fp + fn)
    tile_iou = [ratio(c["tp"], c["tp"] + c["fp"] + c["fn"]) for c in tile_counts]
    defined = [v for v in tile_iou if not np.isnan(v)]
    with_water = [v for v, w in zip(tile_iou, tile_has_water) if w]
    return {
        "pooled_water_iou": water_iou,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "pooled_nonwater_iou": nonwater_iou,
        "pooled_mean_iou_2class": (water_iou + nonwater_iou) / 2
        if not (np.isnan(water_iou) or np.isnan(nonwater_iou)) else float("nan"),
        # Per-tile IoU is undefined (0/0) on a tile with no water in the label AND no
        # water predicted. Those tiles cannot enter a mean; they are counted separately.
        "mean_tile_iou_all_tiles": float(np.mean(defined)) if defined else float("nan"),
        "n_tiles_iou_defined": len(defined),
        "n_tiles_iou_undefined": len(tile_iou) - len(defined),
        "mean_tile_iou_tiles_with_water": float(np.mean(with_water)) if with_water else float("nan"),
        "n_tiles_with_water": len(with_water),
        "counts": tot,
    }


def main():
    paths = config.get_paths(smoke=False)
    out_json = paths.results / "model1_eval_protocol.json"
    out_md = paths.results / "model1_eval_protocol.md"
    for p in (out_json, out_md):
        if p.exists():
            sys.exit(f"STOP: {p} already exists; this script only adds new files.")

    device = train.get_device()
    if device.type != "cuda":
        sys.exit("STOP: no CUDA GPU available; this evaluation was requested on GPU.")
    use_amp = config.USE_AMP and device.type == "cuda"  # same setting as train.py run_test

    model, ckpt = train.load_trained_model(paths, device)
    stats = config.load_norm_stats(paths)  # results/norm_stats.json
    for k in ("vv_mean", "vv_std", "vh_mean", "vh_std"):
        if abs(stats[k] - ckpt["norm_stats"][k]) > 1e-9:
            sys.exit(f"STOP: results/norm_stats.json {k} differs from the checkpoint copy.")
    if ckpt["epoch"] != 44:
        sys.exit(f"STOP: checkpoint epoch is {ckpt['epoch']}, expected 44.")

    test_ids = train.get_split_ids(paths, "test", smoke=False)
    bolivia_ids = train.get_split_ids(paths, "bolivia", smoke=False)
    groups = {"test_all": test_ids,
              "test_india": [t for t in test_ids if t.startswith("India")],
              "bolivia": bolivia_ids}

    with torch.no_grad():
        probs = train.predict_tiles(model, paths, sorted(set(test_ids + bolivia_ids)),
                                    stats, device, use_amp)

    results = {}
    for group, ids in groups.items():
        per_mask = {}
        for mask_name in ("A", "B"):
            unet_c, otsu_c, has_water, n_valid = [], [], [], 0
            n_otsu_undefined_in_mask = 0
            for tid in ids:
                label = train.read_label(paths, tid)
                otsu = train.read_label(paths, tid, "S1OtsuLabelHand")
                otsu_defined = (otsu == 0) | (otsu == 1)
                valid = label >= 0
                if mask_name == "A":
                    valid = valid & otsu_defined
                truth = label == 1
                n_valid += int(valid.sum())
                n_otsu_undefined_in_mask += int((valid & ~otsu_defined).sum())
                has_water.append(bool((truth & valid).any()))
                unet_c.append(counts(probs[tid] >= THRESHOLD, truth, valid))
                otsu_c.append(counts(otsu == 1, truth, valid))  # undefined Otsu -> not water
            per_mask[mask_name] = {
                "n_tiles": len(ids),
                "n_valid_pixels": n_valid,
                "n_tiles_zero_water_in_label": has_water.count(False),
                "n_valid_pixels_with_otsu_undefined": n_otsu_undefined_in_mask,
                "unet": summarize(unet_c, has_water),
                "otsu": summarize(otsu_c, has_water),
            }
        results[group] = per_mask

    # Step 1: sanity check against the saved numbers (Mask A = current protocol).
    sanity = {}
    for group, expected in EXPECTED_MASK_A_UNET_IOU.items():
        got = results[group]["A"]["unet"]["pooled_water_iou"]
        sanity[group] = {"expected": expected, "reproduced": got,
                         "abs_diff": abs(got - expected), "ok": abs(got - expected) <= TOLERANCE}
    print("Step 1 sanity check (Mask A, U-Net pooled water IoU):")
    for g, s in sanity.items():
        print(f"  {g:<11} expected {s['expected']:.4f}  reproduced {s['reproduced']:.6f}  "
              f"diff {s['abs_diff']:.6f}  {'OK' if s['ok'] else 'FAIL'}")
    saved = json.loads((paths.results / "model1_metrics.json").read_text())
    for g in groups:
        for m in ("unet", "otsu"):
            sv, rv = saved["groups"][g][m]["iou"], results[g]["A"][m]["pooled_water_iou"]
            print(f"  vs model1_metrics.json {g:<11} {m:<5} saved {sv:.6f}  now {rv:.6f}")
    if not all(s["ok"] for s in sanity.values()):
        sys.exit("STOP: Mask A does not reproduce the saved numbers within 0.001. "
                 "Nothing was written.")

    prior_runs = count_full_test_runs(paths)
    sha = sha256_of(paths.checkpoint)
    out = {
        "description": "Model 1 re-scored under two pixel masks; no retraining, no tuning.",
        "checkpoint_path": str(paths.checkpoint.relative_to(config.ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_iou": ckpt["val_iou"],
        "norm_stats": "results/norm_stats.json (identical to the copy stored in the checkpoint)",
        "threshold": THRESHOLD,
        "amp": use_amp,
        "device": torch.cuda.get_device_name(0),
        "git": git_info(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "masks": {
            "A": "hand label >= 0 AND Otsu label in {0,1} (current protocol, train.py run_test)",
            "B": "hand label >= 0 only (standard). Otsu: pixels with undefined Otsu label "
                 "are counted as predicted not-water.",
        },
        "definitions": {
            "pooled_water_iou": "TP/(TP+FP+FN) with TP, FP, FN summed over all valid pixels of all tiles",
            "pooled_mean_iou_2class": "mean of pooled water IoU and pooled non-water IoU TN/(TN+FP+FN)",
            "mean_tile_iou_all_tiles": "mean of per-tile water IoU over every tile where it is defined; "
                                       "tiles with no water in label and none predicted have 0/0 IoU "
                                       "and are excluded, counted in n_tiles_iou_undefined",
            "mean_tile_iou_tiles_with_water": "mean of per-tile water IoU over tiles with >= 1 water "
                                              "pixel inside the mask",
            "n_tiles_zero_water_in_label": "tiles with no water pixel inside the mask",
        },
        "sanity_check_step1": sanity,
        "test_split_scorings": {
            "full_runs_in_outputs_logs_test_log": prior_runs,
            "this_run": 1,
            "total": (prior_runs + 1) if prior_runs is not None else None,
        },
        "groups": results,
    }

    def f(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

    cols = [("pooled water IoU", "pooled_water_iou"), ("precision", "precision"),
            ("recall", "recall"), ("mean tile IoU (all)", "mean_tile_iou_all_tiles"),
            ("mean tile IoU (tiles w/ water)", "mean_tile_iou_tiles_with_water"),
            ("pooled 2-class mIoU", "pooled_mean_iou_2class")]
    header = ("| Group | Mask | n tiles | tiles w/o water | valid pixels | "
              + " | ".join(f"U-Net {c} | Otsu {c}" for c, _ in cols) + " |")
    sep = "|" + "---|" * (5 + 2 * len(cols))
    rows = []
    for g in groups:
        for mk in ("A", "B"):
            r = results[g][mk]
            cells = []
            for _, key in cols:
                cells += [f(r["unet"][key]), f(r["otsu"][key])]
            rows.append(f"| {GROUP_NAMES[g]} | {mk} | {r['n_tiles']} | "
                        f"{r['n_tiles_zero_water_in_label']} | {r['n_valid_pixels']:,} | "
                        + " | ".join(cells) + " |")
    undefined_lines = []
    for g in groups:
        for mk in ("A", "B"):
            r = results[g][mk]
            undefined_lines.append(
                f"- {GROUP_NAMES[g]}, mask {mk}: per-tile IoU undefined (0/0) on "
                f"{r['unet']['n_tiles_iou_undefined']} tile(s) for U-Net, "
                f"{r['otsu']['n_tiles_iou_undefined']} for Otsu; "
                f"{r['n_valid_pixels_with_otsu_undefined']:,} valid pixels have no Otsu label.")
    md = "\n".join([
        "# Model 1 evaluation under two masks",
        "",
        f"Generated by eval_protocol.py on {out['date']}. Checkpoint `{out['checkpoint_path']}` "
        f"(epoch {ckpt['epoch']}, SHA256 `{sha}`), git commit `{out['git'].get('commit')}`, "
        f"threshold {THRESHOLD}, no retraining.",
        "",
        "- **Mask A** (current protocol): hand label >= 0 AND Otsu label defined.",
        "- **Mask B** (standard): hand label >= 0 only. Otsu pixels with no Otsu label are "
        "counted as predicted not-water.",
        "- Water IoU pooled = one IoU from TP/FP/FN summed over all valid pixels. 2-class mIoU = "
        "mean of pooled water and non-water IoU.",
        "- Mean tile IoU (all) excludes tiles where per-tile IoU is 0/0 (no water in label, none "
        "predicted); see counts below.",
        "",
        header, sep, *rows,
        "",
        "Tiles with undefined per-tile IoU and pixels without an Otsu label:",
        "",
        *undefined_lines,
        "",
        f"Times the test split has been scored (full runs): {prior_runs} in outputs/logs/test.log "
        f"+ this run = {out['test_split_scorings']['total']}.",
        "",
    ])

    out_json.write_text(json.dumps(nan_to_none(out), indent=2, allow_nan=False) + "\n")
    out_md.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
