"""
summary.py - stage summary: writes results/SUMMARY.md from the saved result files.

Every number in SUMMARY.md is read from a file in results/ by this code. If
a file is missing, the section says which stage has not produced it yet.
Nothing is typed by hand.

Run alone:  python summary.py [--smoke]
"""

import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import common
import config

STAGES_IN_THIS_FILE = ["summary"]


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return "n/a" if not np.isfinite(v) else f"{v:.{digits}f}"
    return str(v)


def git_info(root):
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return commit + (" (with uncommitted changes)" if dirty else "")


def missing(stage):
    return [f"_Not available: stage `{stage}` has not produced its results yet "
            "(see STATUS.md)._", ""]


def model1_section(res):
    lines = ["## Model 1: flood detection (U-Net vs Otsu on Sen1Floods11)", ""]
    metrics = load_json(res / "model1_metrics.json")
    train = load_json(res / "model1_train_summary.json")
    if train:
        lines.append(f"- training: {train['epochs_run']} epoch(s) on {train['device']}, best "
                     f"validation IoU {fmt(train['best_val_iou'])} at epoch {train['best_epoch']}, "
                     f"encoder weights: {train['encoder_weights_used']}, batch {train['batch_size']}, "
                     f"early stopped: {train['early_stopped']}")
        lines.append(f"- training curve check: {train['verdict']}")
    if not metrics:
        return lines + [""] + missing("test")
    names = {"test_all": "Test (all)", "test_india": "Test (India only)", "bolivia": "Bolivia holdout"}
    lines += ["", "| Group | n tiles | U-Net IoU | Otsu IoU | U-Net precision | U-Net recall |",
              "|---|---|---|---|---|---|"]
    for g, m in metrics["groups"].items():
        lines.append(f"| {names.get(g, g)} | {m['n_tiles']} | {fmt(m['unet']['iou'])} | "
                     f"{fmt(m['otsu']['iou'])} | {fmt(m['unet']['precision'])} | {fmt(m['unet']['recall'])} |")
    for w in metrics.get("warnings", []):
        lines.append(f"\nWarning: {w}")
    return lines + [""]


def domain_section(res):
    lines = ["## Domain shift (Assam imagery vs training normalization)", ""]
    shift = load_json(res / "domain_shift.json")
    if not shift:
        return lines + missing("export_s1")
    lines += ["| Data | VV mean (dB) | VV std | VH mean (dB) | VH std |", "|---|---|---|---|---|"]
    for key, label in [("train", "Sen1Floods11 train"), ("assam_normal", "Assam normal years"),
                       ("assam_flood", "Assam flood window")]:
        s = shift.get(key)
        if s:
            lines.append(f"| {label} | {fmt(s['vv_mean'], 2)} | {fmt(s['vv_std'], 2)} | "
                         f"{fmt(s['vh_mean'], 2)} | {fmt(s['vh_std'], 2)} |")
    lines += ["", shift.get("note", ""), ""]
    return lines


def flood_section(res):
    lines = ["## Flooded area per district (maximum extent, excluding normal water)", ""]
    fm = load_json(res / "flood_map_summary.json")
    exp = load_json(res / "export_s1_summary.json")
    if exp:
        aoi = exp["aoi"]
        lines.append(f"- districts: {', '.join(aoi['districts'])} (boundaries: {aoi['district_source']})")
        lines.append(f"- orbit: {aoi['orbit']['orbit_state']} relative orbit {aoi['orbit']['relative_orbit']}, "
                     f"collection {aoi['collection']}")
        lines.append(f"- flood passes: {', '.join(p['date'] for p in exp['passes']['flood'])}")
        lines.append(f"- normal passes: {', '.join(p['date'] for p in exp['passes']['normal'])}")
    if not fm:
        return lines + [""] + missing("flood_map")
    lines += ["", "| District | Flooded km2 (max extent) | Observed km2 | Normal water km2 | Max duration (days) |",
              "|---|---|---|---|---|"]
    for d, v in fm["per_district"].items():
        lines.append(f"| {d} | {fmt(v['max_extent_flooded_km2'], 1)} | {fmt(v['observed_km2'], 1)} | "
                     f"{fmt(v['normal_water_km2'], 1)} | {v['max_duration_days']} |")
    dist = fm["plausibility"].get("distance_to_normal_water_m")
    elev = fm["plausibility"].get("elevation_m")
    lines.append("")
    if dist:
        lines.append(f"- flooded cells: mean distance to normal water {fmt(dist['flooded_mean'], 0)} m vs "
                     f"{fmt(dist['random_land_mean'], 0)} m for random land cells (permutation p = "
                     f"{fmt(dist['p_value'])})")
    if elev:
        lines.append(f"- flooded cells: mean elevation {fmt(elev['flooded_mean'], 1)} m vs "
                     f"{fmt(elev['random_land_mean'], 1)} m for random land cells (p = {fmt(elev['p_value'])})")
    lines.append(f"- revisit interval used for duration: {fm['revisit_interval_days']} days")
    return lines + [""]


def sanity_section(res):
    lines = ["## Sanity checks", ""]
    sc = load_json(res / "sanity_check.json")
    if not sc:
        return lines + missing("sanity_check")
    s2 = sc.get("sentinel2_optical", {})
    j = sc.get("jrc_gsw_normal_water", {})
    lines.append(f"- Sentinel-2 optical check: {s2.get('status')}"
                 + (f", IoU {fmt(s2['metrics_model_vs_s2']['iou'])} on clear pixels "
                    f"({s2['s2_date']} vs pass {s2['flood_pass']})" if s2.get("status") == "DONE"
                    else f" ({s2.get('reason', '')})"))
    lines.append(f"- Normal water vs JRC GSW: {j.get('status')}"
                 + (f", IoU {fmt(j['metrics_normal_water_vs_gsw']['iou'])}, agreement "
                    f"{fmt(j['metrics_normal_water_vs_gsw']['agreement'])}" if j.get("status") == "DONE"
                    else f" ({j.get('reason', '')})"))
    lines.append(f"- Visual figure: {sc.get('visual_figure', {}).get('status')}")
    return lines + [""]


def risk_section(res):
    lines = ["## Top 10 village proxies by risk", ""]
    path = res / "village_risk_ranked.csv"
    meta = load_json(res / "risk_meta.json")
    if not path.exists() or not meta:
        return lines + missing("risk")
    ranked = pd.read_csv(path, comment="#")
    lines.append(f"- {meta['n_units']} village proxies; units: {meta.get('unit_source')}")
    lines.append(f"- components used: {', '.join(meta['components'])}; dropped: "
                 f"{', '.join(meta['dropped']) or 'none'}")
    lines += ["", "| Rank | Unit | Name | District | Score | Flooded fraction | Max duration (days) |",
              "|---|---|---|---|---|---|---|"]
    for r in ranked.head(10).itertuples():
        name = "" if pd.isna(r.name) else str(r.name)
        lines.append(f"| {r.rank} | {r.unit_id} | {name} | {r.district} | {fmt(r.score, 1)} | "
                     f"{fmt(r.flooded_frac)} | {fmt(r.max_duration_days, 0)} |")
    evo_path = res / "ranking_evolution.csv"
    if evo_path.exists():
        evo = pd.read_csv(evo_path)
        lines += ["", "Ranking after each pass vs the final ranking:", "",
                  "| Pass | Date | Overlap with final top-K | Spearman vs final |", "|---|---|---|---|"]
        for r in evo.itertuples():
            lines.append(f"| {r.pass_index} | {r.date} | {r.overlap_with_final_top_k}/{r.top_k} | "
                         f"{fmt(r.spearman_rank_vs_final)} |")
    return lines + [""]


def validation_section(res):
    lines = ["## Validation against IDSP outbreak reports", ""]
    v = load_json(res / "validation.json")
    if not v:
        return lines + missing("validate")
    if v.get("status") != "DONE":
        return lines + [f"**VALIDATION NOT RUN**: {v.get('reason')}", ""]
    lines += ["| Method | Outcome | n districts | Spearman | 95% interval | p-value |",
              "|---|---|---|---|---|---|"]
    for r in v["district_results"]:
        lines.append(f"| {r['method']} | {r['outcome']} | {r['n']} | {fmt(r['rho'])} | "
                     f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {fmt(r['p_value'])} |")
    if v.get("district_week_results"):
        lines += ["", "| District-week method | rows | positives | AUC |", "|---|---|---|---|"]
        for r in v["district_week_results"]:
            lines.append(f"| {r['method']} | {r['n_rows']} | {r['n_positive']} | {fmt(r['auc'])} |")
    lines += ["", "Limitations (generated): "] + [f"- {x}" for x in v.get("limitations", [])]
    return lines + [""]


def stages_section(paths):
    status = common.load_status(paths)
    import run_all
    lines = ["## Stages", "", "| Stage | Status | Runtime (s) | Reason |", "|---|---|---|---|"]
    for name in run_all.STAGE_ORDER:
        e = status.get("stages", {}).get(name)
        if e is None:
            lines.append(f"| {name} | NOT RUN | | not run yet |")
        else:
            lines.append(f"| {name} | {e['status']} | {e.get('runtime_s', '')} | "
                         f"{str(e.get('reason', '')).replace('|', '/')[:200]} |")
    return lines + [""]


def pitch_numbers(res):
    """Short list of headline numbers, each read from a file (source shown)."""
    out = []
    m = load_json(res / "model1_metrics.json")
    if m:
        g = m["groups"]["test_all"]
        out.append(f"U-Net water IoU on {g['n_tiles']} held-out test tiles: {fmt(g['unet']['iou'])} "
                   f"(Otsu baseline {fmt(g['otsu']['iou'])}) [model1_metrics.json]")
        gi = m["groups"]["test_india"]
        out.append(f"India-only test tiles ({gi['n_tiles']}): U-Net IoU {fmt(gi['unet']['iou'])} vs "
                   f"Otsu {fmt(gi['otsu']['iou'])} [model1_metrics.json]")
    fm = load_json(res / "flood_map_summary.json")
    if fm:
        total = sum(v["max_extent_flooded_km2"] for v in fm["per_district"].values())
        out.append(f"Flooded area mapped across {len(fm['per_district'])} district(s): {fmt(total, 1)} km2 "
                   f"from {len(fm['flood_dates'])} Sentinel-1 passes [flood_map_summary.json]")
    meta = load_json(res / "risk_meta.json")
    if meta:
        out.append(f"Village proxies ranked: {meta['n_units']} [risk_meta.json]")
    evo_path = res / "ranking_evolution.csv"
    if evo_path.exists():
        evo = pd.read_csv(evo_path)
        if len(evo):
            first = evo.iloc[0]
            out.append(f"After the first pass alone, {first['overlap_with_final_top_k']} of the final "
                       f"top-{first['top_k']} were already in the top-{first['top_k']} [ranking_evolution.csv]")
    v = load_json(res / "validation.json")
    if v and v.get("status") == "DONE":
        best = [r for r in v["district_results"] if r["method"].startswith("Clearwell") and r["outcome"] == "outbreaks"]
        for r in best[:1]:
            out.append(f"District-level Spearman (index vs outbreaks): {fmt(r['rho'])}, n = {r['n']} "
                       "(indicative only) [validation.json]")
    return out


def run_summary(smoke=False, logger=None, **kwargs):
    paths = config.get_paths(smoke)
    res = paths.results
    env = load_json(res / "env.json") or config.environment_summary()
    lines = ["# Clearwell results summary" + (" (SMOKE RUN: numbers are NOT meaningful)" if smoke else ""), "",
             f"Generated by summary.py at {datetime.now().isoformat(timespec='seconds')} from the files "
             "in this folder. No number here was typed by hand.", "",
             f"- git commit: {git_info(paths.root)}",
             f"- torch {env.get('torch')}, CUDA available: {env.get('cuda_available')}, GPU: "
             f"{env.get('gpu_name', 'none')}, python {env.get('python')}", ""]
    lines += ["## Pitch numbers (each read from the file in brackets)", ""]
    pitch = pitch_numbers(res)
    lines += [f"- {p}" for p in pitch] if pitch else ["- none yet: the stages that produce them have not run"]
    lines.append("")
    for section in (model1_section, domain_section, flood_section, sanity_section, risk_section,
                    validation_section):
        lines += section(res)
    lines += stages_section(paths)
    (res / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {res / 'SUMMARY.md'}")
    return common.result("OK")


if __name__ == "__main__":
    import run_all
    sys.exit(run_all.main(default_stages=STAGES_IN_THIS_FILE))
