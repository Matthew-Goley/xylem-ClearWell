"""
run_all.py - run the Clearwell pipeline stages in order.

    python run_all.py                       # all stages, skip the ones already done
    python run_all.py --smoke               # tiny version of everything (code check)
    python run_all.py --stage train         # one stage (comma list also works)
    python run_all.py --stage train --force # rerun even if it finished before

Each finished stage writes a marker file in outputs/markers/ (or
outputs/smoke/markers/). A stage that fails is recorded in results/STATUS.md
and the pipeline keeps going with every later stage that does not depend on
it. Logs go to outputs/logs/<stage>.log.
"""

import argparse
import importlib
import sys
import time
from datetime import datetime

import common
import config

# (stage name, module, function, stages it needs)
STAGES = [
    ("check_env", "run_all", "run_check_env", []),
    ("get_sen1floods11", "train", "run_get_data", []),
    ("train", "train", "run_train", ["get_sen1floods11"]),
    ("test", "train", "run_test", ["train"]),
    ("export_s1", "export_s1", "run_export", []),
    ("flood_map", "flood_map", "run_flood_map", ["train", "export_s1"]),
    ("sanity_check", "flood_map", "run_sanity_check", ["flood_map"]),
    ("villages", "build_villages", "run_villages", ["flood_map"]),
    ("risk", "risk_score", "run_risk", ["villages"]),
    ("validate", "validate_risk", "run_validate", ["risk"]),
    ("summary", "summary", "run_summary", []),
]
STAGE_ORDER = [s[0] for s in STAGES]
# These are cheap and should always reflect the current state, so they never skip.
ALWAYS_RUN = {"check_env", "summary"}


# ---------------------------------------------------------------------------
# Stage: check_env
# ---------------------------------------------------------------------------
def run_check_env(smoke=False, logger=None, **kwargs):
    """Print torch / CUDA / GPU info and save it to results/env.json."""
    import json
    paths = config.get_paths(smoke)
    info = config.environment_summary()
    for key, value in info.items():
        logger.info(f"{key:>20}: {value}")
    warnings = []
    if "torch_error" in info:
        return common.result("FAILED", f"torch could not be imported: {info['torch_error']}")
    if not info.get("cuda_available"):
        msg = ("NO CUDA GPU FOUND. Training will run on the CPU, which is only sensible for "
               "--smoke runs. On the home desktop check `nvidia-smi` and the torch install.")
        common.loud_warning(logger, msg)
        warnings.append("no CUDA GPU on this machine (CPU only)")
    with open(paths.results / "env.json", "w") as f:
        json.dump(info, f, indent=2)
    return common.result("OK", "", warnings, info)


# ---------------------------------------------------------------------------
# Running stages
# ---------------------------------------------------------------------------
def run_stage(name, smoke=False, force=False, extra=None):
    """Run one stage with marker / dependency / status bookkeeping."""
    paths = config.get_paths(smoke)
    logger = common.get_logger(paths, name)
    _, module_name, func_name, deps = next(s for s in STAGES if s[0] == name)
    marker = common.marker_path(paths, name)

    if marker.exists() and not force and name not in ALWAYS_RUN:
        logger.info(f"Stage {name}: already done ({marker}). Use --force to rerun.")
        return "SKIPPED"

    missing = [d for d in deps if not common.marker_path(paths, d).exists()]
    if missing:
        res = common.result("NOT RUN", f"needs stage(s) {', '.join(missing)} to finish first")
        logger.warning(f"Stage {name}: NOT RUN, {res['reason']}")
        common.record_stage(paths, name, res, 0.0, STAGE_ORDER)
        return res["status"]

    logger.info(f"===== Stage {name} ({'SMOKE' if smoke else 'FULL'}) started =====")
    start = time.time()
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        res = func(smoke=smoke, logger=logger, **(extra or {}))
    except Exception as err:  # noqa: BLE001 - any failure must be recorded, not crash the run
        short, tb = common.format_exception(err)
        logger.error(f"Stage {name} FAILED: {short}\n{tb}")
        res = common.result("FAILED", short)
    runtime = time.time() - start
    for warning in res.get("warnings", []):
        logger.warning(f"[{name}] {warning}")
    logger.info(f"===== Stage {name}: {res['status']} in {runtime:.1f} s "
                f"{('(' + res['reason'] + ')') if res.get('reason') else ''} =====")
    if res["status"] == "OK" and name not in ALWAYS_RUN:
        common.write_marker(paths, name, {"finished_at": datetime.now().isoformat(),
                                          "runtime_s": runtime, "smoke": smoke})
    common.record_stage(paths, name, res, runtime, STAGE_ORDER)
    return res["status"]


def run_stages(names, smoke=False, force=False, extra=None):
    outcomes = {}
    for name in names:
        outcomes[name] = run_stage(name, smoke=smoke, force=force, extra=extra)
    # Always refresh the summary at the end of a multi-stage run.
    if len(names) > 1 and "summary" not in names:
        outcomes["summary"] = run_stage("summary", smoke=smoke)
    return outcomes


def parse_args(argv=None, default_stages=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default=None,
                        help="stage name or comma separated list. Stages: "
                             + ", ".join(STAGE_ORDER))
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (few tiles, 1 epoch, small area) to check the code")
    parser.add_argument("--force", action="store_true", help="rerun stages that are done")
    parser.add_argument("--extra-normal", action="store_true",
                        help="export_s1: also use config.EXTRA_NORMAL_YEARS as normal years")
    args = parser.parse_args(argv)
    if args.stage:
        names = [s.strip() for s in args.stage.split(",") if s.strip()]
    else:
        names = default_stages or STAGE_ORDER
    unknown = [n for n in names if n not in STAGE_ORDER]
    if unknown:
        parser.error(f"unknown stage(s): {unknown}. Valid: {STAGE_ORDER}")
    return args, names


def main(argv=None, default_stages=None):
    args, names = parse_args(argv, default_stages)
    extra = {"extra_normal": args.extra_normal}
    outcomes = run_stages(names, smoke=args.smoke, force=args.force, extra=extra)
    print("\nStage outcomes:")
    for name, status in outcomes.items():
        print(f"  {name:<18} {status}")
    paths = config.get_paths(args.smoke)
    print(f"\nSee {paths.results / 'STATUS.md'} and {paths.results / 'SUMMARY.md'}")
    # Exit code 1 if anything failed, so scripts can notice.
    return 1 if any(s == "FAILED" for s in outcomes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
