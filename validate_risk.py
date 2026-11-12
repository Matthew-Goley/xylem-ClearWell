"""
validate_risk.py - MODEL 2 validation against IDSP outbreak reports (district level).

Input: data/manual/idsp_weather.csv (your friend's file: IDSP weekly outbreak
reports plus weather). Its exact columns are unknown, so the loader:
  * prints the columns, normalizes their names, finds district / date /
    disease / cases / rainfall columns by keywords,
  * removes duplicate rows,
  * keeps Assam and water-borne diseases (config.WATERBORNE_KEYWORDS),
  * fuzzy-matches district spellings to the boundary names (rapidfuzz) and
    saves the mapping to results/idsp_district_mapping.csv for you to check,
  * builds a district x week table where a missing week means zero outbreaks,
    but ONLY for weeks inside the file's own date coverage.

Comparisons (Spearman, permutation p-value, bootstrap 95% interval):
  Clearwell index (district mean of top-decile units, and overall mean)
  B1 flooded area alone, B2 rainfall alone (if the file has rainfall),
  B3 people (or buildings) in flooded area alone.
With only 2-3 mapped districts the statistical power is very low; this is
said plainly in results/validation.md. A district-week design (mapped
districts x weeks around the flood) is also reported.

If the file is missing or does not cover the flood year, the stage reports
"VALIDATION NOT RUN" with the reason. No data are ever invented.

Run alone:  python validate_risk.py [--smoke]
"""

import itertools
import json
import re
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

import common
import config

STAGES_IN_THIS_FILE = ["validate"]


# ---------------------------------------------------------------------------
# Loading the IDSP file
# ---------------------------------------------------------------------------
def normalize_column(name):
    """'Name of District ' -> 'name_of_district'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(columns, include, exclude=()):
    """First column whose name contains any include word and no exclude word."""
    for col in columns:
        if any(w in col for w in include) and not any(w in col for w in exclude):
            return col
    return None


def detect_columns(columns):
    """Guess which normalized column holds what. Values may be None."""
    date_col = (find_column(columns, ["start", "onset", "outbreak_date"], ["week", "year"])
                or find_column(columns, ["report"], ["week", "year", "status"])
                or find_column(columns, ["date"], ["update"]))
    return {
        "district": find_column(columns, ["district"]),
        "state": find_column(columns, ["state"]),
        "disease": find_column(columns, ["disease", "illness", "syndrome"]),
        "cases": find_column(columns, ["case"], ["death", "fatal"]),
        "deaths": find_column(columns, ["death"]),
        "date": date_col,
        "year": find_column(columns, ["year"]),
        "week": find_column(columns, ["week"], ["year"]),
        "rainfall": find_column(columns, ["rain", "precip", "prcp"]),
    }


def is_waterborne(disease):
    text = str(disease).lower()
    for kw in config.WATERBORNE_KEYWORDS:
        if len(kw) <= 3:  # short ones like "add" must be a whole word
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        elif kw in text:
            return True
    return False


def read_csv_any_encoding(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def monday(d):
    d = pd.Timestamp(d)
    return (d - pd.Timedelta(days=d.weekday())).normalize()


def load_idsp(path, district_names, logger):
    """Clean outbreak events: one row per outbreak with district (matched), date, week_start.

    Returns (events DataFrame, info dict, mapping DataFrame).
    """
    raw = read_csv_any_encoding(path)
    info = {"file": str(path), "raw_rows": len(raw), "raw_columns": list(map(str, raw.columns))}
    logger.info(f"IDSP file {path}: {len(raw)} rows, columns: {list(raw.columns)}")
    logger.info("First rows:\n" + raw.head(3).to_string())
    df = raw.copy()
    df.columns = [normalize_column(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    before = len(df)
    df = df.drop_duplicates()
    info["exact_duplicates_removed"] = before - len(df)
    cols = detect_columns(list(df.columns))
    info["detected_columns"] = cols
    logger.info(f"Detected columns: {cols}")
    if cols["district"] is None:
        raise ValueError("no district column found in the IDSP file")

    # Dates: a date column, else ISO year + week.
    dates = pd.Series(pd.NaT, index=df.index)
    if cols["date"]:
        dates = pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True)
    if dates.isna().all() and cols["year"] and cols["week"]:
        years = pd.to_numeric(df[cols["year"]], errors="coerce")
        weeks = pd.to_numeric(df[cols["week"]], errors="coerce")
        dates = pd.Series([pd.Timestamp(date.fromisocalendar(int(y), int(w), 1))
                           if pd.notna(y) and pd.notna(w) and 1 <= w <= 53 else pd.NaT
                           for y, w in zip(years, weeks)], index=df.index)
    df["date"] = dates
    info["rows_without_date"] = int(df["date"].isna().sum())
    df = df[df["date"].notna()].copy()

    if cols["state"]:
        states = df[cols["state"]].dropna().unique().tolist()
        m = common.match_names(states, [config.STATE_NAME])
        keep = {x["query"] for x in m if x["match"] is not None}
        info["state_values_kept"] = sorted(keep)
        df = df[df[cols["state"]].isin(keep)].copy()
    info["rows_in_state"] = len(df)

    if cols["disease"] and config.FILTER_WATERBORNE:
        mask = df[cols["disease"]].apply(is_waterborne)
        info["diseases_kept"] = sorted(df.loc[mask, cols["disease"]].astype(str).unique().tolist())[:50]
        info["diseases_dropped"] = sorted(df.loc[~mask, cols["disease"]].astype(str).unique().tolist())[:50]
        df = df[mask].copy()
    else:
        info["disease_filter"] = "not applied (no disease column or FILTER_WATERBORNE False)"
    info["rows_waterborne"] = len(df)

    raw_names = df[cols["district"]].dropna().unique().tolist()
    matches = common.match_names(raw_names, district_names)
    mapping = pd.DataFrame(matches)
    lookup = {m["query"]: m["match"] for m in matches}
    df["district_matched"] = df[cols["district"]].map(lookup)
    unmatched = mapping[mapping["match"].isna()]["query"].tolist()
    if unmatched:
        logger.warning(f"District names not matched (rows dropped): {unmatched}")
    info["unmatched_districts"] = unmatched
    df = df[df["district_matched"].notna()].copy()

    key = ["district_matched", "date"] + [cols[k] for k in ("disease", "cases") if cols[k]]
    before = len(df)
    df = df.drop_duplicates(subset=key)
    info["key_duplicates_removed"] = before - len(df)
    df["week_start"] = df["date"].apply(monday)
    df["cases"] = pd.to_numeric(df[cols["cases"]], errors="coerce") if cols["cases"] else np.nan
    df["rainfall"] = pd.to_numeric(df[cols["rainfall"]], errors="coerce") if cols["rainfall"] else np.nan
    df["disease"] = df[cols["disease"]] if cols["disease"] else ""
    events = df[["district_matched", "date", "week_start", "disease", "cases", "rainfall"]].rename(
        columns={"district_matched": "district"}).reset_index(drop=True)
    if len(events):
        info["coverage_start"] = events["date"].min().date().isoformat()
        info["coverage_end"] = events["date"].max().date().isoformat()
        info["rows_per_year"] = {str(k): int(v) for k, v in events["date"].dt.year.value_counts().sort_index().items()}
        info["rows_per_month"] = {str(k): int(v) for k, v in
                                  events["date"].dt.to_period("M").value_counts().sort_index().items()}
        logger.info(f"IDSP coverage (Assam, water-borne): {info['coverage_start']} to "
                    f"{info['coverage_end']}; rows per year {info['rows_per_year']}")
    info["events_after_cleaning"] = len(events)
    return events, info, mapping


def district_week_table(events, districts, start, end):
    """Every district x week (Monday) from start to end; missing = 0 outbreaks."""
    weeks = pd.date_range(monday(start), monday(end), freq="7D")
    index = pd.MultiIndex.from_product([districts, weeks], names=["district", "week_start"])
    counts = events.groupby(["district", "week_start"]).size()
    table = counts.reindex(index, fill_value=0).rename("outbreaks").reset_index()
    return table


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def spearman_with_uncertainty(x, y, n_perm=None, n_boot=None, seed=None):
    """Spearman rho, two-sided permutation p-value, bootstrap 95% interval.

    Exact permutations when n <= 8, otherwise Monte Carlo.
    """
    n_perm = n_perm or config.N_PERMUTATIONS
    n_boot = n_boot or config.N_BOOTSTRAP
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    out = {"n": n, "rho": float("nan"), "p_value": float("nan"), "ci_low": float("nan"),
           "ci_high": float("nan"), "n_valid_bootstrap": 0, "permutation": ""}
    if n < 3:
        out["note"] = "fewer than 3 districts: correlation not computable"
        return out
    rho = spearman(x, y)
    out["rho"] = rho
    if not np.isfinite(rho):
        out["note"] = "one of the variables is constant"
        return out
    if n <= 8:
        perms = [spearman(x, np.array(p)) for p in itertools.permutations(y)]
        out["permutation"] = f"exact ({len(perms)} permutations)"
    else:
        perms = [spearman(x, rng.permutation(y)) for _ in range(n_perm)]
        out["permutation"] = f"Monte Carlo ({n_perm})"
    perms = np.array(perms)
    perms = perms[np.isfinite(perms)]
    as_extreme = int(np.sum(np.abs(perms) >= abs(rho) - 1e-12))
    if n <= 8:
        # Exact test: the observed order is one of the permutations.
        out["p_value"] = as_extreme / len(perms)
    else:
        out["p_value"] = (1 + as_extreme) / (len(perms) + 1)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearman(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(r)
    out["n_valid_bootstrap"] = len(boots)
    if len(boots) >= 20:
        out["ci_low"], out["ci_high"] = map(float, np.percentile(boots, [2.5, 97.5]))
    else:
        out["note"] = "too few valid bootstrap samples for an interval"
    return out


def auc_score(y_true, scores):
    """ROC AUC (probability a positive week scores higher than a negative week)."""
    from sklearn.metrics import roc_auc_score
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    ok = np.isfinite(scores)
    y_true, scores = y_true[ok], scores[ok]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def top_decile_mean(values):
    v = np.sort(np.asarray(values, float)[np.isfinite(values)])[::-1]
    if len(v) == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.1 * len(v))))
    return float(v[:k].mean())


def district_level_methods(ranked, events, window, districts):
    """District values for the index and each baseline."""
    people_col = "exposed_pop" if "exposed_pop" in ranked else (
        "buildings_flooded" if "buildings_flooded" in ranked else None)
    ranked = ranked.copy()
    ranked["flooded_area_km2"] = ranked["flooded_frac"].fillna(0) * ranked["area_km2"]
    unit_methods = {"Clearwell index": "score", "B1 flooded area": "flooded_area_km2"}
    if people_col:
        unit_methods[f"B3 {'people' if people_col == 'exposed_pop' else 'buildings'} in flooded area"] = people_col
    rows = {}
    for d in districts:
        g = ranked[ranked["district"] == d]
        row = {}
        for label, col in unit_methods.items():
            row[f"{label} (top-decile mean)"] = top_decile_mean(g[col].values)
            row[f"{label} (mean)"] = float(g[col].mean()) if len(g) else float("nan")
        rows[d] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    w0, w1 = window
    in_window = events[(events["date"] >= w0) & (events["date"] <= w1)]
    if in_window["rainfall"].notna().any():
        rain = in_window.groupby("district")["rainfall"].mean()
        table["B2 rainfall (mean of report rows)"] = rain.reindex(table.index)
    return table


# ---------------------------------------------------------------------------
# Stage validate
# ---------------------------------------------------------------------------
def write_not_run(paths, reason, logger):
    text = (f"# Validation\n\n**VALIDATION NOT RUN**: {reason}\n\n"
            "Nothing was fabricated. Put the IDSP file at data/manual/idsp_weather.csv and rerun "
            "`python run_all.py --stage validate`.\n")
    (paths.results / "validation.md").write_text(text, encoding="utf-8")
    with open(paths.results / "validation.json", "w") as f:
        json.dump({"status": "NOT RUN", "reason": reason}, f, indent=2)
    logger.warning(f"VALIDATION NOT RUN: {reason}")
    return common.result("NOT RUN", f"VALIDATION NOT RUN: {reason}")


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def run_validate(smoke=False, logger=None, **kwargs):
    paths = config.get_paths(smoke)
    idsp_path = paths.manual / config.IDSP_FILE
    if not idsp_path.exists():
        return write_not_run(paths, f"{idsp_path.relative_to(paths.root)} not found", logger)
    aoi = common.load_aoi(paths)
    ranked = pd.read_csv(paths.results / "village_risk_ranked.csv", comment="#")
    features = pd.read_csv(paths.outputs / "village_features.csv")
    by_pass = pd.read_csv(paths.results / "ranking_by_pass.csv")
    gdf, _ = common.get_districts(paths, logger)
    all_districts = sorted(gdf["district"].tolist()) if gdf is not None else sorted(ranked["district"].unique())
    mapped = sorted(ranked["district"].unique().tolist())

    events, info, mapping = load_idsp(idsp_path, all_districts, logger)
    mapping.to_csv(paths.results / "idsp_district_mapping.csv", index=False)
    if not len(events):
        return write_not_run(paths, "no Assam water-borne outbreak rows left after cleaning", logger)
    cov0, cov1 = pd.Timestamp(info["coverage_start"]), pd.Timestamp(info["coverage_end"])
    w0 = pd.Timestamp(config.FLOOD_START)
    w1 = pd.Timestamp(config.FLOOD_END) + pd.Timedelta(weeks=config.OUTCOME_WEEKS_AFTER_FLOOD)
    if str(config.FLOOD_YEAR) not in info.get("rows_per_year", {}):
        return write_not_run(paths, f"the file has no rows in the flood year {config.FLOOD_YEAR} "
                                    f"(coverage {info['coverage_start']} to {info['coverage_end']})", logger)
    if w0 > cov1 or w1 < cov0:
        return write_not_run(paths, f"outcome window {w0.date()} to {w1.date()} lies outside the file "
                                    f"coverage {cov0.date()} to {cov1.date()}", logger)
    warnings = []
    if w1 > cov1 or w0 < cov0:
        warnings.append(f"outcome window truncated to the file coverage ({cov0.date()} to {cov1.date()})")
    w0c, w1c = max(w0, cov0), min(w1, cov1)

    table = district_week_table(events, all_districts, cov0, cov1)
    table.to_csv(paths.outputs / "idsp_district_week.csv", index=False)
    win = table[(table["week_start"] >= monday(w0c)) & (table["week_start"] <= monday(w1c))]
    outcome = win.groupby("district")["outbreaks"].agg(outbreaks="sum",
                                                       outbreak_weeks=lambda s: int((s > 0).sum()))

    # District-level comparison (mapped districts only).
    methods = district_level_methods(ranked, events, (w0c, w1c), mapped)
    methods = methods.join(outcome, how="left").fillna({"outbreaks": 0, "outbreak_weeks": 0})
    methods.to_csv(paths.results / "validation_district_table.csv")
    results = []
    for col in [c for c in methods.columns if c not in ("outbreaks", "outbreak_weeks")]:
        for target in ("outbreaks", "outbreak_weeks"):
            st = spearman_with_uncertainty(methods[col].values, methods[target].values)
            results.append({"design": "district", "method": col, "outcome": target, **st})

    # District-week design (mapped districts, weeks from flood start to end + 4 weeks).
    dw_rows = []
    pass_dates = sorted(aoi["flood_dates"])
    weeks = pd.date_range(monday(w0c), monday(w1c), freq="7D")
    for d in mapped:
        units = by_pass["district"] == d
        feat_d = features[features["district"] == d]
        for wk in weeks:
            seen = [p for p in pass_dates if pd.Timestamp(p) <= wk + pd.Timedelta(days=6)]
            if not seen:
                continue  # no satellite information yet: no exposure value
            tag = seen[-1].replace("-", "")
            future = table[(table["district"] == d) & (table["week_start"] > wk)
                           & (table["week_start"] <= wk + pd.Timedelta(weeks=3))]
            if future.empty or (wk + pd.Timedelta(weeks=3)) > cov1:
                continue  # outcome weeks outside the file coverage
            row = {"district": d, "week_start": wk.date().isoformat(), "last_pass": seen[-1],
                   "index_top_decile": top_decile_mean(by_pass.loc[units, f"score_{tag}"].values),
                   "B1_flooded_km2": float((feat_d[f"cum_flooded_frac_{tag}"].fillna(0) * feat_d["area_km2"]).sum()),
                   "outbreak_next_3_weeks": int(future["outbreaks"].sum() > 0)}
            if f"exposed_pop_{tag}" in feat_d:
                row["B3_exposed_pop"] = float(feat_d[f"exposed_pop_{tag}"].sum())
            dw_rows.append(row)
    dw = pd.DataFrame(dw_rows)
    dw.to_csv(paths.results / "validation_district_week.csv", index=False)
    dw_results = []
    if len(dw):
        for col in [c for c in dw.columns if c.startswith(("index", "B1", "B3"))]:
            dw_results.append({"design": "district-week", "method": col, "n_rows": len(dw),
                               "n_positive": int(dw["outbreak_next_3_weeks"].sum()),
                               "auc": auc_score(dw["outbreak_next_3_weeks"], dw[col]),
                               "spearman": spearman(dw[col].values, dw["outbreak_next_3_weeks"].values)})

    learned = {"status": "NOT RUN", "reason": "config.RUN_LEARNED_MODEL is False"}
    if config.RUN_LEARNED_MODEL:
        learned = learned_model(dw, logger)

    n = len(mapped)
    limitations = [
        (f"Only {n} district(s) were flood-mapped, so the district-level Spearman correlations have "
        f"n = {n}. With n this small the statistical power is very low and the results are "
        "indicative only; a permutation p-value cannot be small with so few districts."),
        ("IDSP lists reported outbreaks only. Weeks without a report were set to zero, which assumes "
        "complete reporting inside the file's date coverage "
        f"({info['coverage_start']} to {info['coverage_end']})."),
        ("Outbreaks depend on many things besides flooding (sanitation, reporting effort, "
        "population), and the outcome window is fixed in config.py."),
        ("Weather in the file is a single-point value per report row, so rainfall exists only for "
        "districts and weeks that had an outbreak (the B2 baseline is biased towards outbreaks)."),
        (f"{len(info.get('unmatched_districts', []))} district spelling(s) in the file could not be "
        "matched (see results/idsp_district_mapping.csv)."),
        "The index weights were fixed in config.py before looking at outbreak data.",
    ]
    report = {"status": "DONE", "smoke": smoke, "idsp": info, "mapped_districts": mapped,
              "outcome_window": [w0c.date().isoformat(), w1c.date().isoformat()],
              "district_results": results, "district_week_results": dw_results,
              "learned_model": learned, "warnings": warnings, "limitations": limitations,
              "generated_at": datetime.now().isoformat()}
    with open(paths.results / "validation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    lines = ["# Validation against IDSP outbreak reports", "",
             "Generated by validate_risk.py." + (" SMOKE RUN." if smoke else ""), "",
             (f"- IDSP rows after cleaning (Assam, water-borne): {info['events_after_cleaning']} "
             f"(duplicates removed: {info['exact_duplicates_removed']} exact, "
             f"{info['key_duplicates_removed']} same district/date/disease/cases)"),
             f"- file coverage: {info['coverage_start']} to {info['coverage_end']}",
             f"- outcome window: {w0c.date()} to {w1c.date()}",
             f"- mapped districts (n = {n}): {', '.join(mapped)}", "",
             "## District level (Spearman vs outbreak counts in the window)", "",
             "| Method | Outcome | n | Spearman rho | 95% bootstrap interval | p-value | permutations |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['method']} | {r['outcome']} | {r['n']} | {fmt(r['rho'])} | "
                     f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {fmt(r['p_value'])} | "
                     f"{r['permutation'] or r.get('note', '')} |")
    lines += ["", "## District-week design (mapped districts; outbreak in the next 3 weeks)", "",
              "| Method | rows | positive rows | AUC | Spearman |", "|---|---|---|---|---|"]
    for r in dw_results:
        lines.append(f"| {r['method']} | {r['n_rows']} | {r['n_positive']} | {fmt(r['auc'])} | "
                     f"{fmt(r['spearman'])} |")
    if not dw_results:
        lines.append("| (no rows inside the file coverage) | | | | |")
    lines += ["", f"Learned model: {learned['status']} ({learned.get('reason', '')})"
              + (f", leave-one-district-out AUC {fmt(learned.get('auc'))}" if learned.get("auc") is not None else ""),
              "", "## Limitations", ""] + [f"- {x}" for x in limitations]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    (paths.results / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if n < 3:
        warnings.append(f"only {n} mapped district(s): district-level correlation not computable")
    return common.result("OK", "", warnings, {"n_districts": n})


def learned_model(dw, logger):
    """Optional: logistic regression with leave-one-district-out CV (only with enough data)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if len(dw) < config.LEARNED_MIN_ROWS or dw.get("outbreak_next_3_weeks", pd.Series(dtype=int)).sum() < config.LEARNED_MIN_POSITIVES:
        return {"status": "NOT RUN", "reason": f"needs >= {config.LEARNED_MIN_ROWS} rows and >= "
                                               f"{config.LEARNED_MIN_POSITIVES} positives; have {len(dw)} rows"}
    feats = [c for c in dw.columns if c.startswith(("index", "B1", "B3"))]
    preds = pd.Series(np.nan, index=dw.index)
    for d in dw["district"].unique():
        train, test = dw["district"] != d, dw["district"] == d
        if dw.loc[train, "outbreak_next_3_weeks"].nunique() < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(dw.loc[train, feats].fillna(0), dw.loc[train, "outbreak_next_3_weeks"])
        preds[test] = model.predict_proba(dw.loc[test, feats].fillna(0))[:, 1]
    ok = preds.notna()
    if dw.loc[ok, "outbreak_next_3_weeks"].nunique() < 2:
        return {"status": "NOT RUN", "reason": "cross-validated predictions have one class only"}
    auc = float(roc_auc_score(dw.loc[ok, "outbreak_next_3_weeks"], preds[ok]))
    logger.info(f"Learned model leave-one-district-out AUC: {auc:.3f}")
    return {"status": "DONE", "reason": "logistic regression, leave-one-district-out", "auc": auc,
            "features": feats}


if __name__ == "__main__":
    import run_all
    sys.exit(run_all.main(default_stages=STAGES_IN_THIS_FILE))
