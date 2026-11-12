"""
risk_score.py - MODEL 2: transparent flood-contamination risk index per village proxy.

There are no village-level outbreak labels, so NO supervised model is trained
here. The index is a weighted sum of percentile ranks with weights fixed in
advance in config.RISK_WEIGHTS (never tuned on outcomes):

    exposure        flooded_frac
    duration        max_duration_days
    exposed_people  exposed_pop (or buildings_flooded if no population data)
    well_reliance   pct_shallow_well_handpump (district level, if supplied)
    low_elevation   1 - elev_pctile

Each component is turned into a percentile rank (0 to 1) across all units;
ties share the lowest rank, so units with zero flooding get 0. Unavailable
components are dropped and the remaining weights renormalized. If a unit is
missing one value, its weights are renormalized over the values it has.
Score = weighted sum x 100.

Per-pass ranking: for each pass date D only information up to D is used
(cumulative extent, exposed people so far, duration so far; duration is left
out on the first pass because one pass carries no duration information).

Outputs: results/village_risk_ranked.csv, results/top20_map.png,
results/ranking_by_pass.csv, results/ranking_evolution.csv/.png,
results/risk_meta.json

Run alone:  python risk_score.py [--smoke]
"""

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import common
import config

STAGES_IN_THIS_FILE = ["risk"]


def datecol(prefix, day):
    return f"{prefix}_{day.replace('-', '')}"


def pct_rank(values):
    """Percentile rank 0..1. Ties share the lowest rank; NaN stays NaN."""
    s = pd.Series(values, dtype=float)
    n = int(s.notna().sum())
    if n == 0:
        return s
    if n == 1:
        return s.where(s.isna(), 0.0)
    return (s.rank(method="min") - 1) / (n - 1)


def available_components(df, meta):
    """Which components can be computed, and the raw column behind each."""
    comps = {"exposure": "flooded_frac", "duration": "max_duration_days"}
    feats = meta.get("features", {})
    if feats.get("population", {}).get("available") and "exposed_pop" in df:
        comps["exposed_people"] = "exposed_pop"
    elif feats.get("buildings", {}).get("available") and "buildings_flooded" in df:
        comps["exposed_people"] = "buildings_flooded"
    if feats.get("well_reliance", {}).get("available") and "pct_shallow_well_handpump" in df:
        comps["well_reliance"] = "pct_shallow_well_handpump"
    if feats.get("elevation", {}).get("available") and "elev_pctile" in df:
        comps["low_elevation"] = "elev_pctile"
    if len(meta.get("flood_dates", [])) < 2:
        comps.pop("duration")  # one pass carries no duration information
    # Keep only components whose column exists and has at least one value.
    return {k: v for k, v in comps.items() if v in df and df[v].notna().any()}


def component_values(df, comp, column):
    """Raw values for a component (low_elevation is 1 - elev_pctile)."""
    vals = df[column].astype(float)
    return 1.0 - vals if comp == "low_elevation" else vals


def score_units(raw, weights):
    """raw: DataFrame with one column per component (raw values).

    Returns (DataFrame of percentile components c_<name>, score Series 0..100,
    renormalized weights used).
    """
    comps = pd.DataFrame(index=raw.index)
    for comp in raw.columns:
        comps[f"c_{comp}"] = pct_rank(raw[comp]).values
    w = {k: weights[k] for k in raw.columns}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    num = sum(comps[f"c_{k}"].fillna(0) * v for k, v in w.items())
    den = sum(comps[f"c_{k}"].notna() * v for k, v in w.items())
    score = 100 * num / den.replace(0, np.nan)
    return comps, score, w


def rank_order(df, score_col="score"):
    """Rank 1 = highest risk. Ties broken by flooded fraction, then unit id (stable)."""
    keys = [score_col] + [c for c in ("flooded_frac",) if c in df] + ["unit_id"]
    asc = [False] * (len(keys) - 1) + [True]
    order = df.sort_values(keys, ascending=asc, na_position="last").index
    ranks = pd.Series(np.arange(1, len(df) + 1), index=order)
    return ranks.reindex(df.index)


def full_ranking(df, meta, weights=None):
    """Final ranking using all passes. Returns (ranked DataFrame, components used, weights)."""
    weights = weights or config.RISK_WEIGHTS
    comps = available_components(df, meta)
    raw = pd.DataFrame({c: component_values(df, c, col) for c, col in comps.items()})
    parts, score, w = score_units(raw, weights)
    out = df.copy()
    for c in parts:
        out[c] = parts[c]
    out["score"] = score
    out["rank"] = rank_order(out)
    return out.sort_values("rank"), comps, w


def per_pass_rankings(df, meta, weights=None):
    """Ranking after each pass using only information up to that pass.

    Returns (wide DataFrame unit_id + rank/score per pass, list of per-pass info).
    """
    weights = weights or config.RISK_WEIGHTS
    final_comps = available_components(df, meta)
    dates = sorted(meta["flood_dates"])
    wide = pd.DataFrame({"unit_id": df["unit_id"].values}, index=df.index)
    info = []
    for k, day in enumerate(dates):
        raw = {}
        for comp, col in final_comps.items():
            if comp == "exposure":
                src = datecol("cum_flooded_frac", day)
            elif comp == "duration":
                if k == 0:
                    continue  # a single pass says nothing about duration
                src = datecol("max_duration_so_far", day)
            elif comp == "exposed_people":
                src = datecol(col, day)  # exposed_pop_D or buildings_flooded_D
            else:
                src = col  # static: known before any pass
            if src not in df:
                continue
            raw[comp] = component_values(df, comp, src)
        raw = pd.DataFrame(raw)
        _, score, w = score_units(raw, weights)
        tmp = pd.DataFrame({"unit_id": df["unit_id"], "score": score,
                            "flooded_frac": df[datecol("cum_flooded_frac", day)]})
        tag = day.replace("-", "")
        wide[f"score_{tag}"] = score.values
        wide[f"rank_{tag}"] = rank_order(tmp).values
        info.append({"pass_index": k + 1, "date": day, "components": list(raw.columns),
                     "weights": w})
    return wide, info


def evolution_table(wide, info, final_rank, top_k):
    """How many of the final top-K were already in the top-K after each pass."""
    k = min(top_k, len(wide))
    final_top = set(wide.loc[final_rank <= k, "unit_id"])
    rows = []
    for item in info:
        tag = item["date"].replace("-", "")
        top = set(wide.loc[wide[f"rank_{tag}"] <= k, "unit_id"])
        rho = pd.Series(wide[f"rank_{tag}"].values).corr(pd.Series(final_rank.values), method="spearman")
        rows.append({"pass_index": item["pass_index"], "date": item["date"],
                     "components": "+".join(item["components"]), "top_k": k,
                     "overlap_with_final_top_k": len(top & final_top),
                     "overlap_fraction": len(top & final_top) / k if k else np.nan,
                     "spearman_rank_vs_final": rho})
    return pd.DataFrame(rows)


def plot_evolution(evo, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(evo["pass_index"], evo["overlap_fraction"], marker="o", label="share of final top-K")
    ax.plot(evo["pass_index"], evo["spearman_rank_vs_final"], marker="s", label="Spearman vs final ranking")
    for _, r in evo.iterrows():
        ax.annotate(r["date"], (r["pass_index"], r["overlap_fraction"]), fontsize=7,
                    textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xlabel("satellite pass number")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(evo["pass_index"])
    k = int(evo["top_k"].iloc[0]) if len(evo) else 0
    ax.set_title(f"How the ranking settles as passes arrive (K = {k})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def plot_top_map(paths, aoi, ranked, out_png):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    units = gpd.read_file(paths.outputs / "villages.gpkg")
    units = units.merge(ranked[["unit_id", "score", "rank"]], on="unit_id", how="left")
    districts = common.districts_for_aoi(paths, aoi)
    crs, transform, width, height = common.aoi_grid(aoi)
    fig, ax = plt.subplots(figsize=(10, 9))
    with rasterio.open(paths.outputs / "flood_extent.tif") as src:
        f = max(1, int(np.ceil(max(width, height) / 1500)))
        ext = common.block_reduce((src.read(1) == 1).astype(np.uint8), f, "max")
    xmin, ymin, xmax, ymax = common.grid_bounds(transform, ext.shape[1] * f, ext.shape[0] * f)
    ax.imshow(np.where(ext == 1, 1.0, np.nan), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6,
              extent=[xmin, xmax, ymin, ymax], interpolation="nearest")
    units.plot(ax=ax, column="score", cmap="OrRd", alpha=0.55, linewidth=0.1, edgecolor="grey",
               legend=True, legend_kwds={"label": "risk score (0-100)", "shrink": 0.6})
    k = config.TOP_K_MAP
    top = units[units["rank"] <= k]
    top.boundary.plot(ax=ax, color="black", linewidth=1.2)
    for row in top.itertuples():
        p = row.geometry.representative_point()
        ax.annotate(str(int(row.rank)), (p.x, p.y), fontsize=8, ha="center", fontweight="bold")
    districts.boundary.plot(ax=ax, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_title(f"Top {k} village proxies by Clearwell risk score (blue = flooded)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def run_risk(smoke=False, logger=None, **kwargs):
    paths = config.get_paths(smoke)
    aoi = common.load_aoi(paths)
    df = pd.read_csv(paths.outputs / "village_features.csv", keep_default_na=True)
    with open(paths.outputs / "village_features_meta.json") as f:
        meta = json.load(f)
    warnings = []
    ranked, comps, w = full_ranking(df, meta)
    missing = [c for c in config.RISK_WEIGHTS if c not in comps]
    if missing:
        warnings.append(f"components not available and dropped (weights renormalized): {missing}")
    logger.info(f"Components used: {comps}; weights after renormalizing: "
                f"{ {k: round(v, 3) for k, v in w.items()} }")

    cols = (["rank", "unit_id", "name", "district", "score"] + [f"c_{c}" for c in comps]
            + ["flooded_frac", "max_duration_days", "mean_duration_days", "first_flood_date"]
            + [c for c in ("pop", "exposed_pop", "buildings", "buildings_flooded", "elev_mean_m",
                           "pct_shallow_well_handpump", "area_km2", "lon", "lat") if c in ranked])
    header = [f"# Clearwell risk ranking of VILLAGE PROXIES (generated {datetime.now().isoformat(timespec='seconds')})",
              f"# components used: {', '.join(f'{k}={v}' for k, v in comps.items())}",
              f"# weights (fixed in config.py, renormalized): {', '.join(f'{k}={v:.3f}' for k, v in w.items())}",
              f"# components dropped: {', '.join(missing) if missing else 'none'}"]
    if smoke:
        header.append("# SMOKE RUN: numbers are not meaningful")
    out_csv = paths.results / "village_risk_ranked.csv"
    with open(out_csv, "w", newline="") as f:
        f.write("\n".join(header) + "\n")
        ranked[cols].to_csv(f, index=False, float_format="%.4f")
    logger.info(f"Wrote {out_csv} ({len(ranked)} village proxies)")
    for r in ranked.head(10).itertuples():
        logger.info(f"  #{r.rank:<3} {r.unit_id:<28} {str(r.name)[:24]:<24} {r.district:<12} score {r.score:.1f}")

    try:
        plot_top_map(paths, aoi, ranked, paths.results / "top20_map.png")
    except Exception as err:  # noqa: BLE001
        logger.warning(f"top-20 map failed: {common.format_exception(err)[1]}")
        warnings.append(f"top-20 map failed: {err}")

    wide, info = per_pass_rankings(df, meta)
    final_rank = ranked.set_index("unit_id").loc[wide["unit_id"], "rank"]
    final_rank.index = wide.index
    wide.insert(1, "final_rank", final_rank.values)
    wide.insert(1, "district", df["district"].values)
    wide.sort_values("final_rank").to_csv(paths.results / "ranking_by_pass.csv", index=False,
                                          float_format="%.3f")
    evo = evolution_table(wide, info, final_rank, config.TOP_K_EVOLUTION)
    evo.to_csv(paths.results / "ranking_evolution.csv", index=False, float_format="%.4f")
    plot_evolution(evo, paths.results / "ranking_evolution.png")
    last_tag = info[-1]["date"].replace("-", "") if info else None
    if last_tag and not (wide[f"rank_{last_tag}"].values == final_rank.values).all():
        warnings.append("ranking after the last pass differs from the final ranking (check inputs)")
    for r in evo.itertuples():
        logger.info(f"pass {r.pass_index} {r.date}: {r.overlap_with_final_top_k}/{r.top_k} of the final "
                    f"top-{r.top_k} already in the top-{r.top_k}; Spearman vs final {r.spearman_rank_vs_final:.3f}")

    with open(paths.results / "risk_meta.json", "w") as f:
        json.dump({"components": comps, "weights_used": w, "weights_config": config.RISK_WEIGHTS,
                   "dropped": missing, "n_units": len(ranked), "per_pass": info,
                   "unit_source": meta.get("unit_source"), "smoke": smoke,
                   "generated_at": datetime.now().isoformat()}, f, indent=2)
    return common.result("OK", "", warnings, {"n_units": len(ranked)})


if __name__ == "__main__":
    import run_all
    sys.exit(run_all.main(default_stages=STAGES_IN_THIS_FILE))
