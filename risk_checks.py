"""
Checks on the Model 2 risk ranking. Read-only on existing inputs; writes only its own reports.

  1. Case check: IDSP outbreak villages read from case_points.csv
     (name, district, lat, lon, cases, onset, coord_source). Add a row, or fill in
     lat/lon, to check another village; no code change needed. For each row:
       - exact, then fuzzy, name match against the 110 OSM units;
       - if lat/lon are given: a unit built the same way as build_villages.py
         (a 1.5 km circle around the point, clipped to its district), with
         flooded fraction, duration and exposed people from the same flood rasters
         and WorldPop clip, and a cross-check of the flooded fraction against
         outputs/flood_extent.tif; where that flooded fraction would rank among
         the 110 units; the risk score and rank of that unit if it were added as a
         111th unit (default weights and flooded-fraction-only); the nearest OSM
         unit and its distance.
  2. Sensitivity of the ranking to the weights    -> results/risk_sensitivity.md
  3. District comparison, index vs simple baselines
  4. Top-15 table                                   -> results/monday_table.md
  Items 1, 3, the v1 comparison and the summary     -> results/risk_checks.md

The default weights in config.RISK_WEIGHTS are not changed; alternative weight
sets are passed to risk_score.full_ranking as arguments.

Run: python risk_checks.py
"""
import json
import math
import re
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rapidfuzz import fuzz, process
from shapely.geometry import Point

import build_villages as bv
import common
import config
import risk_score

CASE_FILE = "case_points.csv"
TOP_K = 20
# Written after reading the numbers of the run; recheck against the tables if inputs change.
SUMMARY = [
    "1. Well reliance now adds about 1 point to Morigaon units instead of about 13 (v1 gave the 16 tied Morigaon "
    "units 94/109 = 0.862 and Nagaon 0), so Morigaon holds 5 of the top 10 "
    "instead of 10; Kakati Gaon and Basundhari (Nagaon) are now ranks 1-2 and Bhaktagaon is 3.",
    "2. Most of the ordering did not change: Spearman 0.990 with the v1 ranking and 18 of the top 20 are the same "
    "units; flooded fraction still drives it.",
    "3. With the district gap shrunk to 0.07, well reliance barely matters any more: the index is almost identical "
    "to its no-well-reliance version (Spearman 0.9997, same top 20).",
    "4. Morigaon still scores higher on average (74 vs 46) because it flooded more, as the flood-only and "
    "population-only baselines also show; top-decile means are close (84 vs 80). Two districts, no test.",
    "5. Goriabori, scored as a unit around its point, is 15.5% flooded and ranks about 25 of 111 (13 of 17 in "
    "Morigaon), upper-middle, not top; flood-only gives rank 24, the same answer. No OSM unit lies within 4 km.",
    "6. That is one case with approximate coordinates and an outbreak 20 days after the last radar pass; it says "
    "nothing about predictive skill. Sibir is unchecked until it has coordinates.",
]


def load():
    paths = config.get_paths(False)
    df = pd.read_csv(paths.outputs / "village_features.csv")
    with open(paths.outputs / "village_features_meta.json") as f:
        meta = json.load(f)
    published = pd.read_csv(paths.results / "village_risk_ranked.csv", comment="#")
    return paths, df, meta, published


def rank_with(df, meta, weights):
    ranked, _, _ = risk_score.full_ranking(df, meta, weights=weights)
    return ranked.set_index("unit_id")


def only(comp):
    return {**{k: 0.0 for k in config.RISK_WEIGHTS}, comp: 1.0}


def scenarios():
    base = dict(config.RISK_WEIGHTS)
    return {
        "default": base,
        "(a) no well reliance": {**base, "well_reliance": 0.0},
        "(b) equal weights": {k: 1.0 for k in base},
        "(c) no duration": {**base, "duration": 0.0},
        "(d) flooded fraction only": only("exposure"),
    }


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def label(name, uid):
    return str(name) if isinstance(name, str) and name.strip() else f"(unnamed, {uid})"


# ---------------------------------------------------------------- 1. cases
def name_match(target, ranked):
    names = ranked["name"].dropna()
    names = names[names.astype(str).str.strip() != ""]
    exact = names[names.map(norm) == norm(target)]
    if len(exact):
        return "exact", exact.index[0], 100.0, []
    choices = {uid: norm(n) for uid, n in names.items()}
    top = process.extract(norm(target), choices, scorer=fuzz.WRatio, limit=3)
    cands = [(ranked.loc[uid, "name"], round(score, 1)) for _, score, uid in top]
    if top and top[0][1] >= config.FUZZY_MATCH_MIN_SCORE:
        return "fuzzy", top[0][2], top[0][1], cands
    return None, None, None, cands


def worldpop_clip(paths, aoi):
    """The cached WorldPop clip build_villages.py used (never downloads)."""
    manual = paths.manual / "worldpop.tif"
    if manual.exists():
        return manual
    clip = paths.cache / "worldpop" / f"worldpop_aoi_{bv.bbox_key(aoi['bbox_lonlat'])}.tif"
    if not clip.exists():
        raise RuntimeError(f"WorldPop clip {clip} not found; run stage villages first")
    return clip


def point_unit_features(paths, aoi, geom, district, df):
    """Features for one unit geometry, computed exactly like build_villages.run_villages."""
    crs, transform, width, height = common.aoi_grid(aoi)
    unit = gpd.GeoDataFrame({"unit_id": ["point"]}, geometry=[geom], crs=crs)
    unit_idx = bv.unit_index_raster(unit, transform, width, height)
    x, y, pop = bv.population_points(worldpop_clip(paths, aoi), aoi)
    r, c, inside = bv.points_to_pixels(x, y, transform, width, height)
    pop_pixels = (r[inside], c[inside], pop[inside], unit_idx[r[inside], c[inside]])
    feats = bv.flood_features(paths, aoi, unit_idx, 1, pop_pixels, None).iloc[0].to_dict()
    feats["pop"] = float(pop_pixels[2][pop_pixels[3] == 1].sum())
    with rasterio.open(paths.outputs / "dem_on_grid.tif") as src:
        dem = src.read(1).astype(np.float32)
    ok = np.isfinite(dem) & (unit_idx == 1)
    feats["elev_mean_m"] = float(dem[ok].mean()) if ok.any() else np.nan
    del dem
    with rasterio.open(paths.outputs / "flood_extent.tif") as src:
        ext = src.read(1)
    inside_u = unit_idx == 1
    obs = inside_u & (ext != 255)
    feats["extent_check_frac"] = float((inside_u & (ext == 1)).sum() / max(obs.sum(), 1))
    feats["observed_share"] = float(obs.sum() / max(inside_u.sum(), 1))
    feats["district"] = district
    feats["pct_shallow_well_handpump"] = float(df.loc[df["district"] == district,
                                                      "pct_shallow_well_handpump"].iloc[0])
    feats["area_km2"] = geom.area / 1e6
    return feats


def rank_with_extra(df, meta, extra, weights):
    """Add the point unit as a 111th unit (elev_pctile recomputed within district) and rank."""
    row = {c: extra.get(c, np.nan) for c in df.columns}
    row.update(unit_id="__case_point__", name="(case point)", district=extra["district"])
    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df2["elev_pctile"] = df2.groupby("district")["elev_mean_m"].rank(pct=True)
    r = rank_with(df2, meta, weights)
    p = r.loc["__case_point__"]
    in_d = r[r["district"] == extra["district"]].sort_values("rank")
    return {"rank": int(p["rank"]), "n": len(r), "score": float(p["score"]),
            "district_rank": int(list(in_d.index).index("__case_point__")) + 1, "district_n": len(in_d)}


def case_check(paths, aoi, df, meta, ranked, units, districts):
    cases = pd.read_csv(paths.root / CASE_FILE, dtype={"lat": float, "lon": float})
    n = len(ranked)
    out = []
    for case in cases.to_dict("records"):
        row = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in case.items()}
        how, uid, score, cands = name_match(case["name"], ranked)
        row["name_match"] = f"{how} match to {ranked.loc[uid, 'name']} (score {score:.0f})" if how else \
            "no name match (" + ", ".join(f"{a} {b}" for a, b in cands) + f"; threshold {config.FUZZY_MATCH_MIN_SCORE})"
        if row["lat"] is None or row["lon"] is None:
            row["status"] = f"no coordinates yet: add lat/lon to {CASE_FILE}"
            out.append(row)
            continue
        pt = gpd.GeoSeries([Point(row["lon"], row["lat"])], crs=4326).to_crs(units.crs)[0]
        hit = districts[districts.contains(pt)]
        if hit.empty:
            row["status"] = "point is outside the study districts"
            out.append(row)
            continue
        dname = hit.iloc[0]["district"]
        row["point_district"] = dname
        geom = pt.buffer(config.VILLAGE_MAX_RADIUS_M).intersection(hit.iloc[0].geometry)
        f = point_unit_features(paths, aoi, geom, dname, df)
        row["feat"] = f
        row["flood_rank_among_110"] = int((ranked["flooded_frac"] > f["flooded_frac"]).sum()) + 1
        row["n_units"] = n
        row["as_unit_default"] = rank_with_extra(df, meta, f, config.RISK_WEIGHTS)
        row["as_unit_floodonly"] = rank_with_extra(df, meta, f, only("exposure"))
        d = units.geometry.distance(pt)
        j = d.idxmin()
        nu = units.loc[j]
        row["nearest"] = {"unit_id": nu["unit_id"], "name": label(nu["name"], nu["unit_id"]),
                          "district": nu["district"], "edge_m": float(d[j]),
                          "centre_m": float(nu.geometry.centroid.distance(pt)),
                          "rank": int(ranked.loc[nu["unit_id"], "rank"]),
                          "score": float(ranked.loc[nu["unit_id"], "score"]),
                          "flooded_frac": float(ranked.loc[nu["unit_id"], "flooded_frac"])}
        row["status"] = "done"
        out.append(row)
    return out


# ---------------------------------------------------------- 2. sensitivity
def sensitivity(df, meta):
    sc = scenarios()
    ranks = {k: rank_with(df, meta, w) for k, w in sc.items()}
    base = ranks["default"]
    out = [compare(base, r, k, sc[k]) for k, r in ranks.items()]
    return out, ranks


def compare(base, r, name, weights=None):
    r = r.reindex(base.index)
    base_top = set(base.index[base["rank"] <= TOP_K])
    top10 = r[r["rank"] <= 10]
    res = {"scenario": name,
           "spearman": float(base["rank"].corr(r["rank"], method="spearman")),
           "top20_overlap": len(base_top & set(r.index[r["rank"] <= TOP_K])),
           "top10_morigaon": int((top10["district"] == "Morigaon").sum())}
    if weights:
        res["weights"] = {c: round(v / sum(weights.values()), 3) for c, v in weights.items() if v}
    return res


# ------------------------------------------------------- 3. district means
def district_table(df, meta):
    sets = {"Full index (default weights)": config.RISK_WEIGHTS,
            "Flooded area only (flooded_frac)": only("exposure"),
            "Exposed population only (exposed_pop)": only("exposed_people")}
    rows = []
    for lab, w in sets.items():
        r = rank_with(df, meta, w)
        for d, g in r.groupby("district"):
            k = max(1, math.ceil(0.1 * len(g)))
            rows.append({"score": lab, "district": d, "n": len(g), "mean": g["score"].mean(),
                         "top_decile_k": k, "top_decile_mean": g["score"].nlargest(k).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ write
def well_note():
    if config.WELL_RELIANCE_SCORING == "absolute":
        return ("- well_reliance is a district-level Census value (Morigaon 92.1%, Nagaon 85.4%), used on an "
                "absolute scale (pct / 100, config.WELL_RELIANCE_SCORING = \"absolute\"): 0.921 vs 0.854, so at "
                "weight 0.15 it adds about 1 score point in Morigaon's favour. Before 2026-11-15 it was "
                "percentile-ranked (1.0 vs 0.0); see results/risk_sensitivity_v1.md.")
    return ("- well_reliance is a district-level Census value, percentile-ranked "
            "(config.WELL_RELIANCE_SCORING = \"percentile\"): 1.0 for every Morigaon unit, 0.0 for every Nagaon "
            "unit, a flat district bonus.")


def write_sensitivity(paths, sens, ranks):
    lines = ["# Risk ranking sensitivity to the weights", "",
             f"Generated by risk_checks.py on {datetime.now():%Y-%m-%d %H:%M}. 110 village proxies "
             "(16 Morigaon, 94 Nagaon). Same components and scoring as risk_score.py "
             f"(well reliance scored \"{config.WELL_RELIANCE_SCORING}\"); only the weights change.", "",
             "| Scenario | Weights (renormalized) | Spearman vs default | Top-20 overlap with default | "
             "Morigaon units in top 10 |", "|---|---|---|---|---|"]
    for s in sens:
        w = ", ".join(f"{k} {v:.2f}" for k, v in s["weights"].items())
        lines.append(f"| {s['scenario']} | {w} | {s['spearman']:.3f} | {s['top20_overlap']}/20 | "
                     f"{s['top10_morigaon']}/10 |")
    lines += ["", "## Top 10 under each scenario", ""]
    for k, r in ranks.items():
        top = r.sort_values("rank").head(10)
        lines.append(f"- **{k}**: " + "; ".join(f"{int(x['rank'])}. {label(x['name'], uid)} ({x['district'][0]})"
                                                for uid, x in top.iterrows()))
    lines += ["", "## Notes", "", well_note(),
              "- duration has only three values (0, 18, 36 days) because there are two flood passes and the "
              "resolution is one revisit interval; many units tie.",
              "- One flood season, one AOI. These numbers say how much the ordering depends on the weight "
              "choice, not whether any ordering is right."]
    (paths.results / "risk_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_monday(paths, base, no_well):
    top = base.sort_values("rank").head(15)
    lines = ["# Top 15 village proxies, Clearwell risk index", "",
             f"Generated by risk_checks.py on {datetime.now():%Y-%m-%d %H:%M} from the default ranking "
             f"(same as results/village_risk_ranked.csv; well reliance scored \"{config.WELL_RELIANCE_SCORING}\"). "
             "Units are OSM-based village proxies (Voronoi cells clipped to 1.5 km), not census villages.", "",
             "| Rank | Village proxy | District | Score (0-100) | Area flooded | People in flooded area (est.) | "
             "Flood duration (est.) | Rank without well reliance |", "|---|---|---|---|---|---|---|---|"]
    for uid, x in top.iterrows():
        lines.append(f"| {int(x['rank'])} | {label(x['name'], uid)} | {x['district']} | {x['score']:.1f} | "
                     f"{100 * x['flooded_frac']:.0f}% | ~{int(round(x['exposed_pop'], -1)):,} | "
                     f"{int(x['max_duration_days'])} days | {int(no_well.loc[uid, 'rank'])} |")
    lines += ["", "Notes:", "",
              "- People in flooded area: WorldPop gridded population inside flooded pixels, rounded to 10. "
              "An estimate, not a count.",
              "- Duration: longest run of consecutive flooded passes plus one revisit interval, so it can only "
              "be 0, 18 or 36 days here (two flood passes). Coarse.",
              "- Rank without well reliance: the same index with the district-level well-reliance component "
              "removed (see results/risk_sensitivity.md).",
              "- Previous version (well reliance percentile-ranked): results/monday_table_v1.md."]
    (paths.results / "monday_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def case_lines(cases):
    lines = ["## 1. Case check: IDSP outbreak villages", "",
             "**An outbreak at a known location is a case check, not a statistical test.** One or two "
             "villages cannot show that the index predicts outbreaks, and a flooded-area-only ranking is "
             "reported next to each result because it gives the same answer.", "",
             f"Points come from `{CASE_FILE}`. A unit is built around each point the same way as the OSM units "
             f"(a {config.VILLAGE_MAX_RADIUS_M} m circle clipped to the district; unlike the OSM units it is not "
             "cut by neighbouring Voronoi cells). Its rank is what it would get as a 111th unit.", ""]
    for c in cases:
        lines += [f"### {c['name']} ({c['district']}; {c['cases']} cases from {c['onset']})", "",
                  f"- Name: {c['name_match']}."]
        if c["status"] != "done":
            lines += [f"- **Not checked:** {c['status']}.", ""]
            continue
        f, a, b, nu = c["feat"], c["as_unit_default"], c["as_unit_floodonly"], c["nearest"]
        lines += [
            f"- Point {c['lat']:.6f}, {c['lon']:.6f} ({c['coord_source']}), inside {c['point_district']}.",
            f"- Within {config.VILLAGE_MAX_RADIUS_M} m: flooded fraction **{100 * f['flooded_frac']:.1f}%** of "
            f"observed land (cross-check from flood_extent.tif: {100 * f['extent_check_frac']:.1f}%; "
            f"{100 * f['observed_share']:.0f}% of the circle observed); estimated people in the flooded area "
            f"**~{int(round(f['exposed_pop'], -1)):,}** of ~{int(round(f['pop'], -1)):,} (WorldPop); "
            f"duration {int(f['max_duration_days'])} days; first flooded {_ffd(f)}.",
            f"- That flooded fraction would rank **{c['flood_rank_among_110']} of {c['n_units'] + 1}** "
            f"(above {c['n_units'] - c['flood_rank_among_110'] + 1} of the 110 units).",
            f"- As a 111th unit: risk score **{a['score']:.1f}**, rank **{a['rank']} / {a['n']}** "
            f"({a['district_rank']} / {a['district_n']} in {c['point_district']}). Flooded-fraction-only "
            f"ranking: rank {b['rank']} / {b['n']} ({b['district_rank']} / {b['district_n']} in "
            f"{c['point_district']}).",
            f"- Nearest OSM unit: {nu['name']} ({nu['unit_id']}, {nu['district']}), "
            + ("point is inside it" if nu["edge_m"] == 0 else f"{nu['edge_m']:,.0f} m from its edge")
            + f", {nu['centre_m']:,.0f} m from its centre; rank {nu['rank']}, score {nu['score']:.1f}, "
              f"flooded {100 * nu['flooded_frac']:.0f}%.", ""]
    return lines


def _ffd(f):
    for k in sorted(k for k in f if k.startswith("flooded_frac_")):
        if f[k] >= 0.01:
            d = k.split("_")[-1]
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return "never (>= 1%)"


def write_checks(paths, cases, dist, sens, v1cmp):
    lines = ["# Risk ranking checks", "", f"Generated by risk_checks.py on {datetime.now():%Y-%m-%d %H:%M}. "
             f"Well reliance scored \"{config.WELL_RELIANCE_SCORING}\". Previous version: "
             "results/risk_checks_v1.md.", ""]
    lines += case_lines(cases)
    lines += ["## 2. Sensitivity", "", "See results/risk_sensitivity.md. Summary:", "",
              "| Scenario | Spearman | Top-20 overlap | Morigaon in top 10 |", "|---|---|---|---|"]
    lines += [f"| {s['scenario']} | {s['spearman']:.3f} | {s['top20_overlap']}/20 | {s['top10_morigaon']}/10 |"
              for s in sens]
    if v1cmp:
        lines += ["", "Current default vs the v1 default (well reliance percentile-ranked, "
                  "results/village_risk_ranked_v1.csv):", "",
                  "| Comparison | Spearman | Top-20 overlap | Morigaon in top 10 |", "|---|---|---|---|"]
        lines += [f"| {s['scenario']} | {s['spearman']:.3f} | {s['top20_overlap']}/20 | {s['top10_morigaon']}/10 |"
                  for s in v1cmp]
    lines += ["", "## 3. District comparison", "",
              "**Two districts only (16 vs 94 units). This is a description, not a statistical test.** "
              "Scores are 0-100; top decile = top ceil(10%) units in the district (2 in Morigaon, 10 in Nagaon).", "",
              "| Score | District | Units | Mean score | Top-decile mean |", "|---|---|---|---|---|"]
    lines += [f"| {r.score} | {r.district} | {r.n} | {r.mean:.1f} | {r.top_decile_mean:.1f} (top {r.top_decile_k}) |"
              for r in dist.itertuples()]
    lines += ["", "## Summary", ""] + SUMMARY
    (paths.results / "risk_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def v1_comparison(paths, base):
    p = paths.results / "village_risk_ranked_v1.csv"
    if not p.exists():
        return []
    v1 = pd.read_csv(p, comment="#").set_index("unit_id")
    return [compare(v1, v1, "v1 default (percentile well reliance)"),
            compare(v1, base, "current default vs v1")]


def main():
    paths, df, meta, published = load()
    aoi = common.load_aoi(paths)
    base = rank_with(df, meta, config.RISK_WEIGHTS)
    pub = published.set_index("unit_id")["rank"]
    if not (base["rank"].reindex(pub.index) == pub).all():
        raise RuntimeError("recomputed default ranking differs from results/village_risk_ranked.csv "
                           "(re-run the risk stage first)")
    units = gpd.read_file(paths.outputs / "villages.gpkg").reset_index(drop=True)
    districts = common.districts_for_aoi(paths, aoi)

    cases = case_check(paths, aoi, df, meta, base, units, districts)
    sens, ranks = sensitivity(df, meta)
    dist = district_table(df, meta)
    v1cmp = v1_comparison(paths, base)
    write_sensitivity(paths, sens, ranks)
    write_monday(paths, base, ranks["(a) no well reliance"])
    write_checks(paths, cases, dist, sens, v1cmp)

    print(json.dumps(cases, indent=2, default=str))
    print(json.dumps(sens + v1cmp, indent=2))
    print(dist.to_string())


if __name__ == "__main__":
    main()
