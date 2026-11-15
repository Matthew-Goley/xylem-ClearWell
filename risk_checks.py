"""
Checks on the Model 2 risk ranking. Read-only on existing results; writes new files only.

  1. Case check: where do the IDSP outbreak villages Sibir and Goriabori
     (Morigaon, after the July 2019 flood) sit in the ranking? Exact, then fuzzy
     name match against the 110 OSM units; if no match, one Nominatim lookup per
     name restricted to Morigaon, then the containing or nearest unit.
  2. Sensitivity of the ranking to the weights   -> results/risk_sensitivity.md
  3. District comparison, index vs simple baselines
  4. Top-15 table                                  -> results/monday_table.md
  Items 1, 3 and the summary                       -> results/risk_checks.md
  Raw geocoder replies are cached in results/risk_checks_geocode.json so a rerun
  does not query Nominatim again.

The default weights in config.RISK_WEIGHTS are not changed; alternative weight
sets are passed to risk_score.full_ranking as arguments.

Run: python risk_checks.py
"""
import json
import math
import re
import time
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from shapely.geometry import Point

import config
import risk_score

CASES = [
    {"name": "Sibir", "district": "Morigaon", "cases": 31, "onset": "2019-07-25"},
    {"name": "Goriabori", "district": "Morigaon", "cases": 16, "onset": "2019-08-30"},
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ClearWell-risk-checks/1.0 (research script, one request per name)"
TOP_K = 20
# Written after reading the numbers of the first run; recheck against the tables if inputs change.
SUMMARY = [
    "1. Flooding drives the order: every weight variant keeps Spearman >= 0.95 and 16-19 of the default top 20, "
    "though part of that agreement is the 27 unflooded units that sit at the bottom in every version.",
    "2. The all-Morigaon top 10 is fragile: it comes from the district-level well-reliance figure, a flat bonus "
    "for all 16 Morigaon units; without it Morigaon has 4 of the top 10 and three Nagaon units move to ranks 1-3.",
    "3. Morigaon does flood more on average (22% vs 6% of unit area, higher means on both baselines), but the most "
    "flooded Nagaon units are about as flooded as Morigaon's; two districts, no statistical test.",
    "4. Duration adds little: it can only be 0, 18 or 36 days, all top 15 are at 36, and dropping it barely moves "
    "the ranking.",
    "5. The case check could not be done: neither Sibir nor Goriabori matches a unit name and Nominatim found "
    "neither in Morigaon, so the ranking is untested against these two outbreaks.",
    "6. Read it as a reasonable ordering of one season's flood exposure across 110 OSM proxies, not as a validated "
    "predictor of where outbreaks happen.",
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


def scenarios():
    base = dict(config.RISK_WEIGHTS)
    zero = {k: 0.0 for k in base}
    return {
        "default": base,
        "(a) no well reliance": {**base, "well_reliance": 0.0},
        "(b) equal weights": {k: 1.0 for k in base},
        "(c) no duration": {**base, "duration": 0.0},
        "(d) flooded fraction only": {**zero, "exposure": 1.0},
    }


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


# ---------------------------------------------------------------- 1. cases
def name_match(target, ranked):
    names = ranked["name"].dropna()
    exact = names[names.map(norm) == norm(target)]
    if len(exact):
        return "exact", exact.index[0], 100.0, []
    choices = {uid: norm(n) for uid, n in names.items()}
    top = process.extract(norm(target), choices, scorer=fuzz.WRatio, limit=3)
    cands = [(ranked.loc[uid, "name"], round(score, 1)) for _, score, uid in top]
    if top and top[0][1] >= config.FUZZY_MATCH_MIN_SCORE:
        return "fuzzy", top[0][2], top[0][1], cands
    return None, None, None, cands


def geocode(name, district_geom_ll, cache):
    if name in cache:
        return cache[name], True
    minx, miny, maxx, maxy = district_geom_ll.bounds
    params = {"q": f"{name}, Morigaon, Assam, India", "format": "jsonv2", "limit": 5,
              "countrycodes": "in", "viewbox": f"{minx},{maxy},{maxx},{miny}", "bounded": 1}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        reply = {"ok": True, "results": r.json(), "queried_at": datetime.now().isoformat()}
    except Exception as err:  # noqa: BLE001
        reply = {"ok": False, "error": f"{type(err).__name__}: {err}", "queried_at": datetime.now().isoformat()}
    cache[name] = reply
    time.sleep(1.1)  # Nominatim policy: at most 1 request per second
    return reply, False


def case_check(paths, ranked, units, districts):
    n = len(ranked)
    mori = ranked[ranked["district"] == "Morigaon"].sort_values("rank")
    mori_rank = {uid: i + 1 for i, uid in enumerate(mori.index)}
    cache_path = paths.results / "risk_checks_geocode.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    dist_ll = districts.to_crs(4326).set_index("district")
    mori_utm = districts.set_index("district").loc["Morigaon", "geometry"]
    rows = []
    for case in CASES:
        row = {**case}
        how, uid, score, cands = name_match(case["name"], ranked)
        row["name_candidates"] = cands
        if how:
            row.update(method=f"{how} name match (score {score:.0f})", unit_id=uid, distance_m=0.0)
        else:
            reply, cached = geocode(case["name"], dist_ll.loc["Morigaon", "geometry"], cache)
            row["geocode_cached"] = cached
            hits = reply.get("results", []) if reply.get("ok") else []
            if not reply.get("ok"):
                row.update(method="geocoding FAILED", reason=reply["error"])
            elif not hits:
                row.update(method="geocoding FAILED", reason="Nominatim returned no result inside Morigaon")
            else:
                hit = hits[0]
                pt = gpd.GeoSeries([Point(float(hit["lon"]), float(hit["lat"]))], crs=4326).to_crs(units.crs)[0]
                d_poly = units.geometry.distance(pt)
                best = d_poly.idxmin()
                row.update(method="geocoded (Nominatim), then containing/nearest unit",
                           geocode_name=hit.get("display_name"), geocode_type=f"{hit.get('category', '')}/{hit.get('type', '')}",
                           lon=float(hit["lon"]), lat=float(hit["lat"]), n_hits=len(hits),
                           in_morigaon_polygon=bool(mori_utm.contains(pt)),
                           unit_id=units.loc[best, "unit_id"], distance_m=float(d_poly[best]),
                           contains=bool(d_poly[best] == 0),
                           dist_to_unit_point_m=float(units.loc[best].geometry.centroid.distance(pt)))
        if row.get("unit_id"):
            u = ranked.loc[row["unit_id"]]
            r = int(u["rank"])
            row.update(unit_name=u["name"], unit_district=u["district"], rank=r,
                       percentile=100 * (n - r) / (n - 1), score=float(u["score"]),
                       morigaon_rank=mori_rank.get(row["unit_id"]), morigaon_n=len(mori),
                       flooded_frac=float(u["flooded_frac"]), max_duration_days=float(u["max_duration_days"]))
        rows.append(row)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return rows


# ---------------------------------------------------------- 2. sensitivity
def sensitivity(df, meta):
    sc = scenarios()
    ranks = {k: rank_with(df, meta, w) for k, w in sc.items()}
    base = ranks["default"]
    base_top = set(base.index[base["rank"] <= TOP_K])
    out = []
    for k, r in ranks.items():
        r = r.reindex(base.index)
        top10 = r[r["rank"] <= 10]
        out.append({"scenario": k,
                    "weights": {c: round(v / sum(sc[k].values()), 3) for c, v in sc[k].items() if v},
                    "spearman": float(base["rank"].corr(r["rank"], method="spearman")),
                    "top20_overlap": len(base_top & set(r.index[r["rank"] <= TOP_K])),
                    "top10_morigaon": int((top10["district"] == "Morigaon").sum())})
    return out, ranks


# ------------------------------------------------------- 3. district means
def district_table(df, meta):
    sets = {"Full index (default weights)": config.RISK_WEIGHTS,
            "Flooded area only (flooded_frac)": {**{k: 0.0 for k in config.RISK_WEIGHTS}, "exposure": 1.0},
            "Exposed population only (exposed_pop)": {**{k: 0.0 for k in config.RISK_WEIGHTS}, "exposed_people": 1.0}}
    rows = []
    for label, w in sets.items():
        r = rank_with(df, meta, w)
        for d, g in r.groupby("district"):
            k = max(1, math.ceil(0.1 * len(g)))
            rows.append({"score": label, "district": d, "n": len(g), "mean": g["score"].mean(),
                         "top_decile_k": k, "top_decile_mean": g["score"].nlargest(k).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ write
def label(name, uid):
    return str(name) if isinstance(name, str) and name.strip() else f"(unnamed, {uid})"


def write_sensitivity(paths, sens, ranks):
    lines = ["# Risk ranking sensitivity to the weights", "",
             f"Generated by risk_checks.py on {datetime.now():%Y-%m-%d %H:%M}. 110 village proxies "
             "(16 Morigaon, 94 Nagaon). Same components and percentile scoring as risk_score.py; only "
             "the weights change. The default weights in config.py are untouched.", "",
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
    lines += ["", "## Notes", "",
              "- well_reliance is a district-level Census value (Morigaon 92.1%, Nagaon 85.4%), the same for "
              "every unit in a district. Its percentile is 1.0 for every Morigaon unit and 0.0 for every "
              "Nagaon unit, so it works as a flat bonus for Morigaon rather than a village signal.",
              "- duration has only three values (0, 18, 36 days) because there are two flood passes and the "
              "resolution is one revisit interval; many units tie.",
              "- One flood season, one AOI. These numbers say how much the ordering depends on the weight "
              "choice, not whether any ordering is right."]
    (paths.results / "risk_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_monday(paths, base, no_well):
    top = base.sort_values("rank").head(15)
    lines = ["# Top 15 village proxies, Clearwell risk index", "",
             f"Generated by risk_checks.py on {datetime.now():%Y-%m-%d %H:%M} from the default ranking "
             "(same as results/village_risk_ranked.csv). Units are OSM-based village proxies (Voronoi cells "
             "clipped to 1.5 km), not census villages.", "",
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
              "removed (see results/risk_sensitivity.md)."]
    (paths.results / "monday_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_checks(paths, cases, dist, sens, summary):
    lines = ["# Risk ranking checks", "", f"Generated by risk_checks.py on {datetime.now():%Y-%m-%d %H:%M}.", "",
             "## 1. Case check: IDSP outbreak villages", "",
             "Percentile = share of the other 109 units ranked below it (100 = top).", ""]
    for c in cases:
        lines.append(f"### {c['name']} ({c['district']}; {c['cases']} cases from {c['onset']})")
        lines.append("")
        lines.append(f"- Fuzzy name candidates among the 110 units: "
                     + ", ".join(f"{n} ({s})" for n, s in c["name_candidates"]) + " (none passed the "
                     f"{config.FUZZY_MATCH_MIN_SCORE} threshold)" if c["method"].startswith("geocod") else
                     f"- Name match: {c['method']}")
        lines.append(f"- Method: {c['method']}" + (" (cached reply reused)" if c.get("geocode_cached") else ""))
        if "reason" in c:
            lines += [f"- **Not located:** {c['reason']}. No rank reported.", ""]
            continue
        if "geocode_name" in c:
            lines.append(f"- Geocoded to: {c['geocode_name']} [{c['geocode_type']}] at {c['lat']:.5f}, "
                         f"{c['lon']:.5f} ({c['n_hits']} hit(s); first used). Inside Morigaon polygon: "
                         f"{c['in_morigaon_polygon']}.")
            where = "inside" if c["contains"] else f"{c['distance_m']:,.0f} m from the edge of"
            lines.append(f"- Point is {where} unit {c['unit_name'] if isinstance(c['unit_name'], str) else '(unnamed)'} "
                         f"({c['unit_id']}, {c['unit_district']}); {c['dist_to_unit_point_m']:,.0f} m from that "
                         "unit's centre.")
        lines.append(f"- Rank **{c['rank']} / 110** (percentile {c['percentile']:.0f}), score {c['score']:.1f}; "
                     f"rank in Morigaon **{c['morigaon_rank']} / {c['morigaon_n']}**. Unit flooded fraction "
                     f"{100 * c['flooded_frac']:.0f}%, duration {int(c['max_duration_days'])} days.")
        lines.append("")
    lines += ["Caveat: two cases, both in Morigaon, where every unit gets the well-reliance bonus. A nearest-unit "
              "match far from the point says little about the village itself.", "",
              "## 2. Sensitivity", "", "See results/risk_sensitivity.md. Summary:", "",
              "| Scenario | Spearman | Top-20 overlap | Morigaon in top 10 |", "|---|---|---|---|"]
    lines += [f"| {s['scenario']} | {s['spearman']:.3f} | {s['top20_overlap']}/20 | {s['top10_morigaon']}/10 |"
              for s in sens]
    lines += ["", "## 3. District comparison", "",
              "**Two districts only (16 vs 94 units). This is a description, not a statistical test.** "
              "Scores are 0-100 percentile-based within each scoring; top decile = top ceil(10%) units in the "
              "district (2 in Morigaon, 10 in Nagaon).", "",
              "| Score | District | Units | Mean score | Top-decile mean |", "|---|---|---|---|---|"]
    lines += [f"| {r.score} | {r.district} | {r.n} | {r.mean:.1f} | {r.top_decile_mean:.1f} (top {r.top_decile_k}) |"
              for r in dist.itertuples()]
    lines += ["", "## Summary", ""] + summary
    (paths.results / "risk_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    paths, df, meta, published = load()
    base = rank_with(df, meta, config.RISK_WEIGHTS)
    pub = published.set_index("unit_id")["rank"]
    if not (base["rank"].reindex(pub.index) == pub).all():
        raise RuntimeError("recomputed default ranking differs from results/village_risk_ranked.csv")
    units = gpd.read_file(paths.outputs / "villages.gpkg").reset_index(drop=True)
    districts = gpd.read_file(paths.outputs / "aoi_districts.gpkg")

    cases = case_check(paths, base, units, districts)
    sens, ranks = sensitivity(df, meta)
    dist = district_table(df, meta)
    write_sensitivity(paths, sens, ranks)
    write_monday(paths, base, ranks["(a) no well reliance"])

    print(json.dumps(cases, indent=2, default=str))
    print(json.dumps(sens, indent=2))
    print(dist.to_string())
    write_checks(paths, cases, dist, sens, SUMMARY)


if __name__ == "__main__":
    main()
