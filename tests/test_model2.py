"""Tests for build_villages.py, risk_score.py and validate_risk.py (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

import build_villages
import risk_score
import validate_risk

CRS = "EPSG:32646"


def _districts():
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame({"district": ["DistA", "DistB"]},
                            geometry=[box(0, 0, 5000, 5000), box(5000, 0, 10000, 5000)], crs=CRS)


def test_voronoi_units_are_clipped_to_district_and_radius():
    import geopandas as gpd
    rng = np.random.default_rng(0)
    xy = rng.uniform([100, 100], [9900, 4900], size=(40, 2))
    pts = gpd.GeoDataFrame({"osm_id": [f"node/{i}" for i in range(40)], "name": [f"v{i}" for i in range(40)],
                            "place": "village"}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS)
    d = _districts()
    units = build_villages.voronoi_units(pts, d, max_radius=1500)
    assert len(units) == 40
    assert units["unit_id"].is_unique
    for row in units.itertuples():
        dist_geom = d[d["district"] == row.district].geometry.iloc[0]
        assert row.geometry.difference(dist_geom.buffer(1e-6)).area < 1e-3
        assert row.geometry.area <= np.pi * 1500 ** 2 + 1
    # Cells must not overlap each other.
    assert units.geometry.area.sum() == pytest.approx(units.geometry.union_all().area, rel=1e-6)


def test_grid_units_only_populated_cells():
    d = _districts()
    all_cells = build_villages.grid_units(d, 1000)
    assert len(all_cells) == 50
    populated = (np.array([500.0, 7500.0]), np.array([500.0, 2500.0]))
    some = build_villages.grid_units(d, 1000, populated)
    assert len(some) == 2 and set(some["district"]) == {"DistA", "DistB"}


def test_zonal_helpers():
    unit_idx = np.array([[1, 1, 2], [0, 2, 2]], dtype=np.int32)
    mask = np.array([[True, False, True], [True, True, False]])
    assert build_villages.count_per_unit(unit_idx, mask, 2).tolist() == [1, 2]
    vals = np.array([[5, 1, 2], [9, 7, 3]])
    assert build_villages.max_per_unit(vals, unit_idx, 2).tolist() == [5, 7]
    frac = build_villages.neighbourhood_fraction(np.ones((5, 5), bool), 2)
    assert frac.shape == (3, 3) and frac[0, 0] == 1.0 and frac[2, 2] == 0.25


def test_points_to_pixels():
    from affine import Affine
    tr = Affine(10, 0, 0, 0, -10, 100)
    r, c, inside = build_villages.points_to_pixels([5, 15, 200], [95, 85, 50], tr, 10, 10)
    assert r[:2].tolist() == [0, 1] and c[:2].tolist() == [0, 1]
    assert inside.tolist() == [True, True, False]


def test_census_template_written_when_missing(tmp_paths, logger):
    assert build_villages.census_well_reliance(tmp_paths, ["DistA"], logger) is None
    assert (tmp_paths.manual / "census_water_template.csv").exists()
    pd.DataFrame({"district": ["dista"], "pct_shallow_well_handpump": [80.0]}).to_csv(
        tmp_paths.manual / "census_water.csv", index=False)
    assert build_villages.census_well_reliance(tmp_paths, ["DistA"], logger) == {"DistA": 80.0}


def test_pct_rank_ties_and_nan():
    r = risk_score.pct_rank([0, 0, 0, 5, 10, np.nan])
    assert r.iloc[:3].tolist() == [0, 0, 0]
    assert r.iloc[4] == 1.0 and np.isnan(r.iloc[5])
    assert risk_score.pct_rank([3.0]).tolist() == [0.0]


def test_score_units_renormalizes_missing_values():
    raw = pd.DataFrame({"exposure": [0.0, 0.5, 1.0], "duration": [0.0, np.nan, 10.0]})
    comps, score, w = risk_score.score_units(raw, {"exposure": 0.35, "duration": 0.25, "well_reliance": 0.4})
    assert set(w) == {"exposure", "duration"} and sum(w.values()) == pytest.approx(1.0)
    assert score.iloc[0] == 0 and score.iloc[2] == pytest.approx(100)
    # unit 1 has no duration: its score uses exposure only (rank 0.5)
    assert score.iloc[1] == pytest.approx(50)


def _feature_table(dates):
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({"unit_id": [f"u{i}" for i in range(n)], "name": "", "district": ["A", "B", "C"] * 10,
                       "area_km2": 1.0, "elev_pctile": rng.random(n)})
    cum = np.zeros(n)
    longest = np.zeros(n)
    for k, d in enumerate(dates):
        cum = np.maximum(cum, rng.random(n) * (rng.random(n) < 0.5))
        longest = np.maximum(longest, rng.integers(0, 4, n) * 6.0 * (cum > 0))
        tag = d.replace("-", "")
        df[f"cum_flooded_frac_{tag}"] = cum
        df[f"max_duration_so_far_{tag}"] = longest
        df[f"exposed_pop_{tag}"] = cum * 100
    tag = dates[-1].replace("-", "")
    df["flooded_frac"] = df[f"cum_flooded_frac_{tag}"]
    df["max_duration_days"] = df[f"max_duration_so_far_{tag}"]
    df["exposed_pop"] = df[f"exposed_pop_{tag}"]
    return df


def test_last_pass_ranking_equals_final_ranking():
    dates = ["2020-07-01", "2020-07-07", "2020-07-13"]
    df = _feature_table(dates)
    meta = {"flood_dates": dates, "features": {"population": {"available": True},
                                               "elevation": {"available": True},
                                               "well_reliance": {"available": False}}}
    ranked, comps, w = risk_score.full_ranking(df, meta)
    assert set(comps) == {"exposure", "duration", "exposed_people", "low_elevation"}
    assert ranked["score"].between(0, 100).all()
    assert sorted(ranked["rank"]) == list(range(1, 31))
    wide, info = risk_score.per_pass_rankings(df, meta)
    assert "duration" not in info[0]["components"] and "duration" in info[1]["components"]
    final = ranked.set_index("unit_id").loc[wide["unit_id"], "rank"].values
    assert (wide["rank_20200713"].values == final).all()
    evo = risk_score.evolution_table(wide, info, pd.Series(final, index=wide.index), 10)
    assert evo["overlap_with_final_top_k"].iloc[-1] == 10


def test_spearman_uncertainty_small_and_large_n():
    tiny = validate_risk.spearman_with_uncertainty([1, 2], [3, 4])
    assert np.isnan(tiny["rho"]) and tiny["n"] == 2
    three = validate_risk.spearman_with_uncertainty([1, 2, 3], [1, 2, 3], n_boot=200)
    assert three["rho"] == pytest.approx(1.0)
    assert three["p_value"] == pytest.approx(2 / 6)  # exact test, 6 permutations, two-sided
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    big = validate_risk.spearman_with_uncertainty(x, x + rng.normal(scale=0.3, size=40),
                                                  n_perm=500, n_boot=300)
    assert big["rho"] > 0.8 and big["p_value"] < 0.01 and big["ci_low"] < big["rho"] < big["ci_high"]


def test_waterborne_filter():
    assert validate_risk.is_waterborne("Acute Diarrhoeal Disease")
    assert validate_risk.is_waterborne("ADD")
    assert validate_risk.is_waterborne("Hepatitis A")
    assert not validate_risk.is_waterborne("Measles")
    assert not validate_risk.is_waterborne("Address unknown")  # 'add' only as a whole word


def test_idsp_loader_dedups_matches_and_zero_fills(tmp_path, logger):
    from synthetic_idsp import make_fake_idsp_csv
    path = tmp_path / "idsp.csv"
    make_fake_idsp_csv(path, ["Dhemaji", "Lakhimpur", "Morigaon"])
    events, info, mapping = validate_risk.load_idsp(path, ["Dhemaji", "Lakhimpur", "Morigaon", "Jorhat"], logger)
    assert info["exact_duplicates_removed"] == 4
    assert info["detected_columns"]["district"] == "name_of_district"
    assert info["detected_columns"]["rainfall"] == "rainfall_mm"
    assert set(events["district"]) <= {"Dhemaji", "Lakhimpur", "Morigaon"}
    assert "Measles" not in set(events["disease"])          # not water-borne
    assert "Patna" not in set(mapping["query"])              # other state dropped
    assert events["week_start"].dt.day_of_week.eq(0).all()      # weeks start on Monday
    table = validate_risk.district_week_table(events, ["Dhemaji", "Jorhat"], events["date"].min(),
                                              events["date"].max())
    assert (table[table["district"] == "Jorhat"]["outbreaks"] == 0).all()
    assert table["outbreaks"].sum() == (events["district"] == "Dhemaji").sum()


def test_clip_raster_to_bbox(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    src = tmp_path / "pop.tif"
    data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    with rasterio.open(src, "w", driver="GTiff", width=100, height=100, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(90.0, 30.0, 0.01, 0.01)) as dst:
        dst.write(data, 1)
    out = build_villages.clip_raster_to_bbox(src, (90.2, 29.5, 90.4, 29.8), tmp_path / "clip.tif")
    with rasterio.open(out) as c:
        clip = c.read(1)
        assert clip.shape == (30, 20)
        assert c.transform.c == pytest.approx(90.2) and c.transform.f == pytest.approx(29.8)
    assert clip[0, 0] == data[20, 20]
