"""Tests for export_s1.py with local synthetic STAC items (no Planetary Computer)."""
from datetime import datetime, timezone

import numpy as np
import pytest

import common
import config
import export_s1

CRS = "EPSG:32646"


def test_power_to_db_guards_zero_and_negative():
    out = export_s1.power_to_db(np.array([1.0, 0.1, 0.0, -5.0, np.nan], dtype=np.float32))
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(-10.0)
    assert np.isnan(out[2:]).all()


def test_windows_and_normal_years():
    assert export_s1.window_for_year(2018) == ("2018-06-25", "2018-08-10")
    assert export_s1.normal_years() == config.NORMAL_YEARS
    assert export_s1.normal_years(extra_normal=True) == config.NORMAL_YEARS + config.EXTRA_NORMAL_YEARS


def _item(item_id, geom, when, orbit_state, rel_orbit, assets=None):
    import pystac
    from shapely.geometry import mapping
    item = pystac.Item(id=item_id, geometry=mapping(geom), bbox=list(geom.bounds), datetime=when,
                       properties={"sat:orbit_state": orbit_state, "sat:relative_orbit": rel_orbit,
                                   "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]})
    for key, href in (assets or {}).items():
        item.add_asset(key, pystac.Asset(href=href, media_type=pystac.MediaType.COG, roles=["data"]))
    return item


def test_orbit_and_group_choice():
    from shapely.geometry import box
    regions = {"A": box(94.0, 27.0, 94.5, 27.5), "B": box(94.5, 27.0, 95.0, 27.5)}
    to_utm = export_s1.to_utm_func(32646)
    regions_utm = {k: to_utm(v) for k, v in regions.items()}
    full = box(93.9, 26.9, 95.1, 27.6)
    half = box(93.9, 26.9, 94.6, 27.6)  # covers A only
    items = []
    for i, day in enumerate(["2020-07-01", "2020-07-13", "2020-07-25", "2018-07-02", "2021-07-03"]):
        when = datetime.fromisoformat(day + "T00:30:00").replace(tzinfo=timezone.utc)
        items.append(_item(f"desc{i}", full, when, "descending", 5))
        items.append(_item(f"asc{i}", half, when, "ascending", 9))
    assert export_s1.is_iw_dualpol(items[0])
    coverage, groups = export_s1.coverage_by_orbit_and_date(items, regions_utm, to_utm)
    windows = {"flood": ("2020-06-25", "2020-08-10"),
               "normal": {2018: ("2018-06-25", "2018-08-10"), 2021: ("2021-06-25", "2021-08-10")}}
    candidate_groups = export_s1.connected_groups(regions_utm, 3)
    assert ["A", "B"] in candidate_groups
    best, table = export_s1.choose_orbit_and_group(coverage, windows, candidate_groups, common.logging.getLogger("t"))
    assert (best["orbit_state"], best["relative_orbit"]) == ("descending", 5)
    assert best["districts"] == ["A", "B"] and best["n_flood_dates"] == 3


def _write_power_cog(path, value, west, north, width, height, res=0.0005):
    import rasterio
    from rasterio.transform import from_origin
    data = np.full((height, width), value, dtype=np.float32)
    data[0, 0] = 0.0  # a zero must become NaN, not -inf
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, res, res), tiled=True,
                       blockxsize=256, blockysize=256, nodata=None) as dst:
        dst.write(data, 1)
    return str(path)


def test_write_rtc_date_mosaics_two_scenes(tmp_path, monkeypatch):
    """odc-stac loading + chunked dB writing with two scenes of the same date."""
    import rasterio
    from shapely.geometry import box
    monkeypatch.setattr(config, "EXPORT_CHUNK", 128)
    when = datetime(2020, 7, 1, 0, 30, tzinfo=timezone.utc)
    items = []
    # Two side-by-side scenes, each 0.06 deg wide, meeting at 94.26 E.
    for i, (west, vv) in enumerate([(94.20, 0.1), (94.26, 0.01)]):
        a = {"vv": _write_power_cog(tmp_path / f"vv{i}.tif", vv, west, 26.95, 120, 512),
             "vh": _write_power_cog(tmp_path / f"vh{i}.tif", vv / 10, west, 26.95, 120, 512)}
        items.append(_item(f"s{i}", box(west, 26.95 - 0.256, west + 0.06, 26.95), when, "descending", 5, a))
    to_utm = export_s1.to_utm_func(32646)
    grid = export_s1.build_grid(to_utm(box(94.22, 26.80, 94.30, 26.90)), 32646)
    aoi = {"grid": grid}
    crs, transform, width, height = common.aoi_grid(aoi)
    out = tmp_path / "2020-07-01.tif"
    export_s1.write_rtc_date(items, crs, transform, width, height, out)
    with rasterio.open(out) as src:
        assert src.count == 2 and src.dtypes == ("float32", "float32")
        assert src.crs.to_string() == CRS and src.res == (10.0, 10.0)
        vv, vh = src.read(1), src.read(2)
    # left part from scene 0 (-10 dB), right part from scene 1 (-20 dB)
    assert np.nanmax(vv) == pytest.approx(-10.0, abs=0.01)
    assert np.nanmin(vv) == pytest.approx(-20.0, abs=0.01)
    assert np.nanmean(vh) < np.nanmean(vv)
    stats = export_s1.pass_stats(out)
    assert stats["valid_fraction"] > 0.9 and 0 < stats["waterish_fraction"] < 1
    pooled = export_s1.pooled_mean_std([stats])
    assert -20.0 <= pooled["vv_mean"] <= -10.0


CAL_XML = """<calibration><calibrationVectorList count="2">
<calibrationVector><line>0</line><pixel count="3">0 50 99</pixel>
<sigmaNought count="3">100 200 300</sigmaNought></calibrationVector>
<calibrationVector><line>99</line><pixel count="3">0 50 99</pixel>
<sigmaNought count="3">100 200 300</sigmaNought></calibrationVector>
</calibrationVectorList></calibration>"""


def test_calibration_lut_interpolates():
    lut = export_s1.calibration_lut(CAL_XML, (100, 100))
    assert lut.shape == (100, 100)
    assert lut[0, 0] == pytest.approx(100) and lut[50, 50] == pytest.approx(200)
    assert lut[99, 99] == pytest.approx(300) and lut[25, 25] == pytest.approx(150)


def test_grd_fallback_with_gcps(tmp_path, monkeypatch):
    """GRD path: DN -> sigma0 with the LUT, warped with GCPs onto the grid."""
    import rasterio
    from rasterio.control import GroundControlPoint
    from shapely.geometry import box
    n = 200
    dn = np.full((n, n), 100, dtype=np.uint16)  # sigma0 = 100^2 / 200^2 = 0.25 at LUT 200
    gcps = [GroundControlPoint(row=r, col=c, x=94.2 + c * 0.0005, y=26.95 - r * 0.0005)
            for r in (0, n) for c in (0, n)]
    path = tmp_path / "grd_vv.tif"
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="uint16",
                       gcps=gcps, crs="EPSG:4326") as dst:
        dst.write(dn, 1)
    flat = CAL_XML.replace("100 200 300", "200 200 200")

    class FakeSession:
        def get(self, url, timeout=None):
            return type("R", (), {"text": flat})()
    monkeypatch.setattr(common, "http_session", lambda: FakeSession())
    when = datetime(2020, 7, 1, 0, 30, tzinfo=timezone.utc)
    assets = {"vv": str(path), "vh": str(path), "schema-calibration-vv": "x", "schema-calibration-vh": "x"}
    item = _item("grd", box(94.2, 26.85, 94.3, 26.95), when, "descending", 5, assets)
    to_utm = export_s1.to_utm_func(32646)
    grid = export_s1.build_grid(to_utm(box(94.22, 26.87, 94.28, 26.93)), 32646)
    crs, transform, width, height = common.aoi_grid({"grid": grid})
    out = tmp_path / "grd.tif"
    export_s1.write_grd_date([item], crs, transform, width, height, out, common.logging.getLogger("t"))
    with rasterio.open(out) as src:
        vv = src.read(1)
    assert np.nanmedian(vv) == pytest.approx(10 * np.log10(0.25), abs=0.05)
