/* Clearwell static map. Every number shown comes from data/*.json, data/*.geojson or
 * data/villages.csv, written by export_web_data.py. Nothing here is a typed-in result.
 * Copy rules: no em dashes, no exclamation marks. */
(function () {
  "use strict";

  var NA = "not available";

  // Plain-language labels for the five score components.
  var COMPONENTS = [
    { key: "exposure", label: "Share of village area flooded" },
    { key: "duration", label: "How long the flood lasted" },
    { key: "exposed_people", label: "People living in the flooded area" },
    { key: "well_reliance", label: "Use of hand pumps and shallow wells" },
    { key: "low_elevation", label: "Low-lying land" }
  ];
  var COMPONENT_EXPLAIN = {
    exposure: "Percentile of the share of the village shape that was flooded.",
    duration: "Percentile of the longest time water stayed, estimated from the gaps between satellite passes.",
    exposed_people: "Percentile of the estimated number of people living where it flooded (from WorldPop population maps).",
    low_elevation: "Percentile of how low the land is compared with the other villages (from the Copernicus land height map)."
  };

  // Single-hue sequential scale (ColorBrewer Oranges, colour-blind safe), fixed 20-point bins.
  var BINS = [
    { min: 80, color: "#a63603", label: "80 to 100" },
    { min: 60, color: "#e6550d", label: "60 to 80" },
    { min: 40, color: "#fd8d3c", label: "40 to 60" },
    { min: 20, color: "#fdbe85", label: "20 to 40" },
    { min: 0, color: "#feedde", label: "0 to 20" }
  ];
  var FLOOD_COLOR = "#2166ac";

  var state = {
    meta: null,
    villages: [],        // feature properties, in rank order
    layersById: {},      // id -> Leaflet layer
    sortKey: "rank",
    sortDir: 1,
    search: "",
    top20: false,
    selectedId: null,
    floodLayers: {},
    floodId: null,
    floodOpacity: 0.7
  };

  // ---------------------------------------------------------------- helpers
  function $(id) { return document.getElementById(id); }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmt(v, digits) {
    if (!isNum(v)) return NA;
    return v.toLocaleString("en-US", { minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0 });
  }
  function pct(v, digits) { return isNum(v) ? fmt(v * 100, digits === undefined ? 0 : digits) + "%" : NA; }
  function dec(v, digits) { return isNum(v) ? v.toFixed(digits === undefined ? 2 : digits) : NA; }
  function setText(id, text) { var el = $(id); if (el) el.textContent = text; }
  function get(obj, path) {
    var parts = path.split(".");
    for (var i = 0; i < parts.length; i++) {
      if (obj === null || obj === undefined || typeof obj !== "object") return undefined;
      obj = obj[parts[i]];
    }
    return obj;
  }
  function joinAnd(list) {
    if (!list || !list.length) return "";
    if (list.length === 1) return list[0];
    return list.slice(0, -1).join(", ") + " and " + list[list.length - 1];
  }
  var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"];
  function parseDate(s) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s || "");
    return m ? { y: +m[1], m: +m[2], d: +m[3] } : null;
  }
  function niceDate(s) {
    var d = parseDate(s);
    return d ? d.d + " " + MONTHS[d.m - 1] + " " + d.y : (s || NA);
  }
  function niceDateTime(s) {
    if (!s) return NA;
    var t = /T(\d{2}:\d{2})/.exec(s);
    return niceDate(s) + (t ? ", " + t[1] : "");
  }
  function colorFor(score) {
    if (!isNum(score)) return "#bdbdbd";
    for (var i = 0; i < BINS.length; i++) if (score >= BINS[i].min) return BINS[i].color;
    return BINS[BINS.length - 1].color;
  }
  // Villages in the top legend bin get a hazard icon in the list, popup and print view.
  var HIGH_SCORE = BINS[0].min;
  function isHigh(p) { return isNum(p.score) && p.score >= HIGH_SCORE; }
  function hazardIcon() {
    return '<span class="hazard" title="Score ' + HIGH_SCORE + ' or higher">' +
      '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">' +
      '<path d="M8 1.3 15.2 14.4H.8Z" fill="#a63603" stroke="#5a1d01" stroke-width="1" stroke-linejoin="round"/>' +
      '<rect x="7.1" y="5.3" width="1.8" height="5.1" rx=".6" fill="#fff"/><circle cx="8" cy="12.2" r="1" fill="#fff"/>' +
      '</svg><span class="visually-hidden">High score, </span></span>';
  }

  function fetchJSON(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + " " + r.status);
      return r.json();
    });
  }

  // ---------------------------------------------------------------- banner and text
  function eventPhrase(meta) {
    var win = get(meta, "event.flood_window");
    var districts = get(meta, "event.districts");
    var a = win && parseDate(win[0]), b = win && parseDate(win[1]);
    var when = "";
    if (a && b) {
      when = a.m === b.m ? MONTHS[a.m - 1] + " " + a.y
        : MONTHS[a.m - 1] + " to " + MONTHS[b.m - 1] + " " + (a.y === b.y ? a.y : a.y + " to " + b.y);
    }
    if (!when && !(districts && districts.length)) return "";
    return "Shows the " + (when || "") + (when ? " " : "") + "flood" +
      (districts && districts.length ? " in " + joinAnd(districts) : "") + ".";
  }

  function fillBanner(meta) {
    var gen = meta && meta.generated_at ? "Data generated " + niceDateTime(meta.generated_at) + "." : "Data generation date not available.";
    setText("print-meta", gen + " Printed " + (function (n) { return n.getDate() + " " + MONTHS[n.getMonth()] + " " + n.getFullYear(); })(new Date()) + ".");
  }

  function fillReadTab(meta) {
    var passes = (meta && meta.passes) || [];
    if (passes.length) {
      setText("read-passes", passes.map(function (p) { return niceDate(p.date); }).join(" and ") +
        (passes.length < 3 ? " (only " + passes.length + " pass" + (passes.length === 1 ? "" : "es") + ")" : ""));
    }
    var ny = get(meta, "event.normal_years");
    if (ny && ny.length) setText("read-normal-years", joinAnd(ny.map(String)));
    var layer = (get(meta, "flood_layers") || [])[0];
    if (layer && isNum(layer.pixel_ground_m)) setText("read-pixel", fmt(layer.pixel_ground_m) + " m");
    var r = get(meta, "scoring.village_radius_m");
    if (isNum(r)) setText("read-radius", r >= 1000 ? fmt(r / 1000, 1) + " km" : fmt(r) + " m");

    var ul = $("read-components");
    var weights = get(meta, "scoring.weights");
    var used = get(meta, "scoring.components_used");
    if (weights && ul) {
      var wellMode = get(meta, "scoring.well_reliance_scoring");
      ul.innerHTML = COMPONENTS.map(function (c) {
        var w = weights[c.key];
        var inUse = !used || used.indexOf(c.key) >= 0;
        var explain = COMPONENT_EXPLAIN[c.key];
        if (c.key === "well_reliance") {
          explain = wellMode === "absolute"
            ? "The share of households in the district that get drinking water from hand pumps or shallow wells (Census 2011). It is a district number, so every village in a district has the same value. Floods can wash dirty water into these wells."
            : "Percentile of the district share of households that get drinking water from hand pumps or shallow wells (Census 2011). Floods can wash dirty water into these wells.";
        }
        return "<li><strong>" + esc(c.label) + "</strong>: weight " +
          (isNum(w) && inUse ? fmt(w * 100) + "%" : "not used") + ". " + esc(explain) + "</li>";
      }).join("");
    }
  }

  function model1Html(m1) {
    var g = get(m1, "groups");
    if (!g) return "<p>" + NA + "</p>";
    var order = ["test_all", "test_india", "bolivia"];
    var html = "<p>The flood detector is a <strong>U-Net</strong> (a type of image-recognition model). " +
      "It was tested on labelled satellite images it had not seen in training and compared with " +
      "<strong>Otsu thresholding</strong> (a standard method that calls every pixel darker than a cut-off value water). " +
      "The main score is <strong>IoU</strong> (overlap between the predicted water and the hand-labelled water, from 0 to 1, higher is better).</p>";
    var sentences = [];
    order.forEach(function (k) {
      var r = g[k];
      if (!r) return;
      var u = get(r, "unet.iou"), o = get(r, "otsu.iou");
      if (!isNum(u) || !isNum(o)) return;
      var who = u > o ? "The U-Net scored higher" : (u < o ? "The threshold method scored higher" : "Both scored the same");
      sentences.push("<li>" + esc(r.label) + " (" + fmt(r.n_tiles) + " image tiles): U-Net IoU " + dec(u) +
        ", threshold method " + dec(o) + ". " + who + ".</li>");
    });
    if (sentences.length) html += "<ul>" + sentences.join("") + "</ul>";
    html += "<div class=\"table-wrap\"><table><caption>Detector scores by test group</caption><thead><tr><th scope=\"col\">Group</th>" +
      "<th scope=\"col\" class=\"num\">Tiles</th><th scope=\"col\" class=\"num\">U-Net IoU</th><th scope=\"col\" class=\"num\">Otsu IoU</th>" +
      "<th scope=\"col\" class=\"num\">U-Net precision</th><th scope=\"col\" class=\"num\">U-Net recall</th></tr></thead><tbody>";
    order.forEach(function (k) {
      var r = g[k];
      if (!r) return;
      html += "<tr><th scope=\"row\">" + esc(r.label) + "</th><td class=\"num\">" + fmt(r.n_tiles) + "</td><td class=\"num\">" +
        dec(get(r, "unet.iou")) + "</td><td class=\"num\">" + dec(get(r, "otsu.iou")) + "</td><td class=\"num\">" +
        dec(get(r, "unet.precision")) + "</td><td class=\"num\">" + dec(get(r, "unet.recall")) + "</td></tr>";
    });
    html += "</tbody></table></div>";
    html += "<p class=\"note\">Precision: of the pixels called water, the share that really were water. " +
      "Recall: of the real water pixels, the share that were found. These tests use images from many countries, " +
      "not from Assam, so accuracy in Assam may differ.</p>";
    return html;
  }

  function opticalHtml(oc) {
    var s2 = get(oc, "sentinel2");
    if (!s2) return "<p>" + NA + "</p>";
    if (s2.status !== "DONE") return "<p>The optical check was not completed (status: " + esc(s2.status || NA) + ").</p>";
    var m = s2.metrics || {};
    var html = "<p>Optical satellites (cameras, like Sentinel-2) cannot see through cloud, but on clear days they show water well. " +
      "A Sentinel-2 image from " + esc(niceDate(s2.s2_date)) + ", " + fmt(s2.days_apart) + (s2.days_apart === 1 ? " day" : " days") + " after the radar pass of " +
      esc(niceDate(s2.flood_pass)) + ", was compared with the radar water map on the cloud-free part of the area (" +
      pct(s2.clear_fraction) + " of it).</p>";
    html += "<ul><li>Overlap (IoU): " + dec(m.iou) + "</li><li>Precision: " + dec(m.precision) +
      "</li><li>Recall: " + dec(m.recall) + "</li><li>Pixels where both maps agree (water or dry): " + pct(m.agreement) + "</li></ul>";
    html += "<p class=\"note\">This compares all water (flood plus normal water), not flood alone. Water can rise or fall in the days between the two images, and the optical method has its own errors, so this is a rough check, not a measure of accuracy.</p>";
    return html;
  }

  function validationHtml(v) {
    if (!v) return "<p>" + NA + "</p>";
    var html = "<p>" + esc(v.statement || NA) + "</p>";
    var wk = v.district_week || [];
    if (wk.length) {
      html += "<details><summary>Numbers from the validation report</summary>" +
        "<p class=\"note\">AUC (how well a number separates weeks with and without an outbreak; 0.5 is chance). " +
        "With very few outbreak weeks these values can change a lot from one extra case.</p>" +
        "<div class=\"table-wrap\"><table><thead><tr><th scope=\"col\">Method</th><th scope=\"col\" class=\"num\">District-weeks</th>" +
        "<th scope=\"col\" class=\"num\">With outbreak</th><th scope=\"col\" class=\"num\">AUC</th></tr></thead><tbody>" +
        wk.map(function (r) {
          return "<tr><td>" + esc(r.method || NA) + "</td><td class=\"num\">" + fmt(r.n_rows) + "</td><td class=\"num\">" +
            fmt(r.n_positive) + "</td><td class=\"num\">" + dec(r.auc) + "</td></tr>";
        }).join("") + "</tbody></table></div></details>";
    }
    return html;
  }

  function fillAccuracyTab(meta) {
    $("acc-model1").innerHTML = model1Html(meta && meta.model1);
    $("acc-optical").innerHTML = opticalHtml(meta && meta.optical_check);
    $("acc-validation").innerHTML = validationHtml(meta && meta.validation);
    var cc = get(meta, "case_check.statement");
    if (cc) setText("acc-case", cc);
    var lim = (meta && meta.limitations) || [];
    var rep = get(meta, "validation.report_limitations") || [];
    var ul = $("acc-limits");
    if (lim.length) {
      ul.innerHTML = lim.map(function (s) { return "<li>" + esc(s) + "</li>"; }).join("");
      if (rep.length) {
        var li = document.createElement("li");
        li.innerHTML = "<details><summary>Technical notes from the validation report</summary><ul>" +
          rep.map(function (s) { return "<li>" + esc(s) + "</li>"; }).join("") + "</ul></details>";
        ul.appendChild(li);
      }
    }
    setText("acc-generated", meta && meta.generated_at ? niceDateTime(meta.generated_at) : NA);
    setText("acc-commit", meta && meta.git_commit ? meta.git_commit +
      (meta.results_uncommitted_changes ? " (results folder had uncommitted changes)" : "") : NA);
    setText("acc-ranked", get(meta, "scoring.generated_at") ? niceDateTime(get(meta, "scoring.generated_at")) : NA);
  }

  // ---------------------------------------------------------------- tabs
  function initTabs() {
    var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));
    function activate(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        $(t.getAttribute("aria-controls")).hidden = !on;
      });
      if (focus) tab.focus();
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { activate(t, false); });
      t.addEventListener("keydown", function (e) {
        var j = null;
        if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = tabs.length - 1;
        if (j !== null) { e.preventDefault(); activate(tabs[j], true); }
      });
    });
    return function showList() { activate(tabs[0], false); };
  }

  // ---------------------------------------------------------------- map
  var map, villageLayer, districtLayer, showListTab;

  function initMap() {
    map = L.map("map", { zoomControl: true, zoomSnap: 0.25 });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
    map.createPane("flood").style.zIndex = 350;
    map.createPane("districts").style.zIndex = 450;
    map.getPane("districts").style.pointerEvents = "none";
    map.setView([26.3, 92.6], 9);
  }

  function addLegend(meta) {
    var legend = L.control({ position: "bottomleft" });
    legend.onAdd = function () {
      var div = L.DomUtil.create("div", "legend");
      div.setAttribute("role", "group");
      div.setAttribute("aria-label", "Map legend");
      var html = "<h4>Risk score</h4>";
      BINS.forEach(function (b) {
        html += '<div class="row"><span class="sw" style="background:' + b.color + '"></span>' + b.label + "</div>";
      });
      html += '<h4 style="margin-top:6px">Other layers</h4>' +
        '<div class="row"><span class="sw" style="background:' + FLOOD_COLOR + ';opacity:.75"></span>Flood water</div>' +
        '<div class="row"><span class="sw" style="background:transparent;border:2px dashed #333"></span>District outline</div>' +
        '<div class="row">' + hazardIcon() + "Score " + HIGH_SCORE + " or higher</div>";
      div.innerHTML = html;
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    legend.addTo(map);
  }

  function villageStyle(p, hover) {
    var sel = p.id === state.selectedId;
    return {
      color: sel ? "#111" : (hover ? "#222" : "#5b4636"),
      weight: sel ? 3.5 : (hover ? 2.5 : 0.8),
      fillColor: colorFor(p.score),
      fillOpacity: hover || sel ? 0.85 : 0.7
    };
  }

  function popupHtml(p) {
    var n = state.villages.length;
    var revisit = get(state.meta, "event.revisit_interval_days");
    var dur;
    if (!isNum(p.duration_days)) dur = NA;
    else if (p.duration_days === 0) dur = "no flooding seen";
    else dur = "about " + fmt(p.duration_days) + " days";
    var bars = COMPONENTS.map(function (c) {
      var v = p["c_" + c.key];
      var ok = isNum(v);
      return "<li><span>" + esc(c.label) + "</span>" +
        '<span class="bar" role="img" aria-label="' + esc(c.label) + ": " + (ok ? v + " out of 100" : NA) + '"><span style="width:' +
        (ok ? Math.max(0, Math.min(100, v)) : 0) + '%"></span></span><span class="v">' + (ok ? v : "n/a") + "</span></li>";
    }).join("");
    return '<div class="pop">' +
      "<h3>" + (isHigh(p) ? hazardIcon() : "") + esc(p.name) + "</h3>" +
      (isHigh(p) ? '<p class="high-note">High score: ' + HIGH_SCORE + " or higher</p>" : "") +
      '<p class="sub">' + esc(p.district || NA) + " district</p>" +
      "<dl>" +
      "<dt>Risk score</dt><dd>" + (isNum(p.score) ? fmt(p.score, 1) + " of 100" : NA) + "</dd>" +
      "<dt>Rank</dt><dd>" + (isNum(p.rank) ? fmt(p.rank) + " of " + fmt(n) : NA) + "</dd>" +
      "<dt>Area flooded</dt><dd>" + (isNum(p.pct_flooded) ? fmt(p.pct_flooded, 1) + "%" : NA) + "</dd>" +
      "<dt>People in flooded area</dt><dd>" + (isNum(p.people_est) ? "about " + fmt(p.people_est) + " (estimate)" : NA) + "</dd>" +
      "<dt>Flood duration</dt><dd>" + dur + (isNum(p.duration_days) && p.duration_days > 0 ? " (rough estimate" +
        (isNum(revisit) ? ", passes are " + fmt(revisit) + " days apart" : "") + ")" : "") + "</dd>" +
      "</dl>" +
      "<div><strong>Score parts</strong> (0 to 100)</div>" +
      '<ul class="bars">' + bars + "</ul>" +
      '<p class="hint">A ranking of flood exposure, not a water test. Test wells to confirm.</p>' +
      "</div>";
  }

  function addVillages(geo) {
    state.villages = geo.features.map(function (f) { return f.properties; })
      .sort(function (a, b) { return a.rank - b.rank; });
    villageLayer = L.geoJSON(geo, {
      style: function (f) { return villageStyle(f.properties, false); },
      onEachFeature: function (f, layer) {
        var p = f.properties;
        state.layersById[p.id] = layer;
        layer.bindTooltip(esc(p.name) + ": rank " + p.rank + ", score " + fmt(p.score, 1), { sticky: true });
        layer.on("mouseover", function () { layer.setStyle(villageStyle(p, true)); layer.bringToFront(); });
        layer.on("mouseout", function () { layer.setStyle(villageStyle(p, false)); });
        layer.on("click", function () { selectVillage(p.id, { fromMap: true }); });
      }
    }).addTo(map);
  }

  function addDistricts(geo) {
    districtLayer = L.geoJSON(geo, {
      pane: "districts",
      interactive: false,
      style: { color: "#222", weight: 2, dashArray: "6 4", fill: false }
    }).addTo(map);
  }

  function initFlood(meta) {
    var sel = $("flood-select");
    var layers = (meta && meta.flood_layers) || [];
    if (!layers.length) {
      sel.innerHTML = "<option>" + NA + "</option>";
      sel.disabled = true;
      $("toggle-flood").disabled = true;
      $("flood-opacity").disabled = true;
      return;
    }
    sel.innerHTML = layers.map(function (l) {
      var label = l.date ? "Pass " + niceDate(l.date) : (l.label || l.id);
      return '<option value="' + esc(l.id) + '">' + esc(label) + "</option>";
    }).join("");
    layers.forEach(function (l) {
      state.floodLayers[l.id] = L.imageOverlay("data/" + l.file, l.bounds, {
        pane: "flood", opacity: state.floodOpacity, interactive: false,
        className: "flood-overlay", alt: "Flood water, " + (l.label || l.id)
      });
    });
    showFlood(layers[0].id);
    sel.addEventListener("change", function () { showFlood(sel.value); });
    var op = $("flood-opacity");
    op.addEventListener("input", function () {
      state.floodOpacity = op.value / 100;
      $("flood-opacity-val").textContent = op.value + "%";
      if (state.floodId) state.floodLayers[state.floodId].setOpacity(state.floodOpacity);
    });
    $("toggle-flood").addEventListener("change", function () { showFlood(sel.value); });
  }

  function showFlood(id) {
    Object.keys(state.floodLayers).forEach(function (k) {
      if (map.hasLayer(state.floodLayers[k])) map.removeLayer(state.floodLayers[k]);
    });
    state.floodId = id;
    if ($("toggle-flood").checked && state.floodLayers[id]) {
      state.floodLayers[id].setOpacity(state.floodOpacity).addTo(map);
    }
  }

  function initToggles() {
    $("toggle-villages").addEventListener("change", function (e) {
      if (!villageLayer) return;
      if (e.target.checked) villageLayer.addTo(map); else map.removeLayer(villageLayer);
    });
    $("toggle-districts").addEventListener("change", function (e) {
      if (!districtLayer) return;
      if (e.target.checked) districtLayer.addTo(map); else map.removeLayer(districtLayer);
    });
  }

  // ---------------------------------------------------------------- selection
  function selectVillage(id, opts) {
    opts = opts || {};
    var prev = state.selectedId;
    state.selectedId = id;
    if (prev && state.layersById[prev]) {
      var pp = findVillage(prev);
      state.layersById[prev].setStyle(villageStyle(pp, false));
    }
    var p = findVillage(id);
    var layer = state.layersById[id];
    if (!p || !layer) return;
    layer.setStyle(villageStyle(p, false));
    layer.bringToFront();
    if (!opts.fromMap) {
      if (!map.hasLayer(villageLayer)) { $("toggle-villages").checked = true; villageLayer.addTo(map); }
      map.fitBounds(layer.getBounds(), { maxZoom: 14, padding: [40, 40] });
    }
    layer.bindPopup(popupHtml(p), { maxWidth: 340, autoPanPadding: [20, 20] }).openPopup();

    // Make sure the row is visible, then highlight it.
    if (showListTab) showListTab();
    if (!isRowVisible(p)) {
      state.search = ""; $("search").value = "";
      state.top20 = false; $("top20").checked = false;
      renderTable();
    } else {
      markSelectedRow();
    }
    var row = document.querySelector('tr[data-id="' + cssEsc(id) + '"]');
    if (row && opts.fromMap) row.scrollIntoView({ block: "nearest" });
  }

  function cssEsc(s) { return window.CSS && CSS.escape ? CSS.escape(s) : String(s).replace(/"/g, '\\"'); }
  function findVillage(id) {
    for (var i = 0; i < state.villages.length; i++) if (state.villages[i].id === id) return state.villages[i];
    return null;
  }

  // ---------------------------------------------------------------- table
  function isRowVisible(p) {
    if (state.top20 && !(p.rank <= 20)) return false;
    if (state.search && String(p.name).toLowerCase().indexOf(state.search) < 0) return false;
    return true;
  }

  function sortedRows() {
    var k = state.sortKey, d = state.sortDir;
    return state.villages.filter(isRowVisible).sort(function (a, b) {
      var x = a[k], y = b[k];
      if (typeof x === "string" || typeof y === "string") {
        x = String(x || ""); y = String(y || "");
        return d * x.localeCompare(y) || a.rank - b.rank;
      }
      var xn = isNum(x), yn = isNum(y);
      if (!xn && !yn) return a.rank - b.rank;
      if (!xn) return 1;
      if (!yn) return -1;
      return d * (x - y) || a.rank - b.rank;
    });
  }

  function renderTable() {
    var tbody = document.querySelector("#rank-table tbody");
    var rows = sortedRows();
    tbody.innerHTML = rows.map(function (p) {
      return '<tr tabindex="0" data-id="' + esc(p.id) + '" aria-selected="false">' +
        '<td class="num">' + fmt(p.rank) + "</td>" +
        "<td>" + (isHigh(p) ? hazardIcon() : "") + esc(p.name) + "</td>" +
        "<td>" + esc(p.district || NA) + "</td>" +
        '<td class="num"><span class="swatch" style="background:' + colorFor(p.score) + '" aria-hidden="true"></span>' + fmt(p.score, 1) + "</td>" +
        '<td class="num">' + (isNum(p.pct_flooded) ? fmt(p.pct_flooded, 1) : NA) + "</td>" +
        '<td class="num">' + (isNum(p.people_est) ? fmt(p.people_est) : NA) + "</td></tr>";
    }).join("");
    var n = state.villages.length;
    var nHigh = state.villages.filter(isHigh).length;
    setText("list-status", (nHigh ? nHigh + " of " + n + " villages score " + HIGH_SCORE + " or higher (marked with a warning sign). " : "") +
      "Showing " + rows.length + " of " + n + " villages" +
      (state.top20 ? ", top 20 only" : "") + (state.search ? ', matching "' + state.search + '"' : "") +
      ". Select a row to show it on the map.");
    markSelectedRow();
  }

  function markSelectedRow() {
    var rows = document.querySelectorAll("#rank-table tbody tr");
    for (var i = 0; i < rows.length; i++) {
      rows[i].setAttribute("aria-selected", rows[i].getAttribute("data-id") === state.selectedId ? "true" : "false");
    }
  }

  function initTable() {
    var tbody = document.querySelector("#rank-table tbody");
    function activate(e) {
      var tr = e.target.closest("tr[data-id]");
      if (tr) selectVillage(tr.getAttribute("data-id"), { fromMap: false });
    }
    tbody.addEventListener("click", activate);
    tbody.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(e); }
      else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        var tr = e.target.closest("tr");
        var next = tr && (e.key === "ArrowDown" ? tr.nextElementSibling : tr.previousElementSibling);
        if (next) { e.preventDefault(); next.focus(); }
      }
    });
    var ths = document.querySelectorAll("#rank-table thead th");
    Array.prototype.forEach.call(ths, function (th) {
      th.querySelector("button").addEventListener("click", function () {
        var key = th.getAttribute("data-key");
        if (state.sortKey === key) state.sortDir = -state.sortDir;
        else {
          state.sortKey = key;
          // Numbers where bigger means more risk start high-to-low; rank and text start low-to-high.
          state.sortDir = (key === "rank" || key === "name" || key === "district") ? 1 : -1;
        }
        Array.prototype.forEach.call(ths, function (h) {
          h.setAttribute("aria-sort", h === th ? (state.sortDir === 1 ? "ascending" : "descending") : "none");
        });
        renderTable();
      });
    });
    $("search").addEventListener("input", function (e) {
      state.search = e.target.value.trim().toLowerCase();
      renderTable();
    });
    $("top20").addEventListener("change", function (e) {
      state.top20 = e.target.checked;
      renderTable();
    });
    $("print-btn").addEventListener("click", function () { window.print(); });
  }

  function renderPrintTable() {
    var tbody = document.querySelector("#print-table tbody");
    tbody.innerHTML = state.villages.filter(function (p) { return p.rank <= 20; }).map(function (p) {
      return "<tr><td>" + fmt(p.rank) + "</td><td>" + (isHigh(p) ? hazardIcon() : "") + esc(p.name) + "</td><td>" + esc(p.district || NA) + "</td><td>" +
        fmt(p.score, 1) + "</td><td>" + (isNum(p.pct_flooded) ? fmt(p.pct_flooded, 1) : NA) + "</td><td>" +
        (isNum(p.people_est) ? fmt(p.people_est) : NA) + "</td><td>" + (isNum(p.duration_days) ? fmt(p.duration_days) : NA) + "</td></tr>";
    }).join("");
  }

  // ---------------------------------------------------------------- boot
  function fitStudyArea() {
    var b = get(state.meta, "study_bounds");
    if (b) map.fitBounds(b, { padding: [10, 10] });
    else if (districtLayer) map.fitBounds(districtLayer.getBounds());
    else if (villageLayer) map.fitBounds(villageLayer.getBounds());
  }

  function boot() {
    showListTab = initTabs();
    initMap();
    initToggles();
    initTable();

    var metaP = fetchJSON("data/meta.json").catch(function (e) { console.warn(e); return null; });
    var villP = fetchJSON("data/villages.geojson").catch(function (e) { console.warn(e); return null; });
    var distP = fetchJSON("data/districts.geojson").catch(function (e) { console.warn(e); return null; });

    Promise.all([metaP, villP, distP]).then(function (res) {
      var meta = res[0], vill = res[1], dist = res[2];
      state.meta = meta;
      fillBanner(meta);
      fillReadTab(meta);
      fillAccuracyTab(meta);
      initFlood(meta);
      if (dist) addDistricts(dist);
      if (vill) addVillages(vill);
      else setText("list-status", "Village data not available.");
      addLegend(meta);
      renderTable();
      renderPrintTable();
      fitStudyArea();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
