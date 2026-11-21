/* Spectral Market Lab site: every number and chart is read from window.SML (site/data.js). */
(() => {
  "use strict";
  const D = window.SML;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const C = {
    s1: css("--s1"), s2: css("--s2"), s3: css("--s3"), s4: css("--s4"), ref: css("--ref"),
    ink: css("--ink"), forest: css("--forest"), lime: css("--lime"), line: css("--line"), muted: css("--muted"),
  };

  /* ---------- formatting ---------- */
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");
  const sci = (v, digits = 1) => {
    if (v === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(v)));
    const m = v / 10 ** e;
    return `${m.toFixed(digits)} × 10${sup(e)}`;
  };
  const pow10 = (v) => {
    const e = Math.round(Math.log10(v));
    return e === 0 ? "1" : e === 1 ? "10" : `10${sup(e)}`;
  };
  const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
  const signed = (v, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)}%`;
  const num = (v, d = 2) => v.toFixed(d).replace("-", "−");
  const pval = (p) => (p == null ? "n/a" : p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(p < 0.01 ? 3 : 2)}`);
  const fmtDate = d3.utcFormat("%b %Y");
  const parseDate = d3.utcParse("%Y-%m-%d");

  /* ---------- tooltip ---------- */
  const tipEl = document.getElementById("tip");
  const showTip = (event, html) => {
    tipEl.innerHTML = html;
    tipEl.hidden = false;
    const pad = 14;
    const { innerWidth: W, innerHeight: H } = window;
    const r = tipEl.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + r.width > W - 8) x = event.clientX - r.width - pad;
    if (y + r.height > H - 8) y = event.clientY - r.height - pad;
    tipEl.style.left = `${Math.max(8, x)}px`;
    tipEl.style.top = `${Math.max(8, y)}px`;
  };
  const hideTip = () => { tipEl.hidden = true; };
  const row = (color, label, value) => `<div class="row"><i style="background:${color}"></i>${label}: <b>${value}</b></div>`;

  /* ---------- chart scaffolding ---------- */
  const charts = [];
  function mount(id, draw) {
    const el = document.getElementById(id);
    if (!el) return null;
    const chart = { el, draw, render() {
      const { width, height } = el.getBoundingClientRect();
      if (width < 10) return;
      el.innerHTML = "";
      const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${width} ${height}`);
      draw(svg, width, height);
    } };
    charts.push(chart);
    chart.render();
    return chart;
  }
  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => charts.forEach((c) => c.render()), 80);
  }).observe(document.body);

  const M = { top: 14, right: 18, bottom: 34, left: 52 };
  function frame(svg, w, h, x, y, opt = {}) {
    const m = { ...M, ...(opt.xLabel ? { bottom: 50 } : {}), ...(opt.yLabel ? { top: 30 } : {}), ...opt.margin };
    x.range([m.left, w - m.right]);
    y.range([h - m.bottom, m.top]);
    const yTicks = opt.yTicks ?? 5;
    const xTicks = opt.xTicks ?? Math.max(2, Math.floor((w - m.left - m.right) / 90));
    const gy = d3.axisLeft(y).tickSize(-(w - m.left - m.right)).tickPadding(8);
    if (Array.isArray(yTicks)) gy.tickValues(yTicks); else gy.ticks(yTicks);
    if (opt.yFormat) gy.tickFormat(opt.yFormat);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.left},0)`).call(gy)
      .call((g) => g.selectAll(".tick line").attr("stroke", "#eceee6"));
    const gx = d3.axisBottom(x).tickSizeOuter(0).tickPadding(8);
    if (Array.isArray(xTicks)) gx.tickValues(xTicks); else gx.ticks(xTicks);
    if (opt.xFormat) gx.tickFormat(opt.xFormat);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.bottom})`).call(gx)
      .call((g) => g.selectAll(".tick line").attr("stroke", C.line));
    if (opt.xLabel) svg.append("text").attr("x", w - m.right).attr("y", h - 4).attr("text-anchor", "end").text(opt.xLabel);
    if (opt.yLabel) svg.append("text").attr("x", Math.max(0, m.left - 44)).attr("y", m.top - 14).attr("text-anchor", "start").text(opt.yLabel);
    return m;
  }

  function legend(id, items) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = items.map(([color, label, kind = "line"]) =>
      `<span><i class="${kind === "line" ? "" : kind}" style="background:${color}"></i>${label}</span>`).join("");
  }

  /** Crosshair hover over a shared x array: calls html(i) for the nearest index. */
  function crosshair(svg, w, h, m, x, xs, html, opts = {}) {
    const rule = svg.append("line").attr("y1", m.top).attr("y2", h - m.bottom).attr("stroke", C.ink).attr("stroke-width", 1).attr("opacity", 0);
    const dots = svg.append("g");
    svg.append("rect").attr("x", m.left).attr("y", m.top).attr("width", w - m.left - m.right).attr("height", h - m.top - m.bottom)
      .attr("fill", "transparent")
      .on("pointermove", (event) => {
        const [px] = d3.pointer(event);
        const xv = x.invert(px);
        let i = d3.bisectCenter(xs.map(Number), Number(xv));
        i = Math.max(0, Math.min(xs.length - 1, i));
        const cx = x(xs[i]);
        rule.attr("x1", cx).attr("x2", cx).attr("opacity", 0.35);
        dots.selectAll("*").remove();
        (opts.points ? opts.points(i) : []).forEach(([py, color]) => {
          if (py == null || !Number.isFinite(py)) return;
          dots.append("circle").attr("cx", cx).attr("cy", py).attr("r", 4.5).attr("fill", color).attr("stroke", "#fff").attr("stroke-width", 2);
        });
        showTip(event, html(i));
      })
      .on("pointerleave", () => { rule.attr("opacity", 0); dots.selectAll("*").remove(); hideTip(); });
  }

  const lineGen = (x, y, xKey = "x", yKey = "y") => d3.line()
    .defined((d) => d[yKey] != null && Number.isFinite(d[yKey]) && (y.type !== "log" || d[yKey] > 0))
    .x((d) => x(d[xKey])).y((d) => y(d[yKey]));
  const path = (svg, data, gen, color, width = 2) =>
    svg.append("path").datum(data).attr("fill", "none").attr("stroke", color).attr("stroke-width", width)
      .attr("stroke-linejoin", "round").attr("stroke-linecap", "round").attr("d", gen);
  const logY = (domain) => Object.assign(d3.scaleLog().domain(domain).clamp(true), { type: "log" });

  function facts(id, list) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = list.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  }

  function segmented(figureEl, attr, onChange) {
    const buttons = [...figureEl.querySelectorAll(`.seg button[data-${attr}]`)];
    buttons.forEach((b) => b.addEventListener("click", () => {
      buttons.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      onChange(b.dataset[attr]);
    }));
  }

  /* ---------- verdict chips & the ten answers ---------- */
  const ICON = {
    yes: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5 6.5 11.5 12.5 4.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    no: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 4.5l7 7m0-7-7 7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    partly: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 2.5a5.5 5.5 0 0 1 0 11Z" fill="currentColor"/></svg>',
  };
  const LABEL = { yes: "Yes", no: "No", partly: "Partly" };
  const chip = (v) => `<span class="chip chip-${v}">${ICON[v]}${LABEL[v]}</span>`;
  document.querySelectorAll(".verdict-line").forEach((el) => { el.innerHTML = chip(el.dataset.verdict); });

  const S = {
    orth: D.orthogonality, stab: D.stability, trend: D.trend_report, spec: D.spectrum, clus: D.clustering,
    track: D.index_tracking, mv: D.min_variance, sa: D.stat_arb, kal: D.kalman_trend, reg: D.regimes,
  };
  const wf = S.trend.walk_forward_summary;
  const answers = [
    ["q-qr", "Does the choice of QR algorithm matter?", `At condition number 10⁸ classical Gram-Schmidt loses all orthogonality; Householder stays at ${sci(S.orth.at_cond_1e8.Householder)}.`, "yes"],
    ["q-solvers", "Is the normal-equations shortcut safe?", `Not on raw day numbers: below 6 correct digits from degree ${S.stab["first_degree_with_error_above_1e-6"].raw.normal}. Rescaling time fixes it.`, "no"],
    ["q-trend", "Does a fitted price trend predict prices?", `${d3.sum(wf, (r) => r.significant_wins)} of 210 setups beat a random walk across 239,904 forecasts.`, "no"],
    ["q-spectrum", "Is there structure beyond noise?", `${S.spec.n_signal} of ${S.spec.n_stocks} eigenvalues clear the noise edge; the market mode alone explains ${pct(S.spec.explained_top[0])}.`, "yes"],
    ["q-clusters", "Can sectors be found without labels?", `Spectral clustering matches NSE sectors with adjusted Rand index ${num(S.clus.ari)} (${pval(S.clus.permutation_p_value)}).`, "yes"],
    ["q-tracking", "Can ten stocks track the NIFTY 50?", `A greedy 10-stock basket tracks it with ${pct(S.track.greedy_basket_10.te_test)} error out of sample.`, "yes"],
    ["q-minvar", "Does cleaning the covariance matrix lower risk?", `With short windows, by ${pct(-S.mv.unconstrained["Ledoit-Wolf"][0].vol_vs_sample, 0)}. With long windows or no shorting, barely.`, "partly"],
    ["q-statarb", "Is there a stat-arb edge after costs?", `Sharpe ${num(S.sa.performance_by_cost_bps["0"].sharpe)} before costs; it breaks even at ${num(S.sa.breakeven_cost_bps)} bps per trade.`, "no"],
    ["q-kalman", "Does a time-varying trend forecast better?", "Within 1% of the random walk on all seven series.", "no"],
    ["q-regimes", "Does the market switch between regimes?", `Yes: calm (${pct(S.reg.regimes[0].annual_vol, 0)} vol) and turbulent (${pct(S.reg.regimes[1].annual_vol, 0)}). It doesn't forecast volatility better than EWMA.`, "yes"],
  ];
  document.getElementById("answer-list").innerHTML = answers.map(([id, q, a, v], i) =>
    `<li><a href="#${id}"><span class="n">${String(i + 1).padStart(2, "0")}</span><span class="q">${q}</span><span class="a">${a}</span>${chip(v)}</a></li>`).join("");

  /* ---------- hero: eigenvalue spectrum ---------- */
  const eig = D.spectrum_series.eigenvalues;
  const q = S.spec.q;
  const mpLo = (1 - Math.sqrt(q)) ** 2;
  const mpHi = S.spec.mp_edge;
  document.getElementById("hero-ticks").innerHTML = eig.map((v) => `<i class="${v > mpHi ? "on" : ""}"></i>`).join("");

  mount("chart-hero", (svg, w, h) => {
    const x = d3.scaleLinear().domain([0.3, eig.length + 0.7]);
    const y = logY([0.12, 22]);
    const m = frame(svg, w, h, x, y, {
      yTicks: [0.2, 0.5, 1, 2, 5, 10, 20], yFormat: d3.format("~g"), xTicks: [1, 10, 20, 30, 40, 51],
      margin: { left: 40, right: 8, top: 18, bottom: 28 },
    });
    svg.append("rect").attr("x", m.left).attr("width", w - m.left - m.right)
      .attr("y", y(mpHi)).attr("height", y(mpLo) - y(mpHi)).attr("fill", C.lime);
    svg.append("text").attr("class", "lbl").attr("x", w - m.right - 4).attr("y", y(mpHi) - 6).attr("text-anchor", "end")
      .text(`noise edge ${num(mpHi)}`);
    const base = y.range()[0];
    const g = svg.append("g");
    eig.forEach((v, i) => {
      const signal = v > mpHi;
      const color = signal ? C.forest : "#aab3ad";
      g.append("line").attr("x1", x(i + 1)).attr("x2", x(i + 1)).attr("y1", base).attr("y2", y(v))
        .attr("stroke", color).attr("stroke-width", signal ? 2.5 : 1.5).attr("stroke-linecap", "round");
      g.append("circle").attr("cx", x(i + 1)).attr("cy", y(v)).attr("r", signal ? 4.5 : 3).attr("fill", color);
    });
    svg.append("text").attr("class", "lbl-strong").attr("x", x(1) + 10).attr("y", y(eig[0]) + 4)
      .text(`market mode ${num(eig[0], 1)}, ${pct(S.spec.explained_top[0])} of variance`);
    const xs = eig.map((_, i) => i + 1);
    crosshair(svg, w, h, m, x, xs, (i) => {
      const v = eig[i];
      const where = v > mpHi ? "above the noise edge" : v >= mpLo ? "inside the noise band" : "below the noise band";
      const name = i === 0 ? "market mode" : i < 4 ? `PC${i + 1}` : `eigenvalue ${i + 1}`;
      return `${name}<br>λ = <b>${num(v)}</b>, ${pct(v / eig.length)} of variance<br>${where}`;
    }, { points: (i) => [[y(eig[i]), eig[i] > mpHi ? C.forest : "#8c958f"]] });
  });

  /* ---------- numerics ---------- */
  const qrNames = ["Classical Gram-Schmidt", "Modified Gram-Schmidt", "Householder"];
  const qrColors = [C.s3, C.s2, C.s1];
  legend("legend-qr", qrNames.map((n, i) => [qrColors[i], n]));
  facts("facts-qr", qrNames.map((n) => [`${n} at κ = 10⁸`, n === "Classical Gram-Schmidt" ? num(S.orth.at_cond_1e8[n]) : sci(S.orth.at_cond_1e8[n])]));
  mount("chart-qr", (svg, w, h) => {
    const ks = S.orth.condition_numbers;
    const x = d3.scaleLog().domain(d3.extent(ks));
    const y = logY([1e-16, 10]);
    const m = frame(svg, w, h, x, y, {
      xTicks: [1, 1e3, 1e6, 1e9, 1e12], xFormat: pow10, yTicks: [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1], yFormat: pow10,
      xLabel: "condition number κ", yLabel: "‖I − QᵀQ‖",
    });
    const series = qrNames.map((n) => S.orth.loss[n].map((v, i) => ({ x: ks[i], y: v })));
    series.forEach((s, i) => path(svg, s, lineGen(x, y), qrColors[i]));
    crosshair(svg, w, h, m, x, ks, (i) => `κ = <b>${sci(ks[i])}</b>` + qrNames.map((n, j) => row(qrColors[j], n, sci(series[j][i].y))).join(""),
      { points: (i) => series.map((s, j) => [y(s[i].y), qrColors[j]]) });
  });

  const solverNames = { normal: "Normal equations", qr: "Householder QR", svd: "SVD" };
  const solverColors = { normal: C.s3, qr: C.s1, svd: C.s2 };
  legend("legend-solvers", Object.keys(solverNames).map((k) => [solverColors[k], solverNames[k]]));
  const firstBad = S.stab["first_degree_with_error_above_1e-6"].raw;
  const condAt = (kind, d) => S.stab.cond[kind][S.stab.degrees.indexOf(d)];
  facts("facts-solvers", [
    ["Normal equations lose 6 digits at", `degree ${firstBad.normal}`],
    ["κ(A) at degree 3, raw → rescaled", `${sci(condAt("raw", 3))} → ${num(condAt("scaled", 3), 1)}`],
    ["Largest κ(A) with a Legendre basis", num(d3.max(S.stab.cond.legendre), 1)],
  ]);
  let solverScale = "raw";
  const solverChart = mount("chart-solvers", (svg, w, h) => {
    const ds = S.stab.degrees;
    const x = d3.scaleLinear().domain(d3.extent(ds));
    const y = logY([1e-17, 1e18]);
    const m = frame(svg, w, h, x, y, {
      xTicks: ds.filter((d) => d % 2 === 0 || d === 1), yTicks: [1e-15, 1e-10, 1e-5, 1, 1e5, 1e10, 1e15], yFormat: pow10,
      xLabel: "polynomial degree", yLabel: "relative error",
    });
    svg.append("line").attr("x1", m.left).attr("x2", w - m.right).attr("y1", y(1e-6)).attr("y2", y(1e-6)).attr("stroke", C.ink).attr("stroke-opacity", 0.5);
    svg.append("text").attr("class", "lbl").attr("x", m.left + 6).attr("y", y(1e-6) - 6).text("6 correct digits");
    const keys = Object.keys(solverNames);
    const series = keys.map((k) => ds.map((d, i) => ({
      x: d, y: S.stab.failed[solverScale][k][i] ? null : S.stab.error[solverScale][k][i],
    })));
    series.forEach((s, j) => {
      path(svg, s, lineGen(x, y), solverColors[keys[j]]);
      svg.append("g").selectAll("circle").data(s.filter((d) => d.y != null && d.y > 0)).join("circle")
        .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("r", 2.6).attr("fill", solverColors[keys[j]]);
    });
    const fail = ds.find((d, i) => S.stab.failed[solverScale].normal[i]);
    if (fail) {
      svg.append("line").attr("x1", x(fail)).attr("x2", x(fail)).attr("y1", m.top).attr("y2", h - m.bottom).attr("stroke", C.s3).attr("stroke-opacity", 0.6);
      svg.append("text").attr("class", "lbl").attr("x", x(fail) - 6).attr("y", y(1e-12)).attr("text-anchor", "end").text(`Cholesky fails from degree ${fail}`);
    }
    crosshair(svg, w, h, m, x, ds, (i) => `degree <b>${ds[i]}</b>, κ(A) = ${sci(S.stab.cond[solverScale][i])}` +
      keys.map((k, j) => row(solverColors[k], solverNames[k], series[j][i].y == null ? "failed" : sci(series[j][i].y))).join(""),
    { points: (i) => series.map((s, j) => [s[i].y ? y(s[i].y) : null, solverColors[keys[j]]]) });
  });
  segmented(document.getElementById("chart-solvers").closest("figure"), "scale", (v) => { solverScale = v; solverChart.render(); });

  const degName = ["constant (moving average)", "straight line", "quadratic"];
  const degColors = [C.s1, C.s2, C.s3];
  legend("legend-trend", degName.map((n, i) => [degColors[i], `degree ${i}: ${n}`, "dot"]));
  const accs = wf.map((r) => r.mean_directional_accuracy);
  facts("facts-trend", [
    ["Setups significantly better than a random walk", "0 of 210"],
    ["Error vs random walk, median", `${num(d3.min(wf, (r) => r.median_rmse_ratio), 1)}–${num(d3.max(wf, (r) => r.median_rmse_ratio), 1)}×`],
    ["Direction called right, on average", `${(d3.min(accs) * 100).toFixed(0)}–${pct(d3.max(accs), 0)}`],
  ]);
  let horizon = 1;
  const trendChart = mount("chart-trend", (svg, w, h) => {
    const rows = wf.filter((r) => r.horizon === horizon);
    const windows = [...new Set(wf.map((r) => r.window))];
    const x = d3.scaleLog().domain([0.85, d3.max(wf, (r) => r.median_rmse_ratio) * 1.15]);
    const y = d3.scaleBand().domain(windows).padding(0.25);
    const m = { ...M, left: 74 };
    x.range([m.left, w - m.right]);
    y.range([m.top, h - m.bottom]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${h - m.bottom})`)
      .call(d3.axisBottom(x).tickValues([1, 1.5, 2, 3, 5, 8]).tickFormat((d) => `${d}×`).tickSize(-(h - m.top - m.bottom)).tickPadding(8))
      .call((g) => g.selectAll(".tick line").attr("stroke", "#eceee6"));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`)
      .call(d3.axisLeft(y).tickFormat((d) => `${d} days`).tickSize(0).tickPadding(10)).call((g) => g.select("path").remove());
    svg.append("text").attr("x", w - m.right).attr("y", h - 4).attr("text-anchor", "end").text("forecast RMSE ÷ random-walk RMSE (log scale)");
    svg.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", m.top - 4).attr("y2", h - m.bottom).attr("stroke", C.ink).attr("stroke-width", 1.5);
    svg.append("text").attr("class", "lbl").attr("x", x(1) + 6).attr("y", m.top + 6).text("random walk");
    windows.forEach((win) => {
      const cy = y(win) + y.bandwidth() / 2;
      const pts = rows.filter((r) => r.window === win);
      svg.append("line").attr("x1", x(d3.min(pts, (r) => r.median_rmse_ratio))).attr("x2", x(d3.max(pts, (r) => r.median_rmse_ratio)))
        .attr("y1", cy).attr("y2", cy).attr("stroke", C.line).attr("stroke-width", 2);
    });
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", (r) => x(r.median_rmse_ratio)).attr("cy", (r) => y(r.window) + y.bandwidth() / 2)
      .attr("r", 7).attr("fill", (r) => degColors[r.degree]).attr("stroke", "#fff").attr("stroke-width", 2)
      .style("cursor", "default")
      .on("pointermove", (event, r) => showTip(event,
        `${r.window}-day window, ${degName[r.degree]}<br>RMSE <b>${num(r.median_rmse_ratio)}×</b> the random walk<br>direction right ${pct(r.mean_directional_accuracy)}<br>significantly worse on ${r.significant_losses} of ${r.n_series} series`))
      .on("pointerleave", hideTip);
  });
  segmented(document.getElementById("chart-trend").closest("figure"), "h", (v) => { horizon = Number(v); trendChart.render(); });

  /* ---------- market structure ---------- */
  facts("facts-spectrum", [
    ["Market mode λ₁", `${num(S.spec.top_eigenvalues[0], 1)}`],
    ["Correlation of PC1 with NIFTY 50", num(S.spec.pc1_index_correlation)],
    [`Above the noise edge (${S.spec.n_signal_market_adjusted} with the market removed)`, `${S.spec.n_signal} of ${S.spec.n_stocks}`],
  ]);
  const sectorOf = {
    COALINDIA: "Energy", ONGC: "Energy", NTPC: "Utilities", TCS: "IT", INFY: "IT", HINDUNILVR: "FMCG", HCLTECH: "IT", TECHM: "IT",
    BRITANNIA: "FMCG", NESTLEIND: "FMCG", ICICIBANK: "Banks", HDFCBANK: "Banks", BAJFINANCE: "Finance", CIPLA: "Pharma", DRREDDY: "Pharma", SUNPHARMA: "Pharma",
  };
  const poleName = (list) => {
    const counts = d3.rollups(list.slice(0, 3).map((n) => sectorOf[n]).filter(Boolean), (v) => v.length, (d) => d).sort((a, b) => b[1] - a[1]);
    return counts.map((c) => c[0]).slice(0, 2).join(" and ");
  };
  document.getElementById("loadings").innerHTML =
    `<div class="loading-axis"><span></span><span>Negative end</span><span>Positive end</span></div>` +
    Object.entries(S.spec.loadings).map(([pc, l]) =>
      `<div class="loading-row"><span class="pc">${pc.toUpperCase()}</span>
        <div class="pole neg"><b>${poleName(l.negative)}</b>${l.negative.join(", ")}</div>
        <div class="pole pos"><b>${poleName(l.positive)}</b>${l.positive.join(", ")}</div></div>`).join("");

  facts("facts-clusters", [
    ["Adjusted Rand index", num(S.clus.ari)],
    ["Chance level (97.5th pct)", num(S.clus.ari_null_97_5)],
    [`Clusters by eigengap (${S.clus.eigengap_k_without_market_removal} with the market left in)`, `${S.clus.eigengap_k}`],
  ]);
  legend("legend-clusters", [[C.s1, "market mode removed"], [C.s3, "market mode left in"], ["#e9ebe6", "chance", "band"]]);
  mount("chart-clusters", (svg, w, h) => {
    const ks = Object.keys(S.clus.ari_by_k).map(Number);
    const a = ks.map((k) => ({ x: k, y: S.clus.ari_by_k[k] }));
    const b = ks.map((k) => ({ x: k, y: S.clus.ari_by_k_without_market_removal[k] }));
    const x = d3.scaleLinear().domain(d3.extent(ks));
    const y = d3.scaleLinear().domain([0, 0.8]);
    const m = frame(svg, w, h, x, y, { xTicks: [2, 5, 8, 11, 14, 17, 20], yFormat: d3.format(".1f"), xLabel: "number of clusters k", yLabel: "adjusted Rand index" });
    svg.append("rect").attr("x", m.left).attr("width", w - m.left - m.right).attr("y", y(S.clus.ari_null_97_5)).attr("height", y(0) - y(S.clus.ari_null_97_5)).attr("fill", "#e9ebe6");
    path(svg, b, lineGen(x, y), C.s3);
    path(svg, a, lineGen(x, y), C.s1);
    const kk = S.clus.eigengap_k;
    svg.append("circle").attr("cx", x(kk)).attr("cy", y(S.clus.ari)).attr("r", 6).attr("fill", C.s1).attr("stroke", "#fff").attr("stroke-width", 2);
    svg.append("line").attr("x1", x(kk)).attr("x2", x(kk)).attr("y1", y(S.clus.ari) + 7).attr("y2", y(0.24)).attr("stroke", C.ink).attr("stroke-opacity", 0.5);
    svg.append("text").attr("class", "lbl-strong").attr("x", x(kk) + 6).attr("y", y(0.24) + 4).text(`eigengap picks k = ${kk}`);
    svg.append("text").attr("class", "lbl").attr("x", w - m.right).attr("y", y(S.clus.ari_null_97_5) - 6).attr("text-anchor", "end").text("chance");
    crosshair(svg, w, h, m, x, ks, (i) => `k = <b>${ks[i]}</b>` + row(C.s1, "market removed", num(a[i].y)) + row(C.s3, "market left in", num(b[i].y)),
      { points: (i) => [[y(a[i].y), C.s1], [y(b[i].y), C.s3]] });
  });
  document.getElementById("cluster-grid").innerHTML = S.clus.clusters
    .slice().sort((p, r) => r.purity - p.purity || r.members.length - p.members.length)
    .map((c) => `<li><div class="ch"><b>${c.majority_sector}</b><span>${pct(c.purity, 0)} ${c.majority_sector}</span></div>
      <div class="purity"><i style="width:${c.purity * 100}%"></i></div><p class="members">${c.members.join(", ")}</p></li>`).join("");

  /* ---------- portfolios ---------- */
  const T = S.track;
  facts("facts-tracking", [
    ["Greedy, 10 stocks", pct(T.greedy_basket_10.te_test)],
    ["Random 10 stocks, median", pct(T.random_median_te_test["10"])],
    ["All stocks", pct(T.all_stocks_te_test)],
  ]);
  legend("legend-tracking", [[C.s1, "greedy selection"], [C.s2, "LASSO path"], [C.ref, "random baskets, median", "dot"]]);
  mount("chart-tracking", (svg, w, h) => {
    const greedy = T.greedy_te_test.map((v, i) => ({ x: i + 1, y: v }));
    const lasso = T.lasso_n_stocks.map((n, i) => ({ x: n, y: T.lasso_te_test[i] })).filter((d) => d.x <= 40);
    const rand = Object.entries(T.random_median_te_test).map(([k, v]) => ({ x: Number(k), y: v }));
    const x = d3.scaleLinear().domain([1, 40]);
    const y = logY([0.015, 0.6]);
    const m = frame(svg, w, h, x, y, { xTicks: [1, 5, 10, 20, 30, 40], yTicks: [0.02, 0.05, 0.1, 0.2, 0.5], yFormat: (d) => pct(d, 0), xLabel: "stocks in the basket", yLabel: "tracking error, annual" });
    svg.append("line").attr("x1", m.left).attr("x2", w - m.right).attr("y1", y(T.all_stocks_te_test)).attr("y2", y(T.all_stocks_te_test)).attr("stroke", C.ink).attr("stroke-opacity", 0.5);
    svg.append("text").attr("class", "lbl").attr("x", m.left + 6).attr("y", y(T.all_stocks_te_test) + 15).text(`all stocks ${pct(T.all_stocks_te_test)}`);
    path(svg, lasso, lineGen(x, y), C.s2);
    svg.append("g").selectAll("circle").data(lasso).join("circle").attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("r", 2.6).attr("fill", C.s2);
    path(svg, greedy, lineGen(x, y), C.s1, 2.5);
    svg.append("g").selectAll("circle").data(rand).join("circle").attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("r", 5).attr("fill", C.ref).attr("stroke", "#fff").attr("stroke-width", 2);
    const g10 = greedy[9];
    svg.append("circle").attr("cx", x(10)).attr("cy", y(g10.y)).attr("r", 6).attr("fill", C.s1).attr("stroke", "#fff").attr("stroke-width", 2);
    svg.append("text").attr("class", "lbl-strong").attr("x", x(10) - 9).attr("y", y(g10.y) + 4).attr("text-anchor", "end").text(`10 stocks: ${pct(g10.y)}`);
    const xs = d3.range(1, 41);
    const near = (arr, n) => arr.find((d) => d.x === n);
    crosshair(svg, w, h, m, x, xs, (i) => {
      const n = xs[i]; const L = near(lasso, n); const R = near(rand, n);
      return `<b>${n}</b> ${n === 1 ? "stock" : "stocks"}` + row(C.s1, "greedy", pct(greedy[i].y)) +
        (L ? row(C.s2, "LASSO", pct(L.y)) : "") + (R ? row(C.ref, "random median", pct(R.y)) : "");
    }, { points: (i) => [[y(greedy[i].y), C.s1]] });
  });
  const basket = Object.entries(T.greedy_basket_10.basket);
  const maxW = d3.max(basket, (b) => b[1]);
  document.getElementById("basket").innerHTML = `<div><h4>The ten-stock basket</h4><p>Chosen and weighted on 2021–2023, then left alone for two years. Tracking error ${pct(T.greedy_basket_10.te_train)} in sample, ${pct(T.greedy_basket_10.te_test)} out of sample.</p></div>
    <div class="basket-bars">${basket.map(([n, wgt]) => `<div><span>${n}</span><span class="bar"><i style="width:${(wgt / maxW) * 100}%"></i></span><span>${pct(wgt)}</span></div>`).join("")}</div>`;

  const est = ["Sample", "Ledoit-Wolf", "RMT clipping", "PCA factor model"];
  const estColors = [C.s3, C.s1, C.s2, C.s4];
  legend("legend-minvar", [...est.map((e, i) => [estColors[i], e]), [C.ink, "equal weight"]]);
  const lw84 = S.mv.unconstrained["Ledoit-Wolf"][0];
  facts("facts-minvar", [
    ["Ledoit-Wolf vs sample, 84-day window", signed(lw84.vol_vs_sample, 0)],
    ["Sample's realised ÷ predicted risk, 84 days", `${num(S.mv.unconstrained.Sample[0].bias_ratio, 1)}×`],
    ["Long-only, all estimators and windows", `${(d3.min(est.flatMap((e) => S.mv.long_only[e].map((r) => r.vol))) * 100).toFixed(1)}–${pct(d3.max(est.flatMap((e) => S.mv.long_only[e].map((r) => r.vol))))}`],
  ]);
  let constraint = "unconstrained";
  const mvChart = mount("chart-minvar", (svg, w, h) => {
    const win = S.mv.windows;
    const x = d3.scaleLog().domain([win[0], win[win.length - 1]]);
    const y = d3.scaleLinear().domain([0.095, 0.15]);
    const m = frame(svg, w, h, x, y, { xTicks: win, xFormat: (d) => `${d}`, yFormat: (d) => pct(d, 0), xLabel: "estimation window, trading days", yLabel: "realised volatility" });
    svg.append("line").attr("x1", m.left).attr("x2", w - m.right).attr("y1", y(S.mv.equal_weight_vol)).attr("y2", y(S.mv.equal_weight_vol)).attr("stroke", C.ink).attr("stroke-opacity", 0.55);
    svg.append("text").attr("class", "lbl").attr("x", w - m.right).attr("y", y(S.mv.equal_weight_vol) - 6).attr("text-anchor", "end").text(`equal weight ${pct(S.mv.equal_weight_vol)}`);
    const series = est.map((e) => S.mv[constraint][e].map((r) => ({ x: r.window, y: r.vol, r })));
    series.forEach((s, j) => {
      path(svg, s, lineGen(x, y), estColors[j], j === 1 ? 2.75 : 2);
      svg.append("g").selectAll("circle").data(s).join("circle").attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("r", 3.5).attr("fill", estColors[j]).attr("stroke", "#fff").attr("stroke-width", 1.5);
    });
    crosshair(svg, w, h, m, x, win, (i) => `<b>${win[i]}</b>-day window (q = ${num(S.mv.q[i])})` + est.map((e, j) => {
      const r = series[j][i].r;
      return row(estColors[j], e, `${pct(r.vol, 2)}${j ? ` (${signed(r.vol_vs_sample)}, ${pval(r.dm_p_vs_sample)})` : ""}`);
    }).join(""), { points: (i) => series.map((s, j) => [y(s[i].y), estColors[j]]) });
  });
  segmented(document.getElementById("chart-minvar").closest("figure"), "c", (v) => { constraint = v; mvChart.render(); });

  const SA = D.stat_arb_series;
  const saDates = SA.dates.map(parseDate);
  facts("facts-statarb", [
    ["Sharpe before costs", num(S.sa.performance_by_cost_bps["0"].sharpe)],
    ["Break-even cost", `${num(S.sa.breakeven_cost_bps)} bps`],
    ["NIFTY 50 buy and hold Sharpe", num(S.sa.nifty_buy_and_hold.sharpe)],
  ]);
  let cost = "0";
  const statLegend = () => {
    const p = S.sa.performance_by_cost_bps[cost];
    legend("legend-statarb", [[C.s1, `strategy: ${signed(p.annual_return)} a year, Sharpe ${num(p.sharpe)}, max drawdown ${pct(p.max_drawdown)}`], [C.ref, `NIFTY 50: ${signed(S.sa.nifty_buy_and_hold.annual_return)} a year`]]);
  };
  statLegend();
  const saChart = mount("chart-statarb", (svg, w, h) => {
    const strat = SA.equity[cost].map((v, i) => ({ x: saDates[i], y: v }));
    const nifty = SA.nifty.map((v, i) => ({ x: saDates[i], y: v }));
    const x = d3.scaleUtc().domain(d3.extent(saDates));
    const y = d3.scaleLinear().domain([0.6, 1.6]);
    const m = frame(svg, w, h, x, y, { yFormat: (d) => `₹${d.toFixed(1)}`, xTicks: Math.max(3, Math.floor(w / 120)), xFormat: d3.utcFormat("%Y") });
    svg.append("line").attr("x1", m.left).attr("x2", w - m.right).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", C.ink).attr("stroke-opacity", 0.4);
    path(svg, nifty, lineGen(x, y), C.ref, 1.75);
    path(svg, strat, lineGen(x, y), C.s1, 2.25);
    crosshair(svg, w, h, m, x, saDates, (i) => `${d3.utcFormat("%d %b %Y")(saDates[i])}` + row(C.s1, "strategy", `₹${strat[i].y.toFixed(3)}`) + row(C.ref, "NIFTY 50", `₹${nifty[i].y.toFixed(3)}`),
      { points: (i) => [[y(strat[i].y), C.s1], [y(nifty[i].y), C.ref]] });
  });
  segmented(document.getElementById("chart-statarb").closest("figure"), "cost", (v) => { cost = v; statLegend(); saChart.render(); });

  /* ---------- time series ---------- */
  const K = D.kalman_series;
  const kDates = K.dates.map(parseDate);
  const kNames = { MLE: "fitted model", "trend-following": "forced trend-following" };
  const kColors = { MLE: C.s1, "trend-following": C.s3 };
  legend("legend-kalman", Object.keys(kNames).map((k) => [kColors[k], kNames[k]]));
  const ex = S.kal.example;
  const kSeries = Object.entries(S.kal.series);
  const ratios = kSeries.map(([, s]) => s.oos_mle.rmse_ratio);
  facts("facts-kalman", [
    ["5-day error vs random walk", `${num(d3.min(ratios))}–${num(d3.max(ratios))}×`],
    ["Series significantly better at 5%", `${kSeries.filter(([, s]) => s.oos_mle.dm_p < 0.05 && s.oos_mle.rmse_ratio < 1).length} of ${kSeries.length}`],
    ["NIFTY slope standard error, fitted vs forced", `${pct(ex.MLE.final_slope_se_annual, 0)} vs ${pct(ex["trend-following"].final_slope_se_annual, 0)}`],
  ]);
  mount("chart-kalman", (svg, w, h) => {
    const x = d3.scaleUtc().domain(d3.extent(kDates));
    const y = d3.scaleLinear().domain([-0.8, 0.8]).clamp(true);
    const m = frame(svg, w, h, x, y, { yFormat: (d) => signed(d, 0), xTicks: Math.max(3, Math.floor(w / 120)), xFormat: d3.utcFormat("%Y"), yLabel: "slope, per year" });
    svg.append("line").attr("x1", m.left).attr("x2", w - m.right).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", C.ink).attr("stroke-opacity", 0.4);
    const order = ["trend-following", "MLE"];
    order.forEach((name) => {
      const t = K.tracks[name];
      const area = d3.area().x((_, i) => x(kDates[i])).y0((_, i) => y(t.slope[i] - 1.96 * t.se[i])).y1((_, i) => y(t.slope[i] + 1.96 * t.se[i]));
      svg.append("path").datum(t.slope).attr("d", area).attr("fill", kColors[name]).attr("fill-opacity", 0.13);
    });
    order.forEach((name) => {
      const t = K.tracks[name];
      path(svg, t.slope.map((v, i) => ({ x: kDates[i], y: v })), lineGen(x, y), kColors[name], name === "MLE" ? 2.5 : 1.75);
    });
    crosshair(svg, w, h, m, x, kDates, (i) => d3.utcFormat("%d %b %Y")(kDates[i]) + order.slice().reverse().map((n) =>
      row(kColors[n], kNames[n], `${signed(K.tracks[n].slope[i])} ± ${pct(1.96 * K.tracks[n].se[i])}`)).join(""),
    { points: (i) => order.map((n) => [y(K.tracks[n].slope[i]), kColors[n]]) });
  });
  mount("chart-kalman-oos", (svg, w, h) => {
    const rows = kSeries.map(([name, s]) => ({ name, ...s.oos_mle }));
    const x = d3.scaleLinear().domain([0.984, 1.012]);
    const y = d3.scaleBand().domain(rows.map((r) => r.name)).padding(0.2);
    const m = { top: 6, right: 18, bottom: 40, left: 96 };
    x.range([m.left, w - m.right]); y.range([m.top, h - m.bottom]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${h - m.bottom})`)
      .call(d3.axisBottom(x).tickValues([0.985, 0.99, 0.995, 1, 1.005, 1.01]).tickFormat((d) => `${d.toFixed(3)}×`).tickSize(-(h - m.top - m.bottom)).tickPadding(6))
      .call((g) => g.selectAll(".tick line").attr("stroke", "#eceee6"));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).tickSize(0).tickPadding(8)).call((g) => g.select("path").remove());
    svg.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", m.top).attr("y2", h - m.bottom).attr("stroke", C.ink).attr("stroke-width", 1.5);
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", (r) => x(r.rmse_ratio)).attr("cy", (r) => y(r.name) + y.bandwidth() / 2).attr("r", 5.5)
      .attr("fill", C.s1).attr("stroke", "#fff").attr("stroke-width", 1.5)
      .on("pointermove", (event, r) => showTip(event, `${r.name}<br>5-day RMSE <b>${num(r.rmse_ratio, 3)}×</b> the random walk<br>${pval(r.dm_p)}`))
      .on("pointerleave", hideTip);
    svg.append("text").attr("class", "lbl").attr("x", x(1) + 6).attr("y", h - m.bottom - 4).text("random walk");
    const best = rows.reduce((a, b) => (a.dm_p < b.dm_p ? a : b));
    svg.append("text").attr("class", "lbl").attr("x", x(best.rmse_ratio) - 10).attr("y", y(best.name) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").text(pval(best.dm_p));
    svg.append("text").attr("x", w - m.right).attr("y", h - 2).attr("text-anchor", "end").text("5-day RMSE ÷ random walk, 2023–2025");
  });

  const R = D.regimes_series;
  const rDates = R.dates.map(parseDate);
  const spans = [];
  R.turbulent.forEach((t, i) => {
    if (t && (i === 0 || !R.turbulent[i - 1])) spans.push([i, i]);
    if (t) spans[spans.length - 1][1] = i;
  });
  const [calm, turb] = S.reg.regimes;
  legend("legend-regimes", [[C.ink, "NIFTY 50, growth of ₹1"], [C.lime, "turbulent regime", "band"]]);
  facts("facts-regimes", [
    [`Calm volatility (${pct(calm.stationary_share, 0)} of the time)`, pct(calm.annual_vol)],
    [`Turbulent volatility (stays ${Math.round(turb.expected_duration_days)} days on average)`, pct(turb.annual_vol)],
    ["Market-mode share, calm → turbulent", `${num(calm.mean_market_mode_share)} → ${num(turb.mean_market_mode_share)}`],
  ]);
  const shade = (svg, x, m, h) => spans.forEach(([a, b]) => {
    const x0 = x(rDates[a]); const x1 = x(rDates[Math.min(b + 1, rDates.length - 1)]);
    svg.append("rect").attr("x", x0).attr("width", Math.max(1.5, x1 - x0)).attr("y", m.top).attr("height", h - m.top - m.bottom).attr("fill", C.lime);
  });
  const regimeTip = (i) => `${d3.utcFormat("%d %b %Y")(rDates[i])}<br>${R.turbulent[i] ? "<b>turbulent</b> regime" : "calm regime"}<br>NIFTY 50 ₹${R.nifty[i].toFixed(2)}` +
    (R.lam1[i] != null ? `<br>market-mode share ${num(R.lam1[i])}` : "");
  mount("chart-regimes", (svg, w, h) => {
    const x = d3.scaleUtc().domain(d3.extent(rDates));
    const y = d3.scaleLinear().domain([0.9, d3.max(R.nifty) * 1.04]);
    const m = M;
    x.range([m.left, w - m.right]); y.range([h - m.bottom, m.top]);
    shade(svg, x, m, h);
    frame(svg, w, h, x, y, { yFormat: (d) => `₹${d.toFixed(1)}`, xTicks: Math.max(3, Math.floor(w / 110)), xFormat: d3.utcFormat("%Y") });
    svg.selectAll(".grid").lower();
    path(svg, R.nifty.map((v, i) => ({ x: rDates[i], y: v })), lineGen(x, y), C.ink, 1.75);
    crosshair(svg, w, h, m, x, rDates, regimeTip, { points: (i) => [[y(R.nifty[i]), C.ink]] });
  });
  mount("chart-lam1", (svg, w, h) => {
    const x = d3.scaleUtc().domain(d3.extent(rDates));
    const vals = R.lam1.filter((v) => v != null);
    const y = d3.scaleLinear().domain([d3.min(vals) * 0.9, d3.max(vals) * 1.05]);
    const m = { ...M, bottom: 28 };
    x.range([m.left, w - m.right]); y.range([h - m.bottom, m.top]);
    shade(svg, x, m, h);
    frame(svg, w, h, x, y, { yTicks: 3, yFormat: d3.format(".2f"), xTicks: Math.max(3, Math.floor(w / 110)), xFormat: d3.utcFormat("%Y"), margin: { bottom: 28 } });
    svg.selectAll(".grid").lower();
    path(svg, R.lam1.map((v, i) => ({ x: rDates[i], y: v })), lineGen(x, y), C.s2, 2);
    crosshair(svg, w, h, m, x, rDates, regimeTip, { points: (i) => [[R.lam1[i] == null ? null : y(R.lam1[i]), C.s2]] });
  });
})();
