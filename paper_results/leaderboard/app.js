(() => {
  "use strict";

  const data = window.CAFE_LEADERBOARD_DATA;
  if (!data) {
    document.body.innerHTML = "<p>Leaderboard data could not be loaded.</p>";
    return;
  }

  const state = {
    suite: "all",
    overallMetric: "paired_nrmse",
    capabilityMetric: "paired_nrmse",
    capabilitySort: "average",
  };

  const metricIds = ["reference_mase", "probe_mase", "paired_nrmse"];
  const capabilityMetricIds = ["probe_mase", "paired_nrmse"];
  const byId = (id) => document.getElementById(id);
  const suiteSelect = byId("suite-select");
  const metricSwitch = byId("overall-metric-switch");
  const capabilityMetricSelect = byId("capability-metric-select");
  const capabilitySortSelect = byId("capability-sort-select");
  const overallTable = byId("overall-table");
  const capabilityTable = byId("capability-table");

  const labelForSuite = (id) => data.suites.find((suite) => suite.id === id)?.label ?? id;
  const labelForCapability = (id) =>
    data.capabilities.find((capability) => capability.id === id)?.label ?? id;

  function formatScore(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return value < 1 ? value.toFixed(4) : value.toFixed(3);
  }

  function makeOption(value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }

  function initializeControls() {
    data.suites.forEach((suite) => suiteSelect.append(makeOption(suite.id, suite.label)));
    metricIds.forEach((metricId) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.metric = metricId;
      button.textContent = data.metrics[metricId].shortLabel;
      button.setAttribute("aria-pressed", String(metricId === state.overallMetric));
      button.title = data.metrics[metricId].description;
      button.addEventListener("click", () => {
        state.overallMetric = metricId;
        render();
      });
      metricSwitch.append(button);
    });
    capabilityMetricIds.forEach((metricId) =>
      capabilityMetricSelect.append(makeOption(metricId, data.metrics[metricId].label)),
    );
    capabilitySortSelect.append(makeOption("average", "Capability average"));
    data.capabilities.forEach((capability) =>
      capabilitySortSelect.append(makeOption(capability.id, capability.label)),
    );
    suiteSelect.value = state.suite;
    capabilityMetricSelect.value = state.capabilityMetric;
    capabilitySortSelect.value = state.capabilitySort;

    suiteSelect.addEventListener("change", () => {
      state.suite = suiteSelect.value;
      render();
    });
    capabilityMetricSelect.addEventListener("change", () => {
      state.capabilityMetric = capabilityMetricSelect.value;
      renderCapability();
      announce();
    });
    capabilitySortSelect.addEventListener("change", () => {
      state.capabilitySort = capabilitySortSelect.value;
      renderCapability();
      announce();
    });
  }

  function sortedOverallRows() {
    return [...data.overall[state.suite]].sort((left, right) => {
      const a = left.values[state.overallMetric];
      const b = right.values[state.overallMetric];
      if (a === null) return 1;
      if (b === null) return -1;
      return a - b || left.model.localeCompare(right.model);
    });
  }

  function overallHeader() {
    const cells = ["Rank", "Model"];
    const header = document.createElement("tr");
    cells.forEach((label) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = label;
      header.append(th);
    });
    metricIds.forEach((metricId) => {
      const th = document.createElement("th");
      th.scope = "col";
      if (metricId === state.overallMetric) th.classList.add("metric-active");
      const button = document.createElement("button");
      button.className = "sort-button";
      button.type = "button";
      button.textContent = data.metrics[metricId].shortLabel;
      button.title = data.metrics[metricId].description;
      button.setAttribute("aria-current", String(metricId === state.overallMetric));
      button.addEventListener("click", () => {
        state.overallMetric = metricId;
        render();
      });
      th.append(button);
      header.append(th);
    });
    const coverage = document.createElement("th");
    coverage.scope = "col";
    coverage.textContent = "Suite coverage";
    header.append(coverage);
    return header;
  }

  function renderOverall() {
    const rows = sortedOverallRows();
    overallTable.tHead.replaceChildren(overallHeader());
    const body = document.createDocumentFragment();
    const minima = Object.fromEntries(
      metricIds.map((metricId) => [
        metricId,
        Math.min(...rows.map((row) => row.values[metricId]).filter(Number.isFinite)),
      ]),
    );
    rows.forEach((row, index) => {
      const tr = document.createElement("tr");
      const rank = document.createElement("td");
      rank.className = "rank";
      rank.textContent = String(index + 1).padStart(2, "0");
      tr.append(rank);
      const model = document.createElement("th");
      model.scope = "row";
      model.className = "model-cell";
      model.textContent = row.model;
      tr.append(model);
      metricIds.forEach((metricId) => {
        const td = document.createElement("td");
        const value = row.values[metricId];
        td.className = "numeric";
        if (metricId === state.overallMetric) td.classList.add("metric-active");
        if (value === minima[metricId]) {
          td.classList.add("best-cell");
          td.title = `Best ${data.metrics[metricId].label}`;
        }
        if (value === null) td.classList.add("missing");
        td.textContent = formatScore(value);
        tr.append(td);
      });
      const coverage = document.createElement("td");
      coverage.className = "coverage numeric";
      coverage.textContent = `${row.coverage[state.overallMetric]}/${row.suiteCount}`;
      tr.append(coverage);
      body.append(tr);
    });
    overallTable.tBodies[0].replaceChildren(body);
    metricSwitch.querySelectorAll("button").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.metric === state.overallMetric));
    });
    byId("overall-note").textContent =
      state.suite === "all"
        ? "Overall scores are suite-equal means over available suites; coverage is shown for the active metric."
        : "This view reports the frozen task-equal score for the selected benchmark suite.";
  }

  function capabilityValue(row, capabilityId) {
    return capabilityId === "average" ? row.average : row.scores[capabilityId];
  }

  function sortedCapabilityRows() {
    return [...data.capability[state.suite][state.capabilityMetric]].sort((left, right) => {
      const a = capabilityValue(left, state.capabilitySort);
      const b = capabilityValue(right, state.capabilitySort);
      if (a === null) return 1;
      if (b === null) return -1;
      return a - b || left.model.localeCompare(right.model);
    });
  }

  function capabilityHeader() {
    const header = document.createElement("tr");
    ["Rank", "Model"].forEach((label) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = label;
      header.append(th);
    });
    data.capabilities.forEach((capability) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.className = "cap-header";
      if (capability.id === state.capabilitySort) th.classList.add("metric-active");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sort-button";
      button.textContent = capability.label;
      button.setAttribute("aria-current", String(capability.id === state.capabilitySort));
      button.addEventListener("click", () => {
        state.capabilitySort = capability.id;
        capabilitySortSelect.value = capability.id;
        renderCapability();
        announce();
      });
      th.append(button);
      header.append(th);
    });
    const average = document.createElement("th");
    average.scope = "col";
    if (state.capabilitySort === "average") average.classList.add("metric-active");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sort-button";
    button.textContent = "Average";
    button.setAttribute("aria-current", String(state.capabilitySort === "average"));
    button.addEventListener("click", () => {
      state.capabilitySort = "average";
      capabilitySortSelect.value = "average";
      renderCapability();
      announce();
    });
    average.append(button);
    header.append(average);
    const coverage = document.createElement("th");
    coverage.scope = "col";
    coverage.textContent = "Suite coverage";
    header.append(coverage);
    return header;
  }

  function renderCapability() {
    const rows = sortedCapabilityRows();
    capabilityTable.tHead.replaceChildren(capabilityHeader());
    const columnIds = [...data.capabilities.map((capability) => capability.id), "average"];
    const ranges = Object.fromEntries(
      columnIds.map((columnId) => {
        const values = rows.map((row) => capabilityValue(row, columnId)).filter(Number.isFinite);
        return [columnId, { min: Math.min(...values), max: Math.max(...values) }];
      }),
    );
    const body = document.createDocumentFragment();
    rows.forEach((row, index) => {
      const tr = document.createElement("tr");
      const rank = document.createElement("td");
      rank.className = "rank";
      rank.textContent = String(index + 1).padStart(2, "0");
      tr.append(rank);
      const model = document.createElement("th");
      model.scope = "row";
      model.className = "model-cell";
      model.textContent = row.model;
      tr.append(model);
      columnIds.forEach((columnId) => {
        const value = capabilityValue(row, columnId);
        const td = document.createElement("td");
        td.className = "numeric heat-cell";
        if (columnId === state.capabilitySort) td.classList.add("metric-active");
        if (value === null) {
          td.classList.add("missing");
        } else {
          const { min, max } = ranges[columnId];
          const normalized = max === min ? 0 : (value - min) / (max - min);
          const lightness = 88 + normalized * 9;
          td.style.setProperty("--heat", `hsl(174 30% ${lightness.toFixed(1)}%)`);
          if (value === min) {
            td.classList.add("best-cell");
            td.title = `Best ${columnId === "average" ? "capability average" : labelForCapability(columnId)}`;
          }
        }
        td.textContent = formatScore(value);
        tr.append(td);
      });
      const coverage = document.createElement("td");
      coverage.className = "coverage numeric";
      const counts = Object.values(row.coverage).filter(Number.isFinite);
      const minimum = Math.min(...counts);
      const maximum = Math.max(...counts);
      coverage.textContent =
        minimum === maximum
          ? `${minimum}/${row.suiteCount}`
          : `${minimum}–${maximum}/${row.suiteCount}`;
      tr.append(coverage);
      body.append(tr);
    });
    capabilityTable.tBodies[0].replaceChildren(body);
  }

  function renderSummary() {
    const rows = sortedOverallRows();
    const leader = rows.find((row) => Number.isFinite(row.values[state.overallMetric]));
    byId("summary-leader").textContent = leader?.model ?? "No result";
    byId("summary-leader-metric").textContent = leader
      ? `${data.metrics[state.overallMetric].shortLabel} ${formatScore(leader.values[state.overallMetric])}` +
        `${state.suite === "all" ? ` · ${leader.coverage[state.overallMetric]}/${leader.suiteCount} suites` : ""}`
      : data.metrics[state.overallMetric].shortLabel;
    byId("summary-suite").textContent = labelForSuite(state.suite);
    byId("summary-coverage").textContent =
      state.suite === "all" ? "Four frozen benchmark suites" : "One frozen benchmark suite";
  }

  function announce() {
    byId("live-status").textContent =
      `${labelForSuite(state.suite)}. Overall ranked by ${data.metrics[state.overallMetric].label}. ` +
      `Capability table uses ${data.metrics[state.capabilityMetric].label} and ranks by ` +
      `${state.capabilitySort === "average" ? "capability average" : labelForCapability(state.capabilitySort)}.`;
  }

  function render() {
    renderOverall();
    renderCapability();
    renderSummary();
    announce();
  }

  initializeControls();
  render();
})();
