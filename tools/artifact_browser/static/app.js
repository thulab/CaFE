"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const MODEL_COLORS = [
  "#4f8cff",
  "#ffb000",
  "#00bfa6",
  "#e66fb5",
  "#a78bfa",
  "#f97316",
  "#42c978",
  "#38bdf8",
  "#f43f5e",
  "#a3acb9",
];
const DASH_PATTERNS = ["", "6 3", "2 3", "8 3 2 3", "4 2", "1 2", "7 2", "3 3", "9 3", "5 2 1 2"];

const elements = {
  dataset: document.querySelector("#dataset-select"),
  capability: document.querySelector("#capability-select"),
  sample: document.querySelector("#sample-select"),
  previous: document.querySelector("#previous-sample"),
  next: document.querySelector("#next-sample"),
  context: document.querySelector("#context-control"),
  channel: document.querySelector("#channel-select"),
  channelField: document.querySelector("#channel-field"),
  sharedScale: document.querySelector("#shared-scale"),
  modelLegend: document.querySelector("#model-legend"),
  selectAllModels: document.querySelector("#select-all-models"),
  selectNoModels: document.querySelector("#select-no-models"),
  selectionSummary: document.querySelector("#selection-summary"),
  chartMeta: document.querySelector("#chart-meta"),
  bestModelCallout: document.querySelector("#best-model-callout"),
  bestModelName: document.querySelector("#best-model-name"),
  bestModelScore: document.querySelector("#best-model-score"),
  contextBestCallout: document.querySelector("#context-best-callout"),
  contextBestLabel: document.querySelector("#context-best-label"),
  contextBestName: document.querySelector("#context-best-name"),
  contextBestScore: document.querySelector("#context-best-score"),
  chartGrid: document.querySelector("#chart-grid"),
  loading: document.querySelector("#loading-panel"),
  error: document.querySelector("#error-panel"),
  errorMessage: document.querySelector("#error-message"),
  retry: document.querySelector("#retry-button"),
  tooltip: document.querySelector("#chart-tooltip"),
  indexStatus: document.querySelector("#index-status"),
  indexDetail: document.querySelector("#index-detail"),
  liveRegion: document.querySelector("#live-region"),
  chartModal: document.querySelector("#chart-modal"),
  modalBackdrop: document.querySelector("#modal-backdrop"),
  modalClose: document.querySelector("#modal-close"),
  modalTitle: document.querySelector("#chart-modal-title"),
  modalChartHost: document.querySelector("#modal-chart-host"),
  modalChartMeta: document.querySelector("#modal-chart-meta"),
  experimentVersion: document.querySelector("#experiment-version"),
  experimentSubtitle: document.querySelector("#experiment-subtitle"),
};

const queryState = new URLSearchParams(window.location.search);
const state = {
  meta: null,
  groups: [],
  payload: null,
  datasetId: queryState.get("dataset"),
  capabilityId: queryState.get("capability"),
  groupId: queryState.get("sample"),
  context: Number(queryState.get("context")) || 504,
  channel: Number(queryState.get("target")) || 0,
  selectedModels: new Set(),
  sharedScale: queryState.get("scale") !== "local",
  sampleRequest: null,
  groupRequest: null,
  expandedIntensity: null,
  modalTrigger: null,
};

async function fetchJSON(path, signal) {
  const response = await fetch(path, { signal, headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function setOptions(select, options, selectedValue) {
  select.replaceChildren();
  for (const option of options) {
    const node = document.createElement("option");
    node.value = option.value;
    node.textContent = option.label;
    node.selected = option.value === selectedValue;
    select.append(node);
  }
}

function formatInteger(value) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

function formatValue(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const numeric = Number(value);
  if (Math.abs(numeric) >= 1000 || (Math.abs(numeric) > 0 && Math.abs(numeric) < 0.001)) {
    return numeric.toExponential(2);
  }
  return numeric.toFixed(digits);
}

function selectedDataset() {
  return state.meta?.datasets.find((dataset) => dataset.id === state.datasetId);
}

function modelColor(modelId) {
  const index = state.meta.models.findIndex((model) => model.id === modelId);
  return MODEL_COLORS[(index < 0 ? 0 : index) % MODEL_COLORS.length];
}

function modelDash(modelId) {
  const index = state.meta.models.findIndex((model) => model.id === modelId);
  return DASH_PATTERNS[(index < 0 ? 0 : index) % DASH_PATTERNS.length];
}

function initializeModelSelection() {
  const encoded = queryState.get("models");
  const available = new Set(state.meta.models.map((model) => model.id));
  if (encoded === "none") return;
  const requested = encoded ? encoded.split("|").filter((model) => available.has(model)) : [];
  state.selectedModels = new Set(requested.length ? requested : available);
}

function initializeControls() {
  const datasets = state.meta.datasets;
  if (!datasets.some((dataset) => dataset.id === state.datasetId)) {
    state.datasetId = datasets[0]?.id ?? null;
  }
  setOptions(
    elements.dataset,
    datasets.map((dataset) => ({ value: dataset.id, label: dataset.id })),
    state.datasetId,
  );
  elements.dataset.disabled = datasets.length === 0;
  refreshCapabilityOptions();

  if (!state.meta.contexts.includes(state.context)) {
    state.context = state.meta.contexts.at(-1);
  }
  elements.context.replaceChildren();
  for (const context of state.meta.contexts) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "segment-button";
    button.textContent = `L${context}`;
    button.dataset.context = String(context);
    button.setAttribute("aria-pressed", String(context === state.context));
    button.addEventListener("click", () => {
      if (state.context === context) return;
      state.context = context;
      for (const sibling of elements.context.children) {
        sibling.setAttribute("aria-pressed", String(Number(sibling.dataset.context) === context));
      }
      loadSample();
    });
    elements.context.append(button);
  }
  elements.sharedScale.checked = state.sharedScale;
}

function refreshCapabilityOptions() {
  const dataset = selectedDataset();
  const capabilities = dataset?.capabilities ?? [];
  if (!capabilities.some((capability) => capability.id === state.capabilityId)) {
    state.capabilityId = capabilities[0]?.id ?? null;
  }
  setOptions(
    elements.capability,
    capabilities.map((capability) => ({
      value: capability.id,
      label: `${capability.id} · ${capability.sampleCount} groups`,
    })),
    state.capabilityId,
  );
  elements.capability.disabled = capabilities.length === 0;
}

function updateIndexStatus() {
  const index = state.meta.index;
  elements.indexStatus.lastElementChild.textContent = `${formatInteger(index.groupCount)} ${index.groupUnit || "paired samples"}`;
  elements.indexDetail.textContent = `${formatInteger(index.sampleCount)} samples · ${formatInteger(index.predictionCount)} predictions indexed`;
  const experiment = state.meta.experiment;
  if (experiment) {
    elements.experimentVersion.textContent = `CAFE · ${String(experiment.version || "CaFE").toUpperCase()}`;
    elements.experimentSubtitle.textContent = `${experiment.id} · ${experiment.sampleScope || "main/primary"} · 同一种子的五档强度`;
  }
}

function groupOptionLabel(group, index) {
  const prefix = `样本 ${String(index + 1).padStart(3, "0")}`;
  if (Number.isInteger(group.seedIndex)) {
    const member = Number.isInteger(group.counterfactualMember)
      ? ` · member ${group.counterfactualMember}`
      : "";
    return `${prefix} · seed ${String(group.seedIndex).padStart(6, "0")}${member}`;
  }
  return `${prefix} · R${group.roundIndex} · ${group.analysisBlock}${String(group.analysisBlockIndex).padStart(3, "0")}`;
}

async function loadGroups({ preserveGroup = true } = {}) {
  state.groupRequest?.abort();
  state.groupRequest = new AbortController();
  elements.sample.disabled = true;
  elements.previous.disabled = true;
  elements.next.disabled = true;
  const params = new URLSearchParams({ dataset: state.datasetId, capability: state.capabilityId });
  try {
    const response = await fetchJSON(`/api/groups?${params}`, state.groupRequest.signal);
    state.groups = response.groups;
    if (!preserveGroup || !state.groups.some((group) => group.id === state.groupId)) {
      state.groupId = state.groups[0]?.id ?? null;
    }
    setOptions(
      elements.sample,
      state.groups.map((group, index) => ({
        value: group.id,
        label: groupOptionLabel(group, index),
      })),
      state.groupId,
    );
    elements.sample.disabled = state.groups.length === 0;
    updateStepper();
    await loadSample();
  } catch (error) {
    if (error.name !== "AbortError") showError(error);
  }
}

function updateStepper() {
  const index = state.groups.findIndex((group) => group.id === state.groupId);
  elements.previous.disabled = index <= 0;
  elements.next.disabled = index < 0 || index >= state.groups.length - 1;
}

function stepSample(delta) {
  const index = state.groups.findIndex((group) => group.id === state.groupId);
  const nextIndex = Math.max(0, Math.min(state.groups.length - 1, index + delta));
  if (nextIndex === index || !state.groups[nextIndex]) return;
  state.groupId = state.groups[nextIndex].id;
  elements.sample.value = state.groupId;
  updateStepper();
  loadSample();
}

function showLoading() {
  closeExpandedChart(false);
  elements.loading.hidden = false;
  elements.error.hidden = true;
  elements.chartGrid.hidden = true;
  elements.tooltip.hidden = true;
}

function showError(error) {
  elements.loading.hidden = true;
  elements.chartGrid.hidden = true;
  elements.error.hidden = false;
  elements.errorMessage.textContent = error.message || String(error);
  elements.liveRegion.textContent = `读取失败：${error.message || error}`;
}

async function loadSample() {
  if (!state.groupId) return;
  state.sampleRequest?.abort();
  state.sampleRequest = new AbortController();
  showLoading();
  const params = new URLSearchParams({ group: state.groupId, context: String(state.context) });
  try {
    state.payload = await fetchJSON(`/api/sample?${params}`, state.sampleRequest.signal);
    state.channel = Math.min(state.channel, state.payload.targetColumns.length - 1);
    refreshChannelOptions();
    updateSampleLabels();
    updateBestModelAnnotation();
    renderModelLegend();
    renderCharts();
    updateURL();
    elements.loading.hidden = true;
    elements.error.hidden = true;
    elements.chartGrid.hidden = false;
    const selected = state.payload.group.seedIndex ?? state.payload.group.poolIndex;
    elements.liveRegion.textContent = `已读取样本 ${selected}，五档强度。`;
  } catch (error) {
    if (error.name !== "AbortError") showError(error);
  }
}

function refreshChannelOptions() {
  const columns = state.payload.targetColumns;
  setOptions(
    elements.channel,
    columns.map((column, index) => ({ value: String(index), label: column })),
    String(state.channel),
  );
  elements.channelField.hidden = columns.length <= 1;
}

function updateSampleLabels() {
  const group = state.payload.group;
  const visibleIndex = state.groups.findIndex((item) => item.id === group.id) + 1;
  elements.selectionSummary.textContent = [
    group.datasetId,
    group.capabilityId,
    `sample ${visibleIndex}/${state.groups.length}`,
    Number.isInteger(group.seedIndex)
      ? `seed ${group.seedIndex}${Number.isInteger(group.counterfactualMember) ? ` · member ${group.counterfactualMember}` : ""}`
      : `seed pool ${group.poolIndex}`,
  ].join(" / ");
  elements.chartMeta.textContent = `L${state.context} + H${state.payload.horizon} · target dim ${group.targetDim} · season ${group.seasonLength}${group.frequency}`;
}

function updateBestModelAnnotation() {
  const ranking = state.payload?.oracleContextRanking;
  if (!ranking?.best) {
    elements.bestModelCallout.hidden = true;
  } else {
    const best = ranking.best;
    elements.bestModelName.textContent = best.modelId;
    elements.bestModelScore.textContent = `MASE ${formatValue(best.maseMean, 4)} · n=${formatInteger(best.sampleCount)}`;
    elements.bestModelCallout.title = [
      `${state.payload.group.datasetId} / ${state.payload.group.capabilityId}`,
      "每个样本先选择 MASE 最低的 context，再对五档 intensity 的全部样本求均值。",
      ranking.runnerUp ? `第二名 ${ranking.runnerUp.modelId}，差值 ${formatValue(ranking.gapToRunnerUp, 4)}。` : "",
    ].filter(Boolean).join("\n");
    elements.bestModelCallout.hidden = false;
  }

  const intensityFiveRanking = rankCurrentSampleModels(5);
  if (!intensityFiveRanking.length) {
    elements.contextBestCallout.hidden = true;
    return;
  }
  const intensityBest = intensityFiveRanking[0];
  const intensityRunnerUp = intensityFiveRanking[1];
  elements.contextBestLabel.textContent = `Current sample · L${state.context} · Intensity 5`;
  elements.contextBestName.textContent = intensityBest.modelId;
  elements.contextBestScore.textContent = `MASE ${formatValue(intensityBest.maseMean, 4)}`;
  elements.contextBestCallout.title = [
    "当前 seed group，仅比较 intensity=5 在所选 context 下的 MASE。",
    intensityRunnerUp ? `第二名 ${intensityRunnerUp.modelId}，差值 ${formatValue(intensityRunnerUp.maseMean - intensityBest.maseMean, 4)}。` : "",
  ].filter(Boolean).join("\n");
  elements.contextBestCallout.hidden = false;
}

function meanModelMetric(modelId, metricName = "mase") {
  if (!state.payload) return null;
  const values = state.payload.intensities
    .map((intensity) => intensity.models[modelId]?.metrics?.[metricName])
    .filter((value) => Number.isFinite(Number(value)))
    .map(Number);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function rankCurrentSampleModels(intensityNumber = null) {
  if (!state.payload || !state.meta) return [];
  return state.meta.models
    .filter((model) => model.kind === "model")
    .map((model) => {
      const sourceIntensities = intensityNumber === null
        ? state.payload.intensities
        : state.payload.intensities.filter((item) => item.intensity === intensityNumber);
      const values = sourceIntensities
        .map((intensity) => intensity.models[model.id]?.metrics?.mase)
        .filter((value) => Number.isFinite(Number(value)))
        .map(Number);
      return {
        modelId: model.id,
        maseMean: values.length
          ? values.reduce((sum, value) => sum + value, 0) / values.length
          : null,
        sampleCount: values.length,
      };
    })
    .filter((model) => model.maseMean !== null)
    .sort((left, right) => left.maseMean - right.maseMean || left.modelId.localeCompare(right.modelId));
}

function renderModelLegend() {
  if (!state.meta) return;
  elements.modelLegend.replaceChildren();
  const sampleRanking = rankCurrentSampleModels();
  const bestModelId = sampleRanking[0]?.modelId;
  for (const model of state.meta.models) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-chip";
    if (model.id === bestModelId) button.classList.add("is-best-model");
    button.setAttribute("aria-pressed", String(state.selectedModels.has(model.id)));
    button.setAttribute("aria-label", `${state.selectedModels.has(model.id) ? "隐藏" : "显示"} ${model.id}`);
    const key = document.createElement("span");
    key.className = "model-line-key";
    key.style.setProperty("--model-color", modelColor(model.id));
    const label = document.createElement("span");
    label.textContent = model.id;
    const metric = document.createElement("span");
    metric.className = "chip-metric";
    metric.textContent = formatValue(meanModelMetric(model.id, "mase"));
    button.append(key, label, metric);
    if (model.id === bestModelId) {
      const bestLabel = document.createElement("span");
      bestLabel.className = "best-chip-label";
      bestLabel.textContent = "BEST";
      button.append(bestLabel);
      button.title = `当前样本、L${state.context} 下五档 intensity 平均 MASE 最低：${formatValue(sampleRanking[0].maseMean, 4)}`;
    }
    button.addEventListener("click", () => {
      if (state.selectedModels.has(model.id)) state.selectedModels.delete(model.id);
      else state.selectedModels.add(model.id);
      renderModelLegend();
      renderCharts();
      updateURL();
    });
    elements.modelLegend.append(button);
  }
}

function valuesForScale(intensities) {
  const values = [];
  for (const intensity of intensities) {
    for (const row of intensity.history) values.push(row[state.channel]);
    for (const row of intensity.actual) values.push(row[state.channel]);
    for (const modelId of state.selectedModels) {
      for (const row of intensity.models[modelId]?.forecast ?? []) values.push(row[state.channel]);
    }
  }
  return values.filter(Number.isFinite);
}

function extent(values) {
  if (!values.length) return [-1, 1];
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    const padding = Math.abs(minimum) * 0.1 || 1;
    return [minimum - padding, maximum + padding];
  }
  const padding = (maximum - minimum) * 0.08;
  return [minimum - padding, maximum + padding];
}

function svgElement(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [attribute, value] of Object.entries(attributes)) {
    node.setAttribute(attribute, String(value));
  }
  return node;
}

function linePath(values, channel, xForIndex, yForValue, indexOffset = 0) {
  let path = "";
  let drawing = false;
  values.forEach((row, index) => {
    const value = Number(row[channel]);
    if (!Number.isFinite(value)) {
      drawing = false;
      return;
    }
    path += `${drawing ? "L" : "M"}${xForIndex(index + indexOffset).toFixed(2)},${yForValue(value).toFixed(2)}`;
    drawing = true;
  });
  return path;
}

function forecastPath(history, forecast, channel, xForIndex, yForValue) {
  const start = history.at(-1)?.[channel];
  const values = Number.isFinite(Number(start)) ? [[Number(start)], ...forecast.map((row) => [row[channel]])] : forecast.map((row) => [row[channel]]);
  const offset = Number.isFinite(Number(start)) ? history.length - 1 : history.length;
  return linePath(values, 0, xForIndex, yForValue, offset);
}

function addText(svg, text, attributes) {
  const node = svgElement("text", attributes);
  node.textContent = text;
  svg.append(node);
  return node;
}

function pointerXInViewBox(svg, clientX, viewWidth, viewHeight) {
  const bounds = svg.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) return 0;
  const scale = Math.min(bounds.width / viewWidth, bounds.height / viewHeight);
  const renderedWidth = viewWidth * scale;
  const horizontalInset = (bounds.width - renderedWidth) / 2;
  return (clientX - bounds.left - horizontalInset) / scale;
}

function createChart(intensity, yExtent, { expanded = false } = {}) {
  const figure = document.createElement("figure");
  figure.className = "chart-card";
  const header = document.createElement("figcaption");
  header.className = "chart-card-header";
  const titleBlock = document.createElement("div");
  const title = document.createElement("h3");
  title.className = "intensity-title";
  title.append("Intensity ");
  const number = document.createElement("span");
  number.className = "intensity-number";
  number.textContent = String(intensity.intensity);
  title.append(number);
  const subtitle = document.createElement("p");
  subtitle.className = "intensity-subtitle";
  subtitle.textContent = `relative ${formatValue(intensity.targetRelativeLevel, 2)}`;
  titleBlock.append(title, subtitle);
  const strengthBlock = document.createElement("div");
  strengthBlock.className = "strength-block";
  const strengthLabel = document.createElement("span");
  strengthLabel.className = "strength-label";
  strengthLabel.textContent = intensity.targetFeature || "strength";
  const strengthValue = document.createElement("p");
  strengthValue.className = "strength-value";
  const realized = intensity.realizedFeature ?? intensity.targetStrength;
  strengthValue.textContent = formatValue(realized, 4);
  strengthBlock.append(strengthLabel, strengthValue);
  const headerActions = document.createElement("div");
  headerActions.className = "chart-header-actions";
  headerActions.append(strengthBlock);
  if (!expanded) {
    const expandButton = document.createElement("button");
    expandButton.type = "button";
    expandButton.className = "expand-chart-button";
    expandButton.setAttribute("aria-label", `放大查看 Intensity ${intensity.intensity}`);
    expandButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4H4v5m11-5h5v5M9 20H4v-5m11 5h5v-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    expandButton.addEventListener("click", () => openExpandedChart(intensity.intensity, expandButton));
    headerActions.append(expandButton);
  }
  header.append(titleBlock, headerActions);

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-wrap";
  const width = 360;
  const height = 230;
  const margin = { top: 16, right: 10, bottom: 22, left: 12 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const historyLength = intensity.history.length;
  const horizon = intensity.actual.length;
  const totalLength = historyLength + horizon;
  const xForIndex = (index) => margin.left + (index / Math.max(1, totalLength - 1)) * innerWidth;
  const yForValue = (value) => margin.top + ((yExtent[1] - value) / (yExtent[1] - yExtent[0])) * innerHeight;
  const boundaryX = xForIndex(historyLength - 0.5);
  const svg = svgElement("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Intensity ${intensity.intensity} 的历史、真实未来和模型预测曲线`,
  });
  svg.append(svgElement("rect", {
    class: "forecast-zone",
    x: boundaryX,
    y: margin.top,
    width: width - margin.right - boundaryX,
    height: innerHeight,
  }));
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + (innerHeight * index) / 4;
    svg.append(svgElement("line", { class: "chart-grid-line", x1: margin.left, y1: y, x2: width - margin.right, y2: y }));
  }
  svg.append(svgElement("line", {
    class: "forecast-boundary",
    x1: boundaryX,
    y1: margin.top,
    x2: boundaryX,
    y2: margin.top + innerHeight,
  }));
  addText(svg, "FORECAST", { class: "boundary-label", x: boundaryX + 5, y: margin.top + 9 });
  addText(svg, formatValue(yExtent[1], 2), { class: "axis-label", x: margin.left + 2, y: margin.top + 9 });
  addText(svg, formatValue(yExtent[0], 2), { class: "axis-label", x: margin.left + 2, y: margin.top + innerHeight - 4 });
  addText(svg, `−${historyLength}`, { class: "axis-label", x: margin.left, y: height - 6 });
  addText(svg, "0", { class: "axis-label", x: boundaryX - 3, y: height - 6 });
  addText(svg, `+${horizon}`, { class: "axis-label", x: width - margin.right - 18, y: height - 6 });

  svg.append(svgElement("path", {
    class: "chart-series history-series",
    d: linePath(intensity.history, state.channel, xForIndex, yForValue),
  }));
  svg.append(svgElement("path", {
    class: "chart-series actual-series",
    d: forecastPath(intensity.history, intensity.actual, state.channel, xForIndex, yForValue),
  }));
  for (const model of state.meta.models) {
    if (!state.selectedModels.has(model.id)) continue;
    const forecast = intensity.models[model.id]?.forecast;
    if (!forecast) continue;
    const path = svgElement("path", {
      class: "chart-series model-series",
      d: forecastPath(intensity.history, forecast, state.channel, xForIndex, yForValue),
      stroke: modelColor(model.id),
    });
    const dash = modelDash(model.id);
    if (dash) path.setAttribute("stroke-dasharray", dash);
    svg.append(path);
  }
  const hoverLine = svgElement("line", {
    class: "hover-line",
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + innerHeight,
    visibility: "hidden",
  });
  svg.append(hoverLine);
  const hoverTarget = svgElement("rect", {
    class: "hover-target",
    x: margin.left,
    y: margin.top,
    width: innerWidth,
    height: innerHeight,
  });
  hoverTarget.addEventListener("pointermove", (event) => {
    const svgX = pointerXInViewBox(svg, event.clientX, width, height);
    const index = Math.max(0, Math.min(totalLength - 1, Math.round(((svgX - margin.left) / innerWidth) * (totalLength - 1))));
    const x = xForIndex(index);
    hoverLine.setAttribute("x1", x);
    hoverLine.setAttribute("x2", x);
    hoverLine.setAttribute("visibility", "visible");
    showTooltip(event, intensity, index, historyLength);
  });
  hoverTarget.addEventListener("pointerleave", () => {
    hoverLine.setAttribute("visibility", "hidden");
    elements.tooltip.hidden = true;
  });
  svg.append(hoverTarget);
  chartWrap.append(svg);

  const footer = document.createElement("div");
  footer.className = "chart-card-footer";
  const sampleId = document.createElement("span");
  sampleId.textContent = Number.isInteger(state.payload.group.seedIndex)
    ? `seed ${state.payload.group.seedIndex}${Number.isInteger(state.payload.group.counterfactualMember) ? ` · member ${state.payload.group.counterfactualMember}` : ""}`
    : `R${state.payload.group.roundIndex} · pool ${state.payload.group.poolIndex}`;
  const maes = [...state.selectedModels]
    .map((modelId) => intensity.models[modelId]?.metrics?.mae)
    .filter((value) => Number.isFinite(Number(value)))
    .map(Number);
  const metric = document.createElement("span");
  metric.textContent = maes.length ? `MAE ${formatValue(Math.min(...maes))}–${formatValue(Math.max(...maes))}` : "model hidden";
  footer.append(sampleId, metric);
  figure.append(header, chartWrap, footer);
  return figure;
}

function renderCharts() {
  if (!state.payload) return;
  elements.tooltip.hidden = true;
  elements.chartGrid.replaceChildren();
  const sharedExtent = extent(valuesForScale(state.payload.intensities));
  for (const intensity of state.payload.intensities) {
    const yExtent = state.sharedScale ? sharedExtent : extent(valuesForScale([intensity]));
    elements.chartGrid.append(createChart(intensity, yExtent));
  }
  if (!elements.chartModal.hidden) renderExpandedChart();
}

function openExpandedChart(intensityNumber, trigger) {
  state.expandedIntensity = intensityNumber;
  state.modalTrigger = trigger;
  elements.chartModal.hidden = false;
  document.body.classList.add("modal-open");
  renderExpandedChart();
  elements.modalClose.focus();
}

function closeExpandedChart(restoreFocus = true) {
  if (elements.chartModal.hidden) return;
  elements.chartModal.hidden = true;
  elements.modalChartHost.replaceChildren();
  elements.tooltip.hidden = true;
  document.body.classList.remove("modal-open");
  const trigger = state.modalTrigger;
  state.expandedIntensity = null;
  state.modalTrigger = null;
  if (restoreFocus && trigger?.isConnected) trigger.focus();
}

function renderExpandedChart() {
  if (!state.payload || state.expandedIntensity === null) return;
  const intensity = state.payload.intensities.find(
    (item) => item.intensity === state.expandedIntensity,
  );
  if (!intensity) {
    closeExpandedChart(false);
    return;
  }
  const yExtent = state.sharedScale
    ? extent(valuesForScale(state.payload.intensities))
    : extent(valuesForScale([intensity]));
  elements.modalTitle.textContent = `Intensity ${intensity.intensity} · 放大曲线`;
  elements.modalChartMeta.textContent = [
    state.payload.group.datasetId,
    state.payload.group.capabilityId,
    `L${state.context} + H${state.payload.horizon}`,
    state.payload.targetColumns[state.channel],
  ].join(" / ");
  elements.modalChartHost.replaceChildren(
    createChart(intensity, yExtent, { expanded: true }),
  );
}

function tooltipRow(label, value, color) {
  const row = document.createElement("div");
  row.className = "tooltip-row";
  const swatch = document.createElement("span");
  swatch.className = "tooltip-swatch";
  swatch.style.setProperty("--swatch", color);
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("span");
  number.className = "tooltip-value";
  number.textContent = formatValue(value, 4);
  row.append(swatch, name, number);
  return row;
}

function showTooltip(event, intensity, index, historyLength) {
  const tooltip = elements.tooltip;
  tooltip.replaceChildren();
  const heading = document.createElement("div");
  heading.className = "tooltip-heading";
  const phase = document.createElement("span");
  const step = document.createElement("span");
  if (index < historyLength) {
    phase.textContent = `Intensity ${intensity.intensity} · history`;
    step.textContent = `t ${index - historyLength}`;
    heading.append(phase, step);
    tooltip.append(heading, tooltipRow("历史", intensity.history[index]?.[state.channel], "#7d899a"));
  } else {
    const forecastIndex = index - historyLength;
    phase.textContent = `Intensity ${intensity.intensity} · forecast`;
    step.textContent = `t +${forecastIndex + 1}`;
    heading.append(phase, step);
    tooltip.append(heading, tooltipRow("真实未来", intensity.actual[forecastIndex]?.[state.channel], "#f7f9fc"));
    for (const model of state.meta.models) {
      if (!state.selectedModels.has(model.id)) continue;
      const value = intensity.models[model.id]?.forecast?.[forecastIndex]?.[state.channel];
      if (value !== undefined) tooltip.append(tooltipRow(model.id, value, modelColor(model.id)));
    }
  }
  tooltip.hidden = false;
  const padding = 12;
  const bounds = tooltip.getBoundingClientRect();
  const left = Math.min(window.innerWidth - bounds.width - padding, event.clientX + 14);
  const top = Math.min(window.innerHeight - bounds.height - padding, event.clientY + 14);
  tooltip.style.left = `${Math.max(padding, left)}px`;
  tooltip.style.top = `${Math.max(padding, top)}px`;
}

function updateURL() {
  const params = new URLSearchParams();
  params.set("dataset", state.datasetId);
  params.set("capability", state.capabilityId);
  params.set("sample", state.groupId);
  params.set("context", String(state.context));
  if (state.channel) params.set("target", String(state.channel));
  if (!state.sharedScale) params.set("scale", "local");
  const allModels = state.meta.models.map((model) => model.id);
  if (state.selectedModels.size === 0) params.set("models", "none");
  else if (state.selectedModels.size !== allModels.length) {
    params.set("models", allModels.filter((model) => state.selectedModels.has(model)).join("|"));
  }
  history.replaceState(null, "", `${window.location.pathname}?${params}`);
}

function bindEvents() {
  elements.dataset.addEventListener("change", () => {
    state.datasetId = elements.dataset.value;
    state.capabilityId = null;
    state.groupId = null;
    refreshCapabilityOptions();
    loadGroups({ preserveGroup: false });
  });
  elements.capability.addEventListener("change", () => {
    state.capabilityId = elements.capability.value;
    state.groupId = null;
    loadGroups({ preserveGroup: false });
  });
  elements.sample.addEventListener("change", () => {
    state.groupId = elements.sample.value;
    updateStepper();
    loadSample();
  });
  elements.previous.addEventListener("click", () => stepSample(-1));
  elements.next.addEventListener("click", () => stepSample(1));
  elements.channel.addEventListener("change", () => {
    state.channel = Number(elements.channel.value);
    renderCharts();
    updateURL();
  });
  elements.sharedScale.addEventListener("change", () => {
    state.sharedScale = elements.sharedScale.checked;
    renderCharts();
    updateURL();
  });
  elements.selectAllModels.addEventListener("click", () => {
    state.selectedModels = new Set(state.meta.models.map((model) => model.id));
    renderModelLegend();
    renderCharts();
    updateURL();
  });
  elements.selectNoModels.addEventListener("click", () => {
    state.selectedModels.clear();
    renderModelLegend();
    renderCharts();
    updateURL();
  });
  elements.retry.addEventListener("click", loadSample);
  elements.modalClose.addEventListener("click", () => closeExpandedChart());
  elements.modalBackdrop.addEventListener("click", () => closeExpandedChart());
  document.addEventListener("keydown", (event) => {
    if (!elements.chartModal.hidden) {
      if (event.key === "Escape") {
        event.preventDefault();
        closeExpandedChart();
      } else if (event.key === "Tab") {
        event.preventDefault();
        elements.modalClose.focus();
      }
      return;
    }
    const target = event.target;
    if (target instanceof HTMLSelectElement || target instanceof HTMLInputElement || target instanceof HTMLButtonElement) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepSample(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      stepSample(1);
    }
  });
  window.addEventListener("scroll", () => { elements.tooltip.hidden = true; }, { passive: true });
  window.addEventListener("resize", () => { elements.tooltip.hidden = true; }, { passive: true });
}

async function initialize() {
  bindEvents();
  showLoading();
  try {
    state.meta = await fetchJSON("/api/meta");
    initializeModelSelection();
    initializeControls();
    updateIndexStatus();
    renderModelLegend();
    await loadGroups();
  } catch (error) {
    showError(error);
    elements.indexStatus.lastElementChild.textContent = "索引不可用";
  }
}

initialize();
