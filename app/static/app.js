const modelSelect = document.getElementById("modelSelect");
const form = document.getElementById("generateForm");
const seedFileInput = document.getElementById("seedFile");
const dropZone = document.getElementById("dropZone");
const uploadLabel = document.getElementById("uploadLabel");
const statusBox = document.getElementById("statusBox");
const downloadBtn = document.getElementById("downloadBtn");
const generateBtn = document.getElementById("generateBtn");
const assessmentGrid = document.getElementById("assessmentGrid");
const assessmentSummary = document.getElementById("assessmentSummary");
const readyBadge = document.getElementById("readyBadge");
const tokenizerBadge = document.getElementById("tokenizerBadge");
const seedAudio = document.getElementById("seedAudio");
const outputAudio = document.getElementById("outputAudio");
const midiMapImage = document.getElementById("midiMapImage");
const previewPanel = document.getElementById("previewPanel");

const sliders = [
  ["temperature", "temperatureVal", (v) => Number(v).toFixed(2)],
  ["length", "lengthVal", (v) => String(v)],
  ["topP", "topPVal", (v) => Number(v).toFixed(2)],
  ["topK", "topKVal", (v) => String(v)],
];

for (const [id, outId, fmt] of sliders) {
  const input = document.getElementById(id);
  const out = document.getElementById(outId);
  const sync = () => {
    out.textContent = fmt(input.value);
  };
  input.addEventListener("input", sync);
  sync();
}

function setStatus(message, isWarn = false) {
  statusBox.textContent = message;
  statusBox.className = isWarn ? "warn" : "";
}

function setArtifactSource(element, url) {
  if (!element) return;
  if (url) {
    const separator = url.includes("?") ? "&" : "?";
    element.src = `${url}${separator}v=${Date.now()}`;
  } else {
    element.removeAttribute("src");
  }
  if (typeof element.load === "function") {
    element.load();
  }
}

function createStatCard(label, value) {
  const card = document.createElement("article");
  card.className = "stat-card";

  const cardLabel = document.createElement("p");
  cardLabel.className = "stat-label";
  cardLabel.textContent = label;

  const cardValue = document.createElement("p");
  cardValue.className = "stat-value";
  cardValue.textContent = value;

  card.append(cardLabel, cardValue);
  return card;
}

function renderAssessment(status) {
  const ready = Boolean(status?.ready);
  readyBadge.textContent = ready ? "CPU ready" : "Needs assets";
  readyBadge.className = `badge ${ready ? "good" : "warn"}`;

  tokenizerBadge.textContent = status?.tokenizer_kind
    ? `Tokenizer: ${status.tokenizer_kind}`
    : "Tokenizer unknown";
  tokenizerBadge.className = "badge muted";

  assessmentSummary.textContent = status?.guidance
    || (ready
      ? "Local checkpoint and tokenizer are available. Generation will run on CPU with built-in audio preview rendering."
      : "Add the Kaggle checkpoint to app/models and the tokenizer to app/tokenizer, then refresh the page.");

  assessmentGrid.innerHTML = "";
  const modelCount = Array.isArray(status?.models) ? status.models.length : 0;
  const modelList = modelCount
    ? status.models.slice(0, 3).join("\n") + (modelCount > 3 ? "\n…" : "")
    : "No checkpoints found";

  const cards = [
    ["Device", status?.device || "cpu"],
    ["CPU Threads", String(status?.cpu_threads ?? "auto")],
    ["Tokenizer", status?.tokenizer_path ? `${status.tokenizer_kind || "unknown"}\n${status.tokenizer_path}` : "Not found"],
    ["Models", `${modelCount} checkpoint(s)\n${modelList}`],
    ["Audio Preview", status?.audio_backend || "pretty_midi.synthesize"],
    ["MIDI Map", status?.pianoroll_backend || "matplotlib"],
  ];

  for (const [label, value] of cards) {
    assessmentGrid.appendChild(createStatCard(label, value));
  }
}

function buildGenerationStatus(data) {
  const lines = [
    `Done. Tokens: ${data.generated_tokens}.`,
  ];

  if (data.seed_duration != null && data.output_duration != null) {
    lines.push(
      `Seed audio: ${Number(data.seed_duration).toFixed(2)}s | Output audio: ${Number(data.output_duration).toFixed(2)}s`
    );
  }

  if (data.generation_elapsed_seconds != null) {
    lines.push(`Generation time: ${Number(data.generation_elapsed_seconds).toFixed(2)}s`);
  }

  if (data.health_warning) {
    lines.push(`Warning: ${data.health_warning}`);
  }

  if (data.generation_stats) {
    const stats = data.generation_stats;
    lines.push(
      `top1=${Number(stats.final_top1_max || 0).toFixed(4)} | candidates=${stats.candidate_count_min}`
    );
  }

  if (data.health_report) {
    const report = data.health_report;
    lines.push(
      `health_top1=${Number(report.max_final_top1_prob || 0).toFixed(4)} | passed=${Boolean(report.passed)}`
    );
  }

  return lines.join("\n");
}

async function loadStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  renderAssessment(data);

  if (data.ready) {
    setStatus("Local generation ready. Upload a seed MIDI to generate a CPU preview.");
  } else {
    setStatus("Local assets are not ready yet. Add the checkpoint to app/models and tokenizer to app/tokenizer.", true);
  }
}

async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  const previous = modelSelect.value;
  modelSelect.innerHTML = "";

  if (!data.models || !data.models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No checkpoints found";
    modelSelect.appendChild(opt);
    modelSelect.disabled = true;
    setStatus("No checkpoints found. Drop .safetensors/.pt into app/models.", true);
    return;
  }

  for (const modelName of data.models) {
    const opt = document.createElement("option");
    opt.value = modelName;
    opt.textContent = modelName;
    modelSelect.appendChild(opt);
  }

  modelSelect.disabled = false;
  if (previous && data.models.includes(previous)) {
    modelSelect.value = previous;
  } else {
    modelSelect.selectedIndex = 0;
  }
}

function clearPreviews() {
  setArtifactSource(seedAudio, "");
  setArtifactSource(outputAudio, "");
  if (midiMapImage) {
    midiMapImage.removeAttribute("src");
  }
}

dropZone.addEventListener("click", () => seedFileInput.click());

for (const evt of ["dragenter", "dragover"]) {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("drag");
  });
}

for (const evt of ["dragleave", "drop"]) {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drag");
  });
}

dropZone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files && files.length) {
    seedFileInput.files = files;
    uploadLabel.textContent = `Seed: ${files[0].name}`;
  }
});

seedFileInput.addEventListener("change", () => {
  if (seedFileInput.files && seedFileInput.files.length) {
    uploadLabel.textContent = `Seed: ${seedFileInput.files[0].name}`;
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!seedFileInput.files || !seedFileInput.files.length) {
    setStatus("Please provide a seed MIDI file.", true);
    return;
  }
  if (!modelSelect.value) {
    setStatus("Please select a checkpoint.", true);
    return;
  }

  generateBtn.disabled = true;
  downloadBtn.classList.add("disabled");
  downloadBtn.removeAttribute("href");
  clearPreviews();
  setStatus("Generating locally on CPU...");

  const body = new FormData();
  body.append("seed_file", seedFileInput.files[0]);
  body.append("model_name", modelSelect.value);
  body.append("temperature", document.getElementById("temperature").value);
  body.append("length", document.getElementById("length").value);
  body.append("top_p", document.getElementById("topP").value);
  body.append("top_k", document.getElementById("topK").value);

  try {
    const res = await fetch("/api/generate", { method: "POST", body });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Generation failed.");
    }

    setStatus(buildGenerationStatus(data), Boolean(data.health_warning));

    if (data.download_url) {
      downloadBtn.href = data.download_url;
      downloadBtn.classList.remove("disabled");
    }

    setArtifactSource(seedAudio, data.seed_audio_url);
    setArtifactSource(outputAudio, data.output_audio_url);
    if (midiMapImage && data.comparison_image_url) {
      midiMapImage.src = `${data.comparison_image_url}${data.comparison_image_url.includes("?") ? "&" : "?"}v=${Date.now()}`;
    }

    if (data.local_generation_status) {
      renderAssessment(data.local_generation_status);
    }

    if (previewPanel) {
      previewPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    setStatus(String(err), true);
  } finally {
    generateBtn.disabled = false;
  }
});

Promise.all([loadStatus(), loadModels()]).catch((err) => {
  setStatus(`Failed to load local generation status: ${err}`, true);
});
