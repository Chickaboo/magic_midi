const modelSelect = document.getElementById("modelSelect");
const form = document.getElementById("generateForm");
const seedFileInput = document.getElementById("seedFile");
const dropZone = document.getElementById("dropZone");
const uploadLabel = document.getElementById("uploadLabel");
const statusBox = document.getElementById("statusBox");
const downloadBtn = document.getElementById("downloadBtn");
const generateBtn = document.getElementById("generateBtn");

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

async function loadModels() {
  setStatus("Loading checkpoints...");
  const res = await fetch("/api/models");
  const data = await res.json();
  modelSelect.innerHTML = "";

  if (!data.models || !data.models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No checkpoints found in app/models";
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
  setStatus(`Found ${data.models.length} checkpoint(s). Ready.`);
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
  setStatus("Generating...");

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

    let message = `Done. Tokens: ${data.generated_tokens}.`;
    if (data.health_warning) {
      message += `\nWarning: ${data.health_warning}`;
    }
    if (data.generation_stats) {
      const s = data.generation_stats;
      message += `\nmax_final_top1=${Number(s.final_top1_max || 0).toFixed(4)} min_candidates=${s.candidate_count_min}`;
    }
    if (data.health_report) {
      const h = data.health_report;
      message += `\nhealth_max_top1=${Number(h.max_final_top1_prob || 0).toFixed(4)} passed=${Boolean(h.passed)}`;
    }
    setStatus(message, Boolean(data.health_warning));

    if (data.download_url) {
      downloadBtn.href = data.download_url;
      downloadBtn.classList.remove("disabled");
    }
  } catch (err) {
    setStatus(String(err), true);
  } finally {
    generateBtn.disabled = false;
  }
});

loadModels().catch((err) => {
  setStatus(`Failed to load models: ${err}`, true);
});
