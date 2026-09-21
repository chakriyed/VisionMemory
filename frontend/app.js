const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const captureCanvas = document.getElementById("capture-canvas");
const startCameraBtn = document.getElementById("start-camera");
const toggleWatchBtn = document.getElementById("toggle-watch");
const watchIndicator = document.getElementById("watch-indicator");
const registerBtn = document.getElementById("register-btn");
const objectNameInput = document.getElementById("object-name");
const registerStatus = document.getElementById("register-status");
const objectList = document.getElementById("object-list");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const queryAnswer = document.getElementById("query-answer");
const queryEvidence = document.getElementById("query-evidence");
const statusBar = document.getElementById("status-bar");

let watching = false;
let watchTimer = null;
const FRAME_INTERVAL_MS = 1000;

function captureFrameDataUrl(quality = 0.7) {
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", quality);
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    startCameraBtn.disabled = true;
    toggleWatchBtn.disabled = false;
    registerBtn.disabled = false;
  } catch (err) {
    alert("Could not access camera: " + err.message);
  }
}

function setWatchIndicator(on) {
  watchIndicator.textContent = on ? "● live" : "";
  watchIndicator.classList.toggle("live", on);
}

function drawDetections(detections) {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";
  for (const det of detections) {
    const [x1, y1, x2, y2] = det.bbox;
    ctx.strokeStyle = "#4fd1c5";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(79, 209, 197, 0.85)";
    ctx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);
    ctx.fillStyle = "#06201d";
    ctx.fillText(label, x1 + 4, Math.max(12, y1 - 5));
  }
}

async function sendFrame() {
  if (!watching) return;
  const dataUrl = captureFrameDataUrl();
  try {
    const res = await fetch("/api/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    if (res.ok) {
      const data = await res.json();
      drawDetections(data.detections || []);
    }
  } catch (err) {
    console.error("frame send failed", err);
  }
}

function startWatching() {
  watching = true;
  setWatchIndicator(true);
  toggleWatchBtn.textContent = "Stop Watching";
  watchTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
}

function stopWatching() {
  watching = false;
  setWatchIndicator(false);
  toggleWatchBtn.textContent = "Start Watching";
  if (watchTimer) clearInterval(watchTimer);
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
}

async function registerObject() {
  const name = objectNameInput.value.trim();
  if (!name) {
    registerStatus.textContent = "Enter a name first.";
    registerStatus.className = "status-line error";
    return;
  }
  const dataUrl = captureFrameDataUrl();
  registerStatus.textContent = "Registering…";
  registerStatus.className = "status-line";
  try {
    const res = await fetch("/api/objects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, image: dataUrl }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Registration failed");
    registerStatus.textContent = `Registered "${data.display_name}".`;
    registerStatus.className = "status-line ok";
    objectNameInput.value = "";
    loadObjects();
  } catch (err) {
    registerStatus.textContent = err.message;
    registerStatus.className = "status-line error";
  }
}

function formatTimestamp(iso) {
  if (!iso) return "never";
  const d = new Date(iso);
  return d.toLocaleString();
}

async function loadObjects() {
  const res = await fetch("/api/objects");
  const objects = await res.json();
  objectList.innerHTML = "";
  for (const obj of objects) {
    const li = document.createElement("li");
    li.className = "object-item";
    const lastSeen = obj.last_event
      ? `${obj.last_event.event_type} at ${formatTimestamp(obj.last_event.timestamp)}`
      : "no sightings yet";
    li.innerHTML = `
      ${obj.reference_image_url ? `<img src="${obj.reference_image_url}" alt="${obj.display_name}" />` : ""}
      <div class="object-meta">
        <div class="name">${obj.display_name}</div>
        <div class="last-seen">${lastSeen}</div>
      </div>
      <button data-id="${obj.id}">Remove</button>
    `;
    li.querySelector("button").addEventListener("click", () => deleteObject(obj.id));
    objectList.appendChild(li);
  }
}

async function deleteObject(id) {
  await fetch(`/api/objects/${id}`, { method: "DELETE" });
  loadObjects();
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;
  queryAnswer.textContent = "Thinking…";
  queryEvidence.innerHTML = "";
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    queryAnswer.textContent = data.answer;
    for (const ev of data.evidence || []) {
      const card = document.createElement("div");
      card.className = "evidence-card";
      card.innerHTML = `
        ${ev.frame_url ? `<img src="${ev.frame_url}" />` : ""}
        <div class="caption"><span class="type">${ev.event_type}</span><br/>${formatTimestamp(ev.timestamp)}</div>
      `;
      queryEvidence.appendChild(card);
    }
  } catch (err) {
    queryAnswer.textContent = "Error: " + err.message;
  }
}

async function checkStatus() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.detector_ready && data.anthropic_configured) {
      statusBar.textContent = "Backend ready — detector loaded, Claude configured.";
      statusBar.className = "status-bar ok";
    } else if (data.detector_ready) {
      statusBar.textContent = "Detector ready, but ANTHROPIC_API_KEY is not set — queries will return raw event logs only.";
      statusBar.className = "status-bar error";
    } else {
      statusBar.textContent = data.detector_error
        ? `Detector failed to load: ${data.detector_error}`
        : "Loading detector model (first run downloads weights, may take a minute)…";
      statusBar.className = "status-bar error";
    }
  } catch (err) {
    statusBar.textContent = "Backend unreachable.";
    statusBar.className = "status-bar error";
  }
}

startCameraBtn.addEventListener("click", startCamera);
toggleWatchBtn.addEventListener("click", () => (watching ? stopWatching() : startWatching()));
registerBtn.addEventListener("click", registerObject);
askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => { if (e.key === "Enter") askQuestion(); });

checkStatus();
setInterval(checkStatus, 10000);
loadObjects();
