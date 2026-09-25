const STREAM = "https://visionenvironnement.quanteec.com/contents/encodings/live/78e0f372-db6f-420e-746c-7561-6665-64-b4d7-fc979b816efed/master.m3u8";
const FALLBACK = "https://s1.vision-environnement.com/live/modules/timelapse/timelapse/montserein.mp4";

const video = document.querySelector("#player");
if (window.Hls && Hls.isSupported()) {
  const hls = new Hls();
  hls.loadSource(STREAM);
  hls.attachMedia(video);
  hls.on(Hls.Events.ERROR, (_, data) => {
    if (data.fatal) video.src = FALLBACK;
  });
} else {
  video.src = video.canPlayType("application/vnd.apple.mpegurl") ? STREAM : FALLBACK;
}

const list = document.querySelector("#list");
const empty = document.querySelector("#empty");
let events = [];
let filter = "all";

document.querySelectorAll(".filters button").forEach((button) => {
  button.addEventListener("click", () => {
    filter = button.dataset.filter;
    document.querySelectorAll(".filters button").forEach((item) => item.classList.toggle("on", item === button));
    render();
  });
});

function detail(event) {
  const info = event.detail || {};
  if (event.type === "plane") {
    const altitude = info.altitude_m == null ? "" : ` · ${Math.round(info.altitude_m)} m`;
    return `${info.icao24 || ""}${altitude}`.trim();
  }
  if (event.type === "bus" && info.route) {
    return `${info.headsign || info.route} · ${info.scheduled || ""} · ${info.source || ""}`.trim();
  }
  if (event.type === "crowd") return `${info.persons} personnes`;
  if (event.type === "fire") return "Tache chaude qui a grossi. Ce n’est pas une alerte.";
  return "";
}

function render() {
  const shown = events.filter((event) => filter === "all" || event.type === filter);
  empty.hidden = shown.length > 0;
  list.innerHTML = shown.map((event) => {
    const when = new Date(event.t).toLocaleString("fr-FR", { dateStyle: "medium", timeStyle: "short" });
    const picture = event.thumb
      ? `<img src="${event.thumb}" alt="">`
      : `<span class="placeholder"></span>`;
    const clip = event.clip_url ? `<a href="${event.clip_url}">Extrait</a>` : "";
    const extra = detail(event);
    return `<li>${picture}<div><h3>${escapeHtml(event.label)}</h3><p class="meta">${when}${extra ? " · " + escapeHtml(extra) : ""}</p>${clip}</div></li>`;
  }).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

async function load() {
  const response = await fetch("data/events.json", { cache: "no-store" });
  const payload = await response.json();
  events = payload.events || [];
  render();
}

load();
setInterval(load, 60000);
