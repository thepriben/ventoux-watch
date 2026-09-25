const CAMERA = { lat: 44.179, lon: 5.2663 };
const STATIONS = [
  { name: "Avignon", lat: 43.9493, lon: 4.8055 },
  { name: "Carpentras", lat: 44.055, lon: 5.048 },
  { name: "Orange", lat: 44.136, lon: 4.809 },
  { name: "Apt", lat: 43.876, lon: 5.396 },
  { name: "Cavaillon", lat: 43.838, lon: 5.038 },
  { name: "Pertuis", lat: 43.695, lon: 5.503 },
];
const WEATHER = {
  0: "Ciel dégagé", 1: "Dégagé", 2: "Nuageux", 3: "Couvert",
  45: "Brouillard", 48: "Brouillard", 51: "Bruine", 53: "Bruine", 55: "Bruine",
  61: "Pluie légère", 63: "Pluie", 65: "Forte pluie", 71: "Neige légère",
  73: "Neige", 75: "Forte neige", 77: "Grésil", 80: "Averses", 81: "Averses",
  82: "Fortes averses", 85: "Averses de neige", 86: "Averses de neige",
  95: "Orage", 96: "Orage", 99: "Orage violent",
};

function km(a, b) {
  const rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad;
  const dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2
    + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(h));
}

const station = STATIONS.slice().sort((a, b) => km(CAMERA, a) - km(CAMERA, b))[0];

function tick() {
  const now = new Date();
  document.querySelector("#clock").textContent = now.toLocaleTimeString("fr-FR", {
    timeZone: "Europe/Paris", hour: "2-digit", minute: "2-digit",
  });
  document.querySelector("#clock-date").textContent = now.toLocaleDateString("fr-FR", {
    timeZone: "Europe/Paris", weekday: "short", day: "numeric", month: "short",
  });
}

async function loadWeather() {
  const place = document.querySelector("#station");
  place.textContent = `${station.name} · ${Math.round(km(CAMERA, station))} km`;
  try {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${station.lat}&longitude=${station.lon}&current=temperature_2m,weather_code&timezone=Europe/Paris`;
    const data = await fetch(url, { cache: "no-store" }).then((response) => response.json());
    const current = data.current || {};
    const temp = Math.round(current.temperature_2m);
    const label = WEATHER[current.weather_code] || "";
    document.querySelector("#weather").textContent = Number.isFinite(temp) ? `${temp} °C` : "—";
    place.textContent = [station.name, `${Math.round(km(CAMERA, station))} km`, label].filter(Boolean).join(" · ");
  } catch (_) {
    document.querySelector("#weather").textContent = "—";
  }
}

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
  if (event.type === "fire") return info.reading || "Tache chaude qui a grossi. Ce n’est pas une alerte.";
  if (info.reading) return [info.context, info.reading].filter(Boolean).join(" · ");
  if (info.context) return info.context;
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
    const review = reviewControls(event);
    const state = event.review === "accepted" ? "validé" : event.review === "rejected" ? "rejeté" : "";
    return `<li class="${event.review || ""}">${picture}<div><h3>${escapeHtml(event.label)}</h3><p class="meta">${when}${extra ? " · " + escapeHtml(extra) : ""}${state ? " · " + state : ""}</p>${clip}${review}</div></li>`;
  }).join("");
}

function reviewControls(event) {
  const accepted = event.review === "accepted" ? " on" : "";
  const rejected = event.review === "rejected" ? " on" : "";
  return `<p class="verdict"><a class="yes${accepted}" href="${reviewUrl(event, "accepted", "valide")}">Juste</a><a class="no${rejected}" href="${reviewUrl(event, "rejected", "rejete")}">Faux</a></p>`;
}

function reviewUrl(event, verdict, label) {
  const title = `revue ${event.id}`;
  const body = `event_id: ${event.id}\nverdict: ${verdict}\nlecture: ${event.label}\n`;
  return `https://github.com/thepriben/ventoux-watch/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}&labels=${label}`;
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
  try {
    const learning = await fetch("data/learning.json", { cache: "no-store" });
    if (learning.ok) {
      const counts = await learning.json();
      document.querySelector("#seen").textContent = String(counts.seen || 0);
      document.querySelector("#named").textContent = `${counts.named || 0} nommés · ${counts.habits || 0} habitudes`;
    }
  } catch (_) {
    /* Le compteur apparaît quand le Pi a commencé à apprendre. */
  }
}

tick();
setInterval(tick, 1000);
loadWeather();
setInterval(loadWeather, 600000);
load();
setInterval(load, 60000);
