const CAMERA = { lat: 44.183501, lon: 5.2621281, bearing: 140 };
const STATIONS = [
  { name: "Avignon", lat: 43.9493, lon: 4.8055 },
  { name: "Carpentras", lat: 44.055, lon: 5.048 },
  { name: "Orange", lat: 44.136, lon: 4.809 },
  { name: "Apt", lat: 43.876, lon: 5.396 },
  { name: "Cavaillon", lat: 43.838, lon: 5.038 },
  { name: "Pertuis", lat: 43.695, lon: 5.503 },
];
const COPY = {
  en: {
    paris: "Paris time",
    weather: "Weather",
    passes: "Passes",
    namedLine: (named, habits) => `${named} named · ${habits} habits`,
    live: "Live webcam",
    camera: "Camera",
    cameraText: "Fixed, facing 140°.",
    pipeline: "Pipeline",
    image: "Frame",
    imageText: "1 frame/s. ffmpeg reads the HLS stream.",
    motion: "Motion",
    motionText: "OpenCV MOG2, frame scaled to 640 px. If more than 35% changes, it is light, and it is ignored. The red beacon on the summit is excluded.",
    track: "Track",
    trackText: "A compact blob, tracked for at least 3 frames.",
    class: "Class",
    classText: "YOLO11 nano, ONNX, on the crop only. Person, car, bus, truck.",
    plane: "Plane",
    planeText: "The sky does not use the model’s airplane class. OpenSky. A callsign is written only when there is one aircraft, or one much lower than the others.",
    bus: "Bus",
    busText: "Trans'CoVe or ZOU. A single trip within ±15 min gives the route. Otherwise “Bus”, from a confidence of 0.6.",
    crowd: "Crowd",
    crowdText: "4 people or more, held for 8 seconds.",
    fire: "Fire",
    fireText: "On the slope, 20 s, area ×1.5, at least 8% warm pixels. Not at dusk. In rain, fog, or snow, it takes 20%.",
    rest: "Unknown",
    restText: "We don't know what it is. The photo is kept. If it comes back in the same place, one new photo every 6 hours, no more.",
    history: "History",
    all: "All",
    planes: "Planes",
    cars: "Cars",
    buses: "Buses",
    crowds: "Crowds",
    fires: "Fires",
    motions: "Motion",
    habits: "Habits",
    empty: "Nothing yet.",
    around: "Around",
    people: "people",
    clip: "Clip",
    right: "Right",
    wrong: "Wrong",
    confirmed: "confirmed",
    rejected: "rejected",
    fireNote: "A warm patch grew. This is not an alert.",
    weatherCodes: {
      0: "Clear", 1: "Clear", 2: "Cloudy", 3: "Overcast",
      45: "Fog", 48: "Fog", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
      61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow",
      73: "Snow", 75: "Heavy snow", 77: "Graupel", 80: "Showers", 81: "Showers",
      82: "Heavy showers", 85: "Snow showers", 86: "Snow showers",
      95: "Thunderstorm", 96: "Thunderstorm", 99: "Violent thunderstorm",
    },
  },
  fr: {
    paris: "Heure de Paris",
    weather: "Météo",
    passes: "Passages",
    namedLine: (named, habits) => `${named} nommés · ${habits} habitudes`,
    live: "Webcam en direct",
    camera: "Caméra",
    cameraText: "Fixe, vers 140°.",
    pipeline: "Pipeline",
    image: "Image",
    imageText: "1 image/s. ffmpeg lit le flux HLS.",
    motion: "Mouvement",
    motionText: "OpenCV MOG2, image ramenée à 640 px. Si plus de 35 % change, c’est la lumière, on ignore. La balise rouge du sommet est exclue.",
    track: "Suivi",
    trackText: "Une tache compacte, suivie au moins de 3 images.",
    class: "Classe",
    classText: "YOLO11 nano, en ONNX, seulement sur le rectangle. Personne, voiture, bus, camion.",
    plane: "Avion",
    planeText: "Le ciel ne passe pas par la classe avion du modèle. OpenSky. L’indicatif n’est écrit que s’il n’y a qu’un avion, ou un seul beaucoup plus bas.",
    bus: "Bus",
    busText: "Trans'CoVe ou ZOU. Une seule course à ±15 min donne la ligne. Sinon « Bus », à partir d’une confiance de 0,6.",
    crowd: "Attroupement",
    crowdText: "4 personnes ou plus, tenues 8 secondes.",
    fire: "Feu",
    fireText: "Sur la pente, 20 s, surface ×1,5, au moins 8 % de pixels chauds. Au crépuscule, non. Sous la pluie, le brouillard ou la neige, il faut 20 %.",
    rest: "Inconnu",
    restText: "On ne sait pas ce que c’est. La photo est gardée. Si ça revient au même endroit, une nouvelle photo toutes les 6 heures, pas plus.",
    history: "Historique",
    all: "Tout",
    planes: "Avions",
    cars: "Voitures",
    buses: "Bus",
    crowds: "Attroupements",
    fires: "Incendies",
    motions: "Mouvements",
    habits: "Habitudes",
    empty: "Rien pour l’instant.",
    around: "Autour",
    people: "personnes",
    clip: "Extrait",
    right: "Juste",
    wrong: "Faux",
    confirmed: "validé",
    rejected: "rejeté",
    fireNote: "Tache chaude qui a grossi. Ce n’est pas une alerte.",
    weatherCodes: {
      0: "Ciel dégagé", 1: "Dégagé", 2: "Nuageux", 3: "Couvert",
      45: "Brouillard", 48: "Brouillard", 51: "Bruine", 53: "Bruine", 55: "Bruine",
      61: "Pluie légère", 63: "Pluie", 65: "Forte pluie", 71: "Neige légère",
      73: "Neige", 75: "Forte neige", 77: "Grésil", 80: "Averses", 81: "Averses",
      82: "Fortes averses", 85: "Averses de neige", 86: "Averses de neige",
      95: "Orage", 96: "Orage", 99: "Orage violent",
    },
  },
};

const LABELS = {
  "Voiture": "Car",
  "Camion": "Truck",
  "Attroupement": "Crowd",
  "Incendie": "Fire",
  "Habitude du cadrage": "Habit of the frame",
  "Mouvement": "Motion",
  "Mouvement sur la route": "Motion on the road",
  "Mouvement dans le ciel": "Motion in the sky",
  "Masse dans le ciel": "Mass in the sky",
  "Point dans le ciel": "Point in the sky",
  "Presque immobile": "Almost still",
  "Véhicule incertain": "Uncertain vehicle",
  "Lueur du soir": "Evening glow",
  "Lueur dans la météo": "Glow in the weather",
};

let lang = localStorage.getItem("ventoux-lang") === "fr" ? "fr" : "en";
let weatherNow = null;
let counts = null;

function t(key) {
  return COPY[lang][key];
}

function locale() {
  return lang === "fr" ? "fr-FR" : "en-GB";
}

function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const value = t(node.dataset.i18n);
    if (typeof value === "string") node.textContent = value;
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
    node.setAttribute("aria-label", t(node.dataset.i18nAria));
  });
  document.querySelectorAll(".langs button").forEach((button) => {
    button.classList.toggle("on", button.dataset.lang === lang);
  });
  tick();
  paintCamera();
  paintWeather();
  paintCounts();
  render();
  paintSequenceMeta();
}

document.querySelectorAll(".langs button").forEach((button) => {
  button.addEventListener("click", () => {
    lang = button.dataset.lang === "fr" ? "fr" : "en";
    localStorage.setItem("ventoux-lang", lang);
    applyLang();
  });
});

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
  document.querySelector("#clock").textContent = now.toLocaleTimeString(locale(), {
    timeZone: "Europe/Paris", hour: "2-digit", minute: "2-digit",
  });
  document.querySelector("#clock-date").textContent = now.toLocaleDateString(locale(), {
    timeZone: "Europe/Paris", weekday: "short", day: "numeric", month: "short",
  });
}

function paintWeather() {
  const place = document.querySelector("#station");
  const distance = `${Math.round(km(CAMERA, station))} km`;
  if (!weatherNow) {
    place.textContent = `${station.name} · ${distance}`;
    return;
  }
  const label = t("weatherCodes")[weatherNow.code] || "";
  document.querySelector("#weather").textContent = Number.isFinite(weatherNow.temp) ? `${weatherNow.temp} °C` : "—";
  place.textContent = [station.name, distance, label].filter(Boolean).join(" · ");
}

async function loadWeather() {
  paintWeather();
  try {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${station.lat}&longitude=${station.lon}&current=temperature_2m,weather_code&timezone=Europe/Paris`;
    const data = await fetch(url, { cache: "no-store" }).then((response) => response.json());
    const current = data.current || {};
    weatherNow = { temp: Math.round(current.temperature_2m), code: current.weather_code };
    paintWeather();
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
  if (event.type === "crowd") return `${info.persons} ${t("people")}`;
  if (event.type === "fire") return showText(info.reading) || t("fireNote");
  if (info.reading) return [showText(info.context), showText(info.reading)].filter(Boolean).join(" · ");
  if (info.context) return showText(info.context);
  return "";
}

function render() {
  const shown = events.filter((event) => filter === "all" || event.type === filter);
  empty.hidden = shown.length > 0;
  list.innerHTML = shown.map((event) => {
    const when = new Date(event.t).toLocaleString(locale(), { dateStyle: "medium", timeStyle: "short", timeZone: "Europe/Paris" });
    const picture = event.thumb
      ? `<img src="${event.thumb}" alt="">`
      : `<span class="placeholder"></span>`;
    const clip = event.clip_url ? `<a href="${event.clip_url}">${escapeHtml(t("clip"))}</a>` : "";
    const extra = detail(event);
    const review = reviewControls(event);
    const state = event.review === "accepted" ? t("confirmed") : event.review === "rejected" ? t("rejected") : "";
    return `<li class="${event.review || ""}">${picture}<div><h3>${escapeHtml(showText(event.label))}</h3><p class="meta">${when}${extra ? " · " + escapeHtml(extra) : ""}${state ? " · " + state : ""}</p>${clip}${review}</div></li>`;
  }).join("");
}

function reviewControls(event) {
  const accepted = event.review === "accepted" ? " on" : "";
  const rejected = event.review === "rejected" ? " on" : "";
  return `<p class="verdict"><a class="yes${accepted}" href="${reviewUrl(event, "accepted", "valide")}">${escapeHtml(t("right"))}</a><a class="no${rejected}" href="${reviewUrl(event, "rejected", "rejete")}">${escapeHtml(t("wrong"))}</a></p>`;
}

function reviewUrl(event, verdict, label) {
  const title = `revue ${event.id}`;
  const body = `event_id: ${event.id}\nverdict: ${verdict}\nlecture: ${event.label}\n`;
  return `https://github.com/thepriben/ventoux-watch/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}&labels=${label}`;
}

function showText(value) {
  if (!value || lang === "fr") return value || "";
  return LABELS[value] || value;
}

function paintCounts() {
  if (!counts) return;
  document.querySelector("#seen").textContent = String(counts.seen || 0);
  document.querySelector("#named").textContent = t("namedLine")(counts.named || 0, counts.habits || 0);
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
      counts = await learning.json();
      paintCounts();
    }
  } catch (_) {
    /* Le compteur apparaît quand le Pi a commencé à apprendre. */
  }
}

const VIEW_REACH = 600;
const VIEW_ANGLE = 90;

function offset(lat, lon, bearing, meters) {
  const rad = Math.PI / 180;
  const distance = meters / 6371000;
  const br = bearing * rad;
  const lat1 = lat * rad;
  const lon1 = lon * rad;
  const lat2 = Math.asin(Math.sin(lat1) * Math.cos(distance) + Math.cos(lat1) * Math.sin(distance) * Math.cos(br));
  const lon2 = lon1 + Math.atan2(Math.sin(br) * Math.sin(distance) * Math.cos(lat1), Math.cos(distance) - Math.sin(lat1) * Math.sin(lat2));
  return [lat2 / rad, lon2 / rad];
}

function viewWedge() {
  const points = [[CAMERA.lat, CAMERA.lon]];
  const start = CAMERA.bearing - VIEW_ANGLE / 2;
  for (let step = 0; step <= 28; step += 1) {
    points.push(offset(CAMERA.lat, CAMERA.lon, start + (VIEW_ANGLE * step) / 28, VIEW_REACH));
  }
  return points;
}

const ZOOMS = [
  { zoom: 18, along: 0 },
  { zoom: 16, along: VIEW_REACH * 0.35 },
  { zoom: 14, along: VIEW_REACH * 0.4 },
];

const map = L.map("map", {
  scrollWheelZoom: false,
  zoomControl: false,
  doubleClickZoom: false,
  boxZoom: false,
  keyboard: false,
  attributionControl: false,
});
L.control.attribution({ prefix: false }).addTo(map);
L.tileLayer("https://data.geopf.fr/wmts?LAYER=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/jpeg&SERVICE=WMTS&VERSION=1.0.0&REQUEST=GetTile&STYLE=normal&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}", {
  maxZoom: 19,
  attribution: "© IGN",
}).addTo(map);
L.polygon(viewWedge(), {
  color: "#c4b094",
  weight: 1.5,
  fillColor: "#f3efe6",
  fillOpacity: 0.62,
}).addTo(map);
L.polyline([
  [CAMERA.lat, CAMERA.lon],
  offset(CAMERA.lat, CAMERA.lon, CAMERA.bearing, VIEW_REACH),
], { color: "#c4b094", weight: 2, dashArray: "4 6" }).addTo(map);
const cameraMark = L.circleMarker([CAMERA.lat, CAMERA.lon], {
  radius: 6,
  color: "#f4f1ea",
  weight: 2,
  fillColor: "#243f34",
  fillOpacity: 1,
}).addTo(map);

function showZoom(index) {
  const level = ZOOMS[index];
  map.setView(offset(CAMERA.lat, CAMERA.lon, CAMERA.bearing, level.along), level.zoom);
  document.querySelectorAll(".zooms button").forEach((button) => {
    button.classList.toggle("on", Number(button.dataset.zoom) === index);
  });
}

document.querySelectorAll(".zooms button").forEach((button) => {
  button.addEventListener("click", () => showZoom(Number(button.dataset.zoom)));
});
showZoom(1);
requestAnimationFrame(() => map.invalidateSize());

function paintCamera() {
  cameraMark.unbindTooltip();
  cameraMark.bindTooltip(t("camera"), { permanent: true, direction: "top", offset: [0, -8] });
}

applyLang();
setInterval(tick, 1000);
loadWeather();
setInterval(loadWeather, 600000);
let sequenceWhen = null;

function paintSequenceMeta() {
  if (!sequenceWhen) return;
  const date = new Date(sequenceWhen).toLocaleDateString(locale(), { month: "long", year: "numeric" });
  document.querySelector("#sequence-meta").textContent = `Mapillary · ${date}`;
}

async function mly(pathname, params) {
  const url = new URL(pathname ? `https://graph.mapillary.com/${pathname}` : "https://graph.mapillary.com/");
  url.searchParams.set("access_token", "MLY|26158465847163536|0186af2cabb143cd46cccc023e7f0d81");
  Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
  const response = await fetch(url);
  if (!response.ok) throw new Error(String(response.status));
  return response.json();
}

async function loadSequence() {
  const pad = 0.015;
  const bbox = [CAMERA.lon - pad, CAMERA.lat - pad, CAMERA.lon + pad, CAMERA.lat + pad].join(",");
  const nearby = await mly("images", { fields: "id,sequence,computed_geometry", bbox, limit: "100" });
  const closest = new Map();
  (nearby.data || []).forEach((img) => {
    const coords = (img.computed_geometry || {}).coordinates;
    if (!coords || !img.sequence) return;
    const dist = km(CAMERA, { lat: coords[1], lon: coords[0] });
    const known = closest.get(img.sequence);
    if (known == null || dist < known) closest.set(img.sequence, dist);
  });
  const sequence = [...closest.entries()].sort((a, b) => a[1] - b[1])[0];
  if (!sequence) return;
  const listed = await mly("image_ids", { sequence_id: sequence[0] });
  const ids = (listed.data || []).map((item) => item.id);
  if (!ids.length) return;
  const details = await mly("", { fields: "id,thumb_256_url,computed_geometry,captured_at", ids: ids.join(",") });
  const ordered = ids.map((id) => details[id]).filter((img) => img && img.thumb_256_url && img.computed_geometry);
  const within = (limit) => ordered.filter((img) => {
    const [lon, lat] = img.computed_geometry.coordinates;
    return km(CAMERA, { lat, lon }) < limit;
  });
  const shown = within(0.6).length >= 4 ? within(0.6) : within(1.2);
  if (!shown.length) return;
  sequenceWhen = shown[0].captured_at;
  L.polyline(shown.map((img) => {
    const [lon, lat] = img.computed_geometry.coordinates;
    return [lat, lon];
  }), { color: "#243f34", weight: 3, opacity: 0.8 }).addTo(map);
  document.querySelector("#film").innerHTML = shown.map((img) => {
    const href = `https://www.mapillary.com/app/?pKey=${encodeURIComponent(img.id)}&focus=photo`;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer"><img src="${escapeHtml(img.thumb_256_url)}" alt=""></a>`;
  }).join("");
  document.querySelector("#sequence").hidden = false;
  paintSequenceMeta();
}

load();
setInterval(load, 60000);
loadSequence();
