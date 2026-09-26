import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// What grows on the ground, in the order the map codes it. The first is what
// the map has not said: up here that is the ski area in summer, pasture worn
// thin by the pistes, so it is given the colour of grass rather than of sand.
const COVER = [0x8c9a6c, 0x47713b, 0x85a95f, 0xaaa195];

// Tarmac, gravel and beaten earth. A road is drawn at the width the map gives
// it, so the difference between a lane and a footpath is a real difference.
const SURFACE = {
  secondary: 0x5c5c62, tertiary: 0x5c5c62, unclassified: 0x606067,
  residential: 0x606067, living_street: 0x606067, service: 0x66666d,
  track: 0x8a7b5e, path: 0xa18a68, footway: 0xa18a68, steps: 0xa18a68,
  cycleway: 0x8a7b5e, pedestrian: 0x8b8b8f, parking: 0x6a6a71,
  // The island is trodden grass over gravel, drier and paler than the pasture
  // around it. Close enough in colour to be honest, far enough to be seen.
  island: 0xa19e6e,
  playground: 0xb8a24a,
  pool: 0x2f6f9e,
};
// Drawn just clear of the ground, so the tarmac does not fight the slope it
// lies on for the same pixels.
const LIFT_M = 0.35;
const WALL = 0x9a8975;
const ROOF = 0x7a5f52;
const STEEL = 0xb9bec7;
const LAMP_ON = 0xffd9a0; const LAMP_OFF = 0x6b6b66;
const LAMP_REACH_M = 45; const LAMP_POWER = 140;
const BEACON_ON = 0xff2b1e; const BEACON_OFF = 0x5e3a36;
// One colour per kind of thing, so the mark says what it stands for before the
// card is read: warm for anything on wheels, cool for a person, red for fire.
const TYPES = {
  car: 0xffb347, bus: 0xffb347, truck: 0xffb347, vehicle: 0xffb347,
  person: 0x67d5f5, aircraft: 0xc9b6ff, plane: 0xc9b6ff, fire: 0xff4436,
  cycle: 0x8ce99a, animal: 0xf2a2c0, other: 0xe6e6e6,
};
const FOLIAGE = [0x345c2c, 0x3e6b33, 0x4a7a3a, 0x2e5228];
const TRUNK = 0x4a3b2c;
// A tree every twelve metres is what a pine wood looks like up here. Sown at
// that spacing over every parcel the map knows, it comes to a quarter of a
// million trees, which no page should be asked to carry. So the wood is thinned
// with distance, the way the eye thins it: at two kilometres, one tree in
// twenty-five stands for the rest, and nobody can tell.
const SPACING_M = 12;
const THINNING_M = 40;
const SPARSEST_M = 60;
// Below this a trunk is worth drawing. Beyond it a trunk is a tenth of a pixel.
const TRUNKS_M = 350;
const CROWN_M = 10;

// The sky and the light at three moments of the day. What is shown is what the
// sun is really doing over the Ventoux at this instant, so the relief is dark
// when the webcam above it is dark.
const HOURS = {
  day: { top: 0x3f6ea8, low: 0xa9c5de, warm: 0xfff4e2, beam: 1.05, fill: 1.35, cool: 0xe4edf8 },
  dusk: { top: 0x2b3352, low: 0xd98b5a, warm: 0xffc089, beam: 0.8, fill: 0.95, cool: 0xa8b2cc },
  night: { top: 0x0d1524, low: 0x27374e, warm: 0xc2d0ea, beam: 0.3, fill: 1.05, cool: 0x9fb2d4 },
};
// Between these two heights of the sun the light turns over. Above, it is day;
// below, the ground keeps only what the sky still gives it.
const DUSK_LOW = -7;
const DUSK_HIGH = 7;

const host = document.getElementById("relief");
if (host) start(host).catch(() => host.closest(".block")?.setAttribute("hidden", ""));

async function start(host) {
  const relief = await fetch("data/relief.json", { cache: "no-store" }).then((answer) => answer.json());
  const pose = relief.pose;
  const aspect = pose.aspect || 16 / 9;
  const axes = frame(pose);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  host.appendChild(renderer.domElement);

  // Everything the map draws is laid on the heightfield the page itself has
  // built, never on a height computed elsewhere: read from a finer model, a
  // road sinks under the very ground it is supposed to lie on.
  const high = sampler(relief.terrain);
  const world = new THREE.Scene();
  world.add(ground(relief));
  for (const road of relief.roads) world.add(ribbon(road.p, road.w, SURFACE[road.k] ?? SURFACE.track, high));
  for (const area of relief.ribbons) {
    world.add(slab(area.p, SURFACE[area.k] ?? SURFACE.parking, high));
    if (area.h) world.add(standing(area, SURFACE[area.k] ?? SURFACE.parking, high));
  }
  for (const house of relief.buildings) world.add(block(house, high));
  world.add(...wood(relief, high));
  const named = relief.masts.map((mast) => column(mast, high))
    .concat((relief.figures || []).map((figure) => carving(figure, high)));
  for (const mast of named) world.add(mast);
  const lamps = (relief.lamps || []).map((lamp) => streetlight(lamp, high));
  for (const lamp of lamps) world.add(lamp.post);
  const beacons = (relief.beacons || []).map((mark) => obstacle(mark, high));
  for (const mark of beacons) world.add(mark.bulb);
  world.add(here());

  const fill = new THREE.AmbientLight(0xffffff, 1);
  const beam = new THREE.DirectionalLight(0xffffff, 1);
  const star = new THREE.Mesh(new THREE.SphereGeometry(70, 16, 12), new THREE.MeshBasicMaterial());
  world.add(fill, beam, star);
  const daylight = () => paintHour(world, pose, { fill, beam, star, lamps, beacons });
  daylight();
  setInterval(daylight, 60000);

  const camera = new THREE.PerspectiveCamera(vertical(pose.hfov, aspect), aspect, 1, 20000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxDistance = 7000;
  // The far slope, straight ahead: near enough that turning feels like walking
  // round the bowl rather than round a marble held at arm's length.
  const look = axes.forward.clone().multiplyScalar(700);

  function home() {
    camera.position.set(0, 0, 0);
    camera.up.copy(axes.up);
    controls.target.copy(look);
    controls.update();
  }
  home();
  document.getElementById("relief-back")?.addEventListener("click", home);
  // The stage goes full screen, not the canvas: the buttons live inside it, and
  // a full screen that swallowed the one that puts the camera back would strand
  // anyone who had turned the scene around.
  const stage = document.getElementById("relief-stage") || host;
  document.getElementById("relief-wide")?.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else stage.requestFullscreen?.();
  });
  document.addEventListener("fullscreenchange", () => {
    const wide = document.getElementById("relief-wide");
    if (wide) wide.textContent = document.fullscreenElement === stage ? "⤡" : "⤢";
    fit();
  });

  const latest = await lastSeen(pose, aspect, axes, high);
  if (latest) world.add(latest.pin);

  const caption = document.getElementById("relief-name");
  const card = document.getElementById("relief-card");
  const finder = new THREE.Raycaster();
  const cursor = new THREE.Vector2();
  renderer.domElement.addEventListener("pointermove", (event) => {
    const box = renderer.domElement.getBoundingClientRect();
    cursor.x = ((event.clientX - box.left) / box.width) * 2 - 1;
    cursor.y = -((event.clientY - box.top) / box.height) * 2 + 1;
    finder.setFromCamera(cursor, camera);
    if (latest && finder.intersectObject(latest.pin, true).length) {
      card.innerHTML = latest.card;
      card.hidden = false;
      // Kept clear of the edge: a card that ran off the view would be read
      // half, and the tags at the end are the ones worth reading.
      const room = stage.getBoundingClientRect();
      const left = Math.min(event.clientX - room.left + 14, room.width - card.offsetWidth - 8);
      const top = Math.min(event.clientY - room.top + 14, room.height - card.offsetHeight - 8);
      card.style.left = `${Math.max(8, left)}px`;
      card.style.top = `${Math.max(8, top)}px`;
      caption.hidden = true;
      return;
    }
    card.hidden = true;
    const hit = finder.intersectObjects(named)[0];
    caption.textContent = hit ? hit.object.name : "";
    caption.hidden = !hit;
  });
  renderer.domElement.addEventListener("pointerleave", () => {
    card.hidden = true;
    caption.hidden = true;
  });

  function fit() {
    // Sixteen by nine, always. Filling a screen of another shape would widen
    // or crop the field of view, and the whole point is that this is the
    // webcam's field of view and no other.
    const wide = document.fullscreenElement === stage;
    const room = wide ? stage.clientHeight * aspect : Infinity;
    const width = Math.min(wide ? stage.clientWidth : host.clientWidth, room);
    renderer.setSize(width, Math.round(width / aspect));
  }
  window.addEventListener("resize", fit);
  fit();

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(world, camera);
  });
}

function spot(east, north, up) {
  // East, up, south: the way three.js holds the world here.
  return new THREE.Vector3(east, up, -north);
}

function frame(pose) {
  /* The camera's own three directions. The same formula the watcher uses to
     read the ground, so the picture and the model agree. */
  const yaw = THREE.MathUtils.degToRad(pose.yaw);
  const pitch = THREE.MathUtils.degToRad(pose.pitch);
  const flat = [Math.sin(yaw), Math.cos(yaw)];
  return {
    right: spot(Math.cos(yaw), -Math.sin(yaw), 0),
    forward: spot(flat[0] * Math.cos(pitch), flat[1] * Math.cos(pitch), -Math.sin(pitch)),
    up: spot(flat[0] * Math.sin(pitch), flat[1] * Math.sin(pitch), Math.cos(pitch)),
  };
}

function vertical(hfov, aspect) {
  const half = Math.tan(THREE.MathUtils.degToRad(hfov) / 2);
  return THREE.MathUtils.radToDeg(2 * Math.atan(half / aspect));
}

function ground(relief) {
  /* The mountain itself, one square per elevation post, coloured by what the
     map says grows on it. */
  const { grid, step_m: step, reach_m: reach } = relief.terrain;
  const cover = relief.cover.grid;
  const side = grid.length;
  const shape = new THREE.PlaneGeometry(2 * reach, 2 * reach, side - 1, side - 1);
  shape.rotateX(-Math.PI / 2);
  const place = shape.attributes.position;
  const tint = [];
  const paint = new THREE.Color();
  for (let index = 0; index < place.count; index += 1) {
    const col = index % side;
    const row = Math.floor(index / side);
    place.setY(index, grid[row][col]);
    paint.setHex(COVER[cover[row][col]] ?? COVER[0]);
    tint.push(paint.r, paint.g, paint.b);
  }
  shape.setAttribute("color", new THREE.Float32BufferAttribute(tint, 3));
  shape.computeVertexNormals();
  return new THREE.Mesh(shape, new THREE.MeshLambertMaterial({ vertexColors: true }));
}

function ribbon(line, width, colour, high) {
  /* A road from its centre line: the map keeps the line and the width apart,
     and so does this. */
  const half = Math.max(0.8, width / 2);
  const left = [];
  const right = [];
  for (let index = 0; index < line.length; index += 1) {
    const before = line[Math.max(0, index - 1)];
    const after = line[Math.min(line.length - 1, index + 1)];
    let run = new THREE.Vector2(after[0] - before[0], after[1] - before[1]);
    if (run.lengthSq() < 1e-9) run = new THREE.Vector2(1, 0);
    run.normalize();
    const side = new THREE.Vector2(-run.y, run.x).multiplyScalar(half);
    const [east, north] = line[index];
    const a = [east + side.x, north + side.y];
    const b = [east - side.x, north - side.y];
    left.push(spot(a[0], a[1], high(a[0], a[1]) + LIFT_M));
    right.push(spot(b[0], b[1], high(b[0], b[1]) + LIFT_M));
  }
  const place = [];
  for (let index = 0; index + 1 < line.length; index += 1) {
    const quad = [left[index], right[index], right[index + 1], left[index + 1]];
    for (const corner of [0, 1, 2, 0, 2, 3]) place.push(quad[corner].x, quad[corner].y, quad[corner].z);
  }
  const shape = new THREE.BufferGeometry();
  shape.setAttribute("position", new THREE.Float32BufferAttribute(place, 3));
  shape.computeVertexNormals();
  return new THREE.Mesh(shape, tarmac(colour));
}

async function lastSeen(pose, aspect, axes, high) {
  /* The newest published event, put back on the ground it happened on.

     The history keeps where a thing was in the picture, never where it was on
     the hill. Sending that image point back out through the same camera the
     watcher looks through, until it meets the terrain, is what turns a box on a
     photograph into a place you can walk round in. */
  const events = await fetch("data/events.json", { cache: "no-store" })
    .then((answer) => answer.json())
    .then((payload) => payload.events || [])
    .catch(() => []);
  const event = events.find((item) => (item.detail || {}).box);
  if (!event) return null;
  const [cx, cy, w, h] = event.detail.box;
  // The foot of the box, not its middle: a thing stands on the ground, and its
  // middle floats a metre above the spot you want to mark.
  const at = toGround(pose, aspect, axes, cx, cy + h / 2, high);
  if (!at) return null;
  return { pin: marker(at, TYPES[event.type] || TYPES.other), card: tags(event, at) };
}

function toGround(pose, aspect, axes, sx, sy, high) {
  const wide = Math.tan(THREE.MathUtils.degToRad(pose.hfov) / 2);
  const aim = axes.forward.clone()
    .addScaledVector(axes.right, (sx - 0.5) * 2 * wide)
    .addScaledVector(axes.up, (0.5 - sy) * 2 * wide / aspect)
    .normalize();
  // Fine close in, coarse far out. Near the bottom of the frame a metre of
  // step is several metres of ground, and the marks that matter most are the
  // ones by the roundabout, thirty metres away.
  let step = 0.25;
  for (let away = 1.5; away < 3000; away += step) {
    const east = aim.x * away;
    const north = -aim.z * away;
    if (aim.y * away <= high(east, north)) return [east, north, high(east, north), away];
    step = Math.max(0.25, away / 120);
  }
  return null;
}

function marker(at, colour) {
  const [east, north, floor] = at;
  const pin = new THREE.Group();
  const skin = new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.85 });
  // Nothing standing on the ground, and nothing joining the two parts. A post
  // with a bright head is a street lamp, and the one real lamp here was
  // surveyed and measured: a mark that looked like a second one would be
  // inventing a light that does not exist.
  const ring = new THREE.Mesh(new THREE.RingGeometry(1.15, 1.45, 32), new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(east, floor + 0.1, -north);
  const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1.1, 4), skin);
  arrow.rotation.x = Math.PI;
  arrow.position.set(east, floor + 2.9, -north);
  pin.add(ring, arrow);
  return pin;
}

function tags(event, at) {
  // Every hour on this page is the hour it was at the camera, not the hour of
  // whoever is reading: an event at dusk must not be stamped as noon.
  const words = window.ventoux || { locale: () => undefined, place: (k) => k, period: (k) => k };
  const clock = { timeZone: "Europe/Paris", hour: "2-digit", minute: "2-digit", second: "2-digit" };
  const day = { timeZone: "Europe/Paris", day: "2-digit", month: "short", year: "numeric" };
  const when = new Date(event.t);
  const detail = event.detail || {};
  const chips = [
    words.period(detail.period),
    detail.weather,
    words.place(detail.surface || event.zone),
    `${Math.round(at[3])} m`,
  ]
    .filter(Boolean)
    .map((word) => `<span class="tag">${escape(word)}</span>`)
    .join("");
  // The photograph first. The mark on the ground says where, and the words say
  // what and when, but only the picture answers the question anybody actually
  // has about an event on a webcam, which is what it looked like.
  const shot = event.thumb
    ? `<img class="shot" src="${escape(event.thumb)}" alt="" loading="lazy">`
    : "";
  return shot
    + `<div class="when">${escape(when.toLocaleTimeString(words.locale(), clock))} · ${escape(when.toLocaleDateString(words.locale(), day))}</div>`
    + `<div class="what">${escape(event.label || event.type || "")}</div>`
    + `<div class="tags">${chips}</div>`;
}

function escape(word) {
  return String(word).replace(/[&<>"]/g, (mark) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[mark]));
}

function obstacle(mark, high) {
  /* The red lamp an aircraft is meant to see, on top of the summit transmitter.
     Small on purpose: three pixels is all it is in the night picture, and made
     bigger it would look like the fire it must never be mistaken for. */
  const [east, north] = mark.at;
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(2.2, 8, 6), new THREE.MeshBasicMaterial({ color: BEACON_OFF }));
  bulb.position.set(east, high(east, north) + mark.h, -north);
  return { bulb };
}

function streetlight(lamp, high) {
  /* A mast, a head, and a light that only burns after dark. It is the brightest
     thing in the night picture, so the night scene is built around it. */
  const [east, north] = lamp.at;
  const foot = high(east, north);
  const post = new THREE.Group();
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.16, lamp.h, 6), new THREE.MeshLambertMaterial({ color: STEEL }));
  mast.position.set(east, foot + lamp.h / 2, -north);
  post.add(mast);
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.45, 10, 8), new THREE.MeshBasicMaterial({ color: LAMP_OFF }));
  bulb.position.set(east, foot + lamp.h, -north);
  post.add(bulb);
  const glow = new THREE.PointLight(LAMP_ON, 0, LAMP_REACH_M, 2);
  glow.position.set(east, foot + lamp.h - 0.2, -north);
  post.add(glow);
  return { post, bulb, glow };
}

function standing(area, colour, high) {
  /* An open structure standing on its own footprint: a post at each corner and
     a rail joining their tops. Drawn hollow because that is what it is — a
     solid block of the same size would read as a shed and hide what is behind. */
  const ring = area.p;
  const shut = ring.length > 3 && ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1];
  const corners = shut ? ring.slice(0, -1) : ring;
  const group = new THREE.Group();
  const skin = new THREE.MeshLambertMaterial({ color: colour });
  const rail = [];
  for (const [east, north] of corners) {
    const foot = high(east, north) + LIFT_M / 2;
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, area.h, 6), skin);
    post.position.set(east, foot + area.h / 2, -north);
    group.add(post);
    rail.push(new THREE.Vector3(east, foot + area.h, -north));
  }
  rail.push(rail[0].clone());
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rail), new THREE.LineBasicMaterial({ color: colour })));
  return group;
}

function slab(ring, colour, high) {
  /* A car park, drawn corner by corner. Its corners carry their own heights,
     so it lies along the slope instead of hovering over it. */
  // A closed way repeats its first point at the end. Left in, the triangulator
  // sees an edge of length zero and gives back nothing at all.
  const shut = ring.length > 3 && ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1];
  const corners = shut ? ring.slice(0, -1) : ring;
  const flat = corners.map(([east, north]) => new THREE.Vector2(east, north));
  const place = [];
  for (const triangle of THREE.ShapeUtils.triangulateShape(flat, [])) {
    for (const corner of triangle) {
      const [east, north] = corners[corner];
      place.push(east, high(east, north) + LIFT_M / 2, -north);
    }
  }
  const shape = new THREE.BufferGeometry();
  shape.setAttribute("position", new THREE.Float32BufferAttribute(place, 3));
  shape.computeVertexNormals();
  return new THREE.Mesh(shape, tarmac(colour));
}

function tarmac(colour) {
  // Pushed a hair towards the camera in the depth test: tarmac and the slope it
  // lies on are a few centimetres apart and would otherwise flicker.
  return new THREE.MeshLambertMaterial({
    color: colour,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });
}

function block(house, high) {
  /* A building, raised off its own footprint to its own height. */
  const outline = new THREE.Shape(house.p.map(([east, north]) => new THREE.Vector2(east, north)));
  const shape = new THREE.ExtrudeGeometry(outline, { depth: house.h, bevelEnabled: false });
  shape.rotateX(-Math.PI / 2);
  // Founded on the lowest corner, so a house on a slope is dug in rather than
  // left standing on one leg.
  shape.translate(0, Math.min(...house.p.map(([east, north]) => high(east, north))), 0);
  return new THREE.Mesh(shape, [
    new THREE.MeshLambertMaterial({ color: WALL }),
    new THREE.MeshLambertMaterial({ color: ROOF }),
  ]);
}

function wood(relief, high) {
  /* The woods, planted. The map gives the outline of a parcel and nothing of
     what stands in it, so the trees are sown on a fixed pattern, nudged about
     by a seeded roll of the dice: the same wood every time the page is opened,
     and never a plantation in rows. */
  const roll = dice(20260926);
  const standing = [];
  for (const parcel of relief.woods) {
    const ring = parcel.p;
    let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
    for (const [x, y] of ring) {
      west = Math.min(west, x); east = Math.max(east, x);
      south = Math.min(south, y); north = Math.max(north, y);
    }
    for (let x = west; x <= east; x += SPACING_M) {
      for (let y = south; y <= north; y += SPACING_M) {
        const at = [x + (roll() - 0.5) * SPACING_M, y + (roll() - 0.5) * SPACING_M];
        const away = Math.hypot(at[0], at[1]);
        const spacing = Math.min(SPARSEST_M, SPACING_M + away / THINNING_M);
        if (roll() > (SPACING_M * SPACING_M) / (spacing * spacing)) continue;
        if (!inside(at, ring)) continue;
        standing.push([at[0], at[1], high(at[0], at[1]), CROWN_M * (0.65 + roll() * 0.7), away]);
      }
    }
  }
  for (const [x, y, tall] of relief.trees) standing.push([x, y, high(x, y), tall, Math.hypot(x, y)]);
  if (!standing.length) return [];
  const close = standing.filter((tree) => tree[4] <= TRUNKS_M);

  const crowns = new THREE.InstancedMesh(
    new THREE.ConeGeometry(0.34, 1, 7),
    // White, because the colour of each tree is carried by the instance and
    // multiplies this one. Tinting the material would tint the whole wood.
    new THREE.MeshLambertMaterial(),
    standing.length,
  );
  const trunks = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.06, 0.09, 1, 5),
    new THREE.MeshLambertMaterial({ color: TRUNK }),
    close.length,
  );
  const sit = new THREE.Object3D();
  const tint = new THREE.Color();
  standing.forEach(([x, y, z, tall], index) => {
    const crown = tall * 0.78;
    sit.position.set(x, z + tall - crown / 2, -y);
    sit.scale.set(crown, crown, crown);
    sit.rotation.y = roll() * Math.PI;
    sit.updateMatrix();
    crowns.setMatrixAt(index, sit.matrix);
    crowns.setColorAt(index, tint.setHex(FOLIAGE[index % FOLIAGE.length]));
  });
  crowns.instanceColor.needsUpdate = true;
  close.forEach(([x, y, z, tall], index) => {
    const crown = tall * 0.78;
    sit.rotation.y = 0;
    sit.position.set(x, z + (tall - crown) / 2, -y);
    sit.scale.set(tall, tall - crown, tall);
    sit.updateMatrix();
    trunks.setMatrixAt(index, sit.matrix);
  });
  return [trunks, crowns];
}

function sampler(terrain) {
  /* How high the ground is anywhere, read between the elevation posts. */
  const { grid, step_m: step, reach_m: reach } = terrain;
  const side = grid.length;
  return (east, north) => {
    const fx = Math.min(side - 1, Math.max(0, (east + reach) / step));
    const fy = Math.min(side - 1, Math.max(0, (reach - north) / step));
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const x1 = Math.min(side - 1, x0 + 1), y1 = Math.min(side - 1, y0 + 1);
    const tx = fx - x0, ty = fy - y0;
    const top = grid[y0][x0] * (1 - tx) + grid[y0][x1] * tx;
    const low = grid[y1][x0] * (1 - tx) + grid[y1][x1] * tx;
    return top * (1 - ty) + low * ty;
  };
}

function inside(point, ring) {
  let within = false;
  for (let a = 0, b = ring.length - 1; a < ring.length; b = a, a += 1) {
    const [ax, ay] = ring[a];
    const [bx, by] = ring[b];
    if (ay > point[1] !== by > point[1] && point[0] < ((bx - ax) * (point[1] - ay)) / (by - ay) + ax) {
      within = !within;
    }
  }
  return within;
}

function dice(seed) {
  // The same wood every time the page opens, which matters: a forest that
  // reshuffles itself on reload is a forest nobody can point at.
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function column(mast, high) {
  /* A tower or an aerial. Thin, and by far the tallest thing on the skyline. */
  const [east, north] = mast.at;
  const shape = new THREE.CylinderGeometry(mast.h / 22, mast.h / 14, mast.h, 8);
  const piece = new THREE.Mesh(shape, new THREE.MeshLambertMaterial({ color: STEEL }));
  piece.position.set(east, high(east, north) + mast.h / 2, -north);
  piece.name = mast.name;
  return piece;
}

function carving(figure, high) {
  /* A carved figure or a memorial stone: a metre of shoulders on a stump. */
  const [east, north] = figure.at;
  const shape = new THREE.CylinderGeometry(figure.h / 9, figure.h / 7, figure.h, 6);
  const piece = new THREE.Mesh(shape, new THREE.MeshLambertMaterial({ color: TRUNK }));
  piece.position.set(east, high(east, north) + figure.h / 2, -north);
  piece.name = figure.name;
  return piece;
}

function here() {
  /* Where the webcam stands. Invisible from the webcam's own angle, and the
     first thing you look for once you have turned away from it. */
  return new THREE.Mesh(
    new THREE.SphereGeometry(2.5, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0xff5a5a }),
  );
}

function sun(when, lat, lon) {
  /* Where the sun stands over this spot, right now. */
  const rad = Math.PI / 180;
  const days = when.valueOf() / 86400000 - 10957.5;
  const anomaly = rad * (357.5291 + 0.98560028 * days);
  const centre = rad * (1.9148 * Math.sin(anomaly) + 0.02 * Math.sin(2 * anomaly) + 0.0003 * Math.sin(3 * anomaly));
  const ecliptic = anomaly + centre + rad * 102.9372 + Math.PI;
  const tilt = rad * 23.4397;
  const fall = Math.asin(Math.sin(ecliptic) * Math.sin(tilt));
  const ascension = Math.atan2(Math.sin(ecliptic) * Math.cos(tilt), Math.cos(ecliptic));
  const hour = rad * (280.16 + 360.9856235 * days) + rad * lon - ascension;
  const phi = rad * lat;
  const height = Math.asin(Math.sin(phi) * Math.sin(fall) + Math.cos(phi) * Math.cos(fall) * Math.cos(hour));
  const bearing = Math.atan2(Math.sin(hour), Math.cos(hour) * Math.sin(phi) - Math.tan(fall) * Math.cos(phi)) + Math.PI;
  return {
    height: height / rad,
    at: spot(Math.sin(bearing) * Math.cos(height), Math.cos(bearing) * Math.cos(height), Math.sin(height)),
  };
}

function paintHour(world, pose, parts) {
  const now = sun(new Date(), pose.lat, pose.lon);
  const lit = ease(now.height, DUSK_LOW, DUSK_HIGH);
  for (const lamp of parts.lamps || []) {
    // Full power once the sun is properly down, off in daylight, the way a
    // photocell switches it on the real roundabout.
    lamp.glow.intensity = LAMP_POWER * (1 - lit);
    lamp.bulb.material.color.setHex(lit > 0.6 ? LAMP_OFF : LAMP_ON);
  }
  for (const mark of parts.beacons || []) mark.bulb.material.color.setHex(lit > 0.6 ? BEACON_OFF : BEACON_ON);
  const hour = lit > 0.5 ? mix(HOURS.dusk, HOURS.day, (lit - 0.5) * 2) : mix(HOURS.night, HOURS.dusk, lit * 2);

  world.background = wash(hour.top, hour.low);
  parts.fill.color.setHex(hour.cool);
  parts.fill.intensity = hour.fill;
  parts.beam.color.setHex(hour.warm);
  parts.beam.intensity = hour.beam;
  // Below the horizon the sun is turned round and dimmed: what reaches the
  // ground then comes from the sky it has left behind, not from underfoot.
  const from = now.height > 0 ? now.at : now.at.clone().setY(Math.abs(now.at.y) * 0.4).normalize();
  parts.beam.position.copy(from).multiplyScalar(9000);
  parts.star.visible = now.height > -1.5;
  parts.star.position.copy(now.at).multiplyScalar(12000);
  parts.star.material.color.setHex(hour.warm);
}

function ease(value, low, high) {
  const span = Math.min(1, Math.max(0, (value - low) / (high - low)));
  return span * span * (3 - 2 * span);
}

function mix(from, to, amount) {
  const blend = (a, b) => new THREE.Color(a).lerp(new THREE.Color(b), amount).getHex();
  const step = (a, b) => a + (b - a) * amount;
  return {
    top: blend(from.top, to.top), low: blend(from.low, to.low),
    warm: blend(from.warm, to.warm), cool: blend(from.cool, to.cool),
    beam: step(from.beam, to.beam), fill: step(from.fill, to.fill),
  };
}

function wash(top, low) {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 128;
  const paint = canvas.getContext("2d");
  const shade = paint.createLinearGradient(0, 0, 0, 128);
  shade.addColorStop(0, `#${top.toString(16).padStart(6, "0")}`);
  shade.addColorStop(1, `#${low.toString(16).padStart(6, "0")}`);
  paint.fillStyle = shade;
  paint.fillRect(0, 0, 2, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
