import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// The colours the map is read in. Not a palette chosen for itself: a road has
// to look like tarmac from above so the shape of the roundabout is legible.
const PAINT = {
  r: 0x6a6a70, o: 0x75757b, i: 0x8b7f6a, p: 0x7e7e84, t: 0xa9946f,
  m: 0x85a95f, f: 0x47713b, b: 0x9d7d64, e: 0xaaa195, ".": 0x938b77,
};
// The sky, and the light on the ground, at three moments of the day. What is
// shown is what the sun is really doing over the Ventoux at this instant, so
// the relief is dark when the webcam above it is dark.
const HOURS = {
  day: { top: 0x3f6ea8, low: 0xa9c5de, warm: 0xfff4e2, beam: 1.05, fill: 1.35, cool: 0xe4edf8 },
  dusk: { top: 0x2b3352, low: 0xd98b5a, warm: 0xffc089, beam: 0.80, fill: 0.95, cool: 0xa8b2cc },
  night: { top: 0x0d1524, low: 0x27374e, warm: 0xc2d0ea, beam: 0.30, fill: 1.05, cool: 0x9fb2d4 },
};
// Between these two heights of the sun the light turns over. Above, it is day;
// below, the ground keeps only what the sky still gives it.
const DUSK_LOW = -7;
const DUSK_HIGH = 7;
// Beyond this much depth change across one cell, the two corners are not the
// same slope: one is a ridge and the other the valley a kilometre behind it.
// Joining them would stretch a curtain of rock across the gap.
const CLIFF = 1.35;

const host = document.getElementById("relief");
if (host) start(host).catch(() => host.closest(".block")?.setAttribute("hidden", ""));

async function start(host) {
  const scene = await fetch("data/scene.json", { cache: "no-store" }).then((r) => r.json());
  const pose = scene.pose;
  const aspect = pose.aspect || 16 / 9;
  pose.aspect = aspect;
  const axes = frame(pose);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  host.appendChild(renderer.domElement);

  const world = new THREE.Scene();
  world.add(ground(scene, pose, axes));
  world.add(marker());

  const fill = new THREE.AmbientLight(0xffffff, 1);
  const beam = new THREE.DirectionalLight(0xffffff, 1);
  const star = new THREE.Mesh(
    new THREE.SphereGeometry(70, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xfff6dd }),
  );
  world.add(fill, beam, star);
  const daylight = () => paintHour(world, pose, { fill, beam, star });
  daylight();
  setInterval(daylight, 60000);

  const camera = new THREE.PerspectiveCamera(vertical(pose.hfov, aspect), aspect, 1, 12000);
  const look = axes.forward.clone().multiplyScalar(centre(scene) || 500);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxDistance = 6000;
  const marks = pins(scene, pose, axes);
  world.add(marks.points);

  function home() {
    camera.position.set(0, 0, 0);
    camera.up.copy(axes.up);
    controls.target.copy(look);
    controls.update();
  }
  home();
  document.getElementById("relief-back")?.addEventListener("click", home);

  const caption = document.getElementById("relief-name");
  const finder = new THREE.Raycaster();
  finder.params.Points.threshold = 14;
  const cursor = new THREE.Vector2();
  renderer.domElement.addEventListener("pointermove", (event) => {
    const box = renderer.domElement.getBoundingClientRect();
    cursor.x = ((event.clientX - box.left) / box.width) * 2 - 1;
    cursor.y = -((event.clientY - box.top) / box.height) * 2 + 1;
    finder.setFromCamera(cursor, camera);
    const hit = finder.intersectObject(marks.points)[0];
    caption.textContent = hit ? marks.names[hit.index] : "";
  });

  function fit() {
    const width = host.clientWidth;
    renderer.setSize(width, Math.round(width / aspect));
  }
  window.addEventListener("resize", fit);
  fit();

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(world, camera);
  });
}

function frame(pose) {
  /* The camera's own three directions, in east-north-up, then turned into the
     way three.js holds the world: x east, y up, z south. */
  const yaw = THREE.MathUtils.degToRad(pose.yaw);
  const pitch = THREE.MathUtils.degToRad(pose.pitch);
  const flat = [Math.sin(yaw), Math.cos(yaw)];
  const enu = (east, north, up) => new THREE.Vector3(east, up, -north);
  return {
    right: enu(Math.cos(yaw), -Math.sin(yaw), 0),
    forward: enu(flat[0] * Math.cos(pitch), flat[1] * Math.cos(pitch), -Math.sin(pitch)),
    up: enu(flat[0] * Math.sin(pitch), flat[1] * Math.sin(pitch), Math.cos(pitch)),
  };
}

function vertical(hfov, aspect) {
  const half = Math.tan(THREE.MathUtils.degToRad(hfov) / 2);
  return THREE.MathUtils.radToDeg(2 * Math.atan(half / aspect));
}

function ray(pose, axes, sx, sy) {
  /* The direction the camera looks at this point of the picture. The same
     formula the watcher uses to read the ground, so the two agree. */
  const half = Math.tan(THREE.MathUtils.degToRad(pose.hfov) / 2);
  const nx = (sx - 0.5) * 2 * half;
  const ny = ((0.5 - sy) * 2 * half) / (pose.aspect || 16 / 9);
  return axes.forward
    .clone()
    .addScaledVector(axes.right, nx)
    .addScaledVector(axes.up, ny)
    .normalize();
}

function reader(reach) {
  /* How far the ground is at any point of the picture, read between the cells
     so the slope comes out smooth instead of stepped. Zero means sky, and a
     cell that touches sky stays sky: guessing there would drape the skyline. */
  const rows = reach.length;
  const columns = reach[0].length;
  return (sx, sy) => {
    const fx = Math.min(columns - 1, Math.max(0, sx * columns - 0.5));
    const fy = Math.min(rows - 1, Math.max(0, sy * rows - 0.5));
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(columns - 1, x0 + 1);
    const y1 = Math.min(rows - 1, y0 + 1);
    const a = reach[y0][x0];
    const b = reach[y0][x1];
    const c = reach[y1][x0];
    const d = reach[y1][x1];
    if (!a || !b || !c || !d) return 0;
    const tx = fx - x0;
    const ty = fy - y0;
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
  };
}

function ground(scene, pose, axes) {
  const grid = scene.grid;
  const rows = grid.length;
  const columns = grid[0].length;
  const depth = reader(scene.reach);
  const position = [];
  const colour = [];
  const normal = [];
  const tint = new THREE.Color();

  const corner = (col, row) => {
    const sx = col / columns;
    const sy = row / rows;
    const span = depth(sx, sy);
    return span ? ray(pose, axes, sx, sy).multiplyScalar(span) : null;
  };

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const letter = grid[row][col];
      if (letter === "s") continue;
      const quad = [corner(col, row), corner(col + 1, row), corner(col + 1, row + 1), corner(col, row + 1)];
      if (quad.some((point) => point === null)) continue;
      const lengths = quad.map((point) => point.length());
      if (Math.max(...lengths) / Math.min(...lengths) > CLIFF) continue;
      tint.setHex(PAINT[letter] ?? PAINT["."]);
      const facing = upward(quad);
      for (const index of [0, 1, 2, 0, 2, 3]) {
        position.push(quad[index].x, quad[index].y, quad[index].z);
        colour.push(tint.r, tint.g, tint.b);
        normal.push(facing.x, facing.y, facing.z);
      }
    }
  }

  const shape = new THREE.BufferGeometry();
  shape.setAttribute("position", new THREE.Float32BufferAttribute(position, 3));
  shape.setAttribute("color", new THREE.Float32BufferAttribute(colour, 3));
  shape.setAttribute("normal", new THREE.Float32BufferAttribute(normal, 3));
  return new THREE.Mesh(shape, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
}

function upward(quad) {
  /* The way the ground faces here. Turned to the sky whatever order the corners
     came in, so a slope is never lit from underneath. */
  const facing = new THREE.Vector3()
    .subVectors(quad[1], quad[0])
    .cross(new THREE.Vector3().subVectors(quad[3], quad[0]))
    .normalize();
  return facing.y < 0 ? facing.negate() : facing;
}

function centre(scene) {
  return reader(scene.reach)(0.5, 0.5);
}

function pins(scene, pose, axes) {
  /* The named things, put back where they stand. The map already knows where
     each one sits in the picture and how far it is, which is all it takes. */
  const position = [];
  const names = [];
  for (const mark of scene.landmarks || []) {
    const spot = ray(pose, axes, mark.x, mark.y).multiplyScalar(mark.distance_m);
    position.push(spot.x, spot.y, spot.z);
    names.push(mark.name);
  }
  const shape = new THREE.BufferGeometry();
  shape.setAttribute("position", new THREE.Float32BufferAttribute(position, 3));
  const dot = new THREE.PointsMaterial({ color: 0xffd479, size: 9, sizeAttenuation: false });
  return { points: new THREE.Points(shape, dot), names };
}

function marker() {
  /* Where the webcam stands. Invisible from the webcam's own angle, and the
     first thing you look for once you have turned away from it. */
  const post = new THREE.Mesh(
    new THREE.SphereGeometry(1.6, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0xff5a5a }),
  );
  post.position.set(0, 0, 0);
  return post;
}

function sun(when, lat, lon) {
  /* Where the sun stands over this spot, right now. Returned as a direction in
     the same east-north-up frame the camera is described in. */
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
    // East, up, south: the way three.js holds the world here.
    at: new THREE.Vector3(
      Math.sin(bearing) * Math.cos(height),
      Math.sin(height),
      -Math.cos(bearing) * Math.cos(height),
    ),
  };
}

function paintHour(world, pose, parts) {
  const now = sun(new Date(), pose.lat, pose.lon);
  const lit = ease(now.height, DUSK_LOW, DUSK_HIGH);
  const hour = lit > 0.5 ? mix(HOURS.dusk, HOURS.day, (lit - 0.5) * 2) : mix(HOURS.night, HOURS.dusk, lit * 2);

  world.background = wash(hour.top, hour.low);
  parts.fill.color.setHex(hour.cool);
  parts.fill.intensity = hour.fill;
  parts.beam.color.setHex(hour.warm);
  parts.beam.intensity = hour.beam;
  // Below the horizon the sun is turned round and dimmed: what reaches the
  // ground then comes from the sky it has left behind, not from underfoot.
  const from = now.height > 0 ? now.at : now.at.clone().setY(Math.abs(now.at.y) * 0.4).normalize();
  parts.beam.position.copy(from).multiplyScalar(4000);
  parts.star.visible = now.height > -1.5;
  parts.star.position.copy(now.at).multiplyScalar(5500);
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
