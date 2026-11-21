// World packs: many scenarios tiled into ONE disjoint graph (spaced apart, never connected), so one sim
// instance (CPU or GPU) runs hundreds of mini-worlds at once. Each scenario keeps its trips (tagged
// with its tile index: pop.tripTile) so results split back per scenario / family (load.js scenarioReport).
//
//   const w = packWorld(specs.map(([f, seed, o]) => makeScenario(f, seed, o)), { name: 'pack-mixed-s1' });
//   writeScenario(w.finish(), 'data/scenarios/pack-mixed-s1');
import { World } from './builder.js';
import { makeScenario, FAMILIES } from './families.js';
import { makeRng, hashSeed } from '../sim/rng.js';

/** Shelf-pack worlds left to right in rows (row width ~ sqrt(total area)), `gap` m between tile boxes. */
export function packWorld(worlds, { name = 'pack', gap = 400, seed = 1, family = 'pack' } = {}) {
  const boxes = worlds.map((w) => w.bbox());
  const area = boxes.reduce((s, b) => s + (b.maxX - b.minX + gap) * (b.maxZ - b.minZ + gap), 0);
  const rowW = Math.max(3000, Math.sqrt(area) * 1.1);
  const pack = new World({ name, family, seed, params: { tiles: worlds.length, gap } });
  let x = 0, z = 0, rowH = 0;
  worlds.forEach((w, k) => {
    const b = boxes[k], wd = b.maxX - b.minX, ht = b.maxZ - b.minZ;
    if (x > 0 && x + wd > rowW) { x = 0; z += rowH + gap; rowH = 0; }
    pack.append(w, x - b.minX, z - b.minZ, k);
    x += wd + gap; rowH = Math.max(rowH, ht);
  });
  // centre the pack on (0, 0) like the single scenarios
  const bb = pack.bbox(), cx = (bb.minX + bb.maxX) / 2, cz = (bb.minZ + bb.maxZ) / 2;
  for (let i = 0; i < pack.nx.length; i++) { pack.nx[i] -= cx; pack.nz[i] -= cz; }
  for (const t of pack.tiles) { t.offset = [t.offset[0] - cx, t.offset[1] - cz]; if (t.view) { t.view.x -= cx; t.view.z -= cz; } t.bbox = { minX: t.bbox.minX - cx, maxX: t.bbox.maxX - cx, minZ: t.bbox.minZ - cz, maxZ: t.bbox.maxZ - cz }; }
  return pack;
}

/**
 * A mixed pack: `count` scenarios drawn from `families` (default all), seeds from `seed`.
 * spec strings: 'family' or 'family:key=value,key=value' (overrides).
 */
export function mixedPack({ families = Object.keys(FAMILIES), count = 60, seed = 1, rateScale = 1, name = null, gap = 400 } = {}) {
  const R = makeRng(hashSeed(`pack:${seed}`)), worlds = [];
  for (let k = 0; k < count; k++) {
    const spec = families[k % families.length];
    const [fam, ov] = parseSpec(spec);
    worlds.push(makeScenario(fam, 1 + R.int(1e6), ov, { rateScale }));
  }
  return packWorld(worlds, { name: name || `pack-${count}-s${seed}`, seed, gap });
}

/** 'junction:control=signal,movement=left' -> ['junction', {control: 'signal', movement: 'left'}] (numbers parsed). */
export function parseSpec(spec) {
  const [fam, rest] = String(spec).split(':');
  const ov = {};
  if (rest) for (const kv of rest.split(',')) { const [k, v] = kv.split('='); ov[k] = v === 'true' ? true : v === 'false' ? false : isNaN(+v) ? v : +v; }
  return [fam, ov];
}
