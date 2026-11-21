// Node loaders for the world data (road packs, lane geometry, junctions, controls) used by the
// lane-line tools (tools/lines/*).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { RoadSet } from '../../src/roadpack.js';
import { ensureLaneGeometry, JunctionSet, buildJunctions, crosswalkLegs } from '../../src/geom/index.js';

export const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const T = { Float32: Float32Array, Uint32: Uint32Array, Uint16: Uint16Array, Uint8: Uint8Array, Int32: Int32Array, Float64: Float64Array };

export function loadPack(dir, metaFile) {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, metaFile), 'utf8'));
  const bin = fs.readFileSync(path.join(dir, meta.file));
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const arrays = {};
  for (const [name, s] of Object.entries(meta.sections)) arrays[name === 'nodeXYZ' ? 'xyz' : name] = new T[s.type](buf, s.byteOffset, s.length);
  return new RoadSet(meta, arrays);
}

/** Typed-array sections of a `{file, sections}` meta (junctions.bin, lines.bin, controls.bin). */
export function loadSections(dir, metaFile) {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, metaFile), 'utf8'));
  const bin = fs.readFileSync(path.join(dir, meta.file));
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const a = {};
  for (const [k, s] of Object.entries(meta.sections)) a[k] = new T[s.type](buf, s.byteOffset, s.length);
  return { meta, arrays: a };
}

/**
 * The world as the sim sees it: main + local packs, merged lane geometry, the junction pack
 * (data/junctions.bin, built with crosswalk legs; rebuilt in memory with the same legs if it does
 * not match), controls.json and crosswalks.json.
 */
export function loadWorld(dir = path.join(ROOT_DIR, 'data'), { junctions = true, log = () => {} } = {}) {
  const main = loadPack(dir, 'roads.meta.json');
  const local = fs.existsSync(path.join(dir, 'roads-local.meta.json')) ? loadPack(dir, 'roads-local.meta.json') : null;
  const LG = ensureLaneGeometry(main, local);
  const ctlPath = path.join(dir, 'controls.json'), cwPath = path.join(dir, 'crosswalks.json');
  const controls = fs.existsSync(ctlPath) ? JSON.parse(fs.readFileSync(ctlPath, 'utf8')) : null;
  const crosswalks = fs.existsSync(cwPath) ? JSON.parse(fs.readFileSync(cwPath, 'utf8')) : null;
  let J = null;
  if (junctions) {
    const jm = path.join(dir, 'junctions.meta.json');
    if (fs.existsSync(jm)) {
      const { meta, arrays } = loadSections(dir, 'junctions.meta.json');
      if (meta.mainEdgeCount === main.edgeCount && arrays.mouthS.length === 2 * LG.G.edgeCount) J = new JunctionSet(LG.G, LG, arrays);
      else log('data/junctions.bin does not match the roads: building junctions in memory (with crosswalk legs)');
    }
    if (!J) J = buildJunctions(LG, crosswalkLegs(LG, controls ? controls.controls : null, crosswalks ? crosswalks.points : null));
  }
  return { dir, main, local, LG, G: LG.G, J, controls, crosswalks };
}
