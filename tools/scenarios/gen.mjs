// node tools/scenarios/gen.mjs <family[:k=v,...]> [--seed 1] [--out data/scenarios/<name>]   one scenario
// node tools/scenarios/gen.mjs pack [--count 60] [--families a,b,c] [--seed 1] [--rate 1] [--out dir]   a world pack
// node tools/scenarios/gen.mjs all [--seed 1]          one scenario per family (data/scenarios/<family>-s<seed>)
// Writes the real data format (roads*, stations*, controls*, crosswalks, crossings, junctions via
// tools/build_junctions.mjs) + scenario.json.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeScenario, FAMILIES } from '../../src/scenarios/families.js';
import { mixedPack, parseSpec } from '../../src/scenarios/pack.js';
import { writeScenario } from '../../src/scenarios/write.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i < 0 ? d : args[i + 1]; };
const what = args[0] || 'all', seed = +opt('seed', 1);
const outOf = (name) => opt('out', path.join(ROOT, 'data/scenarios', name));
const t0 = Date.now();
if (what === 'pack') {
  const fams = opt('families', null);
  const w = mixedPack({ count: +opt('count', 60), seed, rateScale: +opt('rate', 1), families: fams ? fams.split(',') : undefined, name: opt('name', null) });
  const doc = w.finish(); writeScenario(doc, outOf(w.name));
  console.log(`${w.name}: ${doc.scenario.tiles.length} tiles, ${doc.scenario.edgeCount} edges, ${doc.scenario.trips.reduce((s, t) => s + t.count, 0)} trips -> ${outOf(w.name)} (${Date.now() - t0} ms)`);
} else {
  const specs = what === 'all' ? Object.keys(FAMILIES) : [what];
  for (const s of specs) {
    const [fam, ov] = parseSpec(s);
    const w = makeScenario(fam, seed, ov);
    const doc = w.finish(); const dir = what === 'all' ? path.join(ROOT, 'data/scenarios', w.name) : outOf(w.name);
    writeScenario(doc, dir);
    console.log(`${w.name}: ${doc.scenario.edgeCount} edges, ${doc.scenario.trips.reduce((q, t) => q + t.count, 0)} trips; ${w.notes.join('; ')}`);
  }
}
