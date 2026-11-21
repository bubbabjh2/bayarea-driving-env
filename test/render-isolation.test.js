// The sim never loads rendering code: no file under src/sim, src/gpu, src/nav, src/geom, src/scenarios (or the
// top-level data modules and examples) imports three.js, src/look, src/stream, src/viewer or vendor/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIRS = ['src/sim', 'src/gpu', 'src/nav', 'src/geom', 'src/scenarios', 'examples'];
const FILES = ['src/roadpack.js', 'src/proj.js'];
const RENDER = /(?:from\s*|import\s*\(\s*)['"](three(?:\/[^'"]*)?|[^'"]*\/(?:look|viewer|vendor)\/[^'"]*|(?:\.\.\/)+stream\/[^'"]*)['"]/g;

function* walk(d) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) { if (e.name !== 'cache') yield* walk(p); }
    else if (/\.(m?js)$/.test(e.name)) yield p;
  }
}

test('sim / GPU / nav code never imports rendering modules', () => {
  const files = [...DIRS.flatMap((d) => [...walk(path.join(ROOT, d))]), ...FILES.map((f) => path.join(ROOT, f))];
  assert.ok(files.length > 20);
  const hits = [];
  for (const f of files) for (const m of fs.readFileSync(f, 'utf8').matchAll(RENDER)) hits.push(`${path.relative(ROOT, f)}: ${m[1]}`);
  assert.deepEqual(hits, []);
});
