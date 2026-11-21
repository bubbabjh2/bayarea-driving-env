// Write a finished World (builder.js finish()) as a data directory in the real world's format and
// build its junctions with the world's own pipeline (tools/build_junctions.mjs: lane geometry,
// connectors, conflict matrix, hulls, gores, crosswalk set-backs, and controls' stop lines moved to
// the junction mouths). Node only.
//
//   writeScenario(world.finish(), 'data/scenarios/left-signal-s1')
//
// Files (same names / formats as data/): roads.{bin,meta.json,json}, roads-local.{bin,meta.json,json},
// stations.{bin,meta.json,json}, controls.{bin,meta.json,json}, crosswalks.json, crossings.json,
// junctions.{bin,meta.json} (+ scenario.json: trips, obstacles, tiles, params).
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

export function writeScenario(doc, dir, { junctions = true, json = true, quiet = true } = {}) {
  fs.mkdirSync(dir, { recursive: true });
  const w = (f, data) => fs.writeFileSync(path.join(dir, f), data);
  const wj = (f, obj, pretty = false) => w(f, JSON.stringify(obj, null, pretty ? 1 : 0));
  for (const [tier, name] of [['main', 'roads'], ['local', 'roads-local']]) {
    const p = doc[tier];
    w(`${name}.bin`, Buffer.from(p.bin));
    wj(`${name}.meta.json`, p.meta, true);
    if (json) wj(`${name}.json`, p.json);
  }
  w('stations.bin', Buffer.from(doc.stations.bin)); wj('stations.meta.json', doc.stations.meta, true); wj('stations.json', doc.stations.json);
  w('controls.bin', Buffer.from(doc.controls.bin)); wj('controls.meta.json', doc.controls.meta, true); wj('controls.json', doc.controls.json);
  wj('crosswalks.json', doc.crosswalks);
  wj('crossings.json', doc.crossings);
  wj('scenario.json', doc.scenario, true);
  if (junctions) {
    const out = execFileSync(process.execPath, [path.join(ROOT, 'tools/build_junctions.mjs'), path.resolve(dir)], { encoding: 'utf8' });
    if (!quiet) process.stdout.write(out);
    // keep the scenario doc's stop lines in sync with the patched controls (informational)
  }
  return dir;
}
