import test from 'node:test';
import assert from 'node:assert/strict';
import { VehicleStore } from '../../src/sim/vehicle.js';
import { SpatialHash, LaneIndex, rayRect } from '../../src/sim/spatial.js';
import { makeRng } from '../../src/sim/rng.js';

function randomStore(n, seed = 1) {
  const rng = makeRng(seed);
  const vs = new VehicleStore(n);
  for (let i = 0; i < n; i++) {
    const s = vs.alloc({ route: Int32Array.from([rng.int(50)]) });
    vs.x[s] = rng.range(-500, 500); vs.z[s] = rng.range(-500, 500); vs.heading[s] = rng.range(-Math.PI, Math.PI);
    vs.edge[s] = rng.int(50); vs.lane[s] = rng.int(3); vs.s[s] = rng.range(0, 400);
    vs.laneLo[s] = vs.lane[s]; vs.laneHi[s] = Math.min(3, vs.lane[s] + (rng.random() < 0.3 ? 1 : 0)); // some straddle two lanes
  }
  return vs;
}

test('spatial hash neighbour queries match brute force', () => {
  const vs = randomStore(1500);
  const hash = new SpatialHash({ cellSize: 25, tableBits: 14, capacity: 1500 });
  hash.rebuild(vs);
  const rng = makeRng(9);
  for (let q = 0; q < 200; q++) {
    const x = rng.range(-520, 520), z = rng.range(-520, 520), r = rng.range(5, 80);
    const got = hash.near(x, z, r).sort((a, b) => a - b);
    const want = [];
    for (let i = 0; i < vs.high; i++) if ((vs.x[i] - x) ** 2 + (vs.z[i] - z) ** 2 <= r * r) want.push(i);
    assert.deepEqual(got, want);
  }
});

test('raycast hits the nearest vehicle box along the ray', () => {
  const vs = new VehicleStore(8);
  const a = vs.alloc({ route: Int32Array.from([0]) }); vs.x[a] = 30; vs.z[a] = 0; vs.heading[a] = 0;
  const b = vs.alloc({ route: Int32Array.from([0]) }); vs.x[b] = 60; vs.z[b] = 0; vs.heading[b] = 0;
  const c = vs.alloc({ route: Int32Array.from([0]) }); vs.x[c] = 30; vs.z[c] = 10; vs.heading[c] = 0;
  const hash = new SpatialHash({ cellSize: 25, tableBits: 10, capacity: 8 });
  hash.rebuild(vs);
  const hit = hash.raycast(0, 0, 1, 0, 100, -1);
  assert.equal(hit.slot, a);
  assert.ok(Math.abs(hit.dist - (30 - 4.5 / 2)) < 1e-6);
  const miss = hash.raycast(0, 0, 0, 1, 100, -1);
  assert.equal(miss.hit, false);
  assert.ok(rayRect(0, 0, 1, 0, 10, 0, Math.PI / 2, 4, 2) > 0);
});

test('footprint lane index: a car is listed in every lane it overlaps, sorted by s; ahead/behind per lane', () => {
  const vs = randomStore(800, 3);
  const li = new LaneIndex(60, 800);
  li.rebuild(vs);
  const inLane = (j, l) => l >= vs.laneLo[j] && l <= vs.laneHi[j];
  let straddlers = 0;
  for (let i = 0; i < vs.high; i++) {
    if (vs.laneHi[i] > vs.laneLo[i]) straddlers++;
    for (let l = vs.laneLo[i]; l <= vs.laneHi[i]; l++) {
      const a = li.ahead(i, l), b = li.behind(i, l);
      let bestA = -1, bestB = -1;
      for (let j = 0; j < vs.high; j++) {
        if (j === i || vs.edge[j] !== vs.edge[i] || !inLane(j, l)) continue;
        const after = vs.s[j] > vs.s[i] || (vs.s[j] === vs.s[i] && j > i);
        if (after) { if (bestA < 0 || vs.s[j] < vs.s[bestA] || (vs.s[j] === vs.s[bestA] && j < bestA)) bestA = j; }
        else if (bestB < 0 || vs.s[j] > vs.s[bestB] || (vs.s[j] === vs.s[bestB] && j > bestB)) bestB = j;
      }
      assert.equal(a, bestA); assert.equal(b, bestB);
    }
  }
  assert.ok(straddlers > 100);
  const [ah, bh] = li.neighborsAt(vs.edge[0], vs.lane[0], vs.s[0] + 0.001);
  if (ah >= 0) assert.ok(vs.s[ah] >= vs.s[0] + 0.001);
  if (bh >= 0) assert.ok(vs.s[bh] < vs.s[0] + 0.001);
  // rebuild after moving everyone to other keys leaves no stale buckets
  for (let i = 0; i < vs.high; i++) vs.edge[i] = 55;
  li.rebuild(vs);
  for (let e = 0; e < 50; e++) for (let l = 0; l < 4; l++) assert.equal(li.first(e, l), -1);
});
