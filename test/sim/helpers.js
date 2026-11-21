import { buildGraph } from '../../src/sim/roadGraph.js';
import { makeSynthGraph } from '../../src/sim/synthGraph.js';
import { synthAxis } from '../../src/sim/population.js';

export function smallGraph(opts = {}) {
  return buildGraph(makeSynthGraph({ blocks: 12, cross: 6, bridges: 2, ...opts }));
}
export function popOpts(graph, extra = {}) {
  return { axis: synthAxis(graph), long: { minM: 5000, maxM: 16000 }, departures: 'spec', ...extra };
}
