// Shared fixtures for the brain/nav tests.
import { loadGraph } from '../src/nav/graph.js';
import { Router } from '../src/nav/astar.js';
import { makeSyntheticGraph } from '../src/nav/synthetic.js';

let cached = null;
export function synthetic() {
  if (!cached) { const graph = loadGraph(makeSyntheticGraph()); cached = { graph, router: new Router(graph) }; }
  return cached;
}
export const idx = (graph, id) => graph.indexOf(id);
