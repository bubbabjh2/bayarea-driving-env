// Shared lane geometry (world-owned). See README.md.
export { LANE_W, LANE_WIDTH_CLASS, CLASSES_ALL, FREEWAY, LaneGraph, graphFromRoadSets } from './graph.js';
export { hopMap, mapLane, mapLatStrict, turnAngle, pointInto } from './hops.js';
export { buildLaneGeometry, LaneGeometry } from './lanes.js';
export { buildJunctions, loadJunctions, crosswalkLegs, JunctionSet, JUNCTION_SECTIONS } from './junctions.js';
import { graphFromRoadSets } from './graph.js';
import { buildLaneGeometry } from './lanes.js';

/** Build (once) and cache the merged-graph lane geometry on the main RoadSet. */
export function ensureLaneGeometry(main, local = null) {
  const key = local ? 'laneGeomMerged' : 'laneGeomMain';
  if (main[key]) return main[key];
  const G = graphFromRoadSets(main, local);
  const LG = buildLaneGeometry(G);
  main[key] = LG;
  if (local) main.laneGeomMain = main.laneGeomMain || LG;   // edge ids of the main tier are identical
  return LG;
}
export { buildLineSet, loadLineSet, lineSetFromBuffers, emitPaint, markingLevel, LineSet, LINE, LINE_NAMES, COLOR, MARK, LEVEL, SPAN, PATTERNS, PAINT, OUTER_R, OUTER_L, OCCLUDED_BIT, LINE_SECTIONS, LINE_GEOM_SECTIONS } from './lines.js';
