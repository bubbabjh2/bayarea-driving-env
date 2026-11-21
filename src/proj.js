// Projection helpers. Mirrors tools/build_roads.py exactly.
//   x = (lon - lon0) * metersPerDegLon    (+x = east)
//   z = -(lat - lat0) * metersPerDegLat   (+z = south, north = -z)
// y (up) is not part of the data; the viewer derives it from bridge/layer.

export const DEFAULT_PROJECTION = {
  lat0: 37.55,
  lon0: -122.15,
  earthRadius: 6371008.8,
};

export function makeProjection(meta) {
  const p = Object.assign({}, DEFAULT_PROJECTION, meta && meta.projection ? meta.projection : {});
  const kz = p.metersPerDegLat || (p.earthRadius * Math.PI) / 180;
  const kx = p.metersPerDegLon || kz * Math.cos((p.lat0 * Math.PI) / 180);
  return {
    lat0: p.lat0,
    lon0: p.lon0,
    kx,
    kz,
    toXZ(lat, lon) {
      return { x: (lon - p.lon0) * kx, z: -(lat - p.lat0) * kz };
    },
    toLatLon(x, z) {
      return { lat: p.lat0 - z / kz, lon: p.lon0 + x / kx };
    },
  };
}

// Right-hand perpendicular (on a north-up map) of travel direction (dx, dz).
export function rightOf(dx, dz) {
  return { x: -dz, z: dx };
}
