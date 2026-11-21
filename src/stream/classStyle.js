// Per-class base colours of the streamed road meshes (vertex colours). src/look matches classes from these colours,
// so both import this one table. Same values as the original full-load renderer.
export const CLASS_STYLE = {
  motorway:       { color: 0xffb340, y: 0.030, glow: true },
  motorway_link:  { color: 0xc98a2c, y: 0.026, glow: false },
  trunk:          { color: 0xff7d6b, y: 0.028, glow: true },
  trunk_link:     { color: 0xb55a4d, y: 0.024, glow: false },
  primary:        { color: 0xe8e2c8, y: 0.020, glow: false },
  primary_link:   { color: 0x9c9888, y: 0.018, glow: false },
  secondary:      { color: 0x7f8bb0, y: 0.016, glow: false },
  secondary_link: { color: 0x5e6886, y: 0.014, glow: false },
  tertiary:       { color: 0x8c96b4, y: 0.010 },
  tertiary_link:  { color: 0x8c96b4, y: 0.010 },
  unclassified:   { color: 0x6f7896, y: 0.006 },
  residential:    { color: 0x6c7593, y: 0.006 },
  living_street:  { color: 0x5f6886, y: 0.004 },
};
