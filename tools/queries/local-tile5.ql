[out:json][timeout:600];
(
  way["highway"~"^(tertiary|tertiary_link|unclassified|residential|living_street)$"](37.65,-122.52,37.85,-122.16);
);
out body;
>;
out skel qt;
