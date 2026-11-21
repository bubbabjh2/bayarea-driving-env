[out:json][timeout:600];
(
  way["highway"~"^(tertiary|tertiary_link|unclassified|residential|living_street)$"](37.65,-122.16,37.85,-121.80);
);
out body;
>;
out skel qt;
