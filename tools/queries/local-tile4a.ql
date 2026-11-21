[out:json][timeout:600];
(
  way["highway"~"^(tertiary|tertiary_link|unclassified|residential|living_street)$"](37.45,-122.16,37.65,-121.98);
);
out body;
>;
out skel qt;
