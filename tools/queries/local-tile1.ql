[out:json][timeout:600];
(
  way["highway"~"^(tertiary|tertiary_link|unclassified|residential|living_street)$"](37.25,-122.52,37.45,-122.16);
);
out body;
>;
out skel qt;
