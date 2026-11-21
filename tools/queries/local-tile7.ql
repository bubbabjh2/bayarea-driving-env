[out:json][timeout:600];
(
  way["highway"~"^(tertiary|tertiary_link|unclassified|residential|living_street)$"](37.83,-122.56,37.98,-122.25);
);
out body;
>;
out skel qt;
