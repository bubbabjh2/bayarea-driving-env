[out:json][timeout:300];
(
  way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link)$"](37.25,-122.52,37.85,-121.80);
);
out body;
>;
out skel qt;
