[out:json][timeout:300];
(
  way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link)$"](37.80,-122.56,37.98,-122.25);
);
out body;
>;
out skel qt;
