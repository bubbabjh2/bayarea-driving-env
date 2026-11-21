[out:json][timeout:600][maxsize:536870912];
(
  node["highway"~"^(stop|give_way)$"](37.25,-122.56,37.97,-121.80);
);
out body;
