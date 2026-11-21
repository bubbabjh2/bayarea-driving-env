[out:json][timeout:60];
way["highway"]["name"~"^Fremont Boulevard$"](around:500,37.5513,-121.9837)->.a;
way["highway"]["name"~"Mowry Avenue$"](around:500,37.5513,-121.9837)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;
