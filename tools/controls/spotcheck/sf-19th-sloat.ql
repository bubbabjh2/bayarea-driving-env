[out:json][timeout:60];
way["highway"]["name"~"^19th Avenue$"](around:500,37.7347,-122.4752)->.a;
way["highway"]["name"~"^Sloat Boulevard$"](around:500,37.7347,-122.4752)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;