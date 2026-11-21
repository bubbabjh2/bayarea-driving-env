[out:json][timeout:60];
way["highway"]["name"~"El Camino Real$"](around:500,37.4876,-122.2290)->.a;
way["highway"]["name"~"Whipple Avenue$"](around:500,37.4876,-122.2290)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;
