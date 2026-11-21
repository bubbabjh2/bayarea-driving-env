[out:json][timeout:60];
way["highway"]["name"~"^The Alameda$"](around:500,37.3327,-121.9124)->.a;
way["highway"]["name"~"Race Street$"](around:500,37.3327,-121.9124)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;