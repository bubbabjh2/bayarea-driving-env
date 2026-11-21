[out:json][timeout:60];
way["highway"]["name"~"^Mission Street$"](around:500,37.7651,-122.4196)->.a;
way["highway"]["name"~"^16th Street$"](around:500,37.7651,-122.4196)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;