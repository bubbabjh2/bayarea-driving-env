[out:json][timeout:60];
way["highway"]["name"~"^Van Ness Avenue$"](around:500,37.8,-122.4245)->.a;
way["highway"]["name"~"^Lombard Street$"](around:500,37.8,-122.4245)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;