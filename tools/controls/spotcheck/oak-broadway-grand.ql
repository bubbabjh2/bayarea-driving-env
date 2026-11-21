[out:json][timeout:60];
way["highway"]["name"~"^Broadway$"](around:500,37.8115,-122.2685)->.a;
way["highway"]["name"~"Grand Avenue$"](around:500,37.8115,-122.2685)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;
