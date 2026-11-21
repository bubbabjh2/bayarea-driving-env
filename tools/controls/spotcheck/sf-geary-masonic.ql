[out:json][timeout:60];
way["highway"]["name"~"^Geary Boulevard$"](around:500,37.7822,-122.4467)->.a;
way["highway"]["name"~"^Masonic Avenue$"](around:500,37.7822,-122.4467)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;