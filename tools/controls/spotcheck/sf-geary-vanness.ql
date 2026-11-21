[out:json][timeout:60];
way["highway"]["name"~"^Geary (Boulevard|Street)$"](around:500,37.7855,-122.4214)->.a;
way["highway"]["name"~"^Van Ness Avenue$"](around:500,37.7855,-122.4214)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;
