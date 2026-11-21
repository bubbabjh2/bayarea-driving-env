[out:json][timeout:60];
way["highway"]["name"~"Santa Clara Street$"](around:500,37.3364,-121.8907)->.a;
way["highway"]["name"~"(1st|First) Street$"](around:500,37.3364,-121.8907)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;
