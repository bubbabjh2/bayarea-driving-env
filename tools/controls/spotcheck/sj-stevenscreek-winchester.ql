[out:json][timeout:60];
way["highway"]["name"~"^Stevens Creek Boulevard$"](around:500,37.3233,-121.95)->.a;
way["highway"]["name"~"Winchester Boulevard$"](around:500,37.3233,-121.95)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;