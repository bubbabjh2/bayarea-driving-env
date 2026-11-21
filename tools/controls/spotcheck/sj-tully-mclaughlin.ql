[out:json][timeout:60];
way["highway"]["name"~"Tully Road$"](around:500,37.3133,-121.8473)->.a;
way["highway"]["name"~"McLaughlin Avenue$"](around:500,37.3133,-121.8473)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;