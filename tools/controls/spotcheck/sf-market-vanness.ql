[out:json][timeout:60];
way["highway"]["name"~"^Market Street$"](around:500,37.7752,-122.4193)->.a;
way["highway"]["name"~"^(South )?Van Ness Avenue$"](around:500,37.7752,-122.4193)->.b;
node(w.a)(w.b)->.x;
.x out;
node(around.x:90)[~"^(highway|crossing|traffic_signals)$"~"^(traffic_signals|stop|give_way)$"];
out;