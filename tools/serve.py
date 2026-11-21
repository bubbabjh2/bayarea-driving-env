import http.server, socketserver, sys, functools
class S(http.server.ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
h = functools.partial(H, directory=sys.argv[2])
H.extensions_map.update({'.js': 'text/javascript', '.mjs': 'text/javascript'})
S(('127.0.0.1', int(sys.argv[1])), h).serve_forever()
