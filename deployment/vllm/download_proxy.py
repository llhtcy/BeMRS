"""Temporary loopback CONNECT proxy for downloading public deployment files.

Run on the workstation, reverse-forward with SSH, stop after setup. No request
bodies, credentials, TLS contents or signed download URLs are logged.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import select
import socket

ALLOWED = ('pythonhosted.org', 'pypi.org', 'huggingface.co', 'hf.co',
           'xethub.hf.co', 'modelscope.cn', 'aliyun.com')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_CONNECT(self):
        host, separator, port = self.path.rpartition(':')
        if not separator or port != '443' or not any(host == domain or host.endswith('.' + domain) for domain in ALLOWED):
            self.send_error(403)
            return
        try:
            upstream = socket.create_connection((host, 443), timeout=15)
        except OSError:
            self.send_error(502)
            return
        with upstream:
            self.send_response(200, 'Connection established')
            self.end_headers()
            self.wfile.flush()
            upstream.settimeout(None)
            endpoints = [self.connection, upstream]
            try:
                while True:
                    ready, _, _ = select.select(endpoints, [], [], 120)
                    if not ready:
                        break
                    for source in ready:
                        data = source.recv(128 * 1024)
                        if not data:
                            return
                        destination = upstream if source is self.connection else self.connection
                        destination.sendall(data)
            except OSError:
                pass
        self.close_connection = True


if __name__ == '__main__':
    print('Public-download proxy listening on 127.0.0.1:18079; Ctrl+C to stop', flush=True)
    with ThreadingHTTPServer(('127.0.0.1', 18079), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
