#!/usr/bin/python

import SimpleHTTPServer
import SocketServer

PORT = 8000

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    '.webapp': 'application/x-web-app-manifest+json',
});

httpd = SocketServer.TCPServer(('0.0.0.0', PORT), Handler)
print "[i] Web server listening at port %s..." %(PORT)

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    httpd.server_close()
    print "[i] Web server stoped"
