"""Serve the built demo locally without retaining old HTML or Worker assets."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DemoHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      '.js': 'text/javascript', '.wasm': 'application/wasm'}

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--directory', type=Path, default=Path('build/web'))
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if not (directory / 'index.html').is_file():
        parser.error('build the demo before starting the server')
    with ThreadingHTTPServer(('127.0.0.1', args.port), partial(DemoHandler, directory=str(directory))) as server:
        print(f'Demo: http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
