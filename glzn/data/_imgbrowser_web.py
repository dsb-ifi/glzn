"""Loopback HTTP frontend for the dataset gallery."""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from secrets import token_urlsafe
from threading import Thread
from urllib.parse import parse_qs, urlsplit

from .imgbrowser import _Gallery


def _create_server(gallery: _Gallery, cols: int, port: int) -> tuple[ThreadingHTTPServer, str]:
    base = f"/{token_urlsafe(24)}/"
    document = files("glzn.data").joinpath("_imgbrowser.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, data: bytes, content_type: str, status=HTTPStatus.OK, **headers) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), str(value))
            try:
                self.end_headers()
                self.wfile.write(data)
            except ConnectionError:
                # Browsers can cancel thumbnail requests when changing pages.
                self.close_connection = True

        def _json(self, data, status=HTTPStatus.OK) -> None:
            self._send(json.dumps(data).encode("utf-8"), "application/json", status)

        def _route(self) -> tuple[str, dict[str, list[str]]]:
            url = urlsplit(self.path)
            if not url.path.startswith(base):
                return "", {}
            return url.path.removeprefix(base) or "index", parse_qs(url.query)

        def do_GET(self) -> None:
            try:
                route, query = self._route()
                if route == "index":
                    self._send(document, "text/html; charset=utf-8")
                elif route == "api/page":
                    number = int(query.get("page", ["1"])[0])
                    page = gallery.page(number, query.get("selected") == ["1"])
                    self._json({
                        "page": page.number,
                        "pages": page.pages,
                        "start": page.start,
                        "total": page.total,
                        "selectionCount": page.selection_count,
                        "cols": cols,
                        "width": gallery.width,
                        "items": [
                            {"index": item.index, "label": item.label, "stem": item.stem,
                             "selected": selected}
                            for item, selected in page.items
                        ],
                    })
                elif route.startswith("thumbnail/"):
                    item = gallery.item(int(route.removeprefix("thumbnail/")))
                    self._send(item.thumbnail, "image/png")
                elif route == "api/export":
                    data, skipped = gallery.export_selection()
                    self._send(data, "application/json",
                               Content_Disposition='attachment; filename="selection.json"',
                               X_Skipped_Selections=skipped)
                else:
                    self._json({"error": "Gallery page not found."}, HTTPStatus.NOT_FOUND)
            except IndexError:
                self._json({"error": "Sample index is out of range."}, HTTPStatus.NOT_FOUND)
            except (TypeError, ValueError, OSError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            try:
                route, _ = self._route()
                if route not in {"api/select", "api/import", "api/clear"}:
                    self._json({"error": "Gallery action not found."}, HTTPStatus.NOT_FOUND)
                    return
                if self.headers.get_content_type() != "application/json":
                    self._json({"error": "Expected JSON."}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                    return
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 <= length <= 8 * 1024 * 1024:
                    self._json({"error": "Selection file is too large."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                    return
                data = self.rfile.read(length)
                if route == "api/select":
                    fields = json.loads(data)
                    if not isinstance(fields, dict):
                        raise ValueError("Expected a sample index and selection state.")
                    index, selected = fields.get("index"), fields.get("selected")
                    if type(index) is not int or type(selected) is not bool:
                        raise ValueError("Expected an integer sample index and a boolean selection state.")
                    self._json({"count": gallery.select(index, selected)})
                elif route == "api/import":
                    imported, skipped = gallery.import_selection(data)
                    self._json({"imported": imported, "skipped": skipped})
                else:
                    gallery.clear()
                    self._json({"count": 0})
            except IndexError:
                self._json({"error": "Sample index is out of range."}, HTTPStatus.NOT_FOUND)
            except (TypeError, ValueError, OSError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, f"http://127.0.0.1:{server.server_port}{base}"


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except (OSError, webbrowser.Error):
        # The printed URL remains available if the browser cannot be opened.
        pass


def serve_gallery(gallery: _Gallery, cols: int, port: int, *, open_browser: bool = False) -> None:
    server, url = _create_server(gallery, cols, port)
    with server:
        print(f"GLZN gallery at {url}", flush=True)
        if open_browser:
            Thread(target=_open_browser, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
