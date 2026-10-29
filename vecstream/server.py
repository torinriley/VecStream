"""Versioned HTTP/JSON API (``/v1``) with authentication, per-tenant isolation,
metrics, health endpoints, and graceful shutdown. Standard library only."""

from __future__ import annotations

import json
import logging
import re
import ssl
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from . import __version__
from .auth import KeyStore, Principal
from .collections import validate_name
from .errors import (
    AuthenticationError,
    AuthorizationError,
    ClosedError,
    CorruptStoreError,
    InvalidVectorError,
    StoreLockedError,
)
from .metrics import Metrics, open_file_descriptors, process_memory_bytes
from .service import VecStreamService

logger = logging.getLogger("vecstream.server")
API_VERSION = "1"
ANONYMOUS = Principal("anonymous", "default", "superuser")
_SEGMENT = r"([^/]+)"


class HTTPError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _bad(message: str) -> HTTPError:
    return HTTPError(400, "invalid_request", message)


class _Route:
    def __init__(self, method: str, template: str, action: Optional[str], handler: str) -> None:
        pattern = re.sub(r"\{(\w+)\}", _SEGMENT, template)
        self.regex = re.compile(f"^{pattern}$")
        self.method, self.template, self.action, self.handler = method, template, action, handler
        self.names = re.findall(r"\{(\w+)\}", template)


# action None => unauthenticated; "operate" => operator keys; others are per-collection.
ROUTES = [
    _Route("GET", "/v1/healthz", None, "healthz"),
    _Route("GET", "/v1/readyz", None, "readyz"),
    _Route("GET", "/v1/metrics", "operate", "metrics"),
    _Route("GET", "/v1/diagnostics", "operate", "diagnostics"),
    _Route("POST", "/v1/admin/checkpoint", "operate", "checkpoint_all"),
    _Route("GET", "/v1/collections", "read", "list_collections"),
    _Route("POST", "/v1/collections", "manage", "create_collection"),
    _Route("GET", "/v1/collections/{collection}", "read", "get_collection"),
    _Route("DELETE", "/v1/collections/{collection}", "manage", "delete_collection"),
    _Route("POST", "/v1/collections/{collection}/checkpoint", "manage", "checkpoint"),
    _Route("POST", "/v1/collections/{collection}/verify", "manage", "verify"),
    _Route("POST", "/v1/collections/{collection}/batch", "write", "batch"),
    _Route("POST", "/v1/collections/{collection}/search", "read", "search"),
    _Route("POST", "/v1/collections/{collection}/search/batch", "read", "search_batch"),
    _Route("PUT", "/v1/collections/{collection}/vectors/{id}", "write", "put_vector"),
    _Route("GET", "/v1/collections/{collection}/vectors/{id}", "read", "get_vector"),
    _Route("DELETE", "/v1/collections/{collection}/vectors/{id}", "write", "delete_vector"),
]


class VecStreamServer:
    """Owns the listener, the service, and the lifecycle (start, drain, stop)."""

    def __init__(
        self,
        data_dir: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        keystore: Optional[KeyStore] = None,
        *,
        allow_anonymous: bool = False,
        tls_cert: Optional[str] = None,
        tls_key: Optional[str] = None,
        max_body_bytes: int = 64 << 20,
        max_batch_operations: int = 10_000,
        max_k: int = 1_000,
        max_batch_queries: int = 1_000,
        drain_timeout: float = 30.0,
        request_timeout: float = 30.0,
        **collection_options: Any,
    ) -> None:
        if keystore is None and not allow_anonymous:
            raise ValueError("authentication is required: pass a KeyStore or allow_anonymous=True")
        self.keystore = keystore
        self.allow_anonymous = allow_anonymous and keystore is None
        self.service = VecStreamService(data_dir, **collection_options)
        self.metrics = Metrics()
        self.max_body_bytes = max_body_bytes
        self.max_batch_operations = max_batch_operations
        self.max_k = max_k
        self.max_batch_queries = max_batch_queries
        self.drain_timeout = drain_timeout
        self.request_timeout = request_timeout
        self.started_at = time.time()
        self.ready = False
        self.draining = False
        self._inflight = 0
        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._stop_requested = threading.Event()
        self.recovered_collections = 0
        for name, kind, text in [
            ("vecstream_http_requests_total", "counter", "HTTP requests by route and status"),
            ("vecstream_http_request_duration_seconds", "histogram", "HTTP request latency"),
            ("vecstream_auth_failures_total", "counter", "Rejected credentials"),
            ("vecstream_errors_total", "counter", "Responses with status >= 500"),
        ]:
            self.metrics.describe(name, kind, text)
        self.httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self.httpd.daemon_threads = True
        self.httpd.request_queue_size = 128
        if tls_cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(tls_cert, tls_key)
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)

    # -- lifecycle ---------------------------------------------------------------------------

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        """Accept connections immediately, finish crash recovery, then report ready."""
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="vecstream-http")
        self._thread.start()
        self.recovered_collections = self.service.recover_all()
        self.ready = True
        logger.info("ready on %s:%s (%d collections)", *self.address, self.recovered_collections)

    def request_stop(self) -> None:
        """Signal-safe: ask the main thread to begin a graceful shutdown."""
        self._stop_requested.set()

    def wait_for_stop_request(self) -> None:
        self._stop_requested.wait()

    def shutdown(self) -> None:
        """Stop accepting, drain in-flight requests, checkpoint, release storage."""
        if self._stopped.is_set():
            return
        self.draining = True
        self.ready = False
        self.httpd.shutdown()
        deadline = time.monotonic() + self.drain_timeout
        with self._condition:
            while self._inflight and time.monotonic() < deadline:
                self._condition.wait(timeout=0.1)
            abandoned = self._inflight
        if abandoned:
            logger.warning("shutdown: %d requests still running after drain timeout", abandoned)
        try:
            self.service.close()  # waits on each collection's write lock, then checkpoints
        finally:
            self.httpd.server_close()
            if self._thread:
                self._thread.join()
            self._stopped.set()

    # -- request plumbing --------------------------------------------------------------------

    def _enter(self) -> bool:
        with self._condition:
            if self.draining:
                return False
            self._inflight += 1
            return True

    def _exit(self) -> None:
        with self._condition:
            self._inflight -= 1
            self._condition.notify_all()

    def _make_handler(self) -> type:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = f"VecStream/{__version__}"
            timeout = server.request_timeout

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                logger.debug("%s - %s", self.address_string(), format % args)

            def do_GET(self) -> None:
                server.dispatch(self)

            do_POST = do_PUT = do_DELETE = do_GET

            def do_PATCH(self) -> None:
                server.dispatch(self)

        return Handler

    def dispatch(self, handler: BaseHTTPRequestHandler) -> None:
        started = time.perf_counter()
        request_id = handler.headers.get("X-Request-ID") or uuid.uuid4().hex
        route_label, status = "unmatched", 500
        payload: Any = None
        content_type = "application/json"
        extra: Dict[str, str] = {}
        counted = self._enter()
        body_read = False
        try:
            if not counted:
                extra["Connection"] = "close"
                raise HTTPError(503, "shutting_down", "server is shutting down")
            path = urlsplit(handler.path).path
            route, params = self._match(handler.command, path)
            route_label = route.template
            principal = self._authenticate(handler, route)
            if route.action is not None:
                principal.require(route.action, params.get("collection"))
                if route.handler not in ("metrics", "diagnostics") and not self.ready:
                    raise HTTPError(503, "starting", "server is recovering")
            body = self._read_body(handler) if handler.command in ("POST", "PUT") else None
            body_read = True
            result = getattr(self, f"_h_{route.handler}")(principal, params, body)
            if isinstance(result, tuple):
                status, payload, content_type = result
            else:
                status, payload = 200, result
        except HTTPError as exc:
            status, payload = exc.status, {"error": {"code": exc.code, "message": exc.message}}
            if exc.status == 401:
                extra["WWW-Authenticate"] = 'Bearer realm="vecstream"'
        except Exception as exc:  # map domain errors; never leak internals
            status, code, message = self._map_error(exc)
            payload = {"error": {"code": code, "message": message}}
        finally:
            if counted:
                self._exit()
        if not body_read and handler.headers.get("Content-Length", "0") not in ("", "0"):
            handler.close_connection = True  # unread body would corrupt keep-alive framing
        if status >= 500:
            self.metrics.inc("vecstream_errors_total", route=route_label)
        self.metrics.inc(
            "vecstream_http_requests_total",
            method=handler.command,
            route=route_label,
            status=str(status),
        )
        self.metrics.observe(
            "vecstream_http_request_duration_seconds",
            time.perf_counter() - started,
            route=route_label,
        )
        self._respond(handler, status, payload, content_type, request_id, extra)

    def _match(self, method: str, path: str) -> Tuple[_Route, Dict[str, str]]:
        allowed = []
        for route in ROUTES:
            found = route.regex.match(path)
            if found:
                if route.method == method:
                    return route, {
                        n: unquote(v) for n, v in zip(route.names, found.groups(), strict=True)
                    }
                allowed.append(route.method)
        if allowed:
            raise HTTPError(405, "method_not_allowed", f"allowed: {', '.join(sorted(allowed))}")
        raise HTTPError(404, "not_found", "no such endpoint")

    def _authenticate(self, handler: BaseHTTPRequestHandler, route: _Route) -> Principal:
        if route.action is None:
            return ANONYMOUS
        if self.keystore is None:
            return ANONYMOUS
        try:
            return self.keystore.authenticate(handler.headers.get("Authorization"))
        except AuthenticationError as exc:
            self.metrics.inc("vecstream_auth_failures_total")
            raise HTTPError(401, "unauthenticated", str(exc)) from exc

    def _read_body(self, handler: BaseHTTPRequestHandler) -> Any:
        if "chunked" in handler.headers.get("Transfer-Encoding", "").lower():
            raise HTTPError(411, "length_required", "chunked bodies are not supported")
        try:
            length = int(handler.headers.get("Content-Length", ""))
        except ValueError:
            raise HTTPError(411, "length_required", "Content-Length is required") from None
        if length < 0 or length > self.max_body_bytes:
            handler.close_connection = True
            raise HTTPError(413, "payload_too_large", f"body exceeds {self.max_body_bytes} bytes")
        raw = handler.rfile.read(length)
        if len(raw) != length:
            handler.close_connection = True
            raise _bad("truncated request body")
        if not raw:
            return {}
        try:
            return json.loads(raw, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            raise _bad(f"body is not valid JSON: {exc}") from exc

    def _respond(
        self,
        handler: BaseHTTPRequestHandler,
        status: int,
        payload: Any,
        content_type: str,
        request_id: str,
        extra: Dict[str, str],
    ) -> None:
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("X-VecStream-API-Version", API_VERSION)
            handler.send_header("X-Request-ID", request_id)
            handler.send_header("Cache-Control", "no-store")
            for key, value in extra.items():
                handler.send_header(key, value)
            if self.draining:
                handler.send_header("Connection", "close")
                handler.close_connection = True
            handler.end_headers()
            handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            handler.close_connection = True

    @staticmethod
    def _map_error(exc: Exception) -> Tuple[int, str, str]:
        if isinstance(exc, AuthorizationError):
            return 403, "forbidden", str(exc)
        if isinstance(exc, KeyError):
            return 404, "not_found", str(exc.args[0]) if exc.args else "not found"
        if isinstance(exc, (InvalidVectorError, ValueError)):
            return 400, "invalid_request", str(exc)
        if isinstance(exc, ClosedError):
            return 503, "unavailable", str(exc)
        if isinstance(exc, StoreLockedError):
            return 503, "unavailable", "storage is locked"
        if isinstance(exc, CorruptStoreError):
            logger.error("data corruption: %s", exc)
            return 500, "data_corruption", "stored data failed integrity checks"
        logger.exception("unhandled error")
        return 500, "internal", "internal server error"

    # -- handlers (principal, path params, body) -----------------------------------------------

    def _manager(self, principal: Principal):  # type: ignore[no-untyped-def]
        assert principal.tenant is not None
        return self.service.manager(principal.tenant)

    def _h_healthz(self, *_: Any) -> Dict[str, Any]:
        return {"status": "ok", "version": __version__}

    def _h_readyz(self, *_: Any) -> Any:
        if self.ready and not self.draining:
            return {"status": "ready", "collections": self.recovered_collections}
        raise HTTPError(503, "not_ready", "draining" if self.draining else "recovering")

    def _h_metrics(self, *_: Any) -> Tuple[int, str, str]:
        memory = process_memory_bytes()
        gauges: List[Tuple[str, Dict[str, str], float]] = [
            ("vecstream_up", {}, 1.0),
            ("vecstream_ready", {}, float(self.ready)),
            ("vecstream_uptime_seconds", {}, time.time() - self.started_at),
            ("vecstream_http_in_flight", {}, float(self._inflight)),
            ("vecstream_process_resident_memory_bytes", {}, memory["rss"]),
            ("vecstream_process_peak_memory_bytes", {}, memory["peak"]),
            ("vecstream_process_open_fds", {}, float(open_file_descriptors())),
            *self.service.gauges(),
        ]
        return 200, self.metrics.render(gauges), "text/plain; version=0.0.4"

    def _h_diagnostics(self, *_: Any) -> Dict[str, Any]:
        rows = []
        for tenant, collection in self.service.collections():
            try:
                rows.append(
                    {"tenant": tenant, "collection": collection.name, **collection.health()}
                )
            except ClosedError:
                continue
        p50 = self.metrics.quantile(
            "vecstream_http_request_duration_seconds",
            0.5,
            route="/v1/collections/{collection}/search",
        )
        p99 = self.metrics.quantile(
            "vecstream_http_request_duration_seconds",
            0.99,
            route="/v1/collections/{collection}/search",
        )
        return {
            "version": __version__,
            "api_version": API_VERSION,
            "ready": self.ready,
            "uptime_seconds": time.time() - self.started_at,
            "memory": process_memory_bytes(),
            "search_latency_seconds": {"p50_upper_bound": p50, "p99_upper_bound": p99},
            "collections": rows,
        }

    def _h_checkpoint_all(self, *_: Any) -> Dict[str, Any]:
        self.service.checkpoint_all()
        return {"status": "ok"}

    def _h_list_collections(self, principal: Principal, *_: Any) -> Dict[str, Any]:
        names = [
            n for n in self._manager(principal).list_collections() if principal.allows("read", n)
        ]
        return {"collections": sorted(names)}

    def _h_create_collection(
        self, principal: Principal, _p: Any, body: Any
    ) -> Tuple[int, Any, str]:
        body = _object(body)
        name = body.get("name")
        try:
            validate_name(name)  # type: ignore[arg-type]
        except ValueError as exc:
            raise _bad(str(exc)) from exc
        principal.require("manage", name)
        use_hnsw = body.get("use_hnsw")
        if use_hnsw is not None and not isinstance(use_hnsw, bool):
            raise _bad("use_hnsw must be a boolean")
        params = body.get("hnsw_params")
        if params is not None and not isinstance(params, dict):
            raise _bad("hnsw_params must be an object")
        manager = self._manager(principal)
        try:
            manager.create_collection(name, use_hnsw, params)
        except ValueError as exc:
            if "already exists" in str(exc):
                raise HTTPError(409, "already_exists", str(exc)) from exc
            raise _bad(str(exc)) from exc
        return 201, manager.get_collection_stats(name), "application/json"

    def _h_get_collection(self, principal: Principal, params: Dict[str, str], _b: Any) -> Any:
        manager = self._manager(principal)
        stats = manager.get_collection_stats(params["collection"])
        stats["health"] = manager.get_collection(params["collection"]).health()
        return stats

    def _h_delete_collection(self, principal: Principal, params: Dict[str, str], _b: Any) -> Any:
        self._manager(principal).delete_collection(params["collection"])
        return {"deleted": params["collection"]}

    def _h_checkpoint(self, principal: Principal, params: Dict[str, str], _b: Any) -> Any:
        self._manager(principal).get_collection(params["collection"]).checkpoint()
        return {"status": "ok"}

    def _h_verify(self, principal: Principal, params: Dict[str, str], _b: Any) -> Any:
        return self._manager(principal).get_collection(params["collection"]).verify()

    def _h_put_vector(self, principal: Principal, params: Dict[str, str], body: Any) -> Any:
        body = _object(body)
        collection = self._manager(principal).get_collection(params["collection"])
        lsn = collection.apply(
            [
                {
                    "op": "put",
                    "id": params["id"],
                    "vector": body.get("vector"),
                    "metadata": body.get("metadata"),
                }
            ]
        )
        return {"id": params["id"], "lsn": lsn}

    def _h_get_vector(self, principal: Principal, params: Dict[str, str], _b: Any) -> Any:
        collection = self._manager(principal).get_collection(params["collection"])
        vector, metadata = collection.get_vector_with_metadata(params["id"])
        return {"id": params["id"], "vector": vector, "metadata": metadata}

    def _h_delete_vector(self, principal: Principal, params: Dict[str, str], _b: Any) -> Any:
        collection = self._manager(principal).get_collection(params["collection"])
        return {"id": params["id"], "lsn": collection.apply([{"op": "del", "id": params["id"]}])}

    def _h_batch(self, principal: Principal, params: Dict[str, str], body: Any) -> Any:
        operations = _object(body).get("operations")
        if not isinstance(operations, list) or not all(isinstance(o, dict) for o in operations):
            raise _bad("operations must be a list of objects")
        if len(operations) > self.max_batch_operations:
            raise HTTPError(413, "too_many_operations", f"at most {self.max_batch_operations}")
        collection = self._manager(principal).get_collection(params["collection"])
        return {"applied": len(operations), "lsn": collection.apply(operations)}

    def _h_search(self, principal: Principal, params: Dict[str, str], body: Any) -> Any:
        body = _object(body)
        options = _search_options(body, self.max_k)
        collection = self._manager(principal).get_collection(params["collection"])
        rows = collection.search_detailed(body.get("vector"), **options)
        include = bool(body.get("include_metadata", False))
        return {
            "results": [
                {"id": i, "score": s, **({"metadata": m} if include else {})} for i, s, m in rows
            ]
        }

    def _h_search_batch(self, principal: Principal, params: Dict[str, str], body: Any) -> Any:
        body = _object(body)
        options = _search_options(body, self.max_k)
        queries = body.get("vectors")
        if not isinstance(queries, list) or not queries:
            raise _bad("vectors must be a non-empty list of vectors")
        if len(queries) > self.max_batch_queries:
            raise HTTPError(413, "too_many_queries", f"at most {self.max_batch_queries}")
        collection = self._manager(principal).get_collection(params["collection"])
        batches = collection.search_batch(queries, **options)
        return {"results": [[{"id": i, "score": s} for i, s in rows] for rows in batches]}


def _search_options(body: Dict[str, Any], max_k: int) -> Dict[str, Any]:
    k = body.get("k", 10)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= max_k:
        raise _bad(f"k must be an integer between 1 and {max_k}")
    threshold = body.get("threshold", 0.0)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise _bad("threshold must be a number")
    ef_search = body.get("ef_search")
    if ef_search is not None and (not isinstance(ef_search, int) or not 1 <= ef_search <= 100_000):
        raise _bad("ef_search must be an integer between 1 and 100000")
    flt = body.get("filter")
    if flt is not None and not isinstance(flt, dict):
        raise _bad("filter must be an object")
    return {"k": k, "threshold": threshold, "ef_search": ef_search, "filter_metadata": flt}


def _reject_constant(value: str) -> Any:
    raise ValueError(f"{value} is not allowed in JSON")


def _object(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise _bad("request body must be a JSON object")
    return body
