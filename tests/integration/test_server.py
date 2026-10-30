import http.client
import json
import os
import threading
import time

import pytest

from vecstream import Collection, KeyStore, VecStreamServer


class Client:
    def __init__(self, server, key=None):
        self.host, self.port = server.address
        self.key = key

    def call(self, method, path, body=None, raw=None, headers=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        merged = dict(headers or {})
        if self.key:
            merged["Authorization"] = f"Bearer {self.key}"
        payload = raw if raw is not None else (json.dumps(body) if body is not None else None)
        if payload is not None:
            merged["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=merged)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        text = data.decode()
        parsed = (
            json.loads(text) if response.getheader("Content-Type") == "application/json" else text
        )
        return response.status, parsed, response


@pytest.fixture
def env(tmp_path):
    keystore = KeyStore(str(tmp_path / "keys.json"))
    keys = {
        "a_admin": keystore.create_key("acme", "admin")[1],
        "a_reader": keystore.create_key("acme", "reader")[1],
        "a_writer_docs": keystore.create_key("acme", "writer", ["docs-*"])[1],
        "b_admin": keystore.create_key("globex", "admin")[1],
        "ops": keystore.create_key(None, "operator")[1],
    }
    server = VecStreamServer(str(tmp_path / "data"), "127.0.0.1", 0, keystore, sync=False)
    server.start()
    yield server, keys, keystore, tmp_path
    server.shutdown()


def test_health_and_readiness_need_no_credentials(env):
    server, *_ = env
    status, body, response = Client(server).call("GET", "/v1/healthz")
    assert (status, body["status"]) == (200, "ok")
    assert response.getheader("X-VecStream-API-Version") == "1"
    assert Client(server).call("GET", "/v1/readyz")[0] == 200


def test_rejects_missing_wrong_and_malformed_credentials(env):
    server, keys, *_ = env
    for key in (None, "vs_bad.token", "garbage", keys["a_admin"] + "x"):
        status, body, response = Client(server, key).call("GET", "/v1/collections")
        assert status == 401 and body["error"]["code"] == "unauthenticated"
        assert response.getheader("WWW-Authenticate")


def test_roles_and_collection_patterns_are_enforced(env):
    server, keys, *_ = env
    admin, reader, writer = (
        Client(server, keys[k]) for k in ("a_admin", "a_reader", "a_writer_docs")
    )
    assert admin.call("POST", "/v1/collections", {"name": "docs-a"})[0] == 201
    assert admin.call("POST", "/v1/collections", {"name": "secrets"})[0] == 201
    vector = {"vector": [1, 0, 0], "metadata": {"k": "v"}}
    assert writer.call("PUT", "/v1/collections/docs-a/vectors/x", vector)[0] == 200
    assert writer.call("PUT", "/v1/collections/secrets/vectors/x", vector)[0] == 403
    assert writer.call("POST", "/v1/collections", {"name": "docs-b"})[0] == 403  # not a manager
    assert reader.call("GET", "/v1/collections/docs-a/vectors/x")[0] == 200
    assert reader.call("PUT", "/v1/collections/docs-a/vectors/y", vector)[0] == 403
    assert reader.call("DELETE", "/v1/collections/docs-a")[0] == 403
    assert reader.call("POST", "/v1/collections/docs-a/search", {"vector": [1, 0, 0]})[0] == 200
    assert writer.call("GET", "/v1/collections")[2].status == 200
    assert writer.call("GET", "/v1/collections")[1]["collections"] == ["docs-a"]
    assert reader.call("GET", "/v1/metrics")[0] == 403


def test_tenants_cannot_see_or_reach_each_others_data(env):
    server, keys, _, tmp_path = env
    acme, globex = Client(server, keys["a_admin"]), Client(server, keys["b_admin"])
    acme.call("POST", "/v1/collections", {"name": "shared-name"})
    acme.call("PUT", "/v1/collections/shared-name/vectors/secret", {"vector": [1, 0]})
    assert globex.call("GET", "/v1/collections")[1]["collections"] == []
    assert globex.call("GET", "/v1/collections/shared-name")[0] == 404
    assert globex.call("GET", "/v1/collections/shared-name/vectors/secret")[0] == 404
    assert globex.call("POST", "/v1/collections", {"name": "shared-name"})[0] == 201
    status, body, _ = globex.call("POST", "/v1/collections/shared-name/search", {"vector": [1, 0]})
    assert body["results"] == []  # same name, different tenant, different data
    for evil in ["../acme", "..%2Facme", "%2e%2e/acme"]:
        assert globex.call("GET", f"/v1/collections/{evil}")[0] in (400, 403, 404)
    assert (tmp_path / "data" / "tenants" / "acme").is_dir()


def test_crud_search_filter_and_atomic_batch(env):
    server, keys, *_ = env
    admin = Client(server, keys["a_admin"])
    admin.call(
        "POST", "/v1/collections", {"name": "c", "hnsw_params": {"M": 8, "ef_construction": 32}}
    )
    batch = {
        "operations": [
            {"op": "put", "id": "a", "vector": [1, 0, 0], "metadata": {"tag": "x"}},
            {"op": "put", "id": "b", "vector": [0.9, 0.1, 0], "metadata": {"tag": "y"}},
            {"op": "put", "id": "c", "vector": [0, 1, 0]},
        ]
    }
    assert admin.call("POST", "/v1/collections/c/batch", batch)[1]["applied"] == 3
    status, body, _ = admin.call(
        "POST", "/v1/collections/c/search", {"vector": [1, 0, 0], "k": 2, "include_metadata": True}
    )
    assert [r["id"] for r in body["results"]] == ["a", "b"]
    assert body["results"][0]["metadata"] == {"tag": "x"}
    status, body, _ = admin.call(
        "POST", "/v1/collections/c/search", {"vector": [1, 0, 0], "filter": {"tag": "y"}}
    )
    assert [r["id"] for r in body["results"]] == ["b"]
    bad = {"operations": [{"op": "del", "id": "a"}, {"op": "del", "id": "nope"}]}
    assert admin.call("POST", "/v1/collections/c/batch", bad)[0] == 404
    assert admin.call("GET", "/v1/collections/c/vectors/a")[0] == 200  # first delete rolled back
    assert admin.call("DELETE", "/v1/collections/c/vectors/a")[0] == 200
    assert admin.call("GET", "/v1/collections/c/vectors/a")[0] == 404
    assert admin.call("POST", "/v1/collections/c/verify")[1]["ok"] is True
    assert admin.call("GET", "/v1/collections/c")[1]["vector_count"] == 2
    assert admin.call("DELETE", "/v1/collections/c")[0] == 200
    assert admin.call("GET", "/v1/collections/c")[0] == 404


@pytest.mark.parametrize(
    "path,body,expected",
    [
        ("/v1/collections/c/search", {"vector": [1, 0, 0], "k": 0}, 400),
        ("/v1/collections/c/search", {"vector": [1, 0, 0], "k": 10**9}, 400),
        ("/v1/collections/c/search", {"vector": "nope"}, 400),
        ("/v1/collections/c/search", {"vector": [1, 0]}, 400),
        ("/v1/collections/c/search", [1, 2], 400),
        ("/v1/collections/c/batch", {"operations": "x"}, 400),
        ("/v1/collections/c/batch", {"operations": []}, 400),
        ("/v1/collections/c/batch", {"operations": [{"op": "zap", "id": "a"}]}, 400),
        ("/v1/collections", {"name": "../x"}, 400),
        ("/v1/collections", {"name": "c"}, 409),
        ("/v1/collections", {"name": "n", "hnsw_params": {"M": 1}}, 400),
        ("/v1/collections", {"name": "n", "hnsw_params": {"bogus": 1}}, 400),
    ],
)
def test_malformed_requests_get_clean_4xx(env, path, body, expected):
    server, keys, *_ = env
    admin = Client(server, keys["a_admin"])
    admin.call("POST", "/v1/collections", {"name": "c"})
    admin.call("PUT", "/v1/collections/c/vectors/a", {"vector": [1, 0, 0]})
    status, payload, _ = admin.call("POST", path, body)
    assert status == expected, payload
    assert "error" in payload


def test_invalid_json_nan_and_oversized_bodies(env):
    server, keys, *_ = env
    admin = Client(server, keys["a_admin"])
    admin.call("POST", "/v1/collections", {"name": "c"})
    assert admin.call("PUT", "/v1/collections/c/vectors/a", raw="{not json")[0] == 400
    assert admin.call("PUT", "/v1/collections/c/vectors/a", raw='{"vector": [NaN, 1]}')[0] == 400
    assert admin.call("PUT", "/v1/collections/c/vectors/a", raw='{"vector": [1e999, 1]}')[0] == 400
    server.max_body_bytes = 100
    assert (
        admin.call("PUT", "/v1/collections/c/vectors/a", raw=json.dumps({"vector": [1.0] * 100}))[0]
        == 413
    )
    assert admin.call("GET", "/v1/nope")[0] == 404
    assert admin.call("PATCH", "/v1/collections")[0] == 405


def test_metrics_expose_latency_errors_memory_and_index_health(env):
    server, keys, *_ = env
    admin, ops = Client(server, keys["a_admin"]), Client(server, keys["ops"])
    admin.call("POST", "/v1/collections", {"name": "m"})
    for i in range(5):
        admin.call("PUT", f"/v1/collections/m/vectors/v{i}", {"vector": [i + 1, 1, 0]})
    admin.call("POST", "/v1/collections/m/search", {"vector": [1, 1, 0]})
    admin.call("GET", "/v1/collections/missing")
    Client(server, "vs_x.y").call("GET", "/v1/collections")
    status, text, response = ops.call("GET", "/v1/metrics")
    assert status == 200 and response.getheader("Content-Type").startswith("text/plain")
    for needle in [
        'vecstream_http_requests_total{method="POST",'
        'route="/v1/collections/{collection}/search",status="200"} 1',
        "vecstream_http_request_duration_seconds_bucket",
        "vecstream_http_request_duration_seconds_count",
        'status="404"',
        "vecstream_auth_failures_total 1",
        "vecstream_process_resident_memory_bytes",
        'vecstream_collection_vectors{collection="m",tenant="acme"} 5',
        'vecstream_index_nodes{collection="m",tenant="acme"} 5',
        "vecstream_index_isolated_nodes",
        "vecstream_wal_bytes",
        "vecstream_ready 1",
    ]:
        assert needle in text, needle
    status, diag, _ = ops.call("GET", "/v1/diagnostics")
    assert diag["collections"][0]["vectors"] == 5
    assert ops.call("GET", "/v1/collections")[0] == 403  # operators never read tenant data
    assert admin.call("GET", "/v1/metrics")[0] == 403  # tenants never read global metrics


def test_revoked_and_expired_keys_stop_working_without_restart(env):
    server, keys, keystore, _ = env
    key_id, short_lived = keystore.create_key("acme", "reader", ttl_seconds=0.2)
    assert Client(server, short_lived).call("GET", "/v1/collections")[0] == 200
    time.sleep(0.3)
    assert Client(server, short_lived).call("GET", "/v1/collections")[0] == 401
    reader_id = next(
        k["key_id"] for k in keystore.list_keys() if k["role"] == "reader" and not k["expires_at"]
    )
    keystore.revoke(reader_id)
    assert Client(server, keys["a_reader"]).call("GET", "/v1/collections")[0] == 401


def test_key_file_stores_only_digests_and_is_private(env):
    _, keys, keystore, tmp_path = env
    path = tmp_path / "keys.json"
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    content = path.read_text()
    for key in keys.values():
        assert key.split(".", 1)[1] not in content
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        KeyStore(str(path))


def test_server_refuses_to_start_unauthenticated_by_default(tmp_path):
    with pytest.raises(ValueError):
        VecStreamServer(str(tmp_path), port=0)
    dev = VecStreamServer(str(tmp_path), port=0, allow_anonymous=True, sync=False)
    dev.start()
    try:
        assert Client(dev).call("POST", "/v1/collections", {"name": "dev"})[0] == 201
    finally:
        dev.shutdown()


def test_restart_recovers_acknowledged_writes_and_graph(env):
    server, keys, keystore, tmp_path = env
    admin = Client(server, keys["a_admin"])
    admin.call("POST", "/v1/collections", {"name": "p"})
    for i in range(20):
        assert (
            admin.call("PUT", f"/v1/collections/p/vectors/v{i}", {"vector": [i + 1, 1, 2]})[0]
            == 200
        )
    server.shutdown()
    again = VecStreamServer(str(tmp_path / "data"), "127.0.0.1", 0, keystore, sync=False)
    again.start()
    try:
        assert again.recovered_collections == 1
        reopened = again.service.manager("acme").get_collection("p")
        assert reopened.recovery["graph_restored"] is True  # clean shutdown checkpointed the graph
        assert (
            Client(again, keys["a_admin"]).call("GET", "/v1/collections/p")[1]["vector_count"] == 20
        )
    finally:
        again.shutdown()
    env_server = server  # fixture teardown calls shutdown again; must be idempotent
    env_server.shutdown()


def test_graceful_shutdown_drains_inflight_requests_then_refuses_new_ones(env, monkeypatch):
    server, keys, keystore, tmp_path = env
    admin = Client(server, keys["a_admin"])
    admin.call("POST", "/v1/collections", {"name": "d"})
    admin.call("PUT", "/v1/collections/d/vectors/a", {"vector": [1, 0]})
    started, release = threading.Event(), threading.Event()
    original = Collection.search_detailed

    def slow(self, *args, **kwargs):
        started.set()
        release.wait(5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "search_detailed", slow)
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(
            r=admin.call("POST", "/v1/collections/d/search", {"vector": [1, 0]})
        )
    )
    worker.start()
    assert started.wait(5)
    stopper = threading.Thread(target=server.shutdown)
    stopper.start()
    deadline = time.time() + 5
    while not server.draining and time.time() < deadline:
        time.sleep(0.01)
    assert server.draining and stopper.is_alive()  # still waiting on the in-flight search
    release.set()
    worker.join(5)
    stopper.join(10)
    assert result["r"][0] == 200 and result["r"][1]["results"][0]["id"] == "a"
    with pytest.raises(OSError):
        admin.call("GET", "/v1/healthz")
    again = VecStreamServer(str(tmp_path / "data"), "127.0.0.1", 0, keystore, sync=False)
    again.start()
    try:
        assert again.service.manager("acme").get_collection("d").get_vector_count() == 1
    finally:
        again.shutdown()


def test_filtered_search_over_http_fills_k_for_selective_filters(env):
    server, keys, *_ = env
    admin = Client(server, keys["a_admin"])
    admin.call("POST", "/v1/collections", {"name": "f"})
    operations = [
        {"op": "put", "id": f"v{i}", "vector": [1, i % 17 + 1, i % 5], "metadata": {"grp": i % 40}}
        for i in range(400)
    ]
    assert admin.call("POST", "/v1/collections/f/batch", {"operations": operations})[0] == 200
    status, body, _ = admin.call(
        "POST",
        "/v1/collections/f/search",
        {"vector": [1, 1, 1], "k": 5, "filter": {"grp": 3}, "include_metadata": True},
    )
    assert status == 200 and len(body["results"]) == 5
    assert all(r["metadata"]["grp"] == 3 for r in body["results"])
    status, body, _ = admin.call(
        "POST", "/v1/collections/f/search", {"vector": [1, 1, 1], "k": 3, "threshold": 0.999}
    )
    assert status == 200 and all(r["score"] >= 0.999 - 1e-6 for r in body["results"])


def test_batch_search_endpoint(env):
    server, keys, *_ = env
    admin, reader = Client(server, keys["a_admin"]), Client(server, keys["a_reader"])
    admin.call("POST", "/v1/collections", {"name": "b"})
    ops = [{"op": "put", "id": f"v{i}", "vector": [1, i, 0]} for i in range(30)]
    admin.call("POST", "/v1/collections/b/batch", {"operations": ops})
    status, body, _ = reader.call(
        "POST", "/v1/collections/b/search/batch", {"vectors": [[1, 0, 0], [1, 29, 0]], "k": 2}
    )
    assert status == 200 and [r["id"] for r in body["results"][0]][:1] == ["v0"]
    assert body["results"][1][0]["id"] == "v29"
    for bad in ({"vectors": []}, {"vectors": [[1, 0, 0], [1]]}, {"vectors": "x"}, {"k": 1}):
        assert reader.call("POST", "/v1/collections/b/search/batch", bad)[0] == 400
    server.max_batch_queries = 1
    assert (
        reader.call("POST", "/v1/collections/b/search/batch", {"vectors": [[1, 0, 0]] * 2})[0]
        == 413
    )
    assert (
        Client(server, keys["a_writer_docs"]).call(
            "POST", "/v1/collections/b/search/batch", {"vectors": [[1, 0, 0]]}
        )[0]
        == 403
    )
