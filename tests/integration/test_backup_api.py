"""Backup and restore over HTTP and the command line: authorization, isolation, errors."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_server import Client

from vecstream import KeyStore, VecStreamServer


@pytest.fixture
def env(tmp_path):
    keystore = KeyStore(str(tmp_path / "keys.json"))
    keys = {
        "admin": keystore.create_key("acme", "admin")[1],
        "docs_admin": keystore.create_key("acme", "admin", ["docs-*"])[1],
        "writer": keystore.create_key("acme", "writer")[1],
        "globex": keystore.create_key("globex", "admin")[1],
    }
    server = VecStreamServer(
        str(tmp_path / "data"), "127.0.0.1", 0, keystore, sync=False, engine="python"
    )
    server.start()
    yield server, keys, tmp_path
    server.shutdown()


def seed(client, name, count=6):
    assert client.call("POST", "/v1/collections", {"name": name})[0] == 201
    ops = [
        {"op": "put", "id": f"v{i}", "vector": [1.0, float(i), 0.5], "metadata": {"i": i}}
        for i in range(count)
    ]
    assert client.call("POST", f"/v1/collections/{name}/batch", {"operations": ops})[0] == 200


def test_backup_then_restore_over_http(env):
    server, keys, tmp = env
    admin = Client(server, keys["admin"])
    seed(admin, "docs-a")
    status, body, _ = admin.call("POST", "/v1/collections/docs-a/backup", {"id": "nightly-1"})
    assert status == 201 and body["id"] == "nightly-1" and body["vectors"] == 6
    assert (tmp / "data" / "backups" / "acme" / "nightly-1" / "backup.json").is_file()

    status, body, _ = admin.call("GET", "/v1/backups")
    assert status == 200 and [b["id"] for b in body["backups"]] == ["nightly-1"]

    status, body, _ = admin.call("POST", "/v1/collections/docs-b/restore", {"backup": "nightly-1"})
    assert status == 201 and body["vector_count"] == 6
    status, body, _ = admin.call(
        "POST", "/v1/collections/docs-b/search", {"vector": [1, 5, 0.5], "k": 1}
    )
    assert status == 200 and body["results"][0]["id"] == "v5"


def test_default_id_and_duplicate_conflict(env):
    server, keys, _ = env
    admin = Client(server, keys["admin"])
    seed(admin, "docs-a")
    status, body, _ = admin.call("POST", "/v1/collections/docs-a/backup")
    assert status == 201 and body["id"].startswith("docs-a-")
    assert admin.call("POST", "/v1/collections/docs-a/backup", {"id": body["id"]})[0] == 409


def test_restore_over_an_existing_collection_is_a_conflict(env):
    server, keys, _ = env
    admin = Client(server, keys["admin"])
    seed(admin, "docs-a")
    admin.call("POST", "/v1/collections/docs-a/backup", {"id": "b1"})
    status, body, _ = admin.call("POST", "/v1/collections/docs-a/restore", {"backup": "b1"})
    assert status == 409 and body["error"]["code"] == "already_exists"


def test_backup_endpoints_validate_input(env):
    server, keys, _ = env
    admin = Client(server, keys["admin"])
    seed(admin, "docs-a")
    for bad in ("../x", "a/b", "", ".hidden", 5, None):
        status, _, _ = admin.call("POST", "/v1/collections/docs-a/backup", {"id": bad})
        assert status == 400, bad
        status, _, _ = admin.call("POST", "/v1/collections/new/restore", {"backup": bad})
        assert status == 400, bad
    assert admin.call("POST", "/v1/collections/new/restore", {"backup": "nope"})[0] == 404
    assert admin.call("POST", "/v1/collections/missing/backup", {})[0] == 404


def test_roles_and_collection_patterns_gate_backups(env):
    server, keys, _ = env
    admin = Client(server, keys["admin"])
    seed(admin, "docs-a")
    seed(admin, "secrets")
    admin.call("POST", "/v1/collections/docs-a/backup", {"id": "docs-bk"})
    admin.call("POST", "/v1/collections/secrets/backup", {"id": "secret-bk"})

    writer = Client(server, keys["writer"])  # writers may not manage
    assert writer.call("POST", "/v1/collections/docs-a/backup", {})[0] == 403
    assert writer.call("GET", "/v1/backups")[0] == 403

    scoped = Client(server, keys["docs_admin"])
    assert scoped.call("POST", "/v1/collections/secrets/backup", {})[0] == 403
    listed = scoped.call("GET", "/v1/backups")[1]["backups"]
    assert [b["id"] for b in listed] == ["docs-bk"]  # the secrets backup is invisible
    # cannot launder protected data into an allowed name
    status, _, _ = scoped.call("POST", "/v1/collections/docs-x/restore", {"backup": "secret-bk"})
    assert status == 403
    assert scoped.call("POST", "/v1/collections/docs-x/restore", {"backup": "docs-bk"})[0] == 201
    # and cannot restore into a name outside its patterns
    status, _, _ = scoped.call("POST", "/v1/collections/other/restore", {"backup": "docs-bk"})
    assert status == 403


def test_tenants_cannot_see_or_restore_each_others_backups(env):
    server, keys, _ = env
    acme, globex = Client(server, keys["admin"]), Client(server, keys["globex"])
    seed(acme, "shared")
    acme.call("POST", "/v1/collections/shared/backup", {"id": "acme-bk"})
    assert globex.call("GET", "/v1/backups")[1]["backups"] == []
    status, _, _ = globex.call("POST", "/v1/collections/stolen/restore", {"backup": "acme-bk"})
    assert status == 404
    assert globex.call("GET", "/v1/collections")[1]["collections"] == []


def test_tampered_backup_is_rejected_with_422_and_leaves_nothing(env):
    server, keys, tmp = env
    admin = Client(server, keys["admin"])
    seed(admin, "docs-a")
    admin.call("POST", "/v1/collections/docs-a/backup", {"id": "bk"})
    target = tmp / "data" / "backups" / "acme" / "bk" / "vectors.npy"
    raw = bytearray(target.read_bytes())
    raw[-3] ^= 0xFF
    target.write_bytes(bytes(raw))
    status, body, _ = admin.call("POST", "/v1/collections/docs-c/restore", {"backup": "bk"})
    assert status == 422 and body["error"]["code"] == "invalid_backup"
    assert "docs-c" not in admin.call("GET", "/v1/collections")[1]["collections"]
    (tmp / "data" / "backups" / "acme" / "bk" / "backup.json").write_text("{not json")
    assert admin.call("POST", "/v1/collections/docs-c/restore", {"backup": "bk"})[0] == 422


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "vecstream", *args], capture_output=True, text=True, check=False
    )


def test_cli_backup_restore_and_verify(tmp_path):
    from vecstream import CollectionManager

    data = tmp_path / "data"
    manager = CollectionManager(str(data / "tenants" / "default"), sync=False, engine="python")
    collection = manager.create_collection("docs")
    collection.apply([{"op": "put", "id": f"v{i}", "vector": [1, i, 2]} for i in range(5)])
    manager.close()

    out = run_cli(
        "backup", "--data-dir", str(data), "--collection", "docs", "--to", str(tmp_path / "bk"),
        "--engine", "python",
    )  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["vectors"] == 5
    assert run_cli("verify-backup", str(tmp_path / "bk")).returncode == 0

    out = run_cli(
        "restore", "--data-dir", str(data), "--collection", "docs2", "--from", str(tmp_path / "bk"),
        "--engine", "python",
    )  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["vector_count"] == 5

    victim = Path(tmp_path / "bk" / "ids.json")
    victim.write_text(victim.read_text().replace("v0", "vX"))
    failed = run_cli("verify-backup", str(tmp_path / "bk"))
    assert failed.returncode == 1 and "failed verification" in failed.stderr
    assert "Traceback" not in failed.stderr
    taken = run_cli(
        "restore", "--data-dir", str(data), "--collection", "docs2", "--from", str(tmp_path / "bk"),
        "--engine", "python",
    )  # fmt: skip
    assert taken.returncode == 1 and "Traceback" not in taken.stderr
