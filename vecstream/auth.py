"""API-key authentication and role/collection authorization.

Keys look like ``vs_<key_id>.<secret>``. Only a SHA-256 digest of the random
secret is stored, so the key file cannot be replayed if it leaks; the full key
is shown exactly once at creation. Each key is bound to one tenant (or is an
``operator`` key with no tenant data access), a role, and collection patterns.
"""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import AuthenticationError, AuthorizationError

TENANT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
ROLE_ACTIONS = {
    "reader": {"read"},
    "writer": {"read", "write"},
    "admin": {"read", "write", "manage"},
    "operator": {"operate"},
    "superuser": {"read", "write", "manage", "operate"},  # anonymous dev mode only
}
KEY_PREFIX = "vs_"


def validate_tenant(tenant: str) -> str:
    if not isinstance(tenant, str) or not TENANT_PATTERN.match(tenant):
        raise ValueError("tenant must match [a-z0-9][a-z0-9_-]{0,62}")
    return tenant


@dataclass(frozen=True)
class Principal:
    key_id: str
    tenant: Optional[str]
    role: str
    collections: Tuple[str, ...] = ("*",)

    def allows(self, action: str, collection: Optional[str] = None) -> bool:
        if action not in ROLE_ACTIONS[self.role]:
            return False
        if collection is None or self.role == "operator":
            return True
        return any(fnmatch.fnmatchcase(collection, pattern) for pattern in self.collections)

    def require(self, action: str, collection: Optional[str] = None) -> None:
        if not self.allows(action, collection):
            raise AuthorizationError(
                f"role {self.role!r} may not {action}"
                + (f" collection {collection!r}" if collection else "")
            )


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class KeyStore:
    """File-backed API-key registry (0600, atomically replaced, re-read on change)."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._mtime: Optional[float] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._cache, self._mtime = {}, None
            return
        stat = self.path.stat()
        if stat.st_mode & 0o077:
            raise PermissionError(f"{self.path} must not be accessible by group or others")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("keys"), dict):
            raise ValueError(f"{self.path} is not a valid key file")
        self._cache, self._mtime = data["keys"], stat.st_mtime_ns

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime_ns if self.path.exists() else None
        except OSError:
            return
        if mtime != self._mtime:
            self._load()  # picks up CLI-side rotation and revocation without a restart

    def _write(self, keys: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".keys-")
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "keys": keys}, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def _mutate(self, change) -> Any:  # type: ignore[no-untyped-def]
        lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(lock_path, "a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            self._load()
            keys = {k: dict(v) for k, v in self._cache.items()}
            result = change(keys)
            self._write(keys)
            self._load()
            return result

    def create_key(
        self,
        tenant: Optional[str],
        role: str,
        collections: Sequence[str] = ("*",),
        ttl_seconds: Optional[float] = None,
        label: str = "",
    ) -> Tuple[str, str]:
        """Create a key; returns ``(key_id, full_key)``. The full key is not recoverable."""
        if role not in ROLE_ACTIONS or role == "superuser":
            raise ValueError("role must be one of ['admin', 'operator', 'reader', 'writer']")
        if role == "operator":
            tenant = None
        else:
            validate_tenant(tenant or "")
        if not collections or any(not isinstance(c, str) or not c for c in collections):
            raise ValueError("collections must be a non-empty list of name patterns")
        key_id, secret = secrets.token_hex(6), secrets.token_urlsafe(32)

        def change(keys: Dict[str, Dict[str, Any]]) -> None:
            keys[key_id] = {
                "hash": _digest(secret),
                "tenant": tenant,
                "role": role,
                "collections": list(collections),
                "created_at": time.time(),
                "expires_at": time.time() + ttl_seconds if ttl_seconds else None,
                "revoked": False,
                "label": label,
            }

        self._mutate(change)
        return key_id, f"{KEY_PREFIX}{key_id}.{secret}"

    def revoke(self, key_id: str) -> None:
        def change(keys: Dict[str, Dict[str, Any]]) -> None:
            if key_id not in keys:
                raise KeyError(key_id)
            keys[key_id]["revoked"] = True

        self._mutate(change)

    def list_keys(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._refresh()
            return [
                {"key_id": k, **{f: v for f, v in rec.items() if f != "hash"}}
                for k, rec in sorted(self._cache.items())
            ]

    def authenticate(self, header: Optional[str]) -> Principal:
        """Resolve an ``Authorization: Bearer`` header or raise ``AuthenticationError``."""
        if not header or not header.startswith("Bearer "):
            raise AuthenticationError("missing bearer credentials")
        token = header[len("Bearer ") :].strip()
        if not token.startswith(KEY_PREFIX) or "." not in token:
            raise AuthenticationError("malformed API key")
        key_id, _, secret = token[len(KEY_PREFIX) :].partition(".")
        with self._lock:
            self._refresh()
            record = self._cache.get(key_id)
        # Always hash and compare so unknown IDs cost the same as wrong secrets.
        expected = record["hash"] if record else _digest("")
        matches = hmac.compare_digest(expected, _digest(secret))
        if record is None or not matches or record.get("revoked"):
            raise AuthenticationError("invalid API key")
        expires = record.get("expires_at")
        if expires is not None and expires < time.time():
            raise AuthenticationError("API key expired")
        return Principal(
            key_id=key_id,
            tenant=record.get("tenant"),
            role=record["role"],
            collections=tuple(record.get("collections", ["*"])),
        )
