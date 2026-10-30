"""Tenant-isolated service layer: one independent storage tree per tenant."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Tuple

from .auth import validate_tenant
from .collections import Collection, CollectionManager
from .errors import ClosedError

logger = logging.getLogger(__name__)


class VecStreamService:
    """Maps each tenant to its own ``CollectionManager`` directory.

    Isolation is structural: a request is resolved to a tenant by its
    credential, and that tenant's manager has no path to another tenant's
    files, collections, or indexes.
    """

    def __init__(self, data_dir: str, **collection_options: Any) -> None:
        self.data_dir = data_dir
        self.tenants_dir = os.path.join(data_dir, "tenants")
        os.makedirs(self.tenants_dir, exist_ok=True)
        self.collection_options = collection_options
        self._managers: Dict[str, CollectionManager] = {}
        self._lock = threading.Lock()
        self._closed = False

    def manager(self, tenant: str) -> CollectionManager:
        validate_tenant(tenant)
        with self._lock:
            if self._closed:
                raise ClosedError("service is shutting down")
            manager = self._managers.get(tenant)
            if manager is None:
                path = os.path.join(self.tenants_dir, tenant)
                os.makedirs(path, exist_ok=True)
                manager = CollectionManager(path, **self.collection_options)
                self._managers[tenant] = manager
            return manager

    def backup_root(self, tenant: str) -> str:
        """Per-tenant directory holding that tenant's backups (outside the collection trees)."""
        validate_tenant(tenant)
        return os.path.join(self.data_dir, "backups", tenant)

    def recover_all(self) -> int:
        """Open every tenant and collection so crash recovery finishes before serving."""
        opened = 0
        for tenant in sorted(os.listdir(self.tenants_dir)):
            if not os.path.isdir(os.path.join(self.tenants_dir, tenant)):
                continue
            manager = self.manager(tenant)
            for name in manager.list_collections():
                manager.get_collection(name)
                opened += 1
        return opened

    def collections(self) -> Iterator[Tuple[str, Collection]]:
        with self._lock:
            managers = list(self._managers.items())
        for tenant, manager in managers:
            for collection in manager.open_collections():
                yield tenant, collection

    def checkpoint_all(self) -> None:
        for _tenant, collection in self.collections():
            collection.checkpoint()

    def gauges(self) -> List[Tuple[str, Dict[str, str], float]]:
        rows: List[Tuple[str, Dict[str, str], float]] = []
        now = time.time()
        for tenant, collection in self.collections():
            labels = {"tenant": tenant, "collection": collection.name}
            try:
                health = collection.health()
            except ClosedError:
                continue
            rows += [
                ("vecstream_collection_vectors", labels, health["vectors"]),
                ("vecstream_collection_vector_bytes", labels, health["vector_bytes"]),
                ("vecstream_wal_bytes", labels, health["wal_bytes"]),
                ("vecstream_wal_pending_ops", labels, health["ops_since_checkpoint"]),
                ("vecstream_index_nodes", labels, health["index_nodes"]),
                ("vecstream_index_levels", labels, health["index_levels"]),
                ("vecstream_index_avg_degree", labels, health["index_avg_degree"]),
                ("vecstream_index_isolated_nodes", labels, health["index_isolated_nodes"]),
                ("vecstream_checkpoint_failures", labels, health["checkpoint_failures"]),
            ]
            if health["last_checkpoint_time"]:
                age = now - health["last_checkpoint_time"]
                rows.append(("vecstream_seconds_since_checkpoint", labels, age))
        return rows

    def close(self) -> None:
        """Checkpoint and release every tenant; safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            managers = list(self._managers.values())
        failures = []
        for manager in managers:
            try:
                manager.close()
            except Exception as exc:
                logger.exception("failed to close tenant storage")
                failures.append(exc)
        if failures:
            raise failures[0]
