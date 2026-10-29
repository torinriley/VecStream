"""A phase-fair readers-writer lock."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class ReadWriteLock:
    """Many concurrent readers or one exclusive writer; not reentrant.

    Waiting writers block new readers, so a steady stream of searches cannot
    starve mutations. When a writer releases, the readers already waiting get
    the next turn, so a steady stream of mutations cannot starve searches.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._waiting_readers = 0
        self._readers_turn = False

    def acquire_read(self) -> None:
        with self._condition:
            self._waiting_readers += 1
            try:
                while self._writer or (self._waiting_writers and not self._readers_turn):
                    self._condition.wait()
            finally:
                self._waiting_readers -= 1
            self._readers += 1
            if not self._waiting_readers:
                self._readers_turn = False
                self._condition.notify_all()

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._waiting_writers += 1
            try:
                while (
                    self._writer or self._readers or (self._readers_turn and self._waiting_readers)
                ):
                    self._condition.wait()
            finally:
                self._waiting_writers -= 1
            self._writer = True

    def release_write(self) -> None:
        with self._condition:
            self._writer = False
            self._readers_turn = self._waiting_readers > 0
            self._condition.notify_all()

    @contextmanager
    def read(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
