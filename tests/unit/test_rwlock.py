import threading
import time

import pytest

from vecstream.rwlock import ReadWriteLock


def run(threads, timeout=20):
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "deadlock or starvation"


def test_readers_share_and_writers_exclude():
    lock = ReadWriteLock()
    state = {"readers": 0, "writers": 0, "max_readers": 0, "violations": 0}
    guard = threading.Lock()

    def reader():
        for _ in range(200):
            with lock.read():
                with guard:
                    state["readers"] += 1
                    state["max_readers"] = max(state["max_readers"], state["readers"])
                    state["violations"] += state["writers"]
                time.sleep(0)
                with guard:
                    state["readers"] -= 1

    def writer():
        for _ in range(100):
            with lock.write():
                with guard:
                    state["violations"] += state["writers"] + state["readers"]
                    state["writers"] += 1
                time.sleep(0)
                with guard:
                    state["writers"] -= 1

    run(
        [threading.Thread(target=reader) for _ in range(6)]
        + [threading.Thread(target=writer) for _ in range(3)]
    )
    assert state["violations"] == 0


def test_readers_make_progress_under_a_continuous_writer():
    lock, stop, reads = ReadWriteLock(), threading.Event(), [0]

    def writer():
        while not stop.is_set():
            with lock.write():
                pass

    def reader():
        for _ in range(300):
            with lock.read():
                reads[0] += 1

    w = threading.Thread(target=writer)
    w.start()
    try:
        run([threading.Thread(target=reader) for _ in range(3)])
    finally:
        stop.set()
        w.join()
    assert reads[0] == 900


def test_writers_make_progress_under_continuous_readers():
    lock, stop, writes = ReadWriteLock(), threading.Event(), [0]

    def reader():
        while not stop.is_set():
            with lock.read():
                time.sleep(0.0005)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for r in readers:
        r.start()
    try:

        def writer():
            for _ in range(50):
                with lock.write():
                    writes[0] += 1

        run([threading.Thread(target=writer)])
    finally:
        stop.set()
        for r in readers:
            r.join()
    assert writes[0] == 50


def test_lock_released_when_body_raises():
    lock = ReadWriteLock()
    with pytest.raises(RuntimeError):
        with lock.write():
            raise RuntimeError
    with pytest.raises(RuntimeError):
        with lock.read():
            raise RuntimeError
    with lock.write():
        pass
