import pytest

from vecstream.errors import WALCorruptionError
from vecstream.wal import MAGIC, WriteAheadLog


def _log(tmp_path, **kwargs):
    return WriteAheadLog(tmp_path / "wal.log", sync=False, **kwargs)


def test_append_and_recover_in_order(tmp_path):
    wal = _log(tmp_path)
    assert [wal.append(b"a"), wal.append(b"bb"), wal.append(b"")] == [1, 2, 3]
    wal.close()
    reopened = _log(tmp_path)
    assert reopened.recovered_records == [(1, b"a"), (2, b"bb"), (3, b"")]
    assert reopened.append(b"c") == 4


def test_records_covered_by_checkpoint_are_skipped(tmp_path):
    wal = _log(tmp_path)
    for payload in (b"1", b"2", b"3"):
        wal.append(payload)
    wal.close()
    assert _log(tmp_path, checkpoint_lsn=2).recovered_records == [(3, b"3")]


@pytest.mark.parametrize("cut", [1, 5, 17, 21])
def test_torn_tail_is_discarded_never_partially_applied(tmp_path, cut):
    wal = _log(tmp_path)
    wal.append(b"first")
    wal.append(b"second-record")
    wal.close()
    path = tmp_path / "wal.log"
    path.write_bytes(path.read_bytes()[:-cut])
    recovered = _log(tmp_path)
    assert recovered.recovered_records == [(1, b"first")]
    assert recovered.truncated_bytes > 0
    assert recovered.append(b"next") == 2
    recovered.close()
    assert _log(tmp_path).recovered_records == [(1, b"first"), (2, b"next")]


def test_mid_log_corruption_is_reported_not_silently_dropped(tmp_path):
    wal = _log(tmp_path)
    for payload in (b"one", b"two", b"three"):
        wal.append(payload)
    wal.close()
    path = tmp_path / "wal.log"
    data = bytearray(path.read_bytes())
    data[len(MAGIC) + 16 + 1] ^= 0xFF  # flip a bit inside the first payload
    path.write_bytes(bytes(data))
    with pytest.raises(WALCorruptionError):
        _log(tmp_path)
    repaired = _log(tmp_path, on_corruption="truncate")
    assert repaired.recovered_records == []


def test_bad_magic_is_corruption(tmp_path):
    (tmp_path / "wal.log").write_bytes(b"NOTAWAL!" + b"x" * 40)
    with pytest.raises(WALCorruptionError):
        _log(tmp_path)


def test_reset_keeps_lsn_monotonic(tmp_path):
    wal = _log(tmp_path)
    wal.append(b"a")
    wal.append(b"b")
    wal.reset(2)
    assert wal.append(b"c") == 3
    wal.close()
    assert _log(tmp_path, checkpoint_lsn=2).recovered_records == [(3, b"c")]
