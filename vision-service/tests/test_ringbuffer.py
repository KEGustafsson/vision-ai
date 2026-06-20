"""EventBuffer.recent boundary behaviour (n<=0 must not dump the whole buffer)."""

from app.util.ringbuffer import EventBuffer


def _fill(n: int) -> EventBuffer:
    buf = EventBuffer(maxlen=200)
    for i in range(n):
        buf.publish({"i": i})
    return buf


def test_recent_returns_last_n():
    buf = _fill(10)
    assert [e["i"] for e in buf.recent(3)] == [7, 8, 9]


def test_recent_zero_returns_empty_not_whole_buffer():
    buf = _fill(10)
    assert buf.recent(0) == []


def test_recent_negative_returns_empty():
    buf = _fill(10)
    assert buf.recent(-5) == []


def test_recent_more_than_available_returns_all():
    buf = _fill(3)
    assert len(buf.recent(50)) == 3
