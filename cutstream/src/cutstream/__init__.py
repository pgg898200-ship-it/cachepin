"""Release the safe prefix of a byte stream at each feed boundary."""

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

from . import _native


class Plan:
    """Compile byte markers once. Reuse the same plan across independent streams."""

    __slots__ = ("_plan",)

    def __init__(self, markers: Iterable[bytes]):
        if isinstance(markers, (str, bytes, bytearray)):
            raise TypeError("pass an iterable of byte markers, e.g. [b'</tool>']")
        self._plan = _native.compile(tuple(markers))

    def stream(self) -> "Stream":
        return Stream(self)

    def filter(self, chunks: Iterable[bytes]) -> Iterator[bytes]:
        """Yield visible nonempty chunks. Stop consuming input at the first match."""
        stream = self.stream()
        for chunk in chunks:
            visible = stream.feed(chunk)
            if visible:
                yield visible
            if stream.done:
                return
        tail = stream.finish()
        if tail:
            yield tail

    async def afilter(self, chunks: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
        """Async form of filter. The caller owns the upstream connection."""
        stream = self.stream()
        async for chunk in chunks:
            visible = stream.feed(chunk)
            if visible:
                yield visible
            if stream.done:
                return
        tail = stream.finish()
        if tail:
            yield tail


class Stream:
    """One ordered response. EOF flushes a partial marker as ordinary bytes."""

    __slots__ = ("_stream",)

    def __init__(self, plan: Plan):
        self._stream = _native.stream(plan._plan)

    def feed(self, chunk: bytes) -> bytes:
        return _native.feed(self._stream, chunk)

    def finish(self) -> bytes:
        return _native.finish(self._stream)

    @property
    def match(self) -> int | None:
        """Original index of the winning marker, or None."""
        return _native.match(self._stream)

    @property
    def consumed(self) -> int:
        """Number of bytes consumed from the most recent feed."""
        return _native.consumed(self._stream)

    @property
    def pending(self) -> bytes:
        """Copy the currently withheld bytes for inspection."""
        return _native.pending(self._stream)

    @property
    def done(self) -> bool:
        return _native.done(self._stream)
