"""Write shared prompt histories using constant buffered memory."""
import os
from pathlib import Path
import shutil
import tempfile


_MAX_ROWS = 2**31 - 2
_MAX_PAGES = 2**64 - 1


class TraceWriter:
    """Append immutable page spans and ordered request tips to a .tree file.

    Handles belong to this trace, with 0 as the empty prompt. Each extend call
    declares *new* page identities. Share an existing handle to model reuse.
    The completed trace replaces path atomically on a successful context exit.
    Instances are single-use and must be called from one ordered producer.
    """

    def __init__(self, path):
        self._path = Path(path)
        self._state = "new"
        self._work = None
        self._edges = self._tips = None
        self._nodes = self._requests = 0

    def __enter__(self):
        if self._state != "new":
            raise RuntimeError("TraceWriter instances are single-use")
        self._state = "closed"
        self._work = tempfile.TemporaryDirectory(prefix=".prefix-atlas-", dir=self._path.parent)
        self._dir = Path(self._work.name)
        try:
            self._edges = (self._dir / "edges").open("w+b")
            self._tips = (self._dir / "tips").open("w+b")
        except BaseException:
            self._cleanup()
            raise
        self._state = "open"
        return self

    def _check(self, handle):
        if self._state != "open":
            raise RuntimeError("TraceWriter needs an open, healthy context")
        if type(handle) is not int or not 0 <= handle <= self._nodes:
            raise ValueError("handle must be 0 or an existing node number")

    def _write(self, stream, record):
        try:
            stream.write(record)
        except BaseException:
            # A caller can catch an I/O error inside the context. Never publish
            # a spool that could contain a partially written row.
            self._state = "failed"
            raise

    def extend(self, parent, pages):
        """Declare an edge of fresh pages after parent and return its tip."""
        self._check(parent)
        if type(pages) is not int or not 1 <= pages <= _MAX_PAGES:
            raise ValueError("pages must be a positive unsigned 64-bit integer")
        if self._nodes == _MAX_ROWS:
            raise OverflowError("too many nodes for the tree format")
        self._write(self._edges, f"{parent} {pages}\n".encode("ascii"))
        self._nodes += 1
        return self._nodes

    def request(self, tip):
        """Record a request for this exact tip, in arrival order."""
        self._check(tip)
        if self._requests == _MAX_ROWS:
            raise OverflowError("too many requests for the tree format")
        self._write(self._tips, f"{tip}\n".encode("ascii"))
        self._requests += 1

    def _cleanup(self):
        try:
            if self._edges is not None:
                self._edges.close()
        finally:
            try:
                if self._tips is not None:
                    self._tips.close()
            finally:
                self._work.cleanup()

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None:
                return False
            if self._state != "open":
                raise RuntimeError("aborting trace publication after a failed write")
            final = self._dir / "complete"
            with final.open("wb") as out:
                out.write(f"{self._nodes} {self._requests}\n".encode("ascii"))
                for spool in (self._edges, self._tips):
                    spool.seek(0)
                    shutil.copyfileobj(spool, out, length=128*1024)
            os.replace(final, self._path)
        finally:
            self._state = "closed"
            self._cleanup()
        return False
