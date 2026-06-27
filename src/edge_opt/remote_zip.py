"""Read single members of a large remote ZIP archive using HTTP range requests.

COCO train2017 is one 19 GB archive; INT8 calibration needs a few hundred of its images.
This module fetches only the central directory and the requested members. Neighbouring
members share the read buffer, so scattered files cost only a few range requests each.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP, backed by Range requests."""

    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url
        self.session = session or requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.size, self.pos + n) - 1
        response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
        response.raise_for_status()
        data = response.content
        if response.status_code == 200:  # server ignored the range and sent the whole file
            data = data[self.pos : end + 1]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def extract_members(
    url: str,
    out_dir: Path,
    contains: str | None = None,
    names: set[str] | None = None,
    buffer_mb: int = 1,
    skip_existing: bool = True,
) -> list[Path]:
    """Extracts members whose name contains ``contains`` or is listed in ``names`` into
    ``out_dir``, in archive order so that consecutive members come from the same buffer."""
    raw = io.BufferedReader(HttpRangeFile(url), buffer_size=buffer_mb << 20)
    written = []
    with zipfile.ZipFile(raw) as archive:
        members = [
            i
            for i in archive.infolist()
            if not i.is_dir()
            and (
                (contains is not None and contains in i.filename)
                or (names is not None and i.filename in names)
            )
        ]
        for info in sorted(members, key=lambda i: i.header_offset):
            target = out_dir / info.filename
            if not (skip_existing and target.exists() and target.stat().st_size == info.file_size):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
            written.append(target)
    return written
