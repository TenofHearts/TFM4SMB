"""Read smbdataset's binary trailers, including its known CR expansion bug."""

from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np

FRAME_RE = re.compile(
    r"^(?P<episode>.+_e\d+_(?P<world>\d+)-(?P<level>\d+))"
    r"_f(?P<frame>\d+)_a(?P<action>\d+)_.+\.(?P<outcome>win|fail)\.png$"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Frame:
    path: Path
    episode: str
    world: int
    level: int
    number: int
    action: int
    outcome: str


def parse_frame(path: Path) -> Frame:
    match = FRAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected smbdataset filename: {path}")
    g = match.groupdict()
    return Frame(path, g['episode'], int(g['world']), int(g['level']),
                 int(g['frame']), int(g['action']), g['outcome'])


def decode_ram(payload: bytes, encoding: str = "dataset-cr") -> np.ndarray:
    # Issue #4 specifies CR LF -> CR, NOT CR LF -> LF. Never truncate/pad.
    if encoding == "dataset-cr":
        payload = payload.replace(b"\r\n", b"\r")
    elif encoding != "raw":
        raise ValueError(f"Unknown RAM encoding: {encoding}")
    if len(payload) != 2048:
        raise ValueError(f"Expected 2048 decoded RAM bytes, got {len(payload)} ({encoding})")
    return np.frombuffer(payload, dtype=np.uint8).copy()


def read_frame(frame: Frame, encoding: str = "dataset-cr") -> np.ndarray:
    data = frame.path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"Not a PNG: {frame.path}")
    # Walk real image chunks to IEND. The malformed trailer lengths cannot
    # be walked this way because CR expansion changes the physical lengths.
    pos = 8
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        pos += length + 12
        if pos > len(data):
            raise ValueError(f"Truncated PNG: {frame.path}")
        if kind == b"IEND":
            break
    else:
        raise ValueError(f"No IEND chunk: {frame.path}")
    trailer = data[pos:]
    ram_header = b"\x00\x00\x08\x04tEXtRAM\x00"
    bp_header = b"\x00\x00\x00\x05tEXtBP1\x00"
    out_header = b"\x00\x00\x00\x09tEXtOUTCOME\x00"
    if not trailer.startswith(ram_header) or trailer.count(bp_header) != 1:
        raise ValueError(f"Missing/ambiguous RAM or BP1 trailer: {frame.path}")
    bp = trailer.index(bp_header)
    if trailer[bp - 4:bp] != b"\x00" * 4:
        raise ValueError(f"Unexpected RAM trailer terminator: {frame.path}")
    payload = trailer[len(ram_header):bp - 4]
    tail = trailer[bp + len(bp_header):]
    expected_outcome = 2 if frame.outcome == "win" else 1
    expected = bytes([frame.action]) + b"\x00" * 4 + out_header + bytes([expected_outcome]) + b"\x00" * 4
    if tail != expected:
        raise ValueError(f"Filename/BP1/OUTCOME metadata mismatch: {frame.path}")
    return decode_ram(payload, encoding)
