from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

from .model import DataError


def container_for(path: str, override: str = "auto") -> str:
    if override != "auto":
        return override
    return "jsonl" if path == "-" or path.lower().endswith((".jsonl", ".ndjson")) else "json"


def _iter_json_array(stream: TextIO) -> Iterator[tuple[int, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    eof = False

    def fill() -> None:
        nonlocal buffer, pos, eof
        if pos:
            buffer = buffer[pos:]
            pos = 0
        chunk = stream.read(65536)
        if chunk:
            buffer += chunk
        else:
            eof = True

    fill()
    while not buffer.strip() and not eof:
        fill()
    if not buffer.lstrip().startswith("["):
        # A single JSON object is accepted as a one-record JSON file.
        raw = buffer + stream.read()
        try:
            yield 1, json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataError("invalid_json", str(exc), "record 1") from exc
        return
    pos = buffer.index("[") + 1
    index = 0
    after_comma = False
    while True:
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer) or eof:
                break
            fill()
        if pos >= len(buffer):
            raise DataError("invalid_json", "Unclosed JSON array", f"record {index + 1}")
        if buffer[pos] == "]":
            if after_comma:
                raise DataError("invalid_json", "Trailing comma in JSON array", f"record {index + 1}")
            pos += 1
            if (buffer[pos:] + stream.read()).strip():
                raise DataError("invalid_json", "Trailing content after array", f"record {index + 1}")
            return
        while True:
            try:
                value, end = decoder.raw_decode(buffer, pos)
                pos = end
                break
            except json.JSONDecodeError as exc:
                if eof:
                    raise DataError("invalid_json", str(exc), f"record {index + 1}") from exc
                fill()
        index += 1
        after_comma = False
        yield index, value
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer) or eof:
                break
            fill()
        if pos >= len(buffer):
            raise DataError("invalid_json", "Unclosed JSON array", f"record {index + 1}")
        if buffer[pos] == ",":
            pos += 1
            after_comma = True
            continue
        if buffer[pos] == "]":
            if after_comma:
                raise DataError("invalid_json", "Trailing comma in JSON array", f"record {index + 1}")
            pos += 1
            if (buffer[pos:] + stream.read()).strip():
                raise DataError("invalid_json", "Trailing content after array", f"record {index + 1}")
            return
        raise DataError("invalid_json", "Expected ',' or ']'", f"record {index + 1}")


def iter_raw(path: str, container: str = "auto", errors: str = "raise") -> Iterator[tuple[str, Any]]:
    kind = container_for(path, container)
    stream = sys.stdin if path == "-" else open(path, "r", encoding="utf-8-sig", newline="")
    try:
        if kind == "jsonl":
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield f"{path}:{number}", json.loads(line)
                except json.JSONDecodeError as exc:
                    issue = DataError("invalid_json", str(exc), f"{path}:{number}")
                    if errors == "yield":
                        yield f"{path}:{number}", issue
                    else:
                        raise issue from exc
        else:
            for number, item in _iter_json_array(stream):
                yield f"{path}#record-{number}", item
    finally:
        if path != "-":
            stream.close()


@contextmanager
def output_writer(path: str, container: str = "auto") -> Iterator["RecordWriter"]:
    kind = container_for(path, container)
    if path == "-":
        writer = RecordWriter(sys.stdout, kind)
        yield writer
        writer.finish()
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            writer = RecordWriter(stream, kind)
            yield writer
            writer.finish()
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


class RecordWriter:
    def __init__(self, stream: TextIO, container: str) -> None:
        self.stream = stream
        self.container = container
        self.count = 0
        if container == "json":
            stream.write("[\n")

    def write(self, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if self.container == "json":
            if self.count:
                self.stream.write(",\n")
            self.stream.write(encoded)
        else:
            self.stream.write(encoded + "\n")
        self.count += 1

    def finish(self) -> None:
        if self.container == "json":
            self.stream.write("\n]\n")


def write_json(path: str, value: Any) -> None:
    with output_writer(path, "jsonl") as writer:
        # Report is a single JSON object, rather than a one-element array.
        writer.write(value)
