"""Streaming JSON codecs used by text storage backends."""

import json
from typing import Any, Iterator, TextIO

from datautil.common.errors import DataError


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
            raise DataError(
                "invalid_json", "Unclosed JSON array", f"record {index + 1}"
            )
        if buffer[pos] == "]":
            if after_comma:
                raise DataError(
                    "invalid_json",
                    "Trailing comma in JSON array",
                    f"record {index + 1}",
                )
            pos += 1
            if (buffer[pos:] + stream.read()).strip():
                raise DataError(
                    "invalid_json",
                    "Trailing content after array",
                    f"record {index + 1}",
                )
            return
        while True:
            try:
                value, end = decoder.raw_decode(buffer, pos)
                pos = end
                break
            except json.JSONDecodeError as exc:
                if eof:
                    raise DataError(
                        "invalid_json", str(exc), f"record {index + 1}"
                    ) from exc
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
            raise DataError(
                "invalid_json", "Unclosed JSON array", f"record {index + 1}"
            )
        if buffer[pos] == ",":
            pos += 1
            after_comma = True
            continue
        if buffer[pos] == "]":
            if after_comma:
                raise DataError(
                    "invalid_json",
                    "Trailing comma in JSON array",
                    f"record {index + 1}",
                )
            pos += 1
            if (buffer[pos:] + stream.read()).strip():
                raise DataError(
                    "invalid_json",
                    "Trailing content after array",
                    f"record {index + 1}",
                )
            return
        raise DataError("invalid_json", "Expected ',' or ']'", f"record {index + 1}")


class RecordWriter:
    def __init__(self, stream: TextIO, container: str) -> None:
        self.stream = stream
        self.container = container
        self.count = 0
        if container == "json":
            stream.write("[\n")

    def write(self, value: Any) -> None:
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise DataError(
                "unsupported_storage_value", str(exc), f"record {self.count + 1}"
            ) from exc
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
