"""Compatibility helpers that delegate raw rows to storage backends."""

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from datautil.common.storage import STORAGE_BACKENDS, RowWriter, storage_for
from datautil.common.textio import RecordWriter as RecordWriter


def container_for(path: str, override: str = "auto", hint: Optional[str] = None) -> str:
    return storage_for(path, override, hint)


def iter_raw(path: str, container: str = "auto", errors: str = "raise") -> Iterator:
    yield from STORAGE_BACKENDS[storage_for(path, container)].read(path, errors)


@contextmanager
def output_writer(path: str, container: str = "auto") -> Iterator[RowWriter]:
    with STORAGE_BACKENDS[storage_for(path, container)].writer(path) as writer:
        yield writer


def write_json(path: str, value: Any) -> None:
    with output_writer(path, "jsonl") as writer:
        writer.write(value)
