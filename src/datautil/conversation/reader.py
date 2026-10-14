from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator

from datautil.common.errors import DataError
from datautil.common.io import container_for, iter_raw
from datautil.conversation.formats import ADAPTERS, detect_record, validate_record
from datautil.conversation.model import DatasetRecord


def read_records(
    path: str,
    format_name: str | None = None,
    container: str = "auto",
    *,
    id_index: Any = None,
    check_duplicate_ids: bool = True,
    location_alias: str | None = None,
) -> Iterable[DatasetRecord]:
    selected = format_name
    seen_ids: set[str] = set()
    if selected and selected not in ADAPTERS:
        raise DataError("unknown_format", f"Unknown format {selected!r}")
    if container == "auto" and selected:
        container = container_for(
            path, "auto", getattr(ADAPTERS[selected], "storage_hint", None)
        )
    for location, raw in iter_raw(path, container):
        if location_alias is not None and location.startswith(path):
            location = location_alias + location[len(path) :]
        if selected is None:
            choices = detect_record(raw)
            if len(choices) != 1:
                raise DataError(
                    "ambiguous_format",
                    f"Expected one matching format, found {choices}",
                    location,
                )
            selected = choices[0]
        record = replace(
            ADAPTERS[selected].parse(raw, location), source_format=selected
        )
        validate_record(record)
        if check_duplicate_ids:
            if id_index is None:
                if record.record_id in seen_ids:
                    raise DataError(
                        "duplicate_record_id",
                        f"Duplicate record ID {record.record_id!r}",
                        location,
                    )
                seen_ids.add(record.record_id)
            elif not id_index.add_unique(record.record_id):
                raise DataError(
                    "duplicate_record_id",
                    f"Duplicate record ID {record.record_id!r}",
                    location,
                )
        yield record


@contextmanager
def replayable_input(path: str) -> Iterator[str]:
    if path != "-":
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="datautil-input-") as directory:
        target = Path(directory) / "stdin.jsonl"
        with target.open("wb") as stream:
            shutil.copyfileobj(sys.stdin.buffer, stream)
        yield str(target)


def source_format(path: str, requested: str | None) -> str:
    if requested:
        if requested not in ADAPTERS:
            raise DataError("unknown_format", f"Unknown format {requested!r}")
        return requested
    first = next(iter_raw(path), None)
    if first is None:
        raise DataError(
            "empty_dataset", "Cannot infer format from an empty dataset; pass --from"
        )
    choices = detect_record(first[1])
    if len(choices) != 1:
        raise DataError(
            "ambiguous_format", f"Cannot infer source format: {choices}", first[0]
        )
    return choices[0]
