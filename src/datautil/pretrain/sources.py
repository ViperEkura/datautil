"""Document records and bounded JSONL/Parquet I/O for pretraining corpora."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from datautil.common.errors import DataError
from datautil.common.io import iter_raw, output_writer
from datautil.pretrain.model import DocumentRecord


def _field(raw: dict[str, Any], path: str) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _remove(raw: dict[str, Any], path: str) -> None:
    value = raw
    parts = path.split(".")
    for part in parts[:-1]:
        value = value.get(part)
        if not isinstance(value, dict):
            return
    value.pop(parts[-1], None)


def _format(path: str, requested: str | None) -> str:
    selected = requested or (
        "parquet" if path.lower().endswith(".parquet") else "jsonl"
    )
    if selected not in {"jsonl", "parquet"}:
        raise DataError(
            "invalid_document_format", "Document format must be jsonl or parquet", path
        )
    return selected


def validate_fields(fields: Any, location: str) -> dict[str, Any]:
    if not isinstance(fields, dict) or set(fields) - {
        "id",
        "text",
        "metadata",
        "preserve_extra",
    }:
        raise DataError(
            "invalid_document_mapping",
            "fields must map id, text, metadata, or preserve_extra",
            location,
        )
    if not isinstance(fields.get("text"), str) or not fields["text"]:
        raise DataError(
            "invalid_document_mapping", "fields.text must be a field path", location
        )
    if "id" in fields and (not isinstance(fields["id"], str) or not fields["id"]):
        raise DataError(
            "invalid_document_mapping", "fields.id must be a field path", location
        )
    if "preserve_extra" in fields and type(fields["preserve_extra"]) is not bool:
        raise DataError(
            "invalid_document_mapping", "preserve_extra must be boolean", location
        )
    metadata = fields.get("metadata", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in metadata.items()
    ):
        raise DataError(
            "invalid_document_mapping",
            "fields.metadata must map names to field paths",
            location,
        )
    for path in [fields["text"], fields.get("id", "id"), *metadata.values()]:
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", part)
            for part in path.split(".")
        ):
            raise DataError(
                "invalid_document_mapping", f"Invalid field path {path!r}", location
            )
    return fields


def _parse(
    raw: Any, location: str, source: str, fields: dict[str, Any], fmt: str
) -> DocumentRecord:
    if not isinstance(raw, dict):
        raise DataError("invalid_document", "Expected a document object", location)
    text = _field(raw, fields["text"])
    if not isinstance(text, str):
        raise DataError(
            "invalid_document", f"Expected text at {fields['text']}", location
        )
    record_id = _field(raw, fields.get("id", "id"))
    if record_id is None:
        record_id = hashlib.sha256(f"{source}:{location}".encode()).hexdigest()[:24]
    if not isinstance(record_id, (str, int)) or not str(record_id):
        raise DataError(
            "invalid_document",
            "Document ID must be a nonempty string or integer",
            location,
        )
    metadata: dict[str, Any] = {}
    if fields.get("preserve_extra", True):
        metadata = copy.deepcopy(raw)
        _remove(metadata, fields["text"])
        _remove(metadata, fields.get("id", "id"))
        for path in fields.get("metadata", {}).values():
            _remove(metadata, path)
    for key, path in fields.get("metadata", {}).items():
        value = _field(raw, path)
        if value is not None:
            metadata[key] = value
    nested = metadata.pop("metadata", None)
    if nested is not None:
        if not isinstance(nested, dict):
            raise DataError("invalid_document", "metadata must be an object", location)
        metadata.update(nested)
    encoded = metadata.pop("metadata_json", None)
    if encoded is not None:
        try:
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise DataError(
                "invalid_document", "metadata_json must contain a JSON object", location
            ) from exc
        if not isinstance(decoded, dict):
            raise DataError(
                "invalid_document", "metadata_json must contain a JSON object", location
            )
        metadata.update(decoded)
    return DocumentRecord(str(record_id), text, location, metadata, fmt)


class DocumentSourceAdapter(Protocol):
    """Read one document storage format into the independent document model."""

    def read(
        self, path: str, source: str, fields: dict[str, Any]
    ) -> Iterator[DocumentRecord]: ...


class JsonlDocumentAdapter:
    def read(
        self, path: str, source: str, fields: dict[str, Any]
    ) -> Iterator[DocumentRecord]:
        for location, raw in iter_raw(path, "jsonl"):
            yield _parse(raw, location, source, fields, "jsonl")


class ParquetDocumentAdapter:
    def read(
        self, path: str, source: str, fields: dict[str, Any]
    ) -> Iterator[DocumentRecord]:
        row_number = 0
        try:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=2048):
                for raw in batch.to_pylist():
                    row_number += 1
                    yield _parse(
                        raw, f"{path}#row-{row_number}", source, fields, "parquet"
                    )
        except pa.ArrowException as exc:
            raise DataError("invalid_parquet", str(exc), path) from exc


_SOURCE_ADAPTERS: dict[str, DocumentSourceAdapter] = {
    "jsonl": JsonlDocumentAdapter(),
    "parquet": ParquetDocumentAdapter(),
}


def iter_documents(
    path: str, source: str, fields: dict[str, Any], fmt: str | None = None
) -> Iterator[DocumentRecord]:
    yield from _SOURCE_ADAPTERS[_format(path, fmt)].read(path, source, fields)


@contextmanager
def document_writer(path: str) -> Iterator[Any]:
    selected = _format(path, None)
    if selected == "jsonl":
        with output_writer(path, "jsonl") as writer:
            yield lambda record: writer.write(
                {
                    "id": record.record_id,
                    "text": record.text,
                    "metadata": record.metadata,
                }
            )
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    schema = pa.schema(
        [("id", pa.string()), ("text", pa.string()), ("metadata_json", pa.string())]
    )
    writer = None
    rows: list[dict[str, str]] = []

    def flush() -> None:
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            rows.clear()

    def append(record: DocumentRecord) -> None:
        try:
            metadata_json = json.dumps(
                record.metadata, ensure_ascii=False, sort_keys=True
            )
        except (TypeError, ValueError) as exc:
            raise DataError(
                "invalid_metadata", str(exc), record.source_location
            ) from exc
        rows.append(
            {
                "id": record.record_id,
                "text": record.text,
                "metadata_json": metadata_json,
            }
        )
        if len(rows) >= 2048:
            flush()

    try:
        writer = pq.ParquetWriter(temporary, schema, compression="snappy")
        yield append
        flush()
        writer.close()
        writer = None
        os.replace(temporary, target)
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
