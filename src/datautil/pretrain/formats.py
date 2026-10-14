"""Document record mappings without file storage dependencies."""

import copy
import hashlib
import json
import re
from typing import Any, Dict, Optional

from datautil.common.errors import DataError
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
    paths = [fields["text"], fields.get("id", "id"), *metadata.values()]
    for index, left in enumerate(paths):
        if any(
            left == right
            or left.startswith(right + ".")
            or right.startswith(left + ".")
            for right in paths[index + 1 :]
        ):
            raise DataError(
                "invalid_document_mapping",
                "Document field paths must not overlap",
                location,
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


class DocumentFormatAdapter:
    name = "document"
    record_kind = "document"
    layout = "document"
    storage_hint = None

    def __init__(
        self,
        fields: Optional[Dict[str, Any]] = None,
        source: str = "document",
        name: str = "document",
    ) -> None:
        self.fields = validate_fields(
            fields if fields is not None else {"id": "id", "text": "text"}, name
        )
        self.source = source
        self.name = name

    def matches(self, raw: Any) -> bool:
        return isinstance(raw, dict) and isinstance(
            _field(raw, self.fields["text"]), str
        )

    def parse(self, raw: Any, location: str) -> DocumentRecord:
        return _parse(raw, location, self.source, self.fields, self.name)

    def serialize(self, record: DocumentRecord) -> Dict[str, Any]:
        return {
            "id": record.record_id,
            "text": record.text,
            "metadata": record.metadata,
        }


def _set(raw: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        child = raw.setdefault(part, {})
        if not isinstance(child, dict):
            raise DataError("metadata_conflict", "Conflicting output field: " + path)
        raw = child
    if parts[-1] in raw and raw[parts[-1]] != value:
        raise DataError("metadata_conflict", "Conflicting output field: " + path)
    raw[parts[-1]] = value


class YamlDocumentAdapter:
    record_kind = "document"
    layout = "document"

    def __init__(self, config: Dict[str, Any], path: str) -> None:
        if (
            type(config.get("version")) is not int
            or config["version"] != 1
            or config.get("layout") != "document"
            or set(config) - {"version", "name", "layout", "fields", "container"}
        ):
            raise DataError(
                "invalid_profile",
                "Document profile needs version, name, layout, and fields",
                path,
            )
        self.name = config.get("name")
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z][a-z0-9-]*", self.name
        ):
            raise DataError("invalid_profile", "Invalid document profile name", path)
        if not isinstance(config.get("fields"), dict):
            raise DataError(
                "invalid_profile", "Document fields must be a mapping", path
            )
        self.codec = DocumentFormatAdapter(config["fields"], self.name, self.name)
        self.fields = self.codec.fields
        hint = config.get("container")
        if hint is not None and (
            not isinstance(hint, str)
            or hint not in {"json", "jsonl", "parquet", "JSONEachRow"}
        ):
            raise DataError("invalid_profile", "Invalid legacy container hint", path)
        self.storage_hint = "jsonl" if hint == "JSONEachRow" else hint

    def matches(self, raw: Any) -> bool:
        return self.codec.matches(raw)

    def parse(self, raw: Any, location: str) -> DocumentRecord:
        return self.codec.parse(raw, location)

    def serialize(self, record: DocumentRecord) -> Dict[str, Any]:
        mapped = self.fields.get("metadata", {})
        output = copy.deepcopy(
            {key: value for key, value in record.metadata.items() if key not in mapped}
        )
        _set(output, self.fields.get("id", "id"), record.record_id)
        _set(output, self.fields["text"], record.text)
        for key, path in mapped.items():
            if key in record.metadata:
                _set(output, path, record.metadata[key])
        return output
