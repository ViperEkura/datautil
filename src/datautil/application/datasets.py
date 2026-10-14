"""Compose raw storage and record codecs without record-class inheritance."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol

import yaml

from datautil.common.errors import DataError
from datautil.common.storage import StorageBackend
from datautil.conversation.formats import ADAPTERS, detect_record, validate_record
from datautil.conversation.profiles import YamlFormatAdapter
from datautil.pretrain.formats import DocumentFormatAdapter, YamlDocumentAdapter


class RecordAdapter(Protocol):
    name: str
    record_kind: str

    def parse(self, raw: Any, location: str) -> Any: ...

    def serialize(self, record: Any) -> Any: ...


def load_dataset_profiles(paths: List[str]) -> List[RecordAdapter]:
    profiles = []
    for path in paths:
        try:
            config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise DataError("invalid_profile", str(exc), path) from exc
        if not isinstance(config, dict):
            raise DataError("invalid_profile", "Expected a profile object", path)
        profiles.append(
            YamlDocumentAdapter(config, path)
            if config.get("layout") == "document"
            else YamlFormatAdapter(config, path)
        )
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)) or "document" in names:
        raise DataError(
            "invalid_profile", "Duplicate or reserved document profile name"
        )
    return profiles


@dataclass(frozen=True)
class RecordCodec:
    adapter: RecordAdapter
    validate: Callable[[Any], None]

    @property
    def name(self) -> str:
        return self.adapter.name

    @property
    def record_kind(self) -> str:
        return self.adapter.record_kind


def _validate_document(record: Any) -> None:
    if (
        not isinstance(record.text, str)
        or not isinstance(record.metadata, dict)
        or not record.record_id
    ):
        raise DataError(
            "invalid_document",
            "Expected document ID, text, and metadata",
            record.source_location,
        )


class RecordFormats:
    def __init__(self, profiles: List[RecordAdapter]) -> None:
        self.codecs: Dict[str, RecordCodec] = {
            name: RecordCodec(adapter, validate_record)
            for name, adapter in ADAPTERS.items()
        }
        self.codecs["document"] = RecordCodec(
            DocumentFormatAdapter(), _validate_document
        )
        for profile in profiles:
            if profile.record_kind == "document":
                if profile.name in self.codecs:
                    raise DataError(
                        "invalid_profile", "Reserved profile name: " + profile.name
                    )
                self.codecs[profile.name] = RecordCodec(profile, _validate_document)

    def get(self, name: str) -> RecordCodec:
        if name not in self.codecs:
            raise DataError("unknown_format", "Unknown record format: " + name)
        return self.codecs[name]

    def matches(self, raw: Any) -> List[str]:
        return detect_record(raw) + [
            name
            for name, codec in self.codecs.items()
            if codec.record_kind == "document" and codec.adapter.matches(raw)
        ]

    def select(self, raw: Any, location: str) -> RecordCodec:
        names = self.matches(raw)
        if len(names) != 1:
            raise DataError(
                "ambiguous_format",
                f"Expected one record format, found {names}; pass --from",
                location,
            )
        return self.get(names[0])


@dataclass(frozen=True)
class DatasetReader:
    backend: StorageBackend
    formats: RecordFormats
    codec: Optional[RecordCodec] = None

    def rows(self, path: str, errors: str) -> Iterator:
        return self.backend.read(path, errors)

    def parse(self, raw: Any, location: str) -> Any:
        codec = self.codec or self.formats.select(raw, location)
        record = codec.adapter.parse(raw, location)
        codec.validate(record)
        return codec, record


@dataclass(frozen=True)
class DatasetWriter:
    backend: StorageBackend
    codec: Optional[RecordCodec] = None

    @contextmanager
    def open(self, path: str) -> Iterator[Callable[[Any], None]]:
        with self.backend.writer(path) as writer:
            yield lambda record: writer.write(
                self.codec.adapter.serialize(record) if self.codec else record
            )
