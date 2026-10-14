from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from datautil.common.errors import DataError
from datautil.common.storage import storage_for
from datautil.pretrain.quality import validate_document_quality
from datautil.pretrain.sources import validate_fields


@dataclass(frozen=True)
class Source:
    name: str
    path: str
    storage: str
    fields: dict[str, Any]
    weight: float

    @property
    def format(self) -> str:
        return self.storage


def _path(base: Path, value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataError("invalid_pretrain_config", f"{label} must be a path")
    return str((base / value).resolve())


def _dedup_settings(value: Any, name: str) -> dict[str, Any] | None:
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict) or set(value) - {
        "threshold",
        "shingle_size",
        "max_candidates",
        "min_chars",
        "normalization",
    }:
        raise DataError(
            "invalid_pretrain_config", f"dedup.{name} has unsupported options"
        )
    threshold = value.get("threshold", 0.8)
    shingle_size = value.get("shingle_size", 5)
    max_candidates = value.get("max_candidates", 10000)
    min_chars = value.get("min_chars", 80 if name == "paragraph" else 0)
    normalization = value.get("normalization", "nfkc")
    if type(threshold) not in {int, float} or not 0 < threshold <= 1:
        raise DataError(
            "invalid_pretrain_config", f"dedup.{name}.threshold must be in (0, 1]"
        )
    if (
        any(
            type(item) is not int or item < 1 for item in (shingle_size, max_candidates)
        )
        or type(min_chars) is not int
        or min_chars < 0
    ):
        raise DataError(
            "invalid_pretrain_config",
            f"dedup.{name} sizes must be nonnegative integers",
        )
    if normalization not in {"nfkc", "nfkc_casefold", "none"}:
        raise DataError(
            "invalid_pretrain_config", f"dedup.{name}.normalization is unsupported"
        )
    return {
        "threshold": float(threshold),
        "shingle_size": shingle_size,
        "max_candidates": max_candidates,
        "min_chars": min_chars,
        "normalization": normalization,
    }


def load_recipe(path: str) -> dict[str, Any]:
    try:
        content = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(content)
    except (OSError, yaml.YAMLError) as exc:
        raise DataError("invalid_pretrain_config", str(exc), path) from exc
    if (
        not isinstance(raw, dict)
        or type(raw.get("version")) is not int
        or raw["version"] != 1
        or set(raw)
        - {
            "version",
            "sources",
            "mix",
            "dedup",
            "quality",
            "output",
            "output_storage",
            "report",
            "decisions",
        }
    ):
        raise DataError(
            "invalid_pretrain_config",
            "Expected version: 1 and supported pretrain keys",
            path,
        )
    base = Path(path).resolve().parent
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise DataError(
            "invalid_pretrain_config", "sources must be a nonempty list", path
        )
    sources: list[Source] = []
    names = set()
    for position, item in enumerate(sources_raw):
        location = f"{path}:sources[{position}]"
        if not isinstance(item, dict) or set(item) - {
            "name",
            "input",
            "format",
            "storage",
            "fields",
            "weight",
        }:
            raise DataError(
                "invalid_pretrain_config", "Unknown source options", location
            )
        name = item.get("name")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_-]*", name)
            or name in names
        ):
            raise DataError(
                "invalid_pretrain_config",
                "Source names must be unique lowercase identifiers",
                location,
            )
        names.add(name)
        source_path = _path(base, item.get("input"), "source input")
        if "format" in item and "storage" in item:
            raise DataError(
                "invalid_pretrain_config",
                "Use storage or the legacy format key, not both",
                location,
            )
        source_format = storage_for(
            source_path, item.get("storage", item.get("format", "auto"))
        )
        fields = validate_fields(
            item.get("fields", {"id": "id", "text": "text"}), location
        )
        weight = item.get("weight", 1)
        if type(weight) not in {int, float} or not math.isfinite(weight) or weight <= 0:
            raise DataError(
                "invalid_pretrain_config",
                "weight must be positive and finite",
                location,
            )
        sources.append(Source(name, source_path, source_format, fields, float(weight)))
    mix = raw.get("mix")
    if (
        not isinstance(mix, dict)
        or set(mix) - {"count", "seed"}
        or type(mix.get("count")) is not int
        or mix["count"] < 0
        or type(mix.get("seed", 42)) is not int
    ):
        raise DataError(
            "invalid_pretrain_config",
            "mix needs a nonnegative integer count and optional integer seed",
            path,
        )
    dedup = raw.get("dedup", {})
    if not isinstance(dedup, dict) or set(dedup) - {"document", "paragraph"}:
        raise DataError(
            "invalid_pretrain_config", "dedup supports document and paragraph", path
        )
    document_dedup = _dedup_settings(dedup.get("document"), "document")
    paragraph_dedup = _dedup_settings(dedup.get("paragraph"), "paragraph")
    quality = validate_document_quality(raw.get("quality", {}), path)
    output = _path(base, raw.get("output"), "output")
    report = _path(base, raw["report"], "report") if "report" in raw else None
    decisions = (
        _path(base, raw["decisions"], "decisions") if "decisions" in raw else None
    )
    all_paths = (
        [item.path for item in sources]
        + [output]
        + ([report] if report else [])
        + ([decisions] if decisions else [])
    )
    if len(all_paths) != len(set(all_paths)):
        raise DataError(
            "invalid_pretrain_config", "Input and output paths must differ", path
        )
    output_storage = storage_for(output, raw.get("output_storage", "auto"))
    return {
        "sources": sources,
        "count": mix["count"],
        "seed": mix.get("seed", 42),
        "document_dedup": document_dedup,
        "paragraph_dedup": paragraph_dedup,
        "quality": quality,
        "output": output,
        "output_storage": output_storage,
        "report": report,
        "decisions": decisions,
        "config_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
