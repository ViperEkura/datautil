from __future__ import annotations

import tempfile
from collections import Counter
from typing import Any

from datautil.common.errors import DataError, Issue
from datautil.common.index import DiskIndex
from datautil.common.io import iter_raw, output_writer
from datautil.conversation.formats import ADAPTERS, detect_record, validate_record


def detect(path: str, container: str = "auto", sample: int = 100) -> dict[str, Any]:
    if sample <= 0:
        raise DataError("invalid_argument", "sample must be positive")
    counts: Counter[str] = Counter()
    seen = 0
    for _, raw in iter_raw(path, container):
        seen += 1
        for name in detect_record(raw):
            counts[name] += 1
        if seen >= sample:
            break
    candidates = [
        {
            "format": name,
            "matches": counts[name],
            "sampled": seen,
            "confidence": counts[name] / seen if seen else 0,
        }
        for name in ADAPTERS
    ]
    ranked = sorted(candidates, key=lambda item: (-item["matches"], item["format"]))
    selected = (
        ranked[0]["format"]
        if seen
        and ranked[0]["matches"] == seen
        and (len(ranked) < 2 or ranked[1]["matches"] == 0)
        else None
    )
    return {
        "container": container
        if container != "auto"
        else (
            "jsonl" if path == "-" or path.endswith((".jsonl", ".ndjson")) else "json"
        ),
        "sampled": seen,
        "selected": selected,
        "candidates": ranked,
    }


def inspect(
    path: str, sample: int = 1000, format_name: str | None = None
) -> dict[str, Any]:
    if sample <= 0:
        raise DataError("invalid_argument", "sample must be positive")
    if format_name and format_name not in ADAPTERS:
        raise DataError("unknown_format", f"Unknown format {format_name!r}")
    container = (
        getattr(ADAPTERS[format_name], "container", "auto") if format_name else "auto"
    )
    fields: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    turns: Counter[int] = Counter()
    issues: list[dict[str, Any]] = []
    sampled = 0
    for location, raw in iter_raw(path, container):
        if sampled >= sample:
            break
        sampled += 1
        if isinstance(raw, dict):
            fields.update(raw.keys())
        try:
            selected = format_name or detect_record(raw)
            if isinstance(selected, list):
                if len(selected) != 1:
                    raise DataError("ambiguous_format", f"Matches {selected}", location)
                selected = selected[0]
            record = ADAPTERS[selected].parse(raw, location)
            validate_record(record)
            turns[len(record.conversation.messages)] += 1
            roles.update(message.role for message in record.conversation.messages)
        except DataError as exc:
            issues.append(Issue(exc.code, str(exc), exc.location or location).as_dict())
    return {
        "sampled": sampled,
        "fields": dict(fields),
        "roles": dict(roles),
        "message_counts": dict(turns),
        "issues": issues[:100],
        "issue_count": len(issues),
    }


def validate(
    path: str,
    source: str | None,
    target: str | None = None,
    on_error: str = "fail",
    max_errors: int = 1000,
) -> dict[str, Any]:
    if max_errors < 0:
        raise DataError("invalid_argument", "max_errors must be non-negative")
    if source and source not in ADAPTERS:
        raise DataError("unknown_format", f"Unknown format {source!r}")
    if target and target not in ADAPTERS:
        raise DataError("unknown_format", f"Unknown format {target!r}")
    container = getattr(ADAPTERS[source], "container", "auto") if source else "auto"
    count = 0
    issue_count = 0
    issues: list[dict[str, Any]] = []

    with (
        tempfile.TemporaryDirectory(prefix="datautil-validate-") as directory,
        DiskIndex(directory) as index,
    ):
        for location, raw in iter_raw(path, container, errors="yield"):
            count += 1
            try:
                if isinstance(raw, DataError):
                    raise raw
                selected = source or detect_record(raw)
                if isinstance(selected, list):
                    if len(selected) != 1:
                        raise DataError(
                            "ambiguous_format", f"Matches {selected}", location
                        )
                    selected = selected[0]
                record = ADAPTERS[selected].parse(raw, location)
                validate_record(record)
                if not index.add_unique(record.record_id, location):
                    raise DataError(
                        "duplicate_record_id",
                        f"Duplicate record ID {record.record_id!r}; first seen at {index.first_location(record.record_id)}",
                        location,
                    )
                if target:
                    ADAPTERS[target].serialize(record)
            except DataError as exc:
                issue_count += 1
                if len(issues) < max_errors:
                    issues.append(
                        Issue(exc.code, str(exc), exc.location or location).as_dict()
                    )
                if on_error == "fail":
                    break
    return {
        "read_count": count,
        "issue_count": issue_count,
        "issues": issues,
        "issues_truncated": issue_count > len(issues),
    }


def convert(
    path: str,
    output: str,
    source: str | None,
    target: str,
    *,
    on_error: str = "fail",
    dry_run: bool = False,
    input_container: str = "auto",
    output_container: str = "auto",
) -> dict[str, Any]:
    if source and source not in ADAPTERS:
        raise DataError("unknown_format", f"Unknown format {source!r}")
    if target not in ADAPTERS:
        raise DataError("unknown_format", f"Unknown format {target!r}")
    if input_container == "auto" and source:
        input_container = getattr(ADAPTERS[source], "container", "auto")
    if output_container == "auto":
        output_container = getattr(ADAPTERS[target], "container", "auto")
    report: dict[str, Any] = {
        "read_count": 0,
        "written_count": 0,
        "skipped_count": 0,
        "issues": [],
    }

    def process(writer: Any = None) -> None:
        for location, raw in iter_raw(
            path, input_container, errors="yield" if on_error == "skip" else "raise"
        ):
            report["read_count"] += 1
            try:
                if isinstance(raw, DataError):
                    raise raw
                selected = source or detect_record(raw)
                if isinstance(selected, list):
                    if len(selected) != 1:
                        raise DataError(
                            "ambiguous_format", f"Matches {selected}", location
                        )
                    selected = selected[0]
                record = ADAPTERS[selected].parse(raw, location)
                validate_record(record)
                encoded = ADAPTERS[target].serialize(record)
                if writer is not None:
                    writer.write(encoded)
                report["written_count"] += 1
            except DataError as exc:
                report["skipped_count"] += 1
                if len(report["issues"]) < 1000:
                    report["issues"].append(
                        Issue(exc.code, str(exc), exc.location or location).as_dict()
                    )
                if on_error == "fail":
                    raise

    if dry_run:
        process()
    else:
        with output_writer(output, output_container) as writer:
            process(writer)
    report["issues_truncated"] = report["skipped_count"] > len(report["issues"])
    return report
