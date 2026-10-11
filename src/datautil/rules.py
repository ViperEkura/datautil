"""Declarative, ordered record cleaning and filtering rules."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any

import yaml

from .formats import ADAPTERS, validate_record
from .io import iter_raw, output_writer
from .model import Conversation, DataError, DatasetRecord
from .streaming import DiskIndex


_ROLES = {"system", "developer", "user", "assistant", "tool"}
_FILTER_KEYS = {"min_messages", "max_messages", "min_chars", "max_chars", "roles_all", "roles_any"}


def load_rules(path: str) -> list[dict[str, Any]]:
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataError("invalid_rules", str(exc), path) from exc
    return validate_rules(document, path)


def validate_rules(document: Any, source: str = "rules") -> list[dict[str, Any]]:
    if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1 or not isinstance(document.get("rules"), list):
        raise DataError("invalid_rules", "Rules need version: 1 and a rules list", source)
    if set(document) - {"version", "rules"}:
        raise DataError("invalid_rules", "Unknown top-level rule options", source)
    rules = document["rules"]
    for index, rule in enumerate(rules):
        location = f"{source}:rules[{index}]"
        if not isinstance(rule, dict) or not isinstance(rule.get("op"), str) or rule["op"] not in {"trim_content", "collapse_whitespace", "replace_text", "drop_empty_messages", "keep_if"}:
            raise DataError("invalid_rules", "Unknown rule operation", location)
        operation = rule["op"]
        allowed = {"op", "roles"}
        if operation == "replace_text":
            allowed |= {"old", "new"}
            if not isinstance(rule.get("old"), str) or not rule["old"] or not isinstance(rule.get("new"), str):
                raise DataError("invalid_rules", "replace_text needs nonempty old and string new", location)
        elif operation == "keep_if":
            allowed |= {"conditions"}
            conditions = rule.get("conditions")
            if not isinstance(conditions, dict) or not conditions or set(conditions) - _FILTER_KEYS:
                raise DataError("invalid_rules", "keep_if needs supported conditions", location)
            for key, value in conditions.items():
                if key in {"roles_all", "roles_any"}:
                    if not isinstance(value, list) or not value or any(not isinstance(item, str) or item not in _ROLES for item in value):
                        raise DataError("invalid_rules", f"{key} must be a list of roles", location)
                elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise DataError("invalid_rules", f"{key} must be a nonnegative integer", location)
        if set(rule) - allowed:
            raise DataError("invalid_rules", f"Unknown options: {sorted(set(rule) - allowed)}", location)
        if operation in {"drop_empty_messages", "keep_if"} and "roles" in rule:
            raise DataError("invalid_rules", "roles is not used by this operation", location)
        if "roles" in rule and (not isinstance(rule["roles"], list) or not rule["roles"] or any(not isinstance(role, str) or role not in _ROLES for role in rule["roles"])):
            raise DataError("invalid_rules", "roles must be a nonempty list of canonical roles", location)
    return rules


def _keep(record: DatasetRecord, conditions: dict[str, Any]) -> bool:
    messages = record.conversation.messages
    count = len(messages)
    chars = sum(len(message.content) for message in messages)
    roles = {message.role for message in messages}
    return (
        ("min_messages" not in conditions or count >= conditions["min_messages"])
        and ("max_messages" not in conditions or count <= conditions["max_messages"])
        and ("min_chars" not in conditions or chars >= conditions["min_chars"])
        and ("max_chars" not in conditions or chars <= conditions["max_chars"])
        and ("roles_all" not in conditions or set(conditions["roles_all"]) <= roles)
        and ("roles_any" not in conditions or bool(set(conditions["roles_any"]) & roles))
    )


def apply_rules(record: DatasetRecord, rules: list[dict[str, Any]]) -> tuple[DatasetRecord | None, int | None]:
    current = record
    for index, rule in enumerate(rules):
        operation = rule["op"]
        if operation == "keep_if":
            if not _keep(current, rule["conditions"]):
                return None, index
            continue
        messages = []
        for message in current.conversation.messages:
            if operation == "drop_empty_messages":
                if not message.content and not message.tool_calls and not message.thinking:
                    continue
                messages.append(message)
                continue
            if "roles" in rule and message.role not in rule["roles"]:
                messages.append(message)
                continue
            if operation == "trim_content":
                content = message.content.strip()
            elif operation == "collapse_whitespace":
                content = " ".join(message.content.split())
            else:
                content = message.content.replace(rule["old"], rule["new"])
            messages.append(replace(message, content=content))
        current = replace(current, conversation=Conversation(tuple(messages)))
    return current, None


def transform_dataset(input_path: str, output_path: str, source_format: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    if source_format not in ADAPTERS:
        raise DataError("unknown_format", f"Unknown format {source_format!r}")
    adapter = ADAPTERS[source_format]
    input_container = getattr(adapter, "container", "auto")
    output_container = getattr(adapter, "container", "auto")
    read_count = written_count = 0
    dropped: Counter[int] = Counter()
    with tempfile.TemporaryDirectory(prefix="datautil-transform-") as directory, DiskIndex(directory) as index, output_writer(output_path, output_container) as writer:
        for location, raw in iter_raw(input_path, input_container):
            read_count += 1
            record = adapter.parse(raw, location)
            if not index.add_unique(record.record_id):
                raise DataError("duplicate_record_id", f"Duplicate record ID {record.record_id!r}", location)
            transformed, drop_index = apply_rules(record, rules)
            if transformed is None:
                dropped[drop_index] += 1
                continue
            validate_record(transformed)
            writer.write(adapter.serialize(transformed))
            written_count += 1
    return {"read_count": read_count, "written_count": written_count, "dropped_count": read_count - written_count, "dropped_by_rule": {str(index): count for index, count in dropped.items()}}
