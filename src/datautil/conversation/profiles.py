"""Declarative JSON record mappings, including ClickHouse JSONEachRow exports.

Profiles describe the shape of one logical record. They do not execute Python,
SQL, JSONPath, templates, or arbitrary expressions from YAML.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from datautil.common.errors import DataError
from datautil.conversation.formats import FormatAdapter, _id
from datautil.conversation.model import Conversation, DatasetRecord, Message, ToolCall

_NAME = re.compile(r"^[a-z][a-z0-9-]*$")
_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_CANONICAL_ROLES = {"system", "developer", "user", "assistant", "tool"}


def _parts(path: str) -> tuple[str, ...]:
    if (
        not isinstance(path, str)
        or not path
        or any(not _SEGMENT.fullmatch(part) for part in path.split("."))
    ):
        raise DataError("invalid_profile", f"Invalid field path {path!r}")
    return tuple(path.split("."))


def _get(value: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in _parts(path):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _set(value: dict[str, Any], path: str, content: Any) -> None:
    current = value
    parts = _parts(path)
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise DataError(
                "invalid_profile", f"Output path {path!r} conflicts with existing data"
            )
        current = child
    if parts[-1] in current and current[parts[-1]] != content:
        raise DataError(
            "metadata_conflict",
            f"Output path {path!r} conflicts with existing metadata",
        )
    current[parts[-1]] = content


def _remove(value: dict[str, Any], path: str) -> None:
    current = value
    stack: list[tuple[dict[str, Any], str]] = []
    for part in _parts(path):
        if not isinstance(current, dict) or part not in current:
            return
        stack.append((current, part))
        current = current[part]
    parent, key = stack.pop()
    del parent[key]
    while stack and not parent:
        parent, key = stack.pop()
        del parent[key]


class YamlFormatAdapter(FormatAdapter):
    @property
    def label_heading(self) -> str:
        return "Source roles" if self.layout == "messages" else "Mapped fields"

    def display_label(self, role: str) -> str:
        if self.layout == "messages":
            return self.inverse_roles.get(role, role)
        return self.fields.get(
            {"system": "system", "user": "prompt", "assistant": "completion"}.get(
                role, ""
            ),
            role,
        )

    def __init__(self, config: dict[str, Any], path: str) -> None:
        self.path = path
        if not isinstance(config, dict) or config.get("version") != 1:
            raise DataError("invalid_profile", "Profile needs version: 1", path)
        allowed_options = {
            "version",
            "name",
            "container",
            "layout",
            "fields",
            "role_map",
            "ignore_message_fields",
        }
        if unknown := set(config) - allowed_options:
            raise DataError(
                "invalid_profile", f"Unknown profile options: {sorted(unknown)}", path
            )
        self.name = config.get("name")
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise DataError(
                "invalid_profile",
                "Profile name must use lowercase letters, digits, and hyphens",
                path,
            )
        self.layout = config.get("layout")
        if self.layout not in {"messages", "pair"}:
            raise DataError("invalid_profile", "layout must be messages or pair", path)
        container = config.get("container", "jsonl")
        if container == "JSONEachRow":
            container = "jsonl"
        if container not in {"json", "jsonl"}:
            raise DataError(
                "invalid_profile", "container must be json, jsonl, or JSONEachRow", path
            )
        self.container = container
        self.fields = config.get("fields")
        if not isinstance(self.fields, dict):
            raise DataError("invalid_profile", "fields must be a mapping", path)
        required = (
            {"messages", "role", "content"}
            if self.layout == "messages"
            else {"prompt", "completion"}
        )
        if not required <= self.fields.keys():
            raise DataError(
                "invalid_profile",
                f"Missing fields: {sorted(required - self.fields.keys())}",
                path,
            )
        allowed = required | (
            {"id", "tool_calls", "tool_call_id", "name", "thinking"}
            if self.layout == "messages"
            else {"id", "system", "thinking"}
        )
        if unknown := set(self.fields) - allowed:
            raise DataError(
                "invalid_profile", f"Unknown fields: {sorted(unknown)}", path
            )
        for field_path in self.fields.values():
            _parts(field_path)
        root_fields = [
            self.fields[key]
            for key in ("id", "messages", "prompt", "completion", "system")
            if key in self.fields
        ]
        if self.layout == "pair" and "thinking" in self.fields:
            root_fields.append(self.fields["thinking"])
        message_fields = (
            [
                self.fields[key]
                for key in (
                    "role",
                    "content",
                    "tool_calls",
                    "tool_call_id",
                    "name",
                    "thinking",
                )
                if key in self.fields
            ]
            if self.layout == "messages"
            else []
        )
        for paths in (root_fields, message_fields):
            for index, left in enumerate(paths):
                for right in paths[index + 1 :]:
                    if (
                        left == right
                        or left.startswith(right + ".")
                        or right.startswith(left + ".")
                    ):
                        raise DataError(
                            "invalid_profile",
                            f"Overlapping field paths: {left!r} and {right!r}",
                            path,
                        )
        role_map = config.get("role_map", {})
        if self.layout == "pair" and role_map:
            raise DataError(
                "invalid_profile", "role_map is only valid for messages layout", path
            )
        if not isinstance(role_map, dict) or any(
            not isinstance(key, str) or value not in _CANONICAL_ROLES
            for key, value in role_map.items()
        ):
            raise DataError(
                "invalid_profile",
                "role_map must map source roles to canonical roles",
                path,
            )
        if len(set(role_map.values())) != len(role_map):
            raise DataError(
                "invalid_profile",
                "role_map must be one-to-one for reversible export",
                path,
            )
        self.role_map: dict[str, str] = role_map
        self.inverse_roles = {value: key for key, value in role_map.items()}
        ignored = config.get("ignore_message_fields", [])
        if not isinstance(ignored, list) or any(
            not isinstance(item, str) for item in ignored
        ):
            raise DataError(
                "invalid_profile", "ignore_message_fields must be a list of paths", path
            )
        for item in ignored:
            _parts(item)
        self.ignored_message_fields = ignored

    def matches(self, raw: Any) -> bool:
        if not isinstance(raw, dict):
            return False
        if self.layout == "pair":
            return all(
                isinstance(_get(raw, self.fields[key]), str)
                for key in ("prompt", "completion")
            )
        messages = _get(raw, self.fields["messages"])
        return (
            isinstance(messages, list)
            and bool(messages)
            and isinstance(messages[0], dict)
            and isinstance(_get(messages[0], self.fields["role"]), str)
        )

    def parse(self, raw: Any, location: str) -> DatasetRecord:
        if not isinstance(raw, dict):
            raise DataError("invalid_structure", "Expected a JSON object", location)
        ident = _get(raw, self.fields["id"]) if "id" in self.fields else None
        record_id = str(ident) if ident is not None else _id(raw, location)
        extra = copy.deepcopy(raw)
        if self.layout == "pair":
            prompt = _get(raw, self.fields["prompt"])
            completion = _get(raw, self.fields["completion"])
            system = (
                _get(raw, self.fields["system"]) if "system" in self.fields else None
            )
            thinking = (
                _get(raw, self.fields["thinking"])
                if "thinking" in self.fields
                else None
            )
            if (
                not isinstance(prompt, str)
                or not isinstance(completion, str)
                or (system is not None and not isinstance(system, str))
                or (thinking is not None and not isinstance(thinking, str))
            ):
                raise DataError(
                    "invalid_structure", "Pair fields must contain text", location
                )
            messages = ([Message("system", system)] if system else []) + [
                Message("user", prompt),
                Message("assistant", completion, thinking=thinking),
            ]
        else:
            source_messages = _get(raw, self.fields["messages"])
            if not isinstance(source_messages, list):
                raise DataError(
                    "invalid_structure",
                    "Configured messages path must contain an array",
                    location,
                )
            messages = []
            for index, source in enumerate(source_messages):
                if not isinstance(source, dict):
                    raise DataError(
                        "invalid_message",
                        f"messages[{index}] must be an object",
                        location,
                    )
                role_value = _get(source, self.fields["role"])
                content = _get(source, self.fields["content"])
                role = self.role_map.get(role_value, role_value)
                if role not in _CANONICAL_ROLES or not isinstance(content, str):
                    raise DataError(
                        "invalid_message",
                        f"messages[{index}] has invalid role or content",
                        location,
                    )
                calls = (
                    _get(source, self.fields["tool_calls"], [])
                    if "tool_calls" in self.fields
                    else []
                )
                if not isinstance(calls, list):
                    raise DataError(
                        "invalid_tool_call",
                        f"messages[{index}] tool_calls must be an array",
                        location,
                    )
                parsed_calls = []
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    if not isinstance(function, dict) or not all(
                        isinstance(value, str)
                        for value in (
                            call.get("id"),
                            function.get("name"),
                            function.get("arguments"),
                        )
                    ):
                        raise DataError(
                            "invalid_tool_call",
                            f"messages[{index}] has invalid tool call",
                            location,
                        )
                    parsed_calls.append(
                        ToolCall(call["id"], function["name"], function["arguments"])
                    )
                call_id = (
                    _get(source, self.fields["tool_call_id"])
                    if "tool_call_id" in self.fields
                    else None
                )
                name = (
                    _get(source, self.fields["name"]) if "name" in self.fields else None
                )
                thinking = (
                    _get(source, self.fields["thinking"])
                    if "thinking" in self.fields
                    else None
                )
                if call_id is not None and not isinstance(call_id, str):
                    raise DataError(
                        "invalid_tool_call",
                        f"messages[{index}] tool_call_id must be text",
                        location,
                    )
                if thinking is not None and not isinstance(thinking, str):
                    raise DataError(
                        "invalid_content",
                        f"messages[{index}] thinking must be text",
                        location,
                    )
                remaining = copy.deepcopy(source)
                for field in (
                    "role",
                    "content",
                    "tool_calls",
                    "tool_call_id",
                    "name",
                    "thinking",
                ):
                    if field in self.fields:
                        _remove(remaining, self.fields[field])
                for field in self.ignored_message_fields:
                    _remove(remaining, field)
                if remaining:
                    raise DataError(
                        "unmapped_message_fields",
                        f"messages[{index}] has unmapped fields: {sorted(remaining)}",
                        location,
                    )
                messages.append(
                    Message(role, content, tuple(parsed_calls), call_id, name, thinking)
                )
        for field in ("id", "messages", "prompt", "completion", "system", "thinking"):
            if field in self.fields:
                _remove(extra, self.fields[field])
        return DatasetRecord(record_id, Conversation(tuple(messages)), location, extra)

    def serialize(self, record: DatasetRecord) -> dict[str, Any]:
        output = copy.deepcopy(
            {
                key: value
                for key, value in record.metadata.items()
                if not key.startswith("_alpaca_")
            }
        )
        if "id" in self.fields:
            _set(output, self.fields["id"], record.record_id)
        messages = record.conversation.messages
        if self.layout == "pair":
            system = None
            if len(messages) == 3 and messages[0].role == "system":
                if messages[0].thinking is not None:
                    raise DataError(
                        "lossy_conversion",
                        "Pair layout cannot represent thinking on a system message",
                        record.source_location,
                    )
                system = messages[0].content
                messages = messages[1:]
            if (
                len(messages) != 2
                or [item.role for item in messages] != ["user", "assistant"]
                or any(item.tool_calls or item.tool_call_id for item in messages)
                or messages[0].thinking is not None
            ):
                raise DataError(
                    "lossy_conversion",
                    "Pair layout needs one user and one assistant message, without tools",
                    record.source_location,
                )
            if system is not None:
                if "system" not in self.fields:
                    raise DataError(
                        "lossy_conversion",
                        "Profile has no system field",
                        record.source_location,
                    )
                _set(output, self.fields["system"], system)
            _set(output, self.fields["prompt"], messages[0].content)
            _set(output, self.fields["completion"], messages[1].content)
            if messages[1].thinking is not None:
                if "thinking" not in self.fields:
                    raise DataError(
                        "lossy_conversion",
                        "Profile has no thinking field",
                        record.source_location,
                    )
                _set(output, self.fields["thinking"], messages[1].thinking)
        else:
            encoded = []
            for message in messages:
                item: dict[str, Any] = {}
                role = self.inverse_roles.get(message.role, message.role)
                _set(item, self.fields["role"], role)
                _set(item, self.fields["content"], message.content)
                if message.tool_calls:
                    if "tool_calls" not in self.fields:
                        raise DataError(
                            "lossy_conversion",
                            "Profile has no tool_calls field",
                            record.source_location,
                        )
                    _set(
                        item,
                        self.fields["tool_calls"],
                        [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                            for call in message.tool_calls
                        ],
                    )
                if message.tool_call_id:
                    if "tool_call_id" not in self.fields:
                        raise DataError(
                            "lossy_conversion",
                            "Profile has no tool_call_id field",
                            record.source_location,
                        )
                    _set(item, self.fields["tool_call_id"], message.tool_call_id)
                if message.name:
                    if "name" not in self.fields:
                        raise DataError(
                            "lossy_conversion",
                            "Profile has no name field",
                            record.source_location,
                        )
                    _set(item, self.fields["name"], message.name)
                if message.thinking is not None:
                    if "thinking" not in self.fields:
                        raise DataError(
                            "lossy_conversion",
                            "Profile has no thinking field",
                            record.source_location,
                        )
                    _set(item, self.fields["thinking"], message.thinking)
                encoded.append(item)
            _set(output, self.fields["messages"], encoded)
        return output


def load_profiles(paths: list[str]) -> list[YamlFormatAdapter]:
    profiles = []
    for path in paths:
        try:
            config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise DataError("invalid_profile", str(exc), path) from exc
        profiles.append(YamlFormatAdapter(config, path))
    names = [profile.name for profile in profiles]
    if len(names) != len(set(names)):
        raise DataError("invalid_profile", "Duplicate profile names in --format-config")
    return profiles
