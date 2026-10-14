from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Protocol

from datautil.common.errors import DataError
from datautil.conversation.model import Conversation, DatasetRecord, Message, ToolCall


def _id(raw: dict[str, Any], location: str) -> str:
    if raw.get("id") is not None:
        return str(raw["id"])
    digest = hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]
    return f"{digest}-{hashlib.sha256(location.encode()).hexdigest()[:8]}"


class FormatAdapter(Protocol):
    name: str
    record_kind: str
    label_heading: str

    def display_label(self, role: str) -> str: ...

    def parse(self, raw: Any, location: str) -> DatasetRecord: ...

    def serialize(self, record: DatasetRecord) -> dict[str, Any]: ...


class OpenAIMessagesAdapter:
    name = "openai-messages"
    record_kind = "conversation"
    label_heading = "Message roles"

    def display_label(self, role: str) -> str:
        return role

    def parse(self, raw: Any, location: str) -> DatasetRecord:
        if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
            raise DataError(
                "invalid_structure",
                "Expected an object with a messages array",
                location,
            )
        messages: list[Message] = []
        for index, item in enumerate(raw["messages"]):
            if not isinstance(item, dict) or not isinstance(item.get("role"), str):
                raise DataError(
                    "invalid_message", f"messages[{index}] needs a role", location
                )
            content = item.get("content")
            if content is None:
                content = ""
            if isinstance(content, list):
                chunks = []
                for part in content:
                    if (
                        not isinstance(part, dict)
                        or part.get("type") not in ("text", "input_text", "output_text")
                        or not isinstance(part.get("text"), str)
                    ):
                        raise DataError(
                            "unsupported_content",
                            f"messages[{index}] has a non-text part",
                            location,
                        )
                    chunks.append(part["text"])
                content = "".join(chunks)
            if not isinstance(content, str):
                raise DataError(
                    "invalid_content",
                    f"messages[{index}].content must be text",
                    location,
                )
            thinking = item.get("reasoning_content")
            if thinking is not None and not isinstance(thinking, str):
                raise DataError(
                    "invalid_content",
                    f"messages[{index}].reasoning_content must be text",
                    location,
                )
            calls = []
            for call in item.get("tool_calls") or []:
                if not isinstance(call, dict) or not isinstance(
                    call.get("function"), dict
                ):
                    raise DataError(
                        "invalid_tool_call",
                        f"messages[{index}] has an invalid tool call",
                        location,
                    )
                function = call["function"]
                if not all(
                    isinstance(x, str)
                    for x in (
                        call.get("id"),
                        function.get("name"),
                        function.get("arguments"),
                    )
                ):
                    raise DataError(
                        "invalid_tool_call",
                        f"messages[{index}] has an invalid tool call",
                        location,
                    )
                calls.append(
                    ToolCall(call["id"], function["name"], function["arguments"])
                )
            messages.append(
                Message(
                    item["role"],
                    content,
                    tuple(calls),
                    item.get("tool_call_id"),
                    item.get("name"),
                    thinking,
                )
            )
        metadata = {
            key: value for key, value in raw.items() if key not in {"id", "messages"}
        }
        return DatasetRecord(
            _id(raw, location), Conversation(tuple(messages)), location, metadata
        )

    def serialize(self, record: DatasetRecord) -> dict[str, Any]:
        messages = []
        for item in record.conversation.messages:
            value: dict[str, Any] = {"role": item.role, "content": item.content}
            if item.tool_calls:
                value["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in item.tool_calls
                ]
            if item.tool_call_id:
                value["tool_call_id"] = item.tool_call_id
            if item.name:
                value["name"] = item.name
            if item.thinking is not None:
                value["reasoning_content"] = item.thinking
            messages.append(value)
        return {"id": record.record_id, "messages": messages, **record.metadata}


class ShareGPTAdapter:
    name = "sharegpt"
    record_kind = "conversation"
    label_heading = "Source roles"
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    inverse = {value: key for key, value in roles.items()}

    def display_label(self, role: str) -> str:
        return self.inverse.get(role, role)

    def parse(self, raw: Any, location: str) -> DatasetRecord:
        if not isinstance(raw, dict) or not isinstance(raw.get("conversations"), list):
            raise DataError(
                "invalid_structure",
                "Expected an object with a conversations array",
                location,
            )
        messages = []
        for index, item in enumerate(raw["conversations"]):
            if (
                not isinstance(item, dict)
                or item.get("from") not in self.roles
                or not isinstance(item.get("value"), str)
            ):
                raise DataError(
                    "invalid_message",
                    f"conversations[{index}] needs from/value",
                    location,
                )
            messages.append(Message(self.roles[item["from"]], item["value"]))
        metadata = {
            key: value
            for key, value in raw.items()
            if key not in {"id", "conversations"}
        }
        return DatasetRecord(
            _id(raw, location), Conversation(tuple(messages)), location, metadata
        )

    def serialize(self, record: DatasetRecord) -> dict[str, Any]:
        conversations = []
        for item in record.conversation.messages:
            if (
                item.role not in self.inverse
                or item.tool_calls
                or item.tool_call_id
                or item.thinking is not None
            ):
                raise DataError(
                    "lossy_conversion",
                    "ShareGPT cannot represent this role, tool call, or thinking field",
                    record.source_location,
                )
            conversations.append(
                {"from": self.inverse[item.role], "value": item.content}
            )
        return {
            "id": record.record_id,
            "conversations": conversations,
            **record.metadata,
        }


class AlpacaAdapter:
    name = "alpaca"
    record_kind = "conversation"
    label_heading = "Mapped fields"

    def display_label(self, role: str) -> str:
        return {"user": "instruction", "assistant": "output"}.get(role, role)

    def parse(self, raw: Any, location: str) -> DatasetRecord:
        if not isinstance(raw, dict) or not all(
            isinstance(raw.get(key), str) for key in ("instruction", "input", "output")
        ):
            raise DataError(
                "invalid_structure",
                "Expected instruction, input, output strings",
                location,
            )
        prompt = raw["instruction"] + (("\n\n" + raw["input"]) if raw["input"] else "")
        messages = (Message("user", prompt), Message("assistant", raw["output"]))
        metadata = {
            key: value
            for key, value in raw.items()
            if key not in {"id", "instruction", "input", "output"}
        }
        metadata["_alpaca_instruction"] = raw["instruction"]
        metadata["_alpaca_input"] = raw["input"]
        return DatasetRecord(
            _id(raw, location), Conversation(messages), location, metadata
        )

    def serialize(self, record: DatasetRecord) -> dict[str, Any]:
        messages = record.conversation.messages
        if (
            len(messages) != 2
            or [m.role for m in messages] != ["user", "assistant"]
            or any(
                m.tool_calls or m.tool_call_id or m.thinking is not None
                for m in messages
            )
        ):
            raise DataError(
                "lossy_conversion",
                "Alpaca requires one text user message and one text assistant message without thinking or tools",
                record.source_location,
            )
        instruction = record.metadata.get("_alpaca_instruction")
        context = record.metadata.get("_alpaca_input")
        if not (
            isinstance(instruction, str)
            and isinstance(context, str)
            and messages[0].content
            == instruction + (("\n\n" + context) if context else "")
        ):
            instruction, context = messages[0].content, ""
        metadata = {
            key: value
            for key, value in record.metadata.items()
            if not key.startswith("_alpaca_")
        }
        return {
            "id": record.record_id,
            "instruction": instruction,
            "input": context,
            "output": messages[1].content,
            **metadata,
        }


_BUILTINS: dict[str, FormatAdapter] = {
    item.name: item
    for item in (OpenAIMessagesAdapter(), ShareGPTAdapter(), AlpacaAdapter())
}
_ACTIVE: ContextVar[Mapping[str, FormatAdapter] | None] = ContextVar(
    "datautil_formats", default=None
)


class FormatRegistry(MappingABC[str, FormatAdapter]):
    """Read-only view of the adapters scoped to the current command."""

    def __getitem__(self, name: str) -> FormatAdapter:
        return (_ACTIVE.get() or _BUILTINS)[name]

    def __iter__(self) -> Iterator[str]:
        return iter(_ACTIVE.get() or _BUILTINS)

    def __len__(self) -> int:
        return len(_ACTIVE.get() or _BUILTINS)


ADAPTERS: Mapping[str, FormatAdapter] = FormatRegistry()


@contextmanager
def use_profiles(profiles: list[FormatAdapter]) -> Iterator[None]:
    registry = dict(_BUILTINS)
    for profile in profiles:
        if profile.name in _BUILTINS:
            raise DataError(
                "invalid_profile", f"Profile name {profile.name!r} is reserved"
            )
        registry[profile.name] = profile
    token = _ACTIVE.set(registry)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def detect_record(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    matches = []
    if isinstance(raw.get("messages"), list):
        matches.append("openai-messages")
    if isinstance(raw.get("conversations"), list):
        matches.append("sharegpt")
    if all(isinstance(raw.get(key), str) for key in ("instruction", "input", "output")):
        matches.append("alpaca")
    for name, adapter in ADAPTERS.items():
        matcher = getattr(adapter, "matches", None)
        if matcher is not None and matcher(raw):
            matches.append(name)
    return matches


def validate_record(record: DatasetRecord) -> None:
    messages = record.conversation.messages
    if not messages:
        raise DataError(
            "empty_conversation", "Conversation has no messages", record.source_location
        )
    allowed = {"system", "developer", "user", "assistant", "tool"}
    calls = {call.id for message in messages for call in message.tool_calls}
    for index, message in enumerate(messages):
        if message.role not in allowed:
            raise DataError(
                "invalid_role",
                f"messages[{index}] has unknown role {message.role!r}",
                record.source_location,
            )
        if not message.content and not message.tool_calls and not message.thinking:
            raise DataError(
                "empty_message",
                f"messages[{index}] has no content, thinking, or tool call",
                record.source_location,
            )
        if message.role == "tool" and (
            not message.tool_call_id or message.tool_call_id not in calls
        ):
            raise DataError(
                "orphan_tool_result",
                f"messages[{index}] has no matching tool call",
                record.source_location,
            )
