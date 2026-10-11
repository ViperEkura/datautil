from __future__ import annotations

from datautil.conversation.model import DatasetRecord


def _text(record: DatasetRecord, scope: str = "conversation") -> str | None:
    if scope == "user":
        messages = [
            message.content
            for message in record.conversation.messages
            if message.role == "user" and message.content.strip()
        ]
        return (
            "\n".join("user:" + content for content in messages) if messages else None
        )
    chunks = []
    for message in record.conversation.messages:
        chunks.append(message.role + ":" + message.content)
        for call in message.tool_calls:
            chunks.append("tool:" + call.name + ":" + call.arguments)
    return "\n".join(chunks)
