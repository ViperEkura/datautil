from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    record_id: str
    text: str
    source_location: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_format: str | None = None
