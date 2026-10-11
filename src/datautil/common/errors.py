from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    location: str
    record_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "record_id": self.record_id,
        }


class DataError(Exception):
    def __init__(self, code: str, message: str, location: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.location = location
