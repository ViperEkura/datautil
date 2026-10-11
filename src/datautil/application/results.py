from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CommandSpec:
    name: str
    options: Mapping[str, Any]


@dataclass
class CommandResult:
    exit_code: int
    summary: dict[str, Any]
    datasets: dict[str, str]
    reports: dict[str, str]
    messages: list[tuple[str, str]] = field(default_factory=list)
