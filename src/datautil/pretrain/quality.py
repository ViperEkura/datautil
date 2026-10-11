from __future__ import annotations

import re
from typing import Any

from datautil.common.errors import DataError
from datautil.pretrain.model import DocumentRecord


def validate_document_quality(value: Any, source: str) -> dict[str, Any]:
    """Validate opt-in document checks; English presets do not run by default."""
    if not isinstance(value, dict) or set(value) - {
        "min_chars",
        "max_chars",
        "presets",
        "language",
    }:
        raise DataError(
            "invalid_pretrain_config",
            "quality supports min_chars, max_chars, presets, and language",
            source,
        )
    for key in ("min_chars", "max_chars"):
        if key in value and (type(value[key]) is not int or value[key] < 0):
            raise DataError(
                "invalid_pretrain_config", f"quality.{key} must be nonnegative", source
            )
    if value.get("min_chars", 0) > value.get("max_chars", float("inf")):
        raise DataError(
            "invalid_pretrain_config", "quality.min_chars exceeds max_chars", source
        )
    presets = value.get("presets", [])
    if (
        not isinstance(presets, list)
        or any(
            not isinstance(item, str) or item not in {"gopher", "c4"}
            for item in presets
        )
        or len(set(presets)) != len(presets)
    ):
        raise DataError(
            "invalid_pretrain_config",
            "quality.presets must list gopher and/or c4",
            source,
        )
    if presets and value.get("language") != "en":
        raise DataError(
            "invalid_pretrain_config",
            "English Gopher/C4 presets require quality.language: en",
            source,
        )
    if "language" in value and not isinstance(value["language"], str):
        raise DataError(
            "invalid_pretrain_config", "quality.language must be a string", source
        )
    return value


def document_quality_issues(
    record: DocumentRecord, quality: dict[str, Any]
) -> list[dict[str, Any]]:
    text = record.text
    issues: list[dict[str, Any]] = []

    def check(rule: str, measured: Any, operator: str, limit: Any) -> None:
        failed = measured < limit if operator == "min" else measured > limit
        if failed:
            issues.append(
                {
                    "rule": rule,
                    "value": measured,
                    "operator": operator,
                    "threshold": limit,
                }
            )

    check("min_chars", len(text), "min", quality.get("min_chars", 1))
    if "max_chars" in quality:
        check("max_chars", len(text), "max", quality["max_chars"])
    presets = quality.get("presets", [])
    if not presets:
        return issues
    words = text.split()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if "gopher" in presets:
        check("gopher.word_count_min", len(words), "min", 50)
        check("gopher.word_count_max", len(words), "max", 100000)
        if words:
            mean_length = sum(len(word) for word in words) / len(words)
            alphabetic_share = sum(
                any(char.isalpha() for char in word) for word in words
            ) / len(words)
            check("gopher.mean_word_length_min", round(mean_length, 4), "min", 3)
            check("gopher.mean_word_length_max", round(mean_length, 4), "max", 10)
            check(
                "gopher.alphabetic_word_share", round(alphabetic_share, 4), "min", 0.8
            )
            stopwords = {word.casefold().strip(".,!?;:\"'()") for word in words} & {
                "the",
                "be",
                "to",
                "of",
                "and",
                "that",
                "have",
                "with",
            }
            check("gopher.stopword_types", len(stopwords), "min", 2)
        if lines:
            bullet_share = sum(bool(re.match(r"[-*•]", line)) for line in lines) / len(
                lines
            )
            ellipsis_share = sum(line.endswith("...") for line in lines) / len(lines)
            check("gopher.bullet_line_share", round(bullet_share, 4), "max", 0.9)
            check("gopher.ellipsis_line_share", round(ellipsis_share, 4), "max", 0.3)
    if "c4" in presets:
        if "lorem ipsum" in text.casefold():
            issues.append(
                {
                    "rule": "c4.lorem_ipsum",
                    "value": True,
                    "operator": "exclude",
                    "threshold": False,
                }
            )
        if "{" in text:
            issues.append(
                {
                    "rule": "c4.curly_brace",
                    "value": True,
                    "operator": "exclude",
                    "threshold": False,
                }
            )
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
        check("c4.sentence_count", sentence_count, "min", 5)
    return issues
