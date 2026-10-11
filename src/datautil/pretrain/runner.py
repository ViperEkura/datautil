from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from datautil.common.errors import DataError
from datautil.common.io import output_writer
from datautil.pretrain.dedup import DocumentDedupIndex
from datautil.pretrain.model import DocumentRecord
from datautil.pretrain.quality import document_quality_issues
from datautil.pretrain.recipe import Source, load_recipe
from datautil.pretrain.sources import document_writer, iter_documents


def _quotas(target: int, weights: list[float], capacity: list[int]) -> list[int]:
    remaining = min(target, sum(capacity))
    result = [0] * len(capacity)
    while remaining:
        active = [i for i in range(len(capacity)) if result[i] < capacity[i]]
        scale = max(weights[i] for i in active)
        relative = {i: weights[i] / scale for i in active}
        total_weight = sum(relative.values())
        desired = {i: remaining * relative[i] / total_weight for i in active}
        increments = {
            i: min(capacity[i] - result[i], math.floor(desired[i])) for i in active
        }
        added = sum(increments.values())
        for i, value in increments.items():
            result[i] += value
        remaining -= added
        if not remaining:
            break
        for i in sorted(
            active,
            key=lambda index: (-(desired[index] - math.floor(desired[index])), index),
        ):
            if remaining and result[i] < capacity[i]:
                result[i] += 1
                remaining -= 1
        if not added and remaining and all(result[i] >= capacity[i] for i in active):
            break
    return result


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_recipe(path: str) -> dict[str, Any]:
    recipe = load_recipe(path)
    sources: list[Source] = recipe["sources"]
    source_stats: list[Counter[str]] = [Counter() for _ in sources]
    with (
        tempfile.TemporaryDirectory(prefix="datautil-pretrain-") as directory,
        ExitStack() as stack,
    ):
        connection = sqlite3.connect(str(Path(directory) / "corpus.sqlite3"))
        stack.callback(connection.close)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-4096")
        connection.execute("CREATE TABLE seen_ids (id TEXT PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE records (seq INTEGER PRIMARY KEY, source_idx INTEGER NOT NULL, id TEXT UNIQUE NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL, location TEXT NOT NULL, score BLOB NOT NULL)"
        )
        connection.execute("CREATE INDEX records_rank ON records (source_idx, score)")

        def make_index(
            settings: dict[str, Any] | None, name: str
        ) -> DocumentDedupIndex | None:
            if settings is None:
                return None
            index_connection = sqlite3.connect(str(Path(directory) / f"{name}.sqlite3"))
            stack.callback(index_connection.close)
            index_connection.execute("PRAGMA journal_mode=MEMORY")
            index_connection.execute("PRAGMA synchronous=OFF")
            index_connection.execute("PRAGMA cache_size=-4096")
            return DocumentDedupIndex(
                index_connection,
                settings["threshold"],
                settings["shingle_size"],
                settings["max_candidates"],
                settings["normalization"],
            )

        document_index = make_index(recipe["document_dedup"], "document")
        paragraph_indexer = make_index(recipe["paragraph_dedup"], "paragraph")
        dedup_indexes = tuple(
            index for index in (document_index, paragraph_indexer) if index is not None
        )

        def finish_dedup_record(keep: bool) -> None:
            for index in dedup_indexes:
                if not keep:
                    index.connection.execute("ROLLBACK TO SAVEPOINT dedup_record")
                index.connection.execute("RELEASE SAVEPOINT dedup_record")

        decision_writer = (
            stack.enter_context(output_writer(recipe["decisions"], "jsonl"))
            if recipe["decisions"]
            else None
        )

        def decide(value: dict[str, Any]) -> None:
            if decision_writer is not None:
                decision_writer.write(value)

        for source_index, source in enumerate(sources):
            stats = source_stats[source_index]
            for record in iter_documents(
                source.path, source.name, source.fields, source.format
            ):
                stats["read"] += 1
                try:
                    connection.execute(
                        "INSERT INTO seen_ids VALUES (?)", (record.record_id,)
                    )
                except sqlite3.IntegrityError as exc:
                    raise DataError(
                        "duplicate_record_id",
                        f"Duplicate document ID {record.record_id!r}",
                        record.source_location,
                    ) from exc
                if recipe["quality"].get("presets") and record.metadata.get(
                    "language"
                ) not in (None, "en"):
                    raise DataError(
                        "quality_language_mismatch",
                        "English quality preset received a non-English document",
                        record.source_location,
                    )
                issues = document_quality_issues(record, recipe["quality"])
                if issues:
                    stats["quality_rejected"] += 1
                    decide(
                        {
                            "record_id": record.record_id,
                            "source": source.name,
                            "action": "drop",
                            "reason": "quality",
                            "checks": issues,
                        }
                    )
                    continue
                for index in dedup_indexes:
                    index.connection.execute("SAVEPOINT dedup_record")
                if recipe["document_dedup"]:
                    match = document_index.check_or_add(
                        record.text, "document", record.record_id
                    )
                    if match:
                        finish_dedup_record(False)
                        stats["document_duplicate"] += 1
                        decide(
                            {
                                "record_id": record.record_id,
                                "source": source.name,
                                "action": "drop",
                                "reason": "document_duplicate",
                                "kept_id": match.kept_id,
                                "match": match.kind,
                                "similarity": match.similarity,
                            }
                        )
                        continue
                text = record.text
                if recipe["paragraph_dedup"]:
                    parts = re.split(r"\n\s*\n", text)
                    kept: list[str] = []
                    for paragraph_index, paragraph in enumerate(parts):
                        if (
                            len(paragraph.strip())
                            < recipe["paragraph_dedup"]["min_chars"]
                        ):
                            kept.append(paragraph)
                            continue
                        match = paragraph_indexer.check_or_add(
                            paragraph,
                            "paragraph",
                            f"{record.record_id}#p{paragraph_index}",
                        )
                        if match:
                            stats["paragraphs_removed"] += 1
                            decide(
                                {
                                    "record_id": record.record_id,
                                    "source": source.name,
                                    "action": "remove_paragraph",
                                    "paragraph_index": paragraph_index,
                                    "kept_id": match.kept_id,
                                    "match": match.kind,
                                    "similarity": match.similarity,
                                }
                            )
                        else:
                            kept.append(paragraph)
                    if len(kept) != len(parts):
                        text = "\n\n".join(kept)
                    if not text.strip():
                        finish_dedup_record(False)
                        stats["empty_after_dedup"] += 1
                        decide(
                            {
                                "record_id": record.record_id,
                                "source": source.name,
                                "action": "drop",
                                "reason": "empty_after_dedup",
                            }
                        )
                        continue
                    if text != record.text:
                        post_dedup_issues = document_quality_issues(
                            replace(record, text=text), recipe["quality"]
                        )
                        if post_dedup_issues:
                            finish_dedup_record(False)
                            stats["quality_rejected_after_dedup"] += 1
                            decide(
                                {
                                    "record_id": record.record_id,
                                    "source": source.name,
                                    "action": "drop",
                                    "reason": "quality_after_paragraph_dedup",
                                    "checks": post_dedup_issues,
                                }
                            )
                            continue
                metadata = dict(record.metadata)
                metadata.setdefault("source", source.name)
                try:
                    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                except (TypeError, ValueError) as exc:
                    raise DataError(
                        "invalid_metadata", str(exc), record.source_location
                    ) from exc
                score = hashlib.blake2b(
                    f"{recipe['seed']}:{source.name}:{record.record_id}".encode(
                        "utf-8"
                    ),
                    digest_size=16,
                ).digest()
                connection.execute(
                    "INSERT INTO records (source_idx, id, text, metadata, location, score) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        source_index,
                        record.record_id,
                        text,
                        encoded,
                        record.source_location,
                        score,
                    ),
                )
                finish_dedup_record(True)
                stats["available"] += 1
        capacity = [stats["available"] for stats in source_stats]
        quota = _quotas(
            recipe["count"], [source.weight for source in sources], capacity
        )
        connection.execute("CREATE TABLE selected (seq INTEGER PRIMARY KEY)")
        for source_index, amount in enumerate(quota):
            connection.execute(
                "INSERT INTO selected SELECT seq FROM records WHERE source_idx=? ORDER BY score, seq LIMIT ?",
                (source_index, amount),
            )
            source_stats[source_index]["selected"] = amount
        with document_writer(recipe["output"]) as append:
            for record_id, text, metadata, location in connection.execute(
                "SELECT id, text, metadata, location FROM records JOIN selected USING(seq) ORDER BY score, seq"
            ):
                append(DocumentRecord(record_id, text, location, json.loads(metadata)))
        count = sum(quota)
        report = {
            "config_sha256": recipe["config_sha256"],
            "output": recipe["output"],
            "requested_count": recipe["count"],
            "written_count": count,
            "shortfall": recipe["count"] - count,
            "seed": recipe["seed"],
            "mix_unit": "documents",
            "dedup": {
                "document": recipe["document_dedup"],
                "paragraph": recipe["paragraph_dedup"],
                "candidate_count": sum(
                    index.candidate_count
                    for index in (document_index, paragraph_indexer)
                    if index
                ),
                "exact_count": sum(
                    index.exact_count
                    for index in (document_index, paragraph_indexer)
                    if index
                ),
                "near_count": sum(
                    index.near_count
                    for index in (document_index, paragraph_indexer)
                    if index
                ),
            },
            "sources": [
                {
                    "name": source.name,
                    "input": source.path,
                    "input_sha256": _sha256(source.path),
                    "weight": source.weight,
                    **source_stats[index],
                    "realized_share": quota[index] / count if count else 0.0,
                }
                for index, source in enumerate(sources)
            ],
        }
        if recipe["report"]:
            with output_writer(recipe["report"], "jsonl") as writer:
                writer.write(report)
        return report
