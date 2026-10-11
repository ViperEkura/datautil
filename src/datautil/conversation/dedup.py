from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

from datautil.common.errors import DataError
from datautil.common.index import DiskIndex
from datautil.common.io import iter_raw, output_writer
from datautil.conversation.formats import ADAPTERS
from datautil.conversation.model import DatasetRecord
from datautil.conversation.reader import read_records, replayable_input, source_format
from datautil.conversation.text import _text


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _shingles(normalized: str, width: int = 5) -> set[str]:
    return {
        normalized[index : index + width]
        for index in range(max(1, len(normalized) - width + 1))
    }


def dedup_scan(
    records: Iterable[DatasetRecord],
    threshold: float = 0.8,
    *,
    scope: str = "conversation",
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from datasketch import MinHash, MinHashLSH

    if not 0 < threshold <= 1:
        raise DataError("invalid_argument", "threshold must be between 0 and 1")
    if scope not in {"conversation", "user"}:
        raise DataError("invalid_argument", "scope must be conversation or user")
    unscannable = 0
    lsh = None
    if threshold < 1:
        try:
            lsh = MinHashLSH(threshold=threshold, num_perm=128)
        except ValueError as exc:
            if "bands are too small" not in str(exc):
                raise DataError("invalid_argument", str(exc)) from exc
            lsh = MinHashLSH(threshold=threshold, num_perm=128, params=(2, 64))
    record_ids: list[str] = []
    parent: list[int] = []
    matches: dict[tuple[int, int], tuple[float, str]] = {}
    candidate_count = 0

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    with tempfile.TemporaryDirectory(prefix="datautil-dedup-") as directory:
        connection = sqlite3.connect(str(Path(directory) / "candidates.sqlite3"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute(
                "CREATE TABLE representatives (position INTEGER PRIMARY KEY, digest BLOB NOT NULL, normalized TEXT NOT NULL, tokens TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX representatives_digest ON representatives (digest)"
            )
            if lsh is None:
                connection.execute(
                    "CREATE INDEX representatives_tokens ON representatives (tokens)"
                )

            for index, record in enumerate(records):
                record_ids.append(record.record_id)
                parent.append(index)
                raw_text = _text(record, scope)
                if raw_text is None:
                    unscannable += 1
                    continue
                value = _normalize(raw_text)
                digest = hashlib.sha256(value.encode("utf-8")).digest()
                found = connection.execute(
                    "SELECT position FROM representatives WHERE digest=? AND normalized=? LIMIT 1",
                    (digest, value),
                ).fetchone()
                if found is not None:
                    representative = found[0]
                    parent[root(index)] = root(representative)
                    matches[(representative, index)] = (1.0, "exact")
                    continue
                tokens = _shingles(value)
                token_key = json.dumps(
                    sorted(tokens), ensure_ascii=False, separators=(",", ":")
                )
                if lsh is None:
                    found = connection.execute(
                        "SELECT position FROM representatives WHERE tokens=? LIMIT 1",
                        (token_key,),
                    ).fetchone()
                    if found is not None:
                        other = found[0]
                        parent[root(index)] = root(other)
                        matches[(other, index)] = (1.0, "near")
                else:
                    signature = MinHash(num_perm=128, seed=1)
                    for token in sorted(tokens):
                        signature.update(token.encode("utf-8"))
                    candidates = sorted(lsh.query(signature))
                    candidate_count += len(candidates)
                    for other in candidates:
                        candidate_tokens = set(
                            json.loads(
                                connection.execute(
                                    "SELECT tokens FROM representatives WHERE position=?",
                                    (other,),
                                ).fetchone()[0]
                            )
                        )
                        union = tokens | candidate_tokens
                        score = (
                            len(tokens & candidate_tokens) / len(union)
                            if union
                            else 1.0
                        )
                        if score >= threshold:
                            parent[root(index)] = root(other)
                            matches[(other, index)] = (round(score, 4), "near")
                    lsh.insert(index, signature)
                connection.execute(
                    "INSERT INTO representatives VALUES (?, ?, ?, ?)",
                    (index, digest, value, token_key),
                )
        finally:
            connection.close()
    groups: dict[int, list[int]] = defaultdict(list)
    matched_indexes = {index for pair in matches for index in pair}
    for index in sorted(matched_indexes):
        groups[root(index)].append(index)
    matched_pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (left, right), (score, match) in sorted(matches.items()):
        matched_pairs[root(left)].append(
            {
                "a": record_ids[left],
                "b": record_ids[right],
                "similarity": score,
                "match": match,
            }
        )
    result = []
    for members in sorted(groups.values(), key=lambda items: items[0]):
        if len(members) > 1:
            result.append(
                {
                    "group_id": f"g{len(result) + 1}",
                    "record_ids": [record_ids[i] for i in members],
                    "pairs": matched_pairs[root(members[0])],
                }
            )
    if stats is not None:
        algorithm = (
            "exact-sha256+datasketch-minhash-lsh+jaccard"
            if lsh
            else "exact-sha256+exact-shingle-jaccard"
        )
        stats.update(
            {
                "record_count": len(record_ids),
                "scope": scope,
                "algorithm": algorithm,
                "datasketch_version": version("datasketch"),
                "shingle_size": 5,
                "num_perm": 128,
                "lsh_bands": lsh.b if lsh else 0,
                "lsh_rows": lsh.r if lsh else 0,
                "normalization": "NFKC+casefold+whitespace",
                "unscannable_count": unscannable,
                "lsh_candidate_count": candidate_count,
                "exact_pair_count": sum(
                    match == "exact" for _, match in matches.values()
                ),
                "near_pair_count": sum(
                    match == "near" for _, match in matches.values()
                ),
            }
        )
    return result


def load_groups(path: str) -> list[dict[str, Any]]:
    return [item for _, item in iter_raw(path)]


def dedup_apply(
    records: list[DatasetRecord], groups: list[dict[str, Any]]
) -> tuple[list[DatasetRecord], list[dict[str, str]]]:
    order = {record.record_id: index for index, record in enumerate(records)}
    remove: set[str] = set()
    decisions = []
    for group in groups:
        ids = group.get("record_ids", [])
        if len(ids) < 2 or any(item not in order for item in ids):
            raise DataError(
                "invalid_groups", "Group references unknown or too few records"
            )
        keep = min(ids, key=lambda item: order[item])
        for item in ids:
            if item != keep:
                remove.add(item)
                decisions.append({"removed": item, "kept": keep})
    return [record for record in records if record.record_id not in remove], decisions


def dedup_apply_file(
    path: str, groups_path: str, output: str, requested: str | None
) -> dict[str, object]:
    with (
        replayable_input(path) as replay,
        tempfile.TemporaryDirectory(prefix="datautil-apply-") as directory,
    ):
        source = source_format(replay, requested)
        index = DiskIndex(directory)
        try:
            index.conn.execute(
                "CREATE TABLE positions (position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL)"
            )
            index.conn.execute("CREATE TABLE removed (id TEXT PRIMARY KEY)")
            index.conn.execute(
                "CREATE TABLE decisions (position INTEGER PRIMARY KEY, removed TEXT, kept TEXT)"
            )
            total = 0
            for total, record in enumerate(
                read_records(replay, source, id_index=index, location_alias=path), 1
            ):
                index.conn.execute(
                    "INSERT INTO positions VALUES (?, ?)", (total, record.record_id)
                )
            decision_count = 0
            for location, group in iter_raw(groups_path):
                if not isinstance(group, dict) or not isinstance(
                    group.get("record_ids"), list
                ):
                    raise DataError(
                        "invalid_groups", "Group needs a record_ids array", location
                    )
                ids = group["record_ids"]
                if len(ids) < 2 or any(not isinstance(item, str) for item in ids):
                    raise DataError(
                        "invalid_groups",
                        "Group references unknown or too few records",
                        location,
                    )
                positions = [
                    index.conn.execute(
                        "SELECT position FROM positions WHERE id=?", (item,)
                    ).fetchone()
                    for item in ids
                ]
                if any(row is None for row in positions):
                    raise DataError(
                        "invalid_groups",
                        "Group references unknown or too few records",
                        location,
                    )
                kept = ids[min(range(len(ids)), key=lambda item: positions[item][0])]
                for item in ids:
                    if item != kept:
                        index.conn.execute(
                            "INSERT OR IGNORE INTO removed VALUES (?)", (item,)
                        )
                        decision_count += 1
                        if decision_count <= 1000:
                            index.conn.execute(
                                "INSERT INTO decisions VALUES (?, ?, ?)",
                                (decision_count, item, kept),
                            )
            written = 0
            with output_writer(
                output, getattr(ADAPTERS[source], "container", "auto")
            ) as writer:
                position = 0
                for position, record in enumerate(
                    read_records(
                        replay, source, check_duplicate_ids=False, location_alias=path
                    ),
                    1,
                ):
                    row = index.conn.execute(
                        "SELECT id FROM positions WHERE position=?", (position,)
                    ).fetchone()
                    if row is None or row[0] != record.record_id:
                        raise DataError(
                            "input_changed",
                            "Input changed between selection and output passes",
                        )
                    if (
                        index.conn.execute(
                            "SELECT 1 FROM removed WHERE id=?", (record.record_id,)
                        ).fetchone()
                        is None
                    ):
                        writer.write(ADAPTERS[source].serialize(record))
                        written += 1
                if position != total:
                    raise DataError(
                        "input_changed",
                        "Input changed between selection and output passes",
                    )
            decisions = [
                {"removed": item, "kept": kept}
                for item, kept in index.conn.execute(
                    "SELECT removed, kept FROM decisions ORDER BY position"
                )
            ]
            return {
                "record_count": total,
                "written_count": written,
                "removed_count": total - written,
                "decision_count": decision_count,
                "decisions": decisions,
                "decisions_truncated": decision_count > len(decisions),
            }
        finally:
            index.close()
