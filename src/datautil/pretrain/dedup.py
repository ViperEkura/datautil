"""Disk-backed exact and MinHash candidate indexes for document text."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
import zlib
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from datautil.common.errors import DataError


@dataclass(frozen=True)
class Match:
    kept_id: str
    similarity: float
    kind: str


class DocumentDedupIndex:
    def __init__(
        self,
        connection: sqlite3.Connection,
        threshold: float = 0.8,
        shingle_size: int = 5,
        max_candidates: int = 10000,
        normalization: str = "nfkc",
    ) -> None:
        if not 0 < threshold <= 1 or shingle_size < 1 or max_candidates < 1:
            raise DataError(
                "invalid_dedup",
                "threshold must be in (0, 1], and shingle_size/max_candidates must be positive",
            )
        if normalization not in {"nfkc", "nfkc_casefold", "none"}:
            raise DataError(
                "invalid_dedup", "normalization must be nfkc, nfkc_casefold, or none"
            )
        self.connection = connection
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.max_candidates = max_candidates
        self.normalization = normalization
        if threshold < 1:
            try:
                layout = MinHashLSH(threshold=threshold, num_perm=128)
                self.bands, self.rows = layout.b, layout.r
            except ValueError as exc:
                raise DataError("invalid_dedup", str(exc)) from exc
        else:
            self.bands = self.rows = 0
        connection.execute(
            "CREATE TABLE dedup_entries (entry_id INTEGER PRIMARY KEY, scope TEXT NOT NULL, "
            "owner_id TEXT NOT NULL, digest BLOB NOT NULL, normalized_bytes INTEGER NOT NULL, "
            "token_digest BLOB, tokens BLOB)"
        )
        connection.execute("CREATE INDEX dedup_exact ON dedup_entries (scope, digest)")
        if threshold == 1:
            connection.execute(
                "CREATE INDEX dedup_tokens ON dedup_entries (scope, token_digest)"
            )
        else:
            connection.execute(
                "CREATE TABLE dedup_buckets (scope TEXT NOT NULL, band INTEGER NOT NULL, "
                "signature BLOB NOT NULL, entry_id INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX dedup_bucket_lookup ON dedup_buckets (scope, band, signature)"
            )
        self.candidate_count = 0
        self.exact_count = 0
        self.near_count = 0

    def normalize(self, text: str) -> str:
        if self.normalization == "none":
            return text
        value = unicodedata.normalize("NFKC", text)
        if self.normalization == "nfkc_casefold":
            value = value.casefold()
        return re.sub(r"\s+", " ", value).strip()

    def _tokens(self, text: str) -> set[str]:
        size = self.shingle_size
        return {
            text[index : index + size] for index in range(max(1, len(text) - size + 1))
        }

    def check_or_add(self, text: str, scope: str, owner_id: str) -> Match | None:
        normalized = self.normalize(text)
        normalized_bytes = normalized.encode("utf-8")
        digest = hashlib.sha256(normalized_bytes).digest()
        row = self.connection.execute(
            "SELECT owner_id FROM dedup_entries WHERE scope=? AND digest=? "
            "AND normalized_bytes=? LIMIT 1",
            (scope, digest, len(normalized_bytes)),
        ).fetchone()
        if row is not None:
            self.exact_count += 1
            return Match(row[0], 1.0, "exact")
        tokens = self._tokens(normalized)
        serialized = json.dumps(
            sorted(tokens), ensure_ascii=False, separators=(",", ":")
        )
        token_digest = hashlib.sha256(serialized.encode("utf-8")).digest()
        if self.threshold == 1:
            row = self.connection.execute(
                "SELECT owner_id FROM dedup_entries WHERE scope=? AND token_digest=? LIMIT 1",
                (scope, token_digest),
            ).fetchone()
            if row is not None:
                self.near_count += 1
                return Match(row[0], 1.0, "near")
            signature = None
        else:
            signature = MinHash(num_perm=128, seed=1)
            for token in sorted(tokens):
                signature.update(token.encode("utf-8"))
            candidates: set[int] = set()
            for band in range(self.bands):
                segment = signature.hashvalues[
                    band * self.rows : (band + 1) * self.rows
                ].tobytes()
                for (entry_id,) in self.connection.execute(
                    "SELECT entry_id FROM dedup_buckets WHERE scope=? AND band=? AND signature=?",
                    (scope, band, segment),
                ):
                    candidates.add(entry_id)
                    if len(candidates) > self.max_candidates:
                        raise DataError(
                            "dedup_candidate_limit",
                            f"More than {self.max_candidates} candidates for {owner_id!r}; increase max_candidates",
                        )
            self.candidate_count += len(candidates)
            for entry_id in sorted(candidates):
                row = self.connection.execute(
                    "SELECT owner_id, tokens FROM dedup_entries WHERE entry_id=?",
                    (entry_id,),
                ).fetchone()
                other = set(json.loads(zlib.decompress(row[1])))
                union = tokens | other
                similarity = len(tokens & other) / len(union) if union else 1.0
                if similarity >= self.threshold:
                    self.near_count += 1
                    return Match(row[0], round(similarity, 4), "near")
        cursor = self.connection.execute(
            "INSERT INTO dedup_entries (scope, owner_id, digest, normalized_bytes, "
            "token_digest, tokens) VALUES (?, ?, ?, ?, ?, ?)",
            (
                scope,
                owner_id,
                digest,
                len(normalized_bytes),
                token_digest if self.threshold == 1 else None,
                zlib.compress(serialized.encode("utf-8"), level=1)
                if self.threshold < 1 else None,
            ),
        )
        if signature is not None:
            for band in range(self.bands):
                segment = signature.hashvalues[
                    band * self.rows : (band + 1) * self.rows
                ].tobytes()
                self.connection.execute(
                    "INSERT INTO dedup_buckets VALUES (?, ?, ?, ?)",
                    (scope, band, segment, cursor.lastrowid),
                )
        return None
