"""Compact, resumable deduplication of an AstrAI pretraining Parquet manifest.

The output is a compact keep index, not a second copy of the corpus. Each
``keep/*.u64`` file contains little-endian row numbers in its source Parquet.
Progress and accepted-document hashes are committed together per row group.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TEXT_FIELDS = (
    "full_content",
    "content",
    "text",
    "document",
    "body",
    "article",
    "markdown",
    "source_text",
    "code",
)
FALLBACK_FIELDS = (
    "title",
    "problem",
    "question",
    "task",
    "instruction",
    "analysis",
    "solution",
    "answer",
    "response",
    "completion",
    "output",
    "test",
)
WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SCHEMA_VERSION = "2"


def extract(row):
    for key in TEXT_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = [
            message.get("content", message.get("text", ""))
            for message in messages
            if isinstance(message, dict)
        ]
        parts = [part for part in parts if isinstance(part, str) and part.strip()]
        if parts:
            return "\n\n".join(parts)
    parts = [row.get(key) for key in FALLBACK_FIELDS]
    return "\n\n".join(part for part in parts if isinstance(part, str) and part.strip())


def normalize(text, is_code):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if is_code:
        return "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n")
    return re.sub(r"\s+", " ", text).strip()


def near_group(category):
    # L3 is exact-only; natural web/code and math get domain-specific near dedup.
    if category.startswith("web2_"):
        return "web:" + category.split("_", 1)[1]
    if category.startswith("code2_"):
        return "code:" + category.split("_", 1)[1]
    if category in {"math1", "math2"}:
        return "math"
    return None


def simhash(text):
    words = WORD.findall(text.casefold())
    n = len(words) - 4
    if n <= 0:
        return None
    positions = sorted({int(i * (n - 1) / 63) for i in range(min(n, 64))})
    digests = b"".join(
        hashlib.blake2b(" ".join(words[pos : pos + 5]).encode(), digest_size=8).digest()
        for pos in positions
    )
    bits = np.unpackbits(
        np.frombuffer(digests, dtype=np.uint8).reshape(len(positions), 8),
        axis=1,
        bitorder="little",
    )
    majority = (bits.sum(axis=0) * 2 >= len(positions)).astype(np.uint8)
    return int.from_bytes(np.packbits(majority, bitorder="little").tobytes(), "little")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ordered_files(manifest):
    files = list(manifest["files"])
    return sorted(
        files,
        key=lambda item: (
            0 if item["category"].startswith(("web3_", "code3_", "math3_")) else 1,
            item["category"],
            item["output_file"],
        ),
    )


def open_db(path, identity):
    conn = sqlite3.connect(str(path), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA cache_size=-262144")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    existing = dict(conn.execute("SELECT key, value FROM meta"))
    if existing and existing != identity:
        raise RuntimeError(
            "Output database belongs to another manifest or dedup configuration"
        )
    if not existing:
        conn.executemany("INSERT INTO meta VALUES (?, ?)", identity.items())
        conn.execute(
            "CREATE TABLE exact_seen (digest BLOB PRIMARY KEY, file_id INTEGER NOT NULL, "
            "row_num INTEGER NOT NULL) WITHOUT ROWID"
        )
        conn.execute("CREATE INDEX exact_source ON exact_seen(file_id, row_num)")
        conn.execute(
            "CREATE TABLE near_seen (grp TEXT NOT NULL, sig BLOB NOT NULL, "
            "file_id INTEGER NOT NULL, row_num INTEGER NOT NULL, b0 INTEGER NOT NULL, "
            "b1 INTEGER NOT NULL, b2 INTEGER NOT NULL, b3 INTEGER NOT NULL)"
        )
        for band in range(4):
            conn.execute(f"CREATE INDEX near_band_{band} ON near_seen(grp, b{band})")
        conn.execute(
            "CREATE TABLE progress (file_id INTEGER NOT NULL, row_group INTEGER NOT NULL, "
            "rows INTEGER NOT NULL, kept INTEGER NOT NULL, exact INTEGER NOT NULL, "
            "near INTEGER NOT NULL, filtered INTEGER NOT NULL, "
            "PRIMARY KEY(file_id, row_group))"
        )
        conn.execute(
            "CREATE TABLE near_examples (category TEXT NOT NULL, file_id INTEGER NOT NULL, "
            "row_num INTEGER NOT NULL, kept_file_id INTEGER NOT NULL, "
            "kept_row_num INTEGER NOT NULL, distance INTEGER NOT NULL)"
        )
        conn.commit()
    return conn


def check_or_add(conn, text, group, file_id, row_num, category, candidate_cap):
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
    if conn.execute("SELECT 1 FROM exact_seen WHERE digest=?", (digest,)).fetchone():
        return "exact"
    signature = simhash(text) if group else None
    if signature is not None:
        bands = tuple((signature >> (16 * band)) & 65535 for band in range(4))
        seen_ids = set()
        for band, value in enumerate(bands):
            query = f"SELECT rowid, sig, file_id, row_num FROM near_seen WHERE grp=? AND b{band}=?"
            for seen_id, old_sig, kept_file, kept_row in conn.execute(
                query, (group, value)
            ):
                if seen_id in seen_ids:
                    continue
                seen_ids.add(seen_id)
                if len(seen_ids) > candidate_cap:
                    raise RuntimeError(
                        f"Near-match candidate cap {candidate_cap} reached at file {file_id}, row {row_num}"
                    )
                distance = (signature ^ int.from_bytes(old_sig, "little")).bit_count()
                if distance <= 3:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM near_examples WHERE category=?",
                        (category,),
                    ).fetchone()[0]
                    if count < 100:
                        conn.execute(
                            "INSERT INTO near_examples VALUES (?, ?, ?, ?, ?, ?)",
                            (category, file_id, row_num, kept_file, kept_row, distance),
                        )
                    return "near"
    conn.execute(
        "INSERT INTO exact_seen VALUES (?, ?, ?)",
        (digest, file_id, row_num),
    )
    if signature is not None:
        conn.execute(
            "INSERT INTO near_seen VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (group, signature.to_bytes(8, "little"), file_id, row_num, *bands),
        )
    return "kept"


def export_keep_index(conn, output, file_id, expected_rows):
    target = output / "keep" / f"{file_id:04d}.u64"
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_kept = conn.execute(
        "SELECT COALESCE(SUM(kept), 0) FROM progress WHERE file_id=?", (file_id,)
    ).fetchone()[0]
    if target.exists() and target.stat().st_size == expected_kept * 8:
        return target
    temporary = target.with_name(target.name + ".tmp")
    count = 0
    with temporary.open("wb") as stream:
        for (row_num,) in conn.execute(
            "SELECT row_num FROM exact_seen WHERE file_id=? ORDER BY row_num",
            (file_id,),
        ):
            if row_num >= expected_rows:
                raise RuntimeError(f"Bad row number {row_num} for file {file_id}")
            stream.write(struct.pack("<Q", row_num))
            count += 1
    if count != expected_kept:
        temporary.unlink()
        raise RuntimeError(
            f"Keep count mismatch for file {file_id}: {count} vs {expected_kept}"
        )
    temporary.replace(target)
    return target


def summary(conn, files, output, complete):
    by_file = {
        file_id: (groups, rows, kept, exact, near, filtered)
        for file_id, groups, rows, kept, exact, near, filtered in conn.execute(
            "SELECT file_id, COUNT(*), SUM(rows), SUM(kept), SUM(exact), SUM(near), "
            "SUM(filtered) FROM progress GROUP BY file_id"
        )
    }
    categories = {}
    file_entries = []
    for file_id, item in enumerate(files):
        groups, rows, kept, exact, near, filtered = by_file.get(
            file_id, (0, 0, 0, 0, 0, 0)
        )
        category = item["category"]
        stats = categories.setdefault(
            category, {"rows": 0, "kept": 0, "exact": 0, "near": 0, "filtered": 0}
        )
        for name, value in (
            ("rows", rows),
            ("kept", kept),
            ("exact", exact),
            ("near", near),
            ("filtered", filtered),
        ):
            stats[name] += value
        file_entries.append(
            {
                "file_id": file_id,
                "category": category,
                "source": item["output_file"],
                "source_bytes": item["bytes"],
                "completed_row_groups": groups,
                "rows": rows,
                "kept": kept,
                "keep_index": f"keep/{file_id:04d}.u64"
                if (output / "keep" / f"{file_id:04d}.u64").exists()
                else None,
            }
        )
    return {
        "status": "complete" if complete else "partial",
        "algorithm": "global blake2b-128 exact; 64-bit sampled 5-gram SimHash distance <=3 for L2 web/code and math1/math2",
        "index_format": "little-endian uint64 source row numbers, zero based",
        "categories": categories,
        "files": file_entries,
    }


def run(args):
    if args.profile != "astrai-100b":
        raise RuntimeError("Unsupported deduplication profile")
    manifest_path = args.manifest.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "downloaded":
        raise RuntimeError("Input manifest is not marked downloaded")
    files = ordered_files(manifest)
    raw_root = manifest_path.parent
    for item in files:
        source = (raw_root / item["output_file"]).resolve()
        if not source.is_relative_to(raw_root):
            raise RuntimeError(f"Source escapes manifest directory: {source}")
        if not source.is_file() or source.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Missing or size mismatch: {source}")
    if not (args.output / "dedup.sqlite").exists() and any(
        path.exists()
        for path in (args.output / "keep", args.output / "status.json", args.output / "manifest.json")
    ):
        raise RuntimeError("Output has dedup results but no matching dedup.sqlite")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": SCHEMA_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "candidate_cap": str(args.candidate_cap),
        "profile": args.profile,
    }
    conn = open_db(args.output / "dedup.sqlite", identity)
    processed_now = 0
    try:
        for file_id, item in enumerate(files):
            if args.max_files and file_id >= args.max_files:
                break
            source = raw_root / item["output_file"]
            parquet = pq.ParquetFile(source)
            completed = {
                row_group
                for (row_group,) in conn.execute(
                    "SELECT row_group FROM progress WHERE file_id=?", (file_id,)
                )
            }
            if len(completed) == parquet.num_row_groups:
                export_keep_index(conn, args.output, file_id, parquet.metadata.num_rows)
                continue
            columns = [
                name
                for name in parquet.schema_arrow.names
                if name in TEXT_FIELDS or name in FALLBACK_FIELDS or name == "messages"
            ]
            if not columns:
                raise RuntimeError(f"No recognized text columns in {source}")
            offset = 0
            for row_group in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(row_group).num_rows
                if row_group in completed:
                    offset += count
                    continue
                stats = Counter()
                conn.execute("BEGIN IMMEDIATE")
                try:
                    row_num = offset
                    for batch in parquet.iter_batches(
                        row_groups=[row_group], batch_size=256, columns=columns
                    ):
                        for row in batch.to_pylist():
                            stats["rows"] += 1
                            text = normalize(
                                extract(row), item["category"].startswith("code")
                            )
                            if (
                                len(text)
                                < (30 if item["category"].startswith("code") else 50)
                                or CTRL.search(text)
                                or "\ufffd" in text
                            ):
                                stats["filtered"] += 1
                            else:
                                result = check_or_add(
                                    conn,
                                    text,
                                    near_group(item["category"]),
                                    file_id,
                                    row_num,
                                    item["category"],
                                    args.candidate_cap,
                                )
                                stats[result] += 1
                            row_num += 1
                    if stats["rows"] != count:
                        raise RuntimeError(
                            f"Row count changed in {source}, group {row_group}"
                        )
                    conn.execute(
                        "INSERT INTO progress VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            file_id,
                            row_group,
                            count,
                            stats["kept"],
                            stats["exact"],
                            stats["near"],
                            stats["filtered"],
                        ),
                    )
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
                offset += count
                processed_now += 1
                print(
                    f"file={file_id + 1}/{len(files)} group={row_group + 1}/{parquet.num_row_groups} "
                    f"category={item['category']} rows={stats['rows']} kept={stats['kept']} "
                    f"exact={stats['exact']} near={stats['near']} filtered={stats['filtered']}",
                    flush=True,
                )
                if processed_now % 10 == 0:
                    atomic_json(
                        args.output / "status.json",
                        summary(conn, files, args.output, False),
                    )
                if args.stop_after_groups and processed_now >= args.stop_after_groups:
                    result = summary(conn, files, args.output, False)
                    atomic_json(args.output / "status.json", result)
                    return result
            export_keep_index(conn, args.output, file_id, parquet.metadata.num_rows)
        total_groups = sum(
            pq.ParquetFile(raw_root / item["output_file"]).num_row_groups
            for item in files
        )
        done_groups = conn.execute("SELECT COUNT(*) FROM progress").fetchone()[0]
        result = summary(conn, files, args.output, done_groups == total_groups)
        atomic_json(args.output / "status.json", result)
        if result["status"] == "complete":
            atomic_json(args.output / "manifest.json", result)
        return result
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["astrai-100b"], required=True)
    parser.add_argument("--candidate-cap", type=int, default=2000)
    parser.add_argument(
        "--max-files", type=int, default=0, help="pilot: process only first N files"
    )
    parser.add_argument(
        "--stop-after-groups",
        type=int,
        default=0,
        help="pilot: commit N groups, then stop",
    )
    args = parser.parse_args()
    if args.candidate_cap < 1 or args.max_files < 0 or args.stop_after_groups < 0:
        parser.error("numeric options must be positive (or zero for no limit)")
    run(args)


if __name__ == "__main__":
    main()
