"""Parallel, resumable exact and content-verified near deduplication."""

import argparse
import hashlib
import heapq
import json
import os
import struct
import tempfile
import time
import zlib
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import lru_cache
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from datautil.pretrain.manifest_index import (
    CTRL,
    FALLBACK_FIELDS,
    SCHEMA_VERSION,
    TEXT_FIELDS,
    atomic_json,
    content_features,
    extract,
    near_group,
    normalize,
    ordered_files,
    processing_order,
    verify_pair,
)

EXACT = np.dtype([("h0", "<u8"), ("h1", "<u8"), ("doc", "<u8")])
SIGNATURE = np.dtype([("doc", "<u8"), ("signature", "<u4", (128,))])
BUCKET = np.dtype([("domain", "<u2"), ("band", "<u2"), ("key", "<u8"), ("doc", "<u8")])
PAIR = np.dtype([("lose", "<u8"), ("keep", "<u8")])
EDGE = np.dtype([("lose", "<u8"), ("keep", "<u8"), ("score", "<f4")])
U64 = np.dtype("<u8")
MASK32 = (1 << 32) - 1
STAGES = (
    "prepare",
    "exact",
    "exact_keep",
    "signatures",
    "buckets",
    "candidates",
    "verify",
)


@lru_cache(maxsize=4)
def plan_at(output):
    plan = json.loads((Path(output) / "plan.json").read_text(encoding="utf-8"))
    if plan["pipeline_code_sha256"] != code_identity():
        raise RuntimeError(
            "Dedup code changed after this output was created; use a fresh output"
        )
    return plan


def code_identity():
    from datautil.pretrain import manifest_index

    return hashlib.sha256(
        Path(__file__).read_bytes() + Path(manifest_index.__file__).read_bytes()
    ).hexdigest()


def job_path(output, stage, job):
    return Path(output) / stage / f"{job:04d}"


def completed(output, stage, job):
    path = job_path(output, stage, job)
    marker = path / "complete.json"
    if not marker.exists():
        return False
    metadata = json.loads(marker.read_text())
    for name, size in metadata["outputs"].items():
        child = path / name
        if not child.is_file() or child.stat().st_size != size:
            raise RuntimeError(f"Incomplete committed stage output: {child}")
    return True


def begin_job(output, stage, job):
    parent = Path(output) / stage
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".tmp-{job:04d}-", dir=parent))


def finish_job(output, stage, job, temporary, stats):
    outputs = {}
    for path in temporary.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            outputs[str(path.relative_to(temporary))] = path.stat().st_size
    atomic_json(temporary / "complete.json", {"outputs": outputs, "stats": stats})
    target = job_path(output, stage, job)
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite committed stage output: {target}")
    temporary.replace(target)
    return stats


def read_array(path, dtype):
    if not path.exists():
        return np.empty(0, dtype=dtype)
    if path.stat().st_size % dtype.itemsize:
        raise RuntimeError(f"Truncated binary stage output: {path}")
    return np.fromfile(path, dtype=dtype)


def write_array(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        values.tofile(stream)


def append_parts(path, values, parts):
    if not len(values):
        return
    order = np.argsort(parts, kind="stable")
    parts, values = parts[order], values[order]
    boundaries = np.r_[0, np.flatnonzero(parts[1:] != parts[:-1]) + 1, len(parts)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        with (path / f"part-{int(parts[start]):04d}.bin").open("ab") as stream:
            values[start:end].tofile(stream)


def columns_for(parquet):
    columns = [
        name
        for name in parquet.schema_arrow.names
        if name in TEXT_FIELDS or name in FALLBACK_FIELDS or name == "messages"
    ]
    if not columns:
        raise RuntimeError("Parquet has no recognized training text fields")
    return columns


def validate_source(plan, file_id):
    item = plan["files"][file_id]
    source = Path(plan["raw_root"]) / item["output_file"]
    metadata = source.stat()
    if metadata.st_size != item["bytes"] or metadata.st_mtime_ns != item["mtime_ns"]:
        raise RuntimeError(f"Input changed during deduplication: {source}")
    return source


def prepare_file(task):
    output, file_id = task
    plan = plan_at(output)
    item = plan["files"][file_id]
    parquet = pq.ParquetFile(validate_source(plan, file_id))
    temporary = begin_job(output, "prepare", file_id)
    stats = Counter(rows=0, valid=0, filtered=0)
    is_code = item["category"].startswith("code")
    for batch in parquet.iter_batches(
        batch_size=2048, columns=columns_for(parquet), use_threads=False
    ):
        records = []
        for row in batch.to_pylist():
            row_num = stats["rows"]
            stats["rows"] += 1
            text = normalize(extract(row), is_code)
            if (
                len(text) < (30 if is_code else 50)
                or CTRL.search(text)
                or "\ufffd" in text
            ):
                stats["filtered"] += 1
                continue
            h0, h1 = struct.unpack(
                "<QQ", hashlib.blake2b(text.encode(), digest_size=16).digest()
            )
            records.append((h0, h1, (file_id << 32) | row_num))
        values = np.array(records, dtype=EXACT)
        stats["valid"] += len(values)
        append_parts(temporary, values, values["h0"] % plan["shards"])
    if stats["rows"] != item["rows"]:
        raise RuntimeError(f"Parquet row count changed: file {file_id}")
    validate_source(plan, file_id)
    return finish_job(output, "prepare", file_id, temporary, dict(stats))


def reduce_exact(task):
    output, shard = task
    plan = plan_at(output)
    arrays = [
        read_array(
            job_path(output, "prepare", file_id) / f"part-{shard:04d}.bin", EXACT
        )
        for file_id in plan["selected"]
    ]
    records = np.concatenate(arrays) if arrays else np.empty(0, dtype=EXACT)
    del arrays
    temporary = begin_job(output, "exact", shard)
    stats = {"rows": len(records), "kept": 0, "exact": 0}
    if len(records):
        ranks = np.array(plan["ranks"], dtype=np.uint32)
        order = np.lexsort(
            (records["doc"], ranks[records["doc"] >> 32], records["h1"], records["h0"])
        )
        records = records[order]
        first = np.r_[
            True,
            (records["h0"][1:] != records["h0"][:-1])
            | (records["h1"][1:] != records["h1"][:-1]),
        ]
        starts = np.maximum.accumulate(np.where(first, np.arange(len(records)), 0))
        duplicates = np.empty(int((~first).sum()), dtype=PAIR)
        duplicates["lose"] = records["doc"][~first]
        duplicates["keep"] = records["doc"][starts[~first]]
        write_array(temporary / "duplicates.bin", duplicates)
        docs = np.sort(records["doc"][first])
        ids = docs >> 32
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(docs)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            write_array(
                temporary / "keep" / f"{int(ids[start]):04d}.u64",
                (docs[start:end] & MASK32).astype(U64),
            )
        stats.update(kept=len(docs), exact=len(duplicates))
    return finish_job(output, "exact", shard, temporary, stats)


def gather_exact_keep(task):
    output, file_id = task
    plan = plan_at(output)
    arrays = [
        read_array(
            job_path(output, "exact", shard) / "keep" / f"{file_id:04d}.u64", U64
        )
        for shard in range(plan["shards"])
    ]
    rows = np.sort(np.concatenate(arrays))
    if len(rows) and (
        rows[-1] >= plan["files"][file_id]["rows"] or np.any(rows[1:] == rows[:-1])
    ):
        raise RuntimeError(f"Invalid exact keep rows for file {file_id}")
    temporary = begin_job(output, "exact_keep", file_id)
    write_array(temporary / "rows.u64", rows)
    return finish_job(output, "exact_keep", file_id, temporary, {"kept": len(rows)})


def signature_for(features, sketch):
    sketch.clear()
    values = list(features)
    for start in range(0, len(values), 1024):
        sketch.update_batch(value.encode() for value in values[start : start + 1024])
    return sketch.hashvalues.astype("<u4")


def signature_file(task):
    from datasketch import MinHash

    output, file_id = task
    plan = plan_at(output)
    item = plan["files"][file_id]
    temporary = begin_job(output, "signatures", file_id)
    stats = Counter(rows=0, skipped_short=0)
    if near_group(item["category"]) is None:
        return finish_job(output, "signatures", file_id, temporary, dict(stats))
    accepted = read_array(job_path(output, "exact_keep", file_id) / "rows.u64", U64)
    parquet = pq.ParquetFile(validate_source(plan, file_id))
    sketch = MinHash(num_perm=128, seed=42, hashfunc=zlib.crc32, scheme="legacy")
    offset = 0
    with (temporary / "signatures.bin").open("wb") as stream:
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            selected = accepted[
                np.searchsorted(accepted, offset) : np.searchsorted(
                    accepted, offset + count
                )
            ]
            if not len(selected):
                offset += count
                continue
            table = parquet.read_row_group(
                group, columns=columns_for(parquet), use_threads=False
            )
            for start in range(0, len(selected), 512):
                row_nums = selected[start : start + 512]
                rows = table.take(
                    pa.array((row_nums - offset).astype(np.int64))
                ).to_pylist()
                records = np.empty(len(rows), dtype=SIGNATURE)
                used = 0
                for row_num, row in zip(row_nums, rows):
                    text = normalize(extract(row), item["category"].startswith("code"))
                    features = content_features(text, item["category"])
                    if len(features) < 20:
                        stats["skipped_short"] += 1
                        continue
                    records[used]["doc"] = (file_id << 32) | int(row_num)
                    records[used]["signature"] = signature_for(features, sketch)
                    used += 1
                records[:used].tofile(stream)
                stats["rows"] += used
            del table
            offset += count
    validate_source(plan, file_id)
    return finish_job(output, "signatures", file_id, temporary, dict(stats))


def band_keys(signatures, band, domain):
    keys = np.full(
        len(signatures), np.uint64(14695981039346656037 ^ (domain << 16) ^ band)
    )
    with np.errstate(over="ignore"):
        for column in range(band * 8, (band + 1) * 8):
            keys ^= signatures[:, column].astype(np.uint64)
            keys *= np.uint64(1099511628211)
        keys ^= keys >> 33
        keys *= np.uint64(0xFF51AFD7ED558CCD)
        keys ^= keys >> 33
    return keys


def bucket_file(task):
    output, file_id = task
    plan = plan_at(output)
    temporary = begin_job(output, "buckets", file_id)
    source = job_path(output, "signatures", file_id) / "signatures.bin"
    stats = {"rows": 0, "bucket_records": 0}
    if not source.exists() or not source.stat().st_size:
        return finish_job(output, "buckets", file_id, temporary, stats)
    values = np.memmap(source, dtype=SIGNATURE, mode="r")
    domain = plan["domains"][near_group(plan["files"][file_id]["category"])]
    for start in range(0, len(values), 65536):
        block = values[start : start + 65536]
        for band in range(16):
            keys = band_keys(block["signature"], band, domain)
            records = np.empty(len(block), dtype=BUCKET)
            records["domain"], records["band"], records["key"], records["doc"] = (
                domain,
                band,
                keys,
                block["doc"],
            )
            append_parts(temporary, records, keys % plan["shards"])
    stats.update(rows=len(values), bucket_records=len(values) * 16)
    return finish_job(output, "buckets", file_id, temporary, stats)


def reduce_buckets(task):
    output, shard = task
    plan = plan_at(output)
    arrays = [
        read_array(
            job_path(output, "buckets", file_id) / f"part-{shard:04d}.bin", BUCKET
        )
        for file_id in plan["selected"]
    ]
    records = np.concatenate(arrays)
    del arrays
    temporary = begin_job(output, "candidates", shard)
    stats = Counter(
        bucket_records=len(records), candidate_pairs=0, large_buckets=0, dense_buckets=0
    )
    if not len(records):
        return finish_job(output, "candidates", shard, temporary, dict(stats))
    order = np.lexsort(
        (records["doc"], records["key"], records["band"], records["domain"])
    )
    records = records[order]
    same = (
        (records["domain"][1:] == records["domain"][:-1])
        & (records["band"][1:] == records["band"][:-1])
        & (records["key"][1:] == records["key"][:-1])
    )
    bounds = np.r_[0, np.flatnonzero(~same) + 1, len(records)]
    ranks = np.array(plan["ranks"], dtype=np.uint32)

    def emit(keep, lose):
        pairs = np.empty(len(lose), dtype=PAIR)
        pairs["keep"], pairs["lose"] = keep, lose
        # Split large source files across workers, keeping nearby rows together.
        parts = ((lose & MASK32) // 1024 + (lose >> 32) * 0x9E37) % plan["shards"]
        append_parts(temporary, pairs, parts)
        stats["candidate_pairs"] += len(pairs)

    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start < 2:
            continue
        docs = np.unique(records["doc"][start:end])
        docs = docs[np.lexsort((docs, ranks[docs >> 32]))]
        size = len(docs)
        if size <= 32:
            a, b = np.triu_indices(size, k=1)
            emit(docs[a], docs[b])
        else:
            # Large buckets use several ranked anchors; unmatched members remain.
            # This bounds quadratic work and never authorizes deletion by itself.
            stats["large_buckets"] += 1
            stats["dense_buckets"] += size > plan["candidate_cap"]
            for anchor in range(min(8, size - 1)):
                for block in range(anchor + 1, size, 262144):
                    lose = docs[block : block + 262144]
                    emit(np.full(len(lose), docs[anchor], dtype=U64), lose)
    return finish_job(output, "candidates", shard, temporary, dict(stats))


class TextReader:
    def __init__(self, plan):
        self.plan = plan
        self.parquets = {}
        self.tables = OrderedDict()
        self.cached_bytes = 0
        self.documents = OrderedDict()
        self.document_bytes = 0
        self.signatures = {}

    def signature(self, doc):
        doc = int(doc)
        file_id = doc >> 32
        if file_id not in self.signatures:
            path = (
                job_path(self.plan["output"], "signatures", file_id) / "signatures.bin"
            )
            self.signatures[file_id] = np.memmap(path, dtype=SIGNATURE, mode="r")
        values = self.signatures[file_id]
        position = int(np.searchsorted(values["doc"], doc))
        if position >= len(values) or int(values[position]["doc"]) != doc:
            raise RuntimeError(f"Candidate has no committed signature: {doc}")
        return values[position]["signature"]

    def text(self, doc):
        doc = int(doc)
        if doc in self.documents:
            self.documents.move_to_end(doc)
            return self.documents[doc]
        file_id, row = doc >> 32, doc & MASK32
        item = self.plan["files"][file_id]
        if file_id not in self.parquets:
            parquet = pq.ParquetFile(validate_source(self.plan, file_id))
            offsets = np.r_[
                0,
                np.cumsum(
                    [
                        parquet.metadata.row_group(i).num_rows
                        for i in range(parquet.num_row_groups)
                    ]
                ),
            ]
            self.parquets[file_id] = parquet, offsets
        parquet, offsets = self.parquets[file_id]
        if row >= offsets[-1]:
            raise RuntimeError(f"Candidate row outside input: {doc}")
        group = int(np.searchsorted(offsets, row, side="right") - 1)
        key = file_id, group
        if key not in self.tables:
            table = parquet.read_row_group(
                group, columns=columns_for(parquet), use_threads=False
            )
            self.tables[key] = table
            self.cached_bytes += table.nbytes
            while len(self.tables) > 1 and (
                len(self.tables) > 3 or self.cached_bytes > 384 * 1024**2
            ):
                _, old = self.tables.popitem(last=False)
                self.cached_bytes -= old.nbytes
        self.tables.move_to_end(key)
        raw = self.tables[key].slice(row - int(offsets[group]), 1).to_pylist()[0]
        text = normalize(extract(raw), item["category"].startswith("code"))
        self.documents[doc] = text
        self.document_bytes += len(text) * 4
        while len(self.documents) > 1 and (
            len(self.documents) > 8192 or self.document_bytes > 64 * 1024**2
        ):
            _, old_text = self.documents.popitem(last=False)
            self.document_bytes -= len(old_text) * 4
        return text


def sample_add(heaps, category, record):
    key = int.from_bytes(
        hashlib.blake2b(
            struct.pack("<QQ", record["lose"], record["keep"]),
            digest_size=8,
            person=b"dedup-audit-v4",
        ).digest(),
        "little",
    )
    record["sample_key"] = key
    entry = (-key, record["lose"], record["keep"], record)
    heap = heaps[category]
    if len(heap) < 100:
        heapq.heappush(heap, entry)
    elif entry[:3] > heap[0][:3]:
        heapq.heapreplace(heap, entry)


def verify_shard(task):
    output, shard = task
    plan = plan_at(output)
    arrays = [
        read_array(
            job_path(output, "candidates", bucket) / f"part-{shard:04d}.bin", PAIR
        )
        for bucket in range(plan["shards"])
    ]
    pairs = np.unique(np.concatenate(arrays))
    del arrays
    temporary = begin_job(output, "verify", shard)
    (temporary / "edges").mkdir()
    reader = TextReader({**plan, "output": output})
    stats = Counter(candidates=len(pairs), verified=0)
    heaps = defaultdict(list)
    buffers = defaultdict(list)

    def flush(file_id):
        values = np.array(buffers[file_id], dtype=EDGE)
        with (temporary / "edges" / f"{file_id:04d}.bin").open("ab") as stream:
            values.tofile(stream)
        buffers[file_id].clear()

    for pair in pairs:
        lose, keep = int(pair["lose"]), int(pair["keep"])
        file_id, kept_file = lose >> 32, keep >> 32
        category = plan["files"][file_id]["category"]
        if near_group(category) != near_group(plan["files"][kept_file]["category"]):
            raise RuntimeError("Candidate crossed domain/language boundary")
        if (plan["ranks"][kept_file], keep & MASK32) >= (
            plan["ranks"][file_id],
            lose & MASK32,
        ):
            raise RuntimeError("Candidate crossed representative priority order")
        agreement = (
            int(np.count_nonzero(reader.signature(lose) == reader.signature(keep)))
            / 128
        )
        if agreement < 0.5:
            matched, score, reason = False, 0.0, "minhash_prefilter"
            left = right = ""
        else:
            left, right = reader.text(lose), reader.text(keep)
            matched, score, reason = verify_pair(left, right, category)
        stats[reason] += 1
        if matched:
            buffers[file_id].append((lose, keep, score))
            if len(buffers[file_id]) >= 8192:
                flush(file_id)
        sample_key = int.from_bytes(
            hashlib.blake2b(
                struct.pack("<QQ", lose, keep), digest_size=8, person=b"dedup-audit-v4"
            ).digest(),
            "little",
        )
        if len(heaps[category]) < 100 or -sample_key > heaps[category][0][0]:
            if reason == "minhash_prefilter":
                left, right = reader.text(lose), reader.text(keep)
            sample_add(
                heaps,
                category,
                {
                    "lose": lose,
                    "keep": keep,
                    "score": score,
                    "estimated_jaccard": agreement,
                    "reason": reason,
                    "matched": matched,
                    "source_excerpt": left[:400],
                    "kept_excerpt": right[:400],
                },
            )
    for file_id, buffer in buffers.items():
        if buffer:
            flush(file_id)
    atomic_json(
        temporary / "samples.json",
        {category: [entry[3] for entry in heap] for category, heap in heaps.items()},
    )
    return finish_job(output, "verify", shard, temporary, dict(stats))


def resolve_edges(keep_flags, offsets, edges, ranks):
    """Delete only when the directly verified representative is still retained."""
    if not len(edges):
        return np.empty(0, dtype=EDGE)
    edges = edges[
        np.lexsort(
            (edges["keep"], np.asarray(ranks)[edges["keep"] >> 32], edges["lose"])
        )
    ]
    bounds = np.r_[
        0, np.flatnonzero(edges["lose"][1:] != edges["lose"][:-1]) + 1, len(edges)
    ]
    removed = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        lose = int(edges[start]["lose"])
        position = int(offsets[lose >> 32]) + (lose & MASK32)
        if not keep_flags[position]:
            continue
        for edge in edges[start:end]:
            keep = int(edge["keep"])
            kept_position = int(offsets[keep >> 32]) + (keep & MASK32)
            if keep_flags[kept_position]:
                keep_flags[position] = False
                removed.append((lose, keep, float(edge["score"])))
                break
    return np.array(removed, dtype=EDGE)


def export_final(output):
    plan = plan_at(output)
    for file_id in plan["selected"]:
        validate_source(plan, file_id)
    marker = Path(output) / "final.done.json"
    if marker.exists():
        expected = json.loads(marker.read_text())["outputs"]
        for name, size in expected.items():
            path = Path(output) / name
            if not path.exists() or path.stat().st_size != size:
                raise RuntimeError(f"Incomplete final output: {path}")
        return
    offsets = np.r_[
        0, np.cumsum([item["rows"] for item in plan["files"]], dtype=np.uint64)
    ]
    flags = np.zeros(int(offsets[-1]), dtype=bool)
    for file_id in plan["selected"]:
        rows = read_array(job_path(output, "exact_keep", file_id) / "rows.u64", U64)
        flags[(offsets[file_id] + rows).astype(np.int64)] = True
    outputs, removed_arrays = {}, []
    final_samples = defaultdict(list)
    audit = Path(output) / "audit"
    audit.mkdir(exist_ok=True)
    keep_dir = Path(output) / "keep"
    keep_dir.mkdir(exist_ok=True)
    for file_id in plan["selected"]:
        category = plan["files"][file_id]["category"]
        edges = np.concatenate(
            [
                read_array(
                    job_path(output, "verify", shard) / "edges" / f"{file_id:04d}.bin",
                    EDGE,
                )
                for shard in range(plan["shards"])
            ]
        )
        removed = resolve_edges(flags, offsets, edges, plan["ranks"])
        removed_arrays.append(removed)
        target = audit / f"near-{file_id:04d}.bin"
        write_array(target, removed)
        outputs[str(target.relative_to(output))] = target.stat().st_size
        for edge in removed:
            sample_add(
                final_samples,
                category,
                {
                    "lose": int(edge["lose"]),
                    "keep": int(edge["keep"]),
                    "score": float(edge["score"]),
                    "reason": "near_removed",
                },
            )
        rows = read_array(job_path(output, "exact_keep", file_id) / "rows.u64", U64)
        rows = rows[flags[(offsets[file_id] + rows).astype(np.int64)]]
        target = keep_dir / f"{file_id:04d}.u64"
        write_array(target, rows)
        outputs[str(target.relative_to(output))] = target.stat().st_size
    aliases = (
        np.concatenate(removed_arrays) if removed_arrays else np.empty(0, dtype=EDGE)
    )
    aliases = aliases[np.argsort(aliases["lose"])]
    for shard in range(plan["shards"]):
        pairs = read_array(job_path(output, "exact", shard) / "duplicates.bin", PAIR)
        if len(aliases) and len(pairs):
            locations = np.searchsorted(aliases["lose"], pairs["keep"])
            valid = locations < len(aliases)
            matches = np.flatnonzero(valid)
            matches = matches[
                aliases["lose"][locations[matches]] == pairs["keep"][matches]
            ]
            pairs["keep"][matches] = aliases["keep"][locations[matches]]
        target = audit / f"exact-{shard:04d}.bin"
        write_array(target, pairs)
        outputs[str(target.relative_to(output))] = target.stat().st_size
    atomic_json(
        audit / "near_samples.json",
        {
            category: [entry[3] for entry in heap]
            for category, heap in final_samples.items()
        },
    )
    candidate_samples = defaultdict(dict)
    for shard in range(plan["shards"]):
        path = job_path(output, "verify", shard) / "samples.json"
        for category, records in json.loads(path.read_text()).items():
            for record in records:
                candidate_samples[category][(record["lose"], record["keep"])] = record
    atomic_json(
        audit / "candidate_samples.json",
        {
            category: sorted(records.values(), key=lambda record: record["sample_key"])[
                :100
            ]
            for category, records in candidate_samples.items()
        },
    )
    for name in outputs:
        with (Path(output) / name).open("rb") as stream:
            os.fsync(stream.fileno())
    atomic_json(marker, {"outputs": outputs})


def summary(output, stage, complete=False):
    plan = plan_at(output)
    categories, entries = {}, []
    for file_id, item in enumerate(plan["files"]):
        prep_path = job_path(output, "prepare", file_id) / "complete.json"
        stats = json.loads(prep_path.read_text())["stats"] if prep_path.exists() else {}
        rows, filtered, valid = (
            stats.get("rows", 0),
            stats.get("filtered", 0),
            stats.get("valid", 0),
        )
        exact_keep = job_path(output, "exact_keep", file_id) / "rows.u64"
        exact_kept = exact_keep.stat().st_size // 8 if exact_keep.exists() else valid
        final_keep = Path(output) / "keep" / f"{file_id:04d}.u64"
        kept = final_keep.stat().st_size // 8 if final_keep.exists() else exact_kept
        totals = categories.setdefault(item["category"], Counter())
        totals.update(
            rows=rows,
            kept=kept,
            exact=valid - exact_kept,
            near=exact_kept - kept,
            filtered=filtered,
        )
        entries.append(
            {
                "file_id": file_id,
                "category": item["category"],
                "source": item["output_file"],
                "source_bytes": item["bytes"],
                "rows": rows,
                "kept": kept,
                "completed_row_groups": item["row_groups"] if rows else 0,
                "keep_index": f"keep/{file_id:04d}.u64"
                if final_keep.exists()
                else None,
            }
        )
    return {
        "status": "complete" if complete else "partial",
        "stage": stage,
        "algorithm": "global blake2b-128 exact; 128-permutation MinHash/LSH candidates with full-content Jaccard and code/math verification; L3 exact-only",
        "schema": SCHEMA_VERSION,
        "config_id": plan["config_id"],
        "index_format": "little-endian uint64 source row numbers, zero based",
        "audit_format": {"exact": PAIR.descr, "near": EDGE.descr},
        "categories": categories,
        "files": entries,
    }


class StageStop(Exception):
    pass


def stage_jobs(output, stage, function, jobs, workers, budget):
    started = time.perf_counter()
    pending_jobs = [(output, job) for job in jobs if not completed(output, stage, job)]
    total, done = len(jobs), len(jobs) - len(pending_jobs)
    if not pending_jobs:
        print(f"stage={stage} completed={done}/{total} resume=skip", flush=True)
        return
    tasks = iter(pending_jobs)
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        futures = {}
        if pool is not None:
            for _ in range(workers * 2):
                task = next(tasks, None)
                if task is None:
                    break
                futures[pool.submit(function, task)] = task
        while futures or pool is None:
            if pool is None:
                task = next(tasks, None)
                if task is None:
                    break
                result = function(task)
            else:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                future = next(iter(ready))
                task = futures.pop(future)
                result = future.result()
                following = next(tasks, None)
                if following is not None:
                    futures[pool.submit(function, following)] = following
            done += 1
            budget["done"] += 1
            print(
                f"stage={stage} task={task[1]} completed={done}/{total} stats={json.dumps(result)}",
                flush=True,
            )
            if done % 10 == 0 or done == total:
                atomic_json(Path(output) / "status.json", summary(output, stage))
            if budget["limit"] and budget["done"] >= budget["limit"]:
                raise StageStop(stage)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        timings_path = Path(output) / "stage_timings.json"
        timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
        previous = timings.get(stage, {})
        elapsed = time.perf_counter() - started
        timings[stage] = {
            "last_wall_seconds": elapsed,
            "total_wall_seconds": previous.get("total_wall_seconds", 0) + elapsed,
            "workers": workers,
            "completed_jobs": sum(completed(output, stage, job) for job in jobs),
            "total_jobs": total,
        }
        atomic_json(timings_path, timings)


def make_plan(args):
    manifest_path = args.manifest.resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != "downloaded":
        raise RuntimeError("Input manifest is not marked downloaded")
    files = ordered_files(manifest)
    if not files:
        raise RuntimeError("Input manifest has no files")
    for item in files:
        source = (manifest_path.parent / item["output_file"]).resolve()
        if (
            not source.is_relative_to(manifest_path.parent)
            or not source.is_file()
            or source.stat().st_size != item["bytes"]
        ):
            raise RuntimeError(
                f"Missing, size mismatch, or source escapes manifest: {source}"
            )
        metadata = pq.ParquetFile(source).metadata
        item["rows"], item["row_groups"] = metadata.num_rows, metadata.num_row_groups
        item["mtime_ns"] = source.stat().st_mtime_ns
        if metadata.num_rows >= 1 << 32:
            raise RuntimeError(
                "A source file exceeds the 32-bit document row identifier"
            )
    order = processing_order(files)
    selected = [file_id for file_id, _ in order[: args.max_files or None]]
    ranks = [0] * len(files)
    for rank, (file_id, _) in enumerate(order):
        ranks[file_id] = rank
    domains = {
        name: i
        for i, name in enumerate(
            sorted(
                {
                    near_group(item["category"])
                    for item in files
                    if near_group(item["category"]) is not None
                }
            )
        )
    }
    shards = getattr(args, "shards", 64)
    if shards < 1 or shards > 256:
        raise RuntimeError("shards must be between 1 and 256")
    plan = {
        "schema": SCHEMA_VERSION,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "pipeline_code_sha256": code_identity(),
        "raw_root": str(manifest_path.parent),
        "files": files,
        "selected": selected,
        "ranks": ranks,
        "domains": domains,
        "shards": shards,
        "candidate_cap": args.candidate_cap,
        "datasketch_version": version("datasketch"),
        "pygments_version": version("Pygments"),
        "minhash_permutations": 128,
        "minhash_scheme": "legacy",
        "feature_hash": "crc32-candidate-only",
        "minhash_seed": 42,
        "bands": 16,
        "hashes_per_band": 8,
        "verification_shard_policy": "1024-row blocks mixed with source file id",
        "verification": {
            "web": 0.9,
            "code": 0.95,
            "math": 0.95,
            "code_guard": "identical lexical tokens excluding comments/whitespace",
            "math_guard": "identical number/formula sequence",
            "min_distinct_features": 20,
            "minhash_prefilter_min_agreement": 0.5,
            "large_bucket_policy": "all pairs up to 32 members; eight priority anchors above 32; unmatched records retained",
        },
    }
    plan["config_id"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    return plan


@contextmanager
def output_lock(path):
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                if not path.stat().st_size:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Another dedup process owns this output directory"
            ) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def run(args):
    if args.profile != "astrai-100b":
        raise RuntimeError("Unsupported deduplication profile")
    output = str(args.output.resolve())
    plan = make_plan(args)
    root = Path(output)
    plan_path = root / "plan.json"
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "run.lock"
    with output_lock(lock):
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise RuntimeError(
                    "Output belongs to a different manifest or dedup configuration"
                )
        elif any(path.name not in {"run.log", "run.lock"} for path in root.iterdir()):
            raise RuntimeError(
                "Output has dedup results but no matching plan.json; use a new output directory"
            )
        else:
            atomic_json(plan_path, plan)
        plan_at.cache_clear()
        budget = {
            "done": 0,
            "limit": getattr(args, "stop_after_tasks", 0)
            or getattr(args, "stop_after_groups", 0),
        }
        workers = args.workers
        atomic_json(root / "status.json", summary(output, "prepare"))
        try:
            stage_jobs(
                output, "prepare", prepare_file, plan["selected"], workers, budget
            )
            stage_jobs(
                output,
                "exact",
                reduce_exact,
                list(range(plan["shards"])),
                workers,
                budget,
            )
            stage_jobs(
                output,
                "exact_keep",
                gather_exact_keep,
                plan["selected"],
                workers,
                budget,
            )
            stage_jobs(
                output, "signatures", signature_file, plan["selected"], workers, budget
            )
            stage_jobs(
                output, "buckets", bucket_file, plan["selected"], workers, budget
            )
            stage_jobs(
                output,
                "candidates",
                reduce_buckets,
                list(range(plan["shards"])),
                workers,
                budget,
            )
            stage_jobs(
                output,
                "verify",
                verify_shard,
                list(range(plan["shards"])),
                workers,
                budget,
            )
            started = time.perf_counter()
            export_final(output)
            timings_path = root / "stage_timings.json"
            timings = json.loads(timings_path.read_text())
            timings["final"] = {"last_wall_seconds": time.perf_counter() - started}
            atomic_json(timings_path, timings)
            complete = len(plan["selected"]) == len(plan["files"])
            result = summary(output, "finished", complete)
            atomic_json(root / "status.json", result)
            if complete:
                atomic_json(root / "manifest.json", result)
            return result
        except StageStop as exc:
            result = summary(output, str(exc))
            atomic_json(root / "status.json", result)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["astrai-100b"], required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--candidate-cap", type=int, default=2000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--stop-after-tasks", type=int, default=0)
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.candidate_cap < 1
        or args.max_files < 0
        or args.stop_after_tasks < 0
    ):
        parser.error("invalid numeric options")
    run(args)


if __name__ == "__main__":
    main()
