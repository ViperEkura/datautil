"""Disk-backed selection for commands that need more than one input pass."""

from __future__ import annotations

import hashlib
import random
import shutil
import sqlite3
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from .formats import ADAPTERS, detect_record
from .io import iter_raw, output_writer
from .model import DataError
from .operations import read_records


@contextmanager
def replayable_input(path: str) -> Iterator[str]:
    if path != "-":
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="datautil-input-") as directory:
        target = Path(directory) / "stdin.jsonl"
        with target.open("wb") as stream:
            shutil.copyfileobj(sys.stdin.buffer, stream)
        yield str(target)


def source_format(path: str, requested: str | None) -> str:
    if requested:
        if requested not in ADAPTERS:
            raise DataError("unknown_format", f"Unknown format {requested!r}")
        return requested
    first = next(iter_raw(path), None)
    if first is None:
        raise DataError("empty_dataset", "Cannot infer format from an empty dataset; pass --from")
    choices = detect_record(first[1])
    if len(choices) != 1:
        raise DataError("ambiguous_format", f"Cannot infer source format: {choices}", first[0])
    return choices[0]


class DiskIndex:
    def __init__(self, directory: str) -> None:
        self.conn = sqlite3.connect(str(Path(directory) / "index.sqlite3"))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA cache_size=-2048")
        self.conn.execute("CREATE TABLE seen (id TEXT PRIMARY KEY, first_location TEXT)")

    def add_unique(self, record_id: str, location: str | None = None) -> bool:
        try:
            self.conn.execute("INSERT INTO seen VALUES (?, ?)", (record_id, location))
            return True
        except sqlite3.IntegrityError:
            return False

    def first_location(self, record_id: str) -> str | None:
        row = self.conn.execute("SELECT first_location FROM seen WHERE id=?", (record_id,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DiskIndex":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def sample_file(path: str, output: str, requested: str | None, count: int, seed: int) -> dict[str, int]:
    if count < 0:
        raise DataError("invalid_argument", "count must be non-negative")
    with replayable_input(path) as replay, tempfile.TemporaryDirectory(prefix="datautil-sample-") as directory:
        source = source_format(replay, requested)
        index = DiskIndex(directory)
        try:
            total = sum(1 for _ in read_records(replay, source, id_index=index, location_alias=path))
        finally:
            index.close()
        selected_count = min(count, total)
        selected = set(random.Random(seed).sample(range(total), selected_count)) if selected_count < total else None
        seen = 0
        with output_writer(output, getattr(ADAPTERS[source], "container", "auto")) as writer:
            for seen, record in enumerate(read_records(replay, source, check_duplicate_ids=False, location_alias=path), 1):
                if selected is None or seen - 1 in selected:
                    writer.write(ADAPTERS[source].serialize(record))
            if seen != total:
                raise DataError("input_changed", "Input changed between selection and output passes")
        return {"record_count": total, "sample_count": selected_count, "seed": seed}


class GroupIndex(DiskIndex):
    def __init__(self, directory: str) -> None:
        super().__init__(directory)
        self.conn.execute("CREATE TABLE parent (id TEXT PRIMARY KEY, parent TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE records (position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, component TEXT NOT NULL)")

    def find(self, item: str) -> str:
        visited = []
        current = item
        while True:
            row = self.conn.execute("SELECT parent FROM parent WHERE id=?", (current,)).fetchone()
            if row is None:
                self.conn.execute("INSERT INTO parent VALUES (?, ?)", (current, current))
                root = current
                break
            if row[0] == current:
                root = current
                break
            visited.append(current)
            current = row[0]
        for member in visited:
            self.conn.execute("UPDATE parent SET parent=? WHERE id=?", (root, member))
        return root

    def load_groups(self, path: str) -> None:
        for location, group in iter_raw(path):
            if not isinstance(group, dict) or not isinstance(group.get("record_ids"), list):
                raise DataError("invalid_groups", "Group needs a record_ids array", location)
            ids = group["record_ids"]
            if not ids:
                continue
            if any(not isinstance(item, str) for item in ids):
                raise DataError("invalid_groups", "Group record IDs must be strings", location)
            first = self.find(ids[0])
            for item in ids[1:]:
                other = self.find(item)
                if other != first:
                    self.conn.execute("UPDATE parent SET parent=? WHERE id=?", (first, other))

    def component(self, record_id: str) -> str:
        exists = self.conn.execute("SELECT 1 FROM parent WHERE id=?", (record_id,)).fetchone()
        return self.find(record_id) if exists else record_id

    def check_group_ids(self) -> None:
        unknown = self.conn.execute("SELECT parent.id FROM parent LEFT JOIN seen ON seen.id=parent.id WHERE seen.id IS NULL LIMIT 1").fetchone()
        if unknown:
            raise DataError("invalid_groups", f"Group references unknown record {unknown[0]!r}")


def split_file(path: str, train_path: str, test_path: str, requested: str | None, ratio: float, seed: int, groups_path: str | None) -> dict[str, int]:
    if not 0 < ratio < 1:
        raise DataError("invalid_argument", "test ratio must be between 0 and 1")
    if train_path == test_path:
        raise DataError("invalid_argument", "Train and test output paths must differ")
    with replayable_input(path) as replay, tempfile.TemporaryDirectory(prefix="datautil-split-") as directory:
        source = source_format(replay, requested)
        index = GroupIndex(directory)
        try:
            if groups_path:
                index.load_groups(groups_path)
            total = 0
            for total, record in enumerate(read_records(replay, source, id_index=index, location_alias=path), 1):
                index.conn.execute("INSERT INTO records VALUES (?, ?, ?)", (total, record.record_id, index.component(record.record_id)))
            index.check_group_ids()
            index.conn.execute("CREATE TABLE components (id TEXT PRIMARY KEY, first_position INTEGER, size INTEGER, score BLOB)")
            index.conn.create_function("selection_score", 1, lambda position: hashlib.blake2b(f"{seed}:{position}".encode(), digest_size=16).digest())
            index.conn.execute("INSERT INTO components SELECT component, MIN(position), COUNT(*), selection_score(MIN(position)) FROM records GROUP BY component")
            index.conn.execute("CREATE TABLE selected (id TEXT PRIMARY KEY)")
            target = max(1, min(total - 1, round(total * ratio))) if total >= 2 else 0
            selected_count = 0
            for component, size in index.conn.execute("SELECT id, size FROM components ORDER BY score, first_position"):
                if selected_count < target and abs(selected_count + size - target) <= abs(selected_count - target):
                    index.conn.execute("INSERT INTO selected VALUES (?)", (component,))
                    selected_count += size
            train_count = test_count = 0
            with ExitStack() as stack:
                container = getattr(ADAPTERS[source], "container", "auto")
                train = stack.enter_context(output_writer(train_path, container))
                test = stack.enter_context(output_writer(test_path, container))
                for position, record in enumerate(read_records(replay, source, check_duplicate_ids=False, location_alias=path), 1):
                    row = index.conn.execute("SELECT id, component FROM records WHERE position=?", (position,)).fetchone()
                    if row is None or row[0] != record.record_id:
                        raise DataError("input_changed", "Input changed between selection and output passes")
                    if index.conn.execute("SELECT 1 FROM selected WHERE id=?", (row[1],)).fetchone():
                        test.write(ADAPTERS[source].serialize(record))
                        test_count += 1
                    else:
                        train.write(ADAPTERS[source].serialize(record))
                        train_count += 1
            if train_count + test_count != total:
                raise DataError("input_changed", "Input changed between selection and output passes")
            return {"record_count": total, "train_count": train_count, "test_count": test_count, "seed": seed}
        finally:
            index.close()


def dedup_apply_file(path: str, groups_path: str, output: str, requested: str | None) -> dict[str, object]:
    with replayable_input(path) as replay, tempfile.TemporaryDirectory(prefix="datautil-apply-") as directory:
        source = source_format(replay, requested)
        index = DiskIndex(directory)
        try:
            index.conn.execute("CREATE TABLE positions (position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL)")
            index.conn.execute("CREATE TABLE removed (id TEXT PRIMARY KEY)")
            index.conn.execute("CREATE TABLE decisions (position INTEGER PRIMARY KEY, removed TEXT, kept TEXT)")
            total = 0
            for total, record in enumerate(read_records(replay, source, id_index=index, location_alias=path), 1):
                index.conn.execute("INSERT INTO positions VALUES (?, ?)", (total, record.record_id))
            decision_count = 0
            for location, group in iter_raw(groups_path):
                if not isinstance(group, dict) or not isinstance(group.get("record_ids"), list):
                    raise DataError("invalid_groups", "Group needs a record_ids array", location)
                ids = group["record_ids"]
                if len(ids) < 2 or any(not isinstance(item, str) for item in ids):
                    raise DataError("invalid_groups", "Group references unknown or too few records", location)
                positions = [index.conn.execute("SELECT position FROM positions WHERE id=?", (item,)).fetchone() for item in ids]
                if any(row is None for row in positions):
                    raise DataError("invalid_groups", "Group references unknown or too few records", location)
                kept = ids[min(range(len(ids)), key=lambda item: positions[item][0])]
                for item in ids:
                    if item != kept:
                        index.conn.execute("INSERT OR IGNORE INTO removed VALUES (?)", (item,))
                        decision_count += 1
                        if decision_count <= 1000:
                            index.conn.execute("INSERT INTO decisions VALUES (?, ?, ?)", (decision_count, item, kept))
            written = 0
            with output_writer(output, getattr(ADAPTERS[source], "container", "auto")) as writer:
                position = 0
                for position, record in enumerate(read_records(replay, source, check_duplicate_ids=False, location_alias=path), 1):
                    row = index.conn.execute("SELECT id FROM positions WHERE position=?", (position,)).fetchone()
                    if row is None or row[0] != record.record_id:
                        raise DataError("input_changed", "Input changed between selection and output passes")
                    if index.conn.execute("SELECT 1 FROM removed WHERE id=?", (record.record_id,)).fetchone() is None:
                        writer.write(ADAPTERS[source].serialize(record))
                        written += 1
                if position != total:
                    raise DataError("input_changed", "Input changed between selection and output passes")
            decisions = [{"removed": item, "kept": kept} for item, kept in index.conn.execute("SELECT removed, kept FROM decisions ORDER BY position")]
            return {"record_count": total, "written_count": written, "removed_count": total - written, "decision_count": decision_count, "decisions": decisions, "decisions_truncated": decision_count > len(decisions)}
        finally:
            index.close()
