from __future__ import annotations

import hashlib
import random
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from typing import Any

from datautil.common.errors import DataError
from datautil.common.index import DiskIndex
from datautil.common.io import container_for, iter_raw, output_writer
from datautil.conversation.formats import ADAPTERS
from datautil.conversation.model import DatasetRecord
from datautil.conversation.reader import read_records, replayable_input, source_format


def sample_records(
    records: list[DatasetRecord], count: int, seed: int = 42
) -> list[DatasetRecord]:
    if count < 0:
        raise DataError("invalid_argument", "count must be non-negative")
    randomizer = random.Random(seed)
    chosen = set(randomizer.sample(range(len(records)), min(count, len(records))))
    return [record for index, record in enumerate(records) if index in chosen]


def split_records(
    records: list[DatasetRecord],
    ratio: float = 0.1,
    seed: int = 42,
    groups: list[dict[str, Any]] | None = None,
) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    if not 0 < ratio < 1:
        raise DataError("invalid_argument", "test ratio must be between 0 and 1")
    ids = {record.record_id for record in records}
    parent = {item: item for item in ids}

    def root(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for group in groups or []:
        members = group.get("record_ids", [])
        if any(item not in ids for item in members):
            raise DataError("invalid_groups", "Group references an unknown record")
        for item in members[1:]:
            parent[root(item)] = root(members[0])
    blocks: dict[str, list[str]] = defaultdict(list)
    for record in records:
        blocks[root(record.record_id)].append(record.record_id)
    chunks = list(blocks.values())
    random.Random(seed).shuffle(chunks)
    target = (
        max(1, min(len(records) - 1, round(len(records) * ratio)))
        if len(records) >= 2
        else 0
    )
    test_ids: set[str] = set()
    for chunk in chunks:
        if len(test_ids) < target and abs(len(test_ids) + len(chunk) - target) <= abs(
            len(test_ids) - target
        ):
            test_ids.update(chunk)
    return (
        [record for record in records if record.record_id not in test_ids],
        [record for record in records if record.record_id in test_ids],
    )


class GroupIndex(DiskIndex):
    def __init__(self, directory: str) -> None:
        super().__init__(directory)
        self.conn.execute(
            "CREATE TABLE parent (id TEXT PRIMARY KEY, parent TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE records (position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, component TEXT NOT NULL)"
        )

    def find(self, item: str) -> str:
        visited = []
        current = item
        while True:
            row = self.conn.execute(
                "SELECT parent FROM parent WHERE id=?", (current,)
            ).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO parent VALUES (?, ?)", (current, current)
                )
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
            if not isinstance(group, dict) or not isinstance(
                group.get("record_ids"), list
            ):
                raise DataError(
                    "invalid_groups", "Group needs a record_ids array", location
                )
            ids = group["record_ids"]
            if not ids:
                continue
            if any(not isinstance(item, str) for item in ids):
                raise DataError(
                    "invalid_groups", "Group record IDs must be strings", location
                )
            first = self.find(ids[0])
            for item in ids[1:]:
                other = self.find(item)
                if other != first:
                    self.conn.execute(
                        "UPDATE parent SET parent=? WHERE id=?", (first, other)
                    )

    def component(self, record_id: str) -> str:
        exists = self.conn.execute(
            "SELECT 1 FROM parent WHERE id=?", (record_id,)
        ).fetchone()
        return self.find(record_id) if exists else record_id

    def check_group_ids(self) -> None:
        unknown = self.conn.execute(
            "SELECT parent.id FROM parent LEFT JOIN seen ON seen.id=parent.id WHERE seen.id IS NULL LIMIT 1"
        ).fetchone()
        if unknown:
            raise DataError(
                "invalid_groups", f"Group references unknown record {unknown[0]!r}"
            )


def sample_file(
    path: str, output: str, requested: str | None, count: int, seed: int
) -> dict[str, int]:
    if count < 0:
        raise DataError("invalid_argument", "count must be non-negative")
    with (
        replayable_input(path) as replay,
        tempfile.TemporaryDirectory(prefix="datautil-sample-") as directory,
    ):
        source = source_format(replay, requested)
        index = DiskIndex(directory)
        try:
            total = sum(
                1
                for _ in read_records(
                    replay, source, id_index=index, location_alias=path
                )
            )
        finally:
            index.close()
        selected_count = min(count, total)
        selected = (
            set(random.Random(seed).sample(range(total), selected_count))
            if selected_count < total
            else None
        )
        seen = 0
        with output_writer(
            output,
            container_for(
                output, "auto", getattr(ADAPTERS[source], "storage_hint", None)
            ),
        ) as writer:
            for seen, record in enumerate(
                read_records(
                    replay, source, check_duplicate_ids=False, location_alias=path
                ),
                1,
            ):
                if selected is None or seen - 1 in selected:
                    writer.write(ADAPTERS[source].serialize(record))
            if seen != total:
                raise DataError(
                    "input_changed", "Input changed between selection and output passes"
                )
        return {"record_count": total, "sample_count": selected_count, "seed": seed}


def split_file(
    path: str,
    train_path: str,
    test_path: str,
    requested: str | None,
    ratio: float,
    seed: int,
    groups_path: str | None,
) -> dict[str, int]:
    if not 0 < ratio < 1:
        raise DataError("invalid_argument", "test ratio must be between 0 and 1")
    if train_path == test_path:
        raise DataError("invalid_argument", "Train and test output paths must differ")
    with (
        replayable_input(path) as replay,
        tempfile.TemporaryDirectory(prefix="datautil-split-") as directory,
    ):
        source = source_format(replay, requested)
        index = GroupIndex(directory)
        try:
            if groups_path:
                index.load_groups(groups_path)
            total = 0
            for total, record in enumerate(
                read_records(replay, source, id_index=index, location_alias=path), 1
            ):
                index.conn.execute(
                    "INSERT INTO records VALUES (?, ?, ?)",
                    (total, record.record_id, index.component(record.record_id)),
                )
            index.check_group_ids()
            index.conn.execute(
                "CREATE TABLE components (id TEXT PRIMARY KEY, first_position INTEGER, size INTEGER, score BLOB)"
            )
            index.conn.create_function(
                "selection_score",
                1,
                lambda position: hashlib.blake2b(
                    f"{seed}:{position}".encode(), digest_size=16
                ).digest(),
            )
            index.conn.execute(
                "INSERT INTO components SELECT component, MIN(position), COUNT(*), selection_score(MIN(position)) FROM records GROUP BY component"
            )
            index.conn.execute("CREATE TABLE selected (id TEXT PRIMARY KEY)")
            target = max(1, min(total - 1, round(total * ratio))) if total >= 2 else 0
            selected_count = 0
            for component, size in index.conn.execute(
                "SELECT id, size FROM components ORDER BY score, first_position"
            ):
                if selected_count < target and abs(
                    selected_count + size - target
                ) <= abs(selected_count - target):
                    index.conn.execute("INSERT INTO selected VALUES (?)", (component,))
                    selected_count += size
            train_count = test_count = 0
            with ExitStack() as stack:
                hint = getattr(ADAPTERS[source], "storage_hint", None)
                train = stack.enter_context(
                    output_writer(train_path, container_for(train_path, "auto", hint))
                )
                test = stack.enter_context(
                    output_writer(test_path, container_for(test_path, "auto", hint))
                )
                for position, record in enumerate(
                    read_records(
                        replay, source, check_duplicate_ids=False, location_alias=path
                    ),
                    1,
                ):
                    row = index.conn.execute(
                        "SELECT id, component FROM records WHERE position=?",
                        (position,),
                    ).fetchone()
                    if row is None or row[0] != record.record_id:
                        raise DataError(
                            "input_changed",
                            "Input changed between selection and output passes",
                        )
                    if index.conn.execute(
                        "SELECT 1 FROM selected WHERE id=?", (row[1],)
                    ).fetchone():
                        test.write(ADAPTERS[source].serialize(record))
                        test_count += 1
                    else:
                        train.write(ADAPTERS[source].serialize(record))
                        train_count += 1
            if train_count + test_count != total:
                raise DataError(
                    "input_changed", "Input changed between selection and output passes"
                )
            return {
                "record_count": total,
                "train_count": train_count,
                "test_count": test_count,
                "seed": seed,
            }
        finally:
            index.close()
