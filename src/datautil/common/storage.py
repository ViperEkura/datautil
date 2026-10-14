"""Raw row storage, independent of conversation and document formats."""

import json
import math
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterator, List, Optional, Protocol, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from datautil.common.errors import DataError
from datautil.common.textio import RecordWriter, _iter_json_array

_SHAPE_KEY = b"datautil.shape_column"
_VERSION_KEY = b"datautil.shape_version"
_BATCH_SIZE = 2048


class RowWriter(Protocol):
    def write(self, value: Any) -> None: ...


class StorageBackend(Protocol):
    name: str

    def read(self, path: str, errors: str = "raise") -> Iterator[Tuple[str, Any]]: ...

    def writer(self, path: str) -> ContextManager[RowWriter]: ...


def storage_for(path: str, requested: str = "auto", hint: Optional[str] = None) -> str:
    if not isinstance(requested, str) or (
        hint is not None and not isinstance(hint, str)
    ):
        raise DataError("unknown_storage", "Storage backend must be a name", path)
    # Explicit choices always win. Parquet extensions override legacy text hints.
    selected = requested
    if selected == "auto":
        selected = "parquet" if path.lower().endswith(".parquet") else (hint or "auto")
    if selected == "auto":
        selected = (
            "jsonl"
            if path == "-" or path.lower().endswith((".jsonl", ".ndjson"))
            else "json"
        )
    if selected not in STORAGE_BACKENDS:
        raise DataError(
            "unknown_storage", "Unknown storage backend: " + str(selected), path
        )
    return selected


@contextmanager
def _atomic_target(path: str) -> Iterator[str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + target.name + ".", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TextStorageBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def read(self, path: str, errors: str = "raise") -> Iterator[Tuple[str, Any]]:
        stream = (
            sys.stdin
            if path == "-"
            else open(path, "r", encoding="utf-8-sig", newline="")
        )
        try:
            if self.name == "jsonl":
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        value = DataError("invalid_json", str(exc), f"{path}:{number}")
                        if errors != "yield":
                            raise value from exc
                    yield f"{path}:{number}", value
            else:
                for number, item in _iter_json_array(stream):
                    yield f"{path}#record-{number}", item
        finally:
            if path != "-":
                stream.close()

    @contextmanager
    def writer(self, path: str) -> Iterator[RowWriter]:
        if path == "-":
            writer = RecordWriter(sys.stdout, self.name)
            yield writer
            writer.finish()
            return
        with _atomic_target(path) as temporary:
            with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
                writer = RecordWriter(stream, self.name)
                yield writer
                writer.finish()
                stream.flush()
                os.fsync(stream.fileno())


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_shape(child) for child in value]
    return {type(None): "n", bool: "b", int: "i", float: "f", str: "s"}.get(
        type(value), "v"
    )


def _restore(value: Any, shape: Any) -> Any:
    if isinstance(shape, dict):
        if value is None and not shape:
            return {}
        if not isinstance(value, dict) or any(key not in value for key in shape):
            raise ValueError("Invalid object shape")
        return {key: _restore(value[key], child) for key, child in shape.items()}
    if isinstance(shape, list):
        if not isinstance(value, list) or len(value) != len(shape):
            raise ValueError("Invalid array shape")
        return [_restore(child, entry) for child, entry in zip(value, shape)]
    if shape == "n":
        if value is not None:
            raise ValueError("Invalid null shape")
        return None
    if shape == "i":
        if type(value) not in {int, float}:
            raise ValueError("Invalid integer shape")
        converted = int(value)
        if converted != value:
            raise ValueError("Invalid integer shape")
        return converted
    if shape == "f":
        if type(value) not in {int, float}:
            raise ValueError("Invalid float shape")
        return float(value)
    if shape == "b" and type(value) is not bool:
        raise ValueError("Invalid boolean shape")
    if shape == "s" and not isinstance(value, str):
        raise ValueError("Invalid string shape")
    if shape in {"b", "s", "v"}:
        return value
    raise ValueError("Unknown scalar shape")


def _equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _column_name(schema: pa.Schema) -> str:
    name = "_datautil_shape"
    while name in schema.names:
        name += "_"
    return name


def _parquet_type(kind: pa.DataType) -> pa.DataType:
    if pa.types.is_struct(kind):
        return (
            pa.struct(
                [pa.field(field.name, _parquet_type(field.type)) for field in kind]
            )
            if len(kind)
            else pa.null()
        )
    if pa.types.is_list(kind):
        return pa.list_(_parquet_type(kind.value_type))
    if pa.types.is_large_list(kind):
        return pa.large_list(_parquet_type(kind.value_type))
    return kind


def _physical(value: Any, kind: pa.DataType) -> Any:
    if value is None or pa.types.is_null(kind):
        return None
    if pa.types.is_struct(kind):
        return {
            field.name: _physical(value.get(field.name), field.type) for field in kind
        }
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        return [_physical(child, kind.value_type) for child in value]
    return value


class ParquetRowWriter:
    """Stage bounded IPC batches to discover a lossless, dataset-wide schema."""

    def __init__(self, target: Any, directory: str, location: str) -> None:
        self.target = target
        self.directory = Path(directory)
        self.location = location
        self.rows: List[Dict[str, Any]] = []
        self.schema: Optional[pa.Schema] = None
        self.batches = 0
        self.count = 0

    def write(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise DataError(
                "invalid_storage_record", "Parquet rows must be objects", self.location
            )
        self.rows.append(value)
        self.count += 1
        if len(self.rows) >= _BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        try:
            # StructArray infers all row fields; Table.from_pylist uses the first row.
            table = pa.Table.from_struct_array(pa.array(self.rows))
            shapes = [_shape(row) for row in self.rows]
            for original, encoded, shape in zip(self.rows, table.to_pylist(), shapes):
                if not _equal(original, _restore(encoded, shape)):
                    raise DataError(
                        "lossy_storage",
                        "Arrow schema inference would change a record",
                        self.location,
                    )
            self.schema = (
                table.schema
                if self.schema is None
                else pa.unify_schemas([self.schema, table.schema])
            )
            shape_column = _column_name(table.schema)
            table = table.append_column(
                shape_column,
                pa.array([json.dumps(shape, ensure_ascii=False) for shape in shapes]),
            )
            table = table.replace_schema_metadata(
                {_SHAPE_KEY: shape_column.encode(), _VERSION_KEY: b"1"}
            )
            with ipc.new_file(
                str(self.directory / f"{self.batches}.arrow"), table.schema
            ) as writer:
                writer.write_table(table)
            self.batches += 1
            self.rows.clear()
        except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
            raise DataError(
                "incompatible_storage_schema", str(exc), self.location
            ) from exc

    def finish(self) -> None:
        self._flush()
        schema = pa.schema(
            [
                pa.field(field.name, _parquet_type(field.type))
                for field in (self.schema or pa.schema([]))
            ]
        )
        shape_column = _column_name(schema)
        schema = schema.append(pa.field(shape_column, pa.string())).with_metadata(
            {_SHAPE_KEY: shape_column.encode(), _VERSION_KEY: b"1"}
        )
        try:
            with pq.ParquetWriter(self.target, schema, compression="snappy") as writer:
                for number in range(self.batches):
                    with pa.memory_map(
                        str(self.directory / f"{number}.arrow"), "r"
                    ) as stream:
                        table = ipc.open_file(stream).read_all()
                        stage_column = table.schema.metadata[_SHAPE_KEY].decode()
                        rows = []
                        for row in table.to_pylist():
                            shape_json = row.pop(stage_column)
                            shape = json.loads(shape_json)
                            row = _restore(row, shape)
                            value = {
                                field.name: _physical(row.get(field.name), field.type)
                                for field in schema
                                if field.name != shape_column
                            }
                            value[shape_column] = shape_json
                            rows.append(value)
                        encoded = pa.Table.from_pylist(rows, schema=schema)
                        for original, physical in zip(rows, encoded.to_pylist()):
                            shape = json.loads(original[shape_column])
                            if not _equal(
                                _restore(original, shape), _restore(physical, shape)
                            ):
                                raise DataError(
                                    "lossy_storage",
                                    "Parquet encoding would change a record",
                                    self.location,
                                )
                        writer.write_table(encoded)
        except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
            raise DataError(
                "incompatible_storage_schema", str(exc), self.location
            ) from exc


class ParquetStorageBackend:
    name = "parquet"

    def read(self, path: str, errors: str = "raise") -> Iterator[Tuple[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="datautil-parquet-input-") as directory:
            source = path
            if path == "-":
                source = str(Path(directory) / "stdin.parquet")
                with open(source, "wb") as stream:
                    shutil.copyfileobj(sys.stdin.buffer, stream)
            parquet = None
            try:
                parquet = pq.ParquetFile(source)
                metadata = parquet.schema_arrow.metadata or {}
                shape_column = metadata.get(_SHAPE_KEY)
                if shape_column is not None:
                    if metadata.get(_VERSION_KEY) != b"1":
                        raise ValueError("Unsupported datautil shape version")
                    shape_column = shape_column.decode()
                number = 0
                for batch in parquet.iter_batches(batch_size=_BATCH_SIZE):
                    for raw in batch.to_pylist():
                        number += 1
                        if shape_column is not None:
                            raw = _restore(raw, json.loads(raw.pop(shape_column)))
                        yield f"{path}#row-{number}", raw
            except (pa.ArrowException, KeyError, TypeError, ValueError) as exc:
                raise DataError("invalid_parquet", str(exc), path) from exc
            finally:
                if parquet is not None:
                    parquet.close()

    @contextmanager
    def writer(self, path: str) -> Iterator[RowWriter]:
        with tempfile.TemporaryDirectory(prefix="datautil-parquet-stage-") as directory:
            if path == "-":
                writer = ParquetRowWriter(sys.stdout.buffer, directory, path)
                yield writer
                writer.finish()
                return
            with _atomic_target(path) as temporary:
                writer = ParquetRowWriter(temporary, directory, path)
                yield writer
                writer.finish()


STORAGE_BACKENDS: Dict[str, StorageBackend] = {
    "json": TextStorageBackend("json"),
    "jsonl": TextStorageBackend("jsonl"),
    "parquet": ParquetStorageBackend(),
}
