"""Storage and record formats compose without silently changing data."""

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from support import DatautilTestCase, write_jsonl

import datautil
from datautil.cli import main
from datautil.common.io import iter_raw, output_writer


class StorageTests(DatautilTestCase):
    def run_cli(self, args):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            return main(args)

    def test_raw_roundtrip_preserves_late_fields_nulls_and_empty_structs(self):
        source = self.root / "raw.jsonl"
        parquet = self.root / "raw.parquet"
        restored = self.root / "restored.jsonl"
        rows = [
            {"id": str(index), "metadata": {}, "items": []} for index in range(2048)
        ]
        rows += [
            {
                "id": "late",
                "metadata": {"new": None},
                "items": [{"x": 1}, {"y": "two"}],
                "late": True,
                "_datautil_shape": "user data",
            },
            {"id": "null", "metadata": None, "items": [None, {}]},
        ]
        write_jsonl(source, rows)
        self.assertEqual(
            self.run_cli(["format", "convert", str(source), "-o", str(parquet)]), 0
        )
        self.assertEqual(
            self.run_cli(["format", "convert", str(parquet), "-o", str(restored)]), 0
        )
        self.assertEqual([raw for _, raw in iter_raw(str(restored))], rows)
        self.assertIn("late", pq.ParquetFile(parquet).schema_arrow.names)

    def test_tools_thinking_and_metadata_survive_parquet_and_sample(self):
        source = self.root / "tools.jsonl"
        parquet = self.root / "tools.parquet"
        restored = self.root / "restored.jsonl"
        sample = self.root / "sample.jsonl"
        rows = [
            {
                "id": "a",
                "messages": [
                    {"role": "user", "content": "weather"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Check the forecast",
                        "tool_calls": [
                            {
                                "id": "call1",
                                "type": "function",
                                "function": {"name": "weather", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call1", "content": "sunny"},
                    {"role": "assistant", "content": "It is sunny"},
                ],
                "source": "test",
            }
        ]
        write_jsonl(source, rows)
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(source),
                    "--from",
                    "openai-messages",
                    "--to",
                    "openai-messages",
                    "-o",
                    str(parquet),
                ]
            ),
            0,
        )
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(parquet),
                    "--to",
                    "openai-messages",
                    "-o",
                    str(restored),
                ]
            ),
            0,
        )
        self.assertEqual([raw for _, raw in iter_raw(str(restored))], rows)
        self.assertEqual(
            self.run_cli(
                ["analyze", "validate", str(parquet), "--from", "openai-messages"]
            ),
            0,
        )
        self.assertEqual(
            self.run_cli(
                ["prepare", "sample", str(parquet), "-o", str(sample), "--count", "1"]
            ),
            0,
        )
        self.assertEqual(next(iter_raw(str(sample)))[1], rows[0])
        blocked = self.root / "sharegpt.jsonl"
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(parquet),
                    "--to",
                    "sharegpt",
                    "-o",
                    str(blocked),
                ]
            ),
            1,
        )
        self.assertFalse(blocked.exists())

    def test_legacy_profile_hint_does_not_prevent_parquet(self):
        profile = self.root / "profile.yaml"
        profile.write_text(
            "version: 1\nname: mapped-pair\ncontainer: JSONEachRow\nlayout: pair\nfields:\n  id: id\n  prompt: prompt\n  completion: answer\n",
            encoding="utf-8",
        )
        source = self.root / "source.parquet"
        output = self.root / "out.bin"
        with output_writer(str(source)) as writer:
            writer.write({"id": "a", "prompt": "Hi", "answer": "Hello", "extra": None})
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(source),
                    "--format-config",
                    str(profile),
                    "--from",
                    "mapped-pair",
                    "--to",
                    "openai-messages",
                    "--output-storage",
                    "parquet",
                    "-o",
                    str(output),
                ]
            ),
            0,
        )
        row = next(iter_raw(str(output), "parquet"))[1]
        self.assertEqual(row["messages"][1]["content"], "Hello")
        self.assertIsNone(row["extra"])

    def test_document_mapping_is_independent_of_storage(self):
        profile = self.root / "document.yaml"
        profile.write_text(
            "version: 1\nname: web-document\nlayout: document\nfields:\n  id: doc_id\n  text: page.body\n  metadata:\n    url: page.url\n",
            encoding="utf-8",
        )
        source = self.root / "web.json"
        source.write_text(
            json.dumps(
                [
                    {
                        "doc_id": "one",
                        "page": {
                            "body": "First document",
                            "url": "https://example.org",
                        },
                        "language": "en",
                    }
                ]
            ),
            encoding="utf-8",
        )
        parquet = self.root / "docs.parquet"
        restored = self.root / "restored.jsonl"
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(source),
                    "--format-config",
                    str(profile),
                    "--from",
                    "web-document",
                    "--to",
                    "document",
                    "-o",
                    str(parquet),
                ]
            ),
            0,
        )
        row = next(iter_raw(str(parquet)))[1]
        self.assertEqual(row["text"], "First document")
        self.assertEqual(row["metadata"]["url"], "https://example.org")
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(parquet),
                    "--format-config",
                    str(profile),
                    "--from",
                    "document",
                    "--to",
                    "web-document",
                    "-o",
                    str(restored),
                ]
            ),
            0,
        )
        row = next(iter_raw(str(restored)))[1]
        self.assertEqual(row["page"]["body"], "First document")
        self.assertEqual(row["page"]["url"], "https://example.org")
        self.assertEqual(row["language"], "en")

    def test_external_parquet_and_native_timestamps(self):
        source = self.root / "external.parquet"
        output = self.root / "out.parquet"
        rows = [
            {
                "text": "Document",
                "timestamp": datetime(2026, 10, 14),
                "_datautil_shape": "ordinary column",
            }
        ]
        pq.write_table(pa.Table.from_pylist(rows), source)
        self.assertEqual(next(iter_raw(str(source)))[1], rows[0])
        self.assertEqual(
            self.run_cli(["format", "convert", str(source), "-o", str(output)]), 0
        )
        self.assertEqual(next(iter_raw(str(output)))[1], rows[0])
        unsupported = self.root / "out.jsonl"
        unsupported.write_text("keep this file", encoding="utf-8")
        self.assertEqual(
            self.run_cli(["format", "convert", str(source), "-o", str(unsupported)]), 1
        )
        self.assertEqual(unsupported.read_text(encoding="utf-8"), "keep this file")

    def test_schema_failure_and_dry_run_leave_existing_output(self):
        source = self.root / "invalid.jsonl"
        target = self.root / "out.parquet"
        write_jsonl(source, [{"value": 1}, {"value": "incompatible"}])
        target.write_bytes(b"original")
        self.assertEqual(
            self.run_cli(["format", "convert", str(source), "-o", str(target)]), 1
        )
        self.assertEqual(target.read_bytes(), b"original")
        write_jsonl(source, [{"value": 1}])
        self.assertEqual(
            self.run_cli(
                ["format", "convert", str(source), "-o", str(target), "--dry-run"]
            ),
            0,
        )
        self.assertEqual(target.read_bytes(), b"original")
        self.assertFalse(list(self.root.glob(".out.parquet.*")))

    def test_skip_bad_jsonl_rows_and_reject_cross_kind_mapping(self):
        source = self.root / "source.jsonl"
        parquet = self.root / "valid.parquet"
        source.write_text(
            '{"id":"a","text":"first"}\ninvalid JSON\n{"id":"b","text":"second"}\n',
            encoding="utf-8",
        )
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(source),
                    "-o",
                    str(parquet),
                    "--on-error",
                    "skip",
                ]
            ),
            1,
        )
        self.assertEqual(len(list(iter_raw(str(parquet)))), 2)
        blocked = self.root / "conversation.jsonl"
        self.assertEqual(
            self.run_cli(
                [
                    "format",
                    "convert",
                    str(parquet),
                    "--from",
                    "document",
                    "--to",
                    "openai-messages",
                    "-o",
                    str(blocked),
                ]
            ),
            1,
        )
        self.assertFalse(blocked.exists())

    def test_empty_dataset_and_empty_object(self):
        source = self.root / "empty.jsonl"
        parquet = self.root / "empty.parquet"
        source.write_text("", encoding="utf-8")
        self.assertEqual(
            self.run_cli(["format", "convert", str(source), "-o", str(parquet)]), 0
        )
        self.assertEqual(list(iter_raw(str(parquet))), [])
        write_jsonl(source, [{}, {"empty": {}}])
        self.assertEqual(
            self.run_cli(["format", "convert", str(source), "-o", str(parquet)]), 0
        )
        self.assertEqual(
            [raw for _, raw in iter_raw(str(parquet))], [{}, {"empty": {}}]
        )

    def test_document_build_uses_storage_keys(self):
        source = self.root / "documents.data"
        source.write_text(
            json.dumps([{"id": "a", "text": "One document"}]), encoding="utf-8"
        )
        config = self.root / "recipe.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "sources": [
                        {"name": "one", "input": "documents.data", "storage": "json"}
                    ],
                    "mix": {"count": 1},
                    "output": "output.data",
                    "output_storage": "parquet",
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(self.run_cli(["prepare", "build", str(config)]), 0)
        self.assertEqual(
            next(iter_raw(str(self.root / "output.data"), "parquet"))[1]["text"],
            "One document",
        )

    def test_binary_standard_streams(self):
        source = self.root / "source.jsonl"
        output = self.root / "restored.jsonl"
        rows = [{"text": "stream", "metadata": {}}]
        write_jsonl(source, rows)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(datautil.__file__).resolve().parent.parent)
        encoded = subprocess.run(
            [
                sys.executable,
                "-m",
                "datautil",
                "format",
                "convert",
                str(source),
                "-o",
                "-",
                "--output-storage",
                "parquet",
            ],
            capture_output=True,
            env=env,
            check=True,
        )
        decoded = subprocess.run(
            [
                sys.executable,
                "-m",
                "datautil",
                "format",
                "convert",
                "-",
                "--input-storage",
                "parquet",
                "-o",
                str(output),
            ],
            input=encoded.stdout,
            capture_output=True,
            env=env,
        )
        self.assertEqual(decoded.returncode, 0, decoded.stderr.decode())
        self.assertEqual([raw for _, raw in iter_raw(str(output))], rows)
