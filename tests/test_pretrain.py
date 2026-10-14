import io
import json
import sqlite3
import struct
from contextlib import redirect_stderr

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from support import DatautilTestCase, write_jsonl

from datautil.cli import main
from datautil.common.io import iter_raw
from datautil.pretrain.dedup import DocumentDedupIndex
from datautil.pretrain.model import DocumentRecord
from datautil.pretrain.quality import document_quality_issues
from datautil.pretrain.sources import iter_documents


class PretrainTests(DatautilTestCase):
    def test_compact_manifest_index_resumes_and_prefers_l3(self):
        shared = "A shared document with enough English words to pass the minimum length filter."
        files = []
        for category, name, texts in (
            ("web2_en", "natural.parquet", [shared, "A separate natural web document with enough words to keep."]),
            ("web3_en_QA-Synthetic", "synthetic.parquet", [shared, "A separate synthetic document with enough words to keep."]),
        ):
            path = self.root / name
            pq.write_table(pa.table({"text": texts}), path)
            files.append({
                "category": category,
                "output_file": name,
                "bytes": path.stat().st_size,
            })
        manifest = self.root / "raw-manifest.json"
        manifest.write_text(json.dumps({"status": "downloaded", "files": files}), encoding="utf-8")
        output = self.root / "compact-index"
        command = [
            "prepare", "pretrain-index", "--profile", "astrai-100b",
            "--manifest", str(manifest), "--output", str(output),
        ]
        stale = self.root / "stale-index"
        (stale / "keep").mkdir(parents=True)
        (stale / "keep" / "0000.u64").write_bytes(struct.pack("<Q", 0))
        with redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(command[:-1] + [str(stale)]), 1)
        self.assertIn("no matching dedup.sqlite", error.getvalue())
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(command + ["--stop-after-groups", "1"]), 0)
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "partial")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(command), 0)
        result = json.loads((output / "manifest.json").read_text())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["categories"]["web2_en"]["exact"], 1)
        self.assertEqual(result["categories"]["web3_en_QA-Synthetic"]["kept"], 2)
        self.assertEqual(
            struct.unpack("<Q", (output / "keep" / "0001.u64").read_bytes()), (1,)
        )
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(command), 0)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), result)

    def _recipe(self, **overrides):
        first = self.root / "web.jsonl"
        second = self.root / "books.parquet"
        common = "A shared paragraph with enough characters to check paragraph deduplication."
        write_jsonl(
            first,
            [
                {
                    "doc_id": f"web-{i}",
                    "payload": {
                        "body": f"Web document {i} has useful text.\n\n{common}"
                    },
                    "url": f"https://example.org/{i}",
                }
                for i in range(8)
            ],
        )
        pq.write_table(
            pa.table(
                {
                    "id": [f"book-{i}" for i in range(8)] + ["book-copy"],
                    "text": [
                        f"Book document {i} has different content." for i in range(8)
                    ]
                    + ["Web document 0 has useful text.\n\n" + common],
                    "genre": ["fiction"] * 9,
                }
            ),
            second,
        )
        recipe = {
            "version": 1,
            "sources": [
                {
                    "name": "web",
                    "input": "web.jsonl",
                    "fields": {
                        "id": "doc_id",
                        "text": "payload.body",
                        "metadata": {"url": "url"},
                    },
                    "weight": 3,
                },
                {"name": "books", "input": "books.parquet", "weight": 1},
            ],
            "mix": {"count": 8, "seed": 42},
            "dedup": {
                "document": {"threshold": 1},
                "paragraph": {"threshold": 1, "min_chars": 30},
            },
            "quality": {"min_chars": 1},
            "output": "mixed.jsonl",
            "report": "report.jsonl",
            "decisions": "decisions.jsonl",
        }
        recipe.update(overrides)
        config = self.root / "pretrain.yaml"
        config.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        return config

    def test_map_parquet_mix_dedup_and_repeat(self):
        config = self._recipe()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["prepare", "build", str(config)]), 0)
        output = self.root / "mixed.jsonl"
        first = output.read_bytes()
        rows = [row for _, row in iter_raw(str(output))]
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["metadata"]["source"] == "web" for row in rows), 6)
        self.assertEqual(sum(row["metadata"]["source"] == "books" for row in rows), 2)
        self.assertTrue(all("text" in row and "id" in row for row in rows))
        self.assertTrue(
            any(row["metadata"].get("url", "").startswith("https://") for row in rows)
        )
        report = next(iter_raw(str(self.root / "report.jsonl")))[1]
        self.assertEqual(report["written_count"], 8)
        self.assertEqual(report["sources"][0]["selected"], 6)
        self.assertEqual(report["sources"][1]["selected"], 2)
        self.assertGreater(report["dedup"]["exact_count"], 0)
        decisions = [row for _, row in iter_raw(str(self.root / "decisions.jsonl"))]
        self.assertTrue(any(row["action"] == "remove_paragraph" for row in decisions))
        self.assertTrue(
            any(
                row["reason"] == "document_duplicate"
                for row in decisions
                if "reason" in row
            )
        )
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["pretrain", "run", str(config)]), 0)
        self.assertEqual(output.read_bytes(), first)

    def test_parquet_output_round_trip(self):
        config = self._recipe(output="mixed.parquet")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["pretrain", "run", str(config)]), 0)
        records = list(
            iter_documents(
                str(self.root / "mixed.parquet"), "mixed", {"id": "id", "text": "text"}
            )
        )
        self.assertEqual(len(records), 8)
        self.assertTrue(
            all(
                isinstance(record, DocumentRecord)
                and record.metadata["source"] in {"web", "books"}
                for record in records
            )
        )

    def test_invalid_mapping_fails_before_output(self):
        config = self._recipe(
            sources=[
                {"name": "bad", "input": "web.jsonl", "fields": {"text": "missing"}}
            ]
        )
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["pretrain", "run", str(config)]), 1)
        self.assertIn("invalid_document", errors.getvalue())
        self.assertFalse((self.root / "mixed.jsonl").exists())

    def test_invalid_parquet_reports_data_error(self):
        (self.root / "broken.parquet").write_bytes(b"not a parquet file")
        config = self._recipe(sources=[{"name": "broken", "input": "broken.parquet"}])
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["pretrain", "run", str(config)]), 1)
        self.assertIn("invalid_parquet", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        self.assertFalse((self.root / "mixed.jsonl").exists())

    def test_quality_presets_require_explicit_english_and_report_measures(self):
        config = self._recipe(quality={"presets": ["gopher", "c4"]})
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["pretrain", "run", str(config)]), 1)
        self.assertIn("quality.language", errors.getvalue())
        record = DocumentRecord("a", "Short text.", "row 1")
        issues = document_quality_issues(
            record, {"language": "en", "presets": ["gopher", "c4"]}
        )
        self.assertIn("gopher.word_count_min", {item["rule"] for item in issues})
        self.assertIn("c4.sentence_count", {item["rule"] for item in issues})

    def test_mix_redistributes_exhausted_source_and_reports_shortfall(self):
        config = self._recipe(mix={"count": 100, "seed": 42})
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["pretrain", "run", str(config)]), 0)
        report = next(iter_raw(str(self.root / "report.jsonl")))[1]
        self.assertEqual(report["requested_count"], 100)
        self.assertEqual(
            report["written_count"], sum(item["selected"] for item in report["sources"])
        )
        self.assertGreater(report["shortfall"], 0)

    def test_disk_lsh_confirms_near_match(self):
        connection = sqlite3.connect(str(self.root / "near.sqlite3"))
        try:
            index = DocumentDedupIndex(connection, threshold=0.7)
            first = " ".join(f"distinct-section-{number:03d}" for number in range(70))
            second = first.replace("distinct-section-035", "different-section-035")
            self.assertIsNone(index.check_or_add(first, "document", "first"))
            match = index.check_or_add(second, "document", "second")
            self.assertIsNotNone(match)
            self.assertEqual((match.kept_id, match.kind), ("first", "near"))
            self.assertGreaterEqual(match.similarity, 0.7)
            self.assertGreater(index.candidate_count, 0)
        finally:
            connection.close()

    def test_duplicate_document_ids_fail_without_output(self):
        source = self.root / "duplicates.jsonl"
        write_jsonl(
            source, [{"id": "same", "text": "First"}, {"id": "same", "text": "Second"}]
        )
        config = self.root / "duplicates.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "sources": [{"name": "one", "input": "duplicates.jsonl"}],
                    "mix": {"count": 1},
                    "output": "out.jsonl",
                }
            ),
            encoding="utf-8",
        )
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["pretrain", "run", str(config)]), 1)
        self.assertIn("duplicate_record_id", errors.getvalue())
        self.assertFalse((self.root / "out.jsonl").exists())

    def test_post_dedup_quality_rejects_and_rolls_back_indexes(self):
        shared = "S" * 100
        tail = "UniqueTail"
        source = self.root / "quality.jsonl"
        write_jsonl(
            source,
            [
                {"id": "first", "text": shared},
                {"id": "rejected", "text": shared + "\n\n" + tail},
                {"id": "rejected-again", "text": shared + "\n\n" + tail},
                {"id": "last", "text": tail + "\n\n" + "T" * 100},
            ],
        )
        config = self.root / "quality.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "sources": [{"name": "web", "input": "quality.jsonl"}],
                    "mix": {"count": 4},
                    "quality": {"min_chars": 80},
                    "dedup": {
                        "document": {"threshold": 1},
                        "paragraph": {"threshold": 1, "min_chars": 1},
                    },
                    "output": "quality-output.jsonl",
                    "report": "quality-report.jsonl",
                    "decisions": "quality-decisions.jsonl",
                }
            ),
            encoding="utf-8",
        )
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["pretrain", "run", str(config)]), 0)
        rows = [row for _, row in iter_raw(str(self.root / "quality-output.jsonl"))]
        self.assertEqual({row["id"] for row in rows}, {"first", "last"})
        self.assertIn(tail, next(row["text"] for row in rows if row["id"] == "last"))
        report = next(iter_raw(str(self.root / "quality-report.jsonl")))[1]
        self.assertEqual(report["sources"][0]["quality_rejected_after_dedup"], 2)
        self.assertEqual(report["sources"][0].get("document_duplicate", 0), 0)
        decisions = [
            row for _, row in iter_raw(str(self.root / "quality-decisions.jsonl"))
        ]
        rejected = [
            row
            for row in decisions
            if row.get("reason") == "quality_after_paragraph_dedup"
        ]
        self.assertEqual(
            {row["record_id"] for row in rejected}, {"rejected", "rejected-again"}
        )
        self.assertTrue(
            all(row["checks"][0]["rule"] == "min_chars" for row in rejected)
        )

    def test_dedup_normalization_can_preserve_code_case(self):
        exact = sqlite3.connect(str(self.root / "case.sqlite3"))
        folded = sqlite3.connect(str(self.root / "folded.sqlite3"))
        try:
            case_index = DocumentDedupIndex(exact, threshold=1, normalization="nfkc")
            fold_index = DocumentDedupIndex(
                folded, threshold=1, normalization="nfkc_casefold"
            )
            self.assertIsNone(case_index.check_or_add("Value = 1", "document", "one"))
            self.assertIsNone(case_index.check_or_add("value = 1", "document", "two"))
            self.assertIsNone(fold_index.check_or_add("Value = 1", "document", "one"))
            self.assertEqual(
                fold_index.check_or_add("value = 1", "document", "two").kind, "exact"
            )
        finally:
            exact.close()
            folded.close()
