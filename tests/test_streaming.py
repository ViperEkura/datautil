import io
import json
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

from support import DatautilTestCase, write_jsonl

from datautil.cli import main
from datautil.common.io import iter_raw
from datautil.conversation.reader import read_records
from datautil.conversation.sampling import sample_records


class StreamingTests(DatautilTestCase):
    def test_sample_preserves_seeded_selection_on_many_records(self):
        source = self.root / "source.jsonl"
        output = self.root / "sample.jsonl"
        write_jsonl(
            source,
            [
                {
                    "id": f"r{index:05d}",
                    "messages": [{"role": "user", "content": str(index)}],
                }
                for index in range(3000)
            ],
        )
        expected = [
            record.record_id
            for record in sample_records(list(read_records(str(source))), 37, 42)
        ]
        self.assertEqual(
            main(
                [
                    "sample",
                    str(source),
                    "-o",
                    str(output),
                    "--count",
                    "37",
                    "--seed",
                    "42",
                ]
            ),
            0,
        )
        self.assertEqual([item["id"] for _, item in iter_raw(str(output))], expected)

    def test_split_keeps_overlapping_groups_together_and_is_repeatable(self):
        source = self.root / "source.jsonl"
        groups = self.root / "groups.jsonl"
        train = self.root / "train.jsonl"
        test = self.root / "test.jsonl"
        write_jsonl(
            source,
            [
                {
                    "id": f"r{index:04d}",
                    "messages": [{"role": "user", "content": str(index)}],
                }
                for index in range(1200)
            ],
        )
        write_jsonl(
            groups,
            [
                {"group_id": "first", "record_ids": ["r0001", "r0002"], "pairs": []},
                {"group_id": "second", "record_ids": ["r0002", "r0003"], "pairs": []},
            ],
        )
        command = [
            "split",
            str(source),
            "--train",
            str(train),
            "--test",
            str(test),
            "--groups",
            str(groups),
            "--test-ratio",
            "0.2",
            "--seed",
            "42",
        ]
        self.assertEqual(main(command), 0)
        first_train = [row["id"] for _, row in iter_raw(str(train))]
        first_test = [row["id"] for _, row in iter_raw(str(test))]
        self.assertEqual(len(first_train) + len(first_test), 1200)
        self.assertTrue(
            {"r0001", "r0002", "r0003"} <= set(first_train)
            or {"r0001", "r0002", "r0003"} <= set(first_test)
        )
        self.assertEqual(main(command), 0)
        self.assertEqual([row["id"] for _, row in iter_raw(str(train))], first_train)
        self.assertEqual([row["id"] for _, row in iter_raw(str(test))], first_test)

    def test_dedup_apply_reads_groups_from_disk_and_limits_report_details(self):
        source = self.root / "source.jsonl"
        groups = self.root / "groups.jsonl"
        output = self.root / "unique.jsonl"
        report = self.root / "report.jsonl"
        write_jsonl(
            source,
            [
                {
                    "id": f"r{index:04d}",
                    "messages": [{"role": "user", "content": str(index)}],
                }
                for index in range(1200)
            ],
        )
        write_jsonl(
            groups,
            [
                {
                    "group_id": "all",
                    "record_ids": [f"r{index:04d}" for index in range(1200)],
                    "pairs": [],
                }
            ],
        )
        self.assertEqual(
            main(
                [
                    "dedup",
                    "apply",
                    str(source),
                    "--groups",
                    str(groups),
                    "-o",
                    str(output),
                    "--keep",
                    "first",
                    "--report",
                    str(report),
                ]
            ),
            0,
        )
        self.assertEqual([row["id"] for _, row in iter_raw(str(output))], ["r0000"])
        summary = next(iter_raw(str(report)))[1]
        self.assertEqual(summary["removed_count"], 1199)
        self.assertEqual(len(summary["decisions"]), 1000)
        self.assertTrue(summary["decisions_truncated"])

    def test_duplicate_ids_fail_before_outputs_are_written(self):
        source = self.root / "source.jsonl"
        output = self.root / "sample.jsonl"
        write_jsonl(
            source,
            [
                {"id": "same", "messages": [{"role": "user", "content": "one"}]},
                {"id": "same", "messages": [{"role": "user", "content": "two"}]},
            ],
        )
        self.assertEqual(
            main(["sample", str(source), "-o", str(output), "--count", "1"]), 1
        )
        self.assertFalse(output.exists())

    def test_validate_reports_later_rows_and_caps_issue_details(self):
        source = self.root / "source.jsonl"
        valid = {"id": "same", "messages": [{"role": "user", "content": "valid"}]}
        other = {
            "id": "other",
            "messages": [{"role": "user", "content": "still valid"}],
        }
        source.write_text(
            "\n".join(
                [
                    json.dumps(valid),
                    "{broken json",
                    json.dumps(valid),
                    json.dumps(
                        {
                            "id": "bad-role",
                            "messages": [{"role": "alien", "content": "x"}],
                        }
                    ),
                    json.dumps(other),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = main(
                [
                    "validate",
                    str(source),
                    "--from",
                    "openai-messages",
                    "--max-errors",
                    "2",
                ]
            )
        summary = json.loads(captured.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual(
            (summary["read_count"], summary["issue_count"], len(summary["issues"])),
            (5, 3, 2),
        )
        self.assertEqual(
            [item["code"] for item in summary["issues"]],
            ["invalid_json", "duplicate_record_id"],
        )
        self.assertTrue(summary["issues_truncated"])

    def test_stdin_records_without_ids_remain_deterministic(self):
        payload = (
            json.dumps({"messages": [{"role": "user", "content": "same input"}]}) + "\n"
        ).encode()
        outputs = [self.root / "first.jsonl", self.root / "second.jsonl"]
        for output in outputs:
            with patch.object(
                sys, "stdin", io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8")
            ):
                self.assertEqual(
                    main(
                        [
                            "sample",
                            "-",
                            "-o",
                            str(output),
                            "--from",
                            "openai-messages",
                            "--count",
                            "1",
                            "--seed",
                            "42",
                        ]
                    ),
                    0,
                )
        self.assertEqual(
            [item for _, item in iter_raw(str(outputs[0]))],
            [item for _, item in iter_raw(str(outputs[1]))],
        )
