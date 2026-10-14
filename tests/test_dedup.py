from support import DatautilTestCase, write_jsonl

from datautil.cli import main
from datautil.common.io import iter_raw
from datautil.conversation.dedup import dedup_apply, dedup_scan
from datautil.conversation.reader import read_records
from datautil.conversation.sampling import sample_records, split_records


class DedupTests(DatautilTestCase):
    def test_dedup_groups_and_split_do_not_leak(self):
        path = self.root / "input.jsonl"
        write_jsonl(
            path,
            [
                {
                    "id": "a",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Explain the Pythagorean theorem clearly",
                        },
                        {
                            "role": "assistant",
                            "content": "a squared plus b squared equals c squared",
                        },
                    ],
                },
                {
                    "id": "b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Explain the Pythagorean theorem clearly",
                        },
                        {
                            "role": "assistant",
                            "content": "a squared plus b squared equals c squared",
                        },
                    ],
                },
                {
                    "id": "c",
                    "messages": [
                        {"role": "user", "content": "How do I bake bread?"},
                        {"role": "assistant", "content": "Start with flour."},
                    ],
                },
            ],
        )
        records = list(read_records(str(path)))
        groups = dedup_scan(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["record_ids"]), {"a", "b"})
        retained, decisions = dedup_apply(records, groups)
        self.assertEqual({item.record_id for item in retained}, {"a", "c"})
        self.assertEqual(decisions, [{"removed": "b", "kept": "a"}])
        train, test = split_records(records, 0.5, 42, groups)
        train_ids, test_ids = {x.record_id for x in train}, {x.record_id for x in test}
        self.assertFalse(
            ("a" in train_ids and "b" in test_ids)
            or ("b" in train_ids and "a" in test_ids)
        )
        self.assertEqual(
            (train_ids, test_ids),
            (
                {x.record_id for x in split_records(records, 0.5, 42, groups)[0]},
                {x.record_id for x in split_records(records, 0.5, 42, groups)[1]},
            ),
        )
        self.assertEqual(
            [x.record_id for x in sample_records(records, 2, 42)],
            [x.record_id for x in sample_records(records, 2, 42)],
        )

    def test_dedup_exact_near_scope_and_unscannable(self):
        path = self.root / "input.jsonl"
        write_jsonl(
            path,
            [
                {
                    "id": "a",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Explain Python list comprehensions with several concrete examples and caveats",
                        },
                        {
                            "role": "assistant",
                            "content": "Use brackets with an expression and an iteration clause.",
                        },
                    ],
                },
                {
                    "id": "b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "ＥＸＰＬＡＩＮ  Python list comprehensions with several concrete examples and caveats",
                        },
                        {
                            "role": "assistant",
                            "content": "A completely different answer about gardening and soil.",
                        },
                    ],
                },
                {
                    "id": "c",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Explain Python list comprehension with several concrete examples and caveats",
                        },
                        {
                            "role": "assistant",
                            "content": "Another unrelated answer about cooking.",
                        },
                    ],
                },
                {
                    "id": "d",
                    "messages": [
                        {"role": "user", "content": "How do stars form?"},
                        {"role": "assistant", "content": "In gas clouds."},
                    ],
                },
                {
                    "id": "e",
                    "messages": [{"role": "assistant", "content": "No user prompt"}],
                },
                {
                    "id": "f",
                    "messages": [{"role": "assistant", "content": "No user prompt"}],
                },
            ],
        )
        records = list(read_records(str(path)))
        full_groups = dedup_scan(records)
        self.assertFalse(
            any({"a", "b"} <= set(group["record_ids"]) for group in full_groups)
        )
        stats = {}
        groups = dedup_scan(records, 0.7, scope="user", stats=stats)
        self.assertEqual(groups, dedup_scan(records, 0.7, scope="user"))
        self.assertEqual(groups[0]["record_ids"], ["a", "b", "c"])
        self.assertEqual(groups[0]["pairs"][0]["match"], "exact")
        self.assertTrue(any(pair["match"] == "near" for pair in groups[0]["pairs"]))
        self.assertEqual(stats["unscannable_count"], 2)
        self.assertEqual(stats["exact_pair_count"], 1)
        self.assertGreaterEqual(stats["near_pair_count"], 1)
        self.assertFalse(
            any(
                "d" in group["record_ids"]
                or "e" in group["record_ids"]
                or "f" in group["record_ids"]
                for group in groups
            )
        )
        old_groups = [{"group_id": "old", "record_ids": ["a", "b"], "pairs": []}]
        retained, _ = dedup_apply(records, old_groups)
        self.assertNotIn("b", {record.record_id for record in retained})
        train, test = split_records(records, 0.5, 42, old_groups)
        self.assertFalse(
            ({"a", "b"} & {record.record_id for record in train})
            and ({"a", "b"} & {record.record_id for record in test})
        )
        short_path = self.root / "short.jsonl"
        write_jsonl(
            short_path,
            [
                {"id": "short-a", "messages": [{"role": "user", "content": "a"}]},
                {"id": "short-b", "messages": [{"role": "user", "content": "Ａ"}]},
            ],
        )
        short_groups = dedup_scan(
            list(read_records(str(short_path))), 1.0, scope="user"
        )
        self.assertEqual(short_groups[0]["record_ids"], ["short-a", "short-b"])
        self.assertEqual(
            dedup_scan(list(read_records(str(short_path))), 0.99, scope="user")[0][
                "record_ids"
            ],
            ["short-a", "short-b"],
        )

    def test_dedup_cli_scope_report_and_pipeline(self):
        source = self.root / "source.jsonl"
        groups = self.root / "groups.jsonl"
        report = self.root / "report.jsonl"
        pipeline = self.root / "pipeline.yaml"
        write_jsonl(
            source,
            [
                {
                    "id": "a",
                    "messages": [
                        {"role": "user", "content": "Same prompt"},
                        {"role": "assistant", "content": "First answer"},
                    ],
                },
                {
                    "id": "b",
                    "messages": [
                        {"role": "user", "content": "same  prompt"},
                        {"role": "assistant", "content": "Second answer"},
                    ],
                },
            ],
        )
        self.assertEqual(
            main(
                [
                    "dedup",
                    "scan",
                    str(source),
                    "-o",
                    str(groups),
                    "--from",
                    "openai-messages",
                    "--scope",
                    "user",
                    "--report",
                    str(report),
                ]
            ),
            0,
        )
        self.assertEqual(next(iter_raw(str(groups)))[1]["pairs"][0]["match"], "exact")
        summary = next(iter_raw(str(report)))[1]
        self.assertEqual(
            (
                summary["scope"],
                summary["exact_pair_count"],
                summary["shingle_size"],
                summary["num_perm"],
            ),
            ("user", 1, 5, 128),
        )
        self.assertIn("datasketch_version", summary)
        pipeline.write_text(
            "steps:\n  - id: duplicates\n    command: prepare dedup scan\n    input: source.jsonl\n    output: pipeline-groups.jsonl\n    from: openai-messages\n    scope: user\n",
            encoding="utf-8",
        )
        self.assertEqual(main(["pipeline", "run", str(pipeline)]), 0)
        self.assertEqual(
            next(iter_raw(str(self.root / "pipeline-groups.jsonl")))[1]["record_ids"],
            ["a", "b"],
        )
