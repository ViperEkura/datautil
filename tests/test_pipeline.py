from support import DatautilTestCase, write_jsonl

from datautil.cli import main
from datautil.common.errors import DataError
from datautil.common.io import iter_raw
from datautil.conversation.transform import validate_rules


class PipelineTests(DatautilTestCase):
    def test_pipeline(self):
        source = self.root / "source.jsonl"
        output = self.root / "out.jsonl"
        config = self.root / "pipeline.yaml"
        write_jsonl(source, [{"instruction": "Say hi", "input": "", "output": "Hi"}])
        config.write_text(
            f"steps:\n  - id: converted\n    command: convert\n    input: '{source.as_posix()}'\n    output: '{output.as_posix()}'\n    from: alpaca\n    to: openai-messages\n  - id: analyzed\n    command: analyze\n    input: '@converted'\n    from: openai-messages\n    output: analysis.jsonl\n",
            encoding="utf-8",
        )
        self.assertEqual(main(["run", str(config)]), 0)
        self.assertEqual(next(iter_raw(str(output)))[1]["messages"][0]["role"], "user")
        self.assertEqual(
            next(iter_raw(str(self.root / "analysis.jsonl")))[1]["message_count"], 2
        )

    def test_transform_rules_clean_filter_and_report(self):
        source = self.root / "source.jsonl"
        output = self.root / "cleaned.jsonl"
        report = self.root / "report.jsonl"
        rules = self.root / "rules.yaml"
        write_jsonl(
            source,
            [
                {
                    "id": "keep",
                    "messages": [
                        {"role": "user", "content": "  Hi  "},
                        {"role": "assistant", "content": "  Hello  "},
                        {"role": "assistant", "content": ""},
                    ],
                },
                {
                    "id": "drop",
                    "messages": [{"role": "user", "content": " Only one message "}],
                },
            ],
        )
        rules.write_text(
            "version: 1\nrules:\n  - op: trim_content\n  - op: drop_empty_messages\n  - op: keep_if\n    conditions:\n      min_messages: 2\n      roles_all: [user, assistant]\n",
            encoding="utf-8",
        )
        self.assertEqual(
            main(
                [
                    "transform",
                    str(source),
                    "-o",
                    str(output),
                    "--from",
                    "openai-messages",
                    "--rules",
                    str(rules),
                    "--report",
                    str(report),
                ]
            ),
            0,
        )
        items = [value for _, value in iter_raw(str(output))]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["messages"][0]["content"], "Hi")
        self.assertEqual(len(items[0]["messages"]), 2)
        self.assertEqual(next(iter_raw(str(report)))[1]["dropped_by_rule"], {"2": 1})

    def test_pipeline_inline_rules_then_sample(self):
        source = self.root / "source.jsonl"
        pipeline = self.root / "pipeline.yaml"
        write_jsonl(
            source,
            [
                {
                    "id": "a",
                    "messages": [
                        {"role": "user", "content": "  Hi "},
                        {"role": "assistant", "content": " Hello  "},
                    ],
                }
            ],
        )
        pipeline.write_text(
            "version: 1\nsteps:\n  - id: cleaned\n    command: transform\n    input: source.jsonl\n    output: cleaned.jsonl\n    from: openai-messages\n    rules:\n      - op: trim_content\n  - id: sample\n    command: sample\n    input: '@cleaned'\n    output: sampled.jsonl\n    from: openai-messages\n    count: 1\n",
            encoding="utf-8",
        )
        self.assertEqual(main(["run", str(pipeline)]), 0)
        self.assertEqual(
            next(iter_raw(str(self.root / "sampled.jsonl")))[1]["messages"][0][
                "content"
            ],
            "Hi",
        )

    def test_invalid_rule_types_report_data_error(self):
        for rule in (
            {"op": ["trim_content"]},
            {"op": "keep_if", "conditions": {"roles_any": [["user"]]}},
        ):
            with self.subTest(rule=rule), self.assertRaises(DataError):
                validate_rules({"version": 1, "rules": [rule]})
