"""Grouped CLI execution, aliases, and pipeline artifact contracts."""

import io
from contextlib import redirect_stderr

import yaml
from support import DatautilTestCase, write_jsonl

from datautil.application.commands import execute
from datautil.application.parser import _parser
from datautil.application.results import CommandSpec
from datautil.cli import main
from datautil.common.io import iter_raw


class CommandGroupTests(DatautilTestCase):
    def test_grouped_pipeline_keeps_dataset_and_analysis_outputs_distinct(self):
        source = self.root / "source.jsonl"
        write_jsonl(
            source, [{"instruction": "  Hi  ", "input": "", "output": "  Hello  "}]
        )
        config = self.root / "pipeline.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "steps": [
                        {
                            "id": "converted",
                            "command": "format convert",
                            "input": "source.jsonl",
                            "output": "converted.jsonl",
                            "from": "alpaca",
                            "to": "openai-messages",
                        },
                        {
                            "id": "cleaned",
                            "command": "prepare clean",
                            "input": "@converted",
                            "output": "cleaned.jsonl",
                            "from": "openai-messages",
                            "rules": [{"op": "trim_content"}],
                            "report": "clean-report.jsonl",
                        },
                        {
                            "id": "summary",
                            "command": "analyze summary",
                            "input": "@cleaned",
                            "output": "analysis.jsonl",
                        },
                        {
                            "id": "sampled",
                            "command": "prepare sample",
                            "input": "@cleaned",
                            "output": "sampled.jsonl",
                            "count": 1,
                        },
                        {
                            "id": "validated",
                            "command": "analyze validate",
                            "input": "@sampled",
                            "report": "validation.jsonl",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        parsed = _parser().parse_args(["pipeline", "run", str(config)])
        result = execute(CommandSpec(parsed.command, vars(parsed)))
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.reports["summary"], str(self.root / "analysis.jsonl"))
        self.assertNotIn("summary", result.datasets)
        sample = next(iter_raw(str(self.root / "sampled.jsonl")))[1]
        self.assertEqual(
            [message["content"] for message in sample["messages"]], ["Hi", "Hello"]
        )
        analysis = next(iter_raw(str(self.root / "analysis.jsonl")))[1]
        self.assertEqual(analysis["message_count"], 2)
        clean_report = next(iter_raw(str(self.root / "clean-report.jsonl")))[1]
        self.assertEqual(clean_report["command"], "clean")
        self.assertEqual(clean_report["written_count"], 1)
        validation = next(iter_raw(str(self.root / "validation.jsonl")))[1]
        self.assertEqual(validation["issue_count"], 0)

    def test_flat_convert_alias_preserves_output_and_report_schema(self):
        source = self.root / "source.jsonl"
        output = self.root / "converted.jsonl"
        report = self.root / "report.jsonl"
        write_jsonl(source, [{"instruction": "Hi", "input": "", "output": "Hello"}])
        options = [
            str(source),
            "-o",
            str(output),
            "--from",
            "alpaca",
            "--to",
            "openai-messages",
            "--report",
            str(report),
        ]
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["format", "convert"] + options), 0)
            grouped_output, grouped_report = output.read_bytes(), report.read_bytes()
            self.assertEqual(main(["convert"] + options), 0)
        self.assertEqual(output.read_bytes(), grouped_output)
        self.assertEqual(report.read_bytes(), grouped_report)

    def test_grouped_classification_requires_explicit_scope(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as status:
            main(
                [
                    "analyze",
                    "classify",
                    "source.jsonl",
                    "-o",
                    "labels.jsonl",
                    "--model",
                    "mock",
                ]
            )
        self.assertEqual(status.exception.code, 2)
