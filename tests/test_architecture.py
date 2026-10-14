"""Architecture contracts and command orchestration regressions."""

import ast
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from support import DatautilTestCase, write_jsonl

from datautil.application.commands import execute
from datautil.application.parser import _parser
from datautil.application.results import CommandSpec
from datautil.cli import main


class ArchitectureTests(DatautilTestCase):
    def test_profiles_are_isolated_between_commands(self):
        first = self.root / "first.yaml"
        second = self.root / "second.yaml"
        for path, name in ((first, "profile-one"), (second, "profile-two")):
            path.write_text(
                f"version: 1\nname: {name}\nlayout: pair\nfields:\n"
                "  prompt: prompt\n  completion: answer\n",
                encoding="utf-8",
            )
        for profile, present, absent in (
            (first, "profile-one", "profile-two"),
            (second, "profile-two", "profile-one"),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(["format", "list", "--format-config", str(profile)]), 0
                )
            names = {item["name"] for item in json.loads(output.getvalue())["formats"]}
            self.assertIn(present, names)
            self.assertNotIn(absent, names)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["formats"]), 0)
        names = {item["name"] for item in json.loads(output.getvalue())["formats"]}
        self.assertNotIn("profile-one", names)
        self.assertNotIn("profile-two", names)

    def test_dispatch_returns_command_result(self):
        source = self.root / "source.jsonl"
        output = self.root / "out.jsonl"
        write_jsonl(
            source, [{"id": "a", "messages": [{"role": "user", "content": "Hi"}]}]
        )
        parsed = _parser().parse_args(
            ["convert", str(source), "-o", str(output), "--to", "openai-messages"]
        )
        result = execute(CommandSpec(parsed.command, vars(parsed)))
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.summary["written_count"], 1)
        self.assertEqual(result.datasets["output"], str(output))

    def test_pipeline_stops_after_failure(self):
        source = self.root / "bad.jsonl"
        labels = self.root / "labels.jsonl"
        config = self.root / "pipeline.yaml"
        later = self.root / "later.jsonl"
        write_jsonl(
            source, [{"id": "bad", "messages": [{"role": "alien", "content": "Hi"}]}]
        )
        write_jsonl(labels, [{"record_id": "ok", "domain": "test"}])
        config.write_text(
            "steps:\n  - id: first\n    command: stats\n"
            "    input: labels.jsonl\n    by: domain\n"
            "  - id: check\n    command: validate\n"
            "    input: bad.jsonl\n    from: openai-messages\n"
            "  - id: later\n    command: convert\n    input: bad.jsonl\n"
            "    output: later.jsonl\n    from: openai-messages\n"
            "    to: openai-messages\n",
            encoding="utf-8",
        )
        output = io.StringIO()
        with redirect_stderr(io.StringIO()), redirect_stdout(output):
            self.assertEqual(main(["run", str(config)]), 1)
        self.assertIn('"command": "stats"', output.getvalue())
        self.assertIn('"command": "validate"', output.getvalue())
        self.assertFalse(later.exists())

    def test_dependency_direction(self):
        package = Path(__file__).resolve().parents[1] / "src" / "datautil"
        for domain in ("common", "conversation", "pretrain"):
            for path in (package / domain).glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        self.assertFalse(
                            module == "cli" or module.endswith(".cli"), path
                        )
                        if domain == "common":
                            self.assertFalse(
                                any(
                                    piece in module.split(".")
                                    for piece in ("conversation", "pretrain")
                                ),
                                path,
                            )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertFalse(
                                alias.name == "datautil.cli"
                                or alias.name.startswith("datautil.cli."),
                                path,
                            )
                            if domain == "common":
                                self.assertFalse(
                                    alias.name.startswith(
                                        ("datautil.conversation", "datautil.pretrain")
                                    ),
                                    path,
                                )
