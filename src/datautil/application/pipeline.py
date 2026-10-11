from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from datautil.application.commands import execute
from datautil.application.parser import _parser
from datautil.application.results import CommandResult, CommandSpec
from datautil.common.errors import DataError


@dataclass
class OperationResult:
    datasets: dict[str, str] = field(default_factory=dict)
    reports: dict[str, str] = field(default_factory=dict)
    messages: list[tuple[str, str]] = field(default_factory=list)


class Operation(Protocol):
    def run(self, input_path: str, context: dict[str, Any]) -> OperationResult: ...


class PipelineRunner:
    """Run a reproducible sequence of CLI operations from a YAML file.

    Each step has an id, command, input, and command-specific options. An input
    beginning with @ references a previous step's primary dataset output.
    """

    def run(
        self, config_path: str, on_step: Callable[[CommandResult], None] | None = None
    ) -> OperationResult:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("steps"), list)
            or config.get("version", 1) != 1
        ):
            raise DataError("invalid_pipeline", "Pipeline needs a steps list")
        result = OperationResult()
        used: set[str] = set()
        base = Path(config_path).resolve().parent
        for step in config["steps"]:
            if (
                not isinstance(step, dict)
                or not isinstance(step.get("id"), str)
                or not isinstance(step.get("command"), str)
            ):
                raise DataError("invalid_pipeline", "Each step needs id and command")
            identifier = step["id"]
            if identifier in used:
                raise DataError("invalid_pipeline", f"Duplicate step ID {identifier}")
            used.add(identifier)
            command = step["command"]
            if command not in {
                "validate",
                "convert",
                "transform",
                "analyze",
                "dedup scan",
                "dedup apply",
                "classify",
                "stats",
                "sample",
                "split",
            }:
                raise DataError(
                    "invalid_pipeline", f"Unsupported pipeline command {command}"
                )
            args = command.split()
            for key, value in step.items():
                if key in {"id", "command"}:
                    continue
                if isinstance(value, str) and value.startswith("@"):
                    reference = value[1:]
                    if reference not in result.datasets:
                        raise DataError(
                            "invalid_pipeline",
                            f"Unknown dataset reference @{reference}",
                        )
                    value = result.datasets[reference]
                elif (
                    key
                    in {
                        "input",
                        "output",
                        "train",
                        "test",
                        "groups",
                        "report",
                        "labels",
                        "taxonomy",
                        "rules",
                    }
                    and isinstance(value, str)
                    and value != "-"
                    and not Path(value).is_absolute()
                ):
                    value = str(base / value)
                elif key == "format_config" and isinstance(value, list):
                    value = [
                        str(base / item) if not Path(item).is_absolute() else item
                        for item in value
                    ]
                elif (
                    key == "format_config"
                    and isinstance(value, str)
                    and not Path(value).is_absolute()
                ):
                    value = str(base / value)
                flag = "--" + key.replace("_", "-")
                if key == "input":
                    args.append(str(value))
                elif (
                    key == "rules"
                    and command == "transform"
                    and isinstance(value, list)
                ):
                    args.extend(
                        ["--rules-json", json.dumps({"version": 1, "rules": value})]
                    )
                elif isinstance(value, bool):
                    if value:
                        args.append(flag)
                elif isinstance(value, list):
                    for item in value:
                        args.extend([flag, str(item)])
                elif value is not None:
                    args.extend([flag, str(value)])
            parsed = _parser().parse_args(args)
            command_result = execute(CommandSpec(parsed.command, vars(parsed)))
            if on_step is None:
                result.messages.extend(command_result.messages)
            else:
                on_step(command_result)
            status = command_result.exit_code
            if status != 0:
                raise DataError(
                    "pipeline_step_failed", f"Step {identifier} exited with {status}"
                )
            if command == "analyze" and "output" in step:
                report_path = step["output"]
                result.reports[identifier] = (
                    str(base / report_path)
                    if not Path(report_path).is_absolute()
                    else str(report_path)
                )
            elif "output" in step:
                result.datasets[identifier] = (
                    str(base / step["output"])
                    if not Path(step["output"]).is_absolute()
                    else str(step["output"])
                )
            elif "train" in step:
                result.datasets[identifier + ".train"] = (
                    str(base / step["train"])
                    if not Path(step["train"]).is_absolute()
                    else str(step["train"])
                )
                result.datasets[identifier + ".test"] = (
                    str(base / step["test"])
                    if not Path(step["test"]).is_absolute()
                    else str(step["test"])
                )
            if "report" in step:
                result.reports[identifier] = (
                    str(base / step["report"])
                    if not Path(step["report"]).is_absolute()
                    else str(step["report"])
                )
        return result
