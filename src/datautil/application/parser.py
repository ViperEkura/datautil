from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Sequence

from datautil import __version__
from datautil.application.routing import COMMAND_PATHS, normalize_argv


def _argument_templates() -> Dict[str, argparse.ArgumentParser]:
    templates: Dict[str, argparse.ArgumentParser] = {}

    def command(name: str, help: str) -> argparse.ArgumentParser:
        template = argparse.ArgumentParser(add_help=False, description=help)
        templates[name] = template
        return template

    def profile_option(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--format-config",
            action="append",
            default=[],
            metavar="YAML",
            help="Load a custom record-format profile; repeatable",
        )

    formats = command("formats", "List supported record formats")
    profile_option(formats)

    detection = command("detect", "Detect input structure")
    detection.add_argument("input")
    detection.add_argument("--sample", type=int, default=100)
    detection.add_argument(
        "--input-container", choices=["auto", "json", "jsonl"], default="auto"
    )
    profile_option(detection)

    inspection = command("inspect", "Summarize conversation records and issues")
    inspection.add_argument("input")
    inspection.add_argument("--sample", type=int, default=1000)
    inspection.add_argument("--from", dest="source_format")
    profile_option(inspection)

    analysis = command("analyze", "Summarize conversations, thinking, and tool chains")
    analysis.add_argument("input")
    analysis.add_argument(
        "-o", "--output", help="Write per-record JSONL; omit for summary only"
    )
    analysis.add_argument("--from", dest="source_format")
    analysis.add_argument("--report")
    analysis.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON summary instead of ASCII charts",
    )
    profile_option(analysis)

    validation = command(
        "validate", "Validate conversation source and target compatibility"
    )
    validation.add_argument("input")
    validation.add_argument("--from", dest="source_format")
    validation.add_argument("--to", dest="target_format")
    validation.add_argument(
        "--max-errors",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum issue details to retain; the full issue count is still reported",
    )
    validation.add_argument("--report")
    profile_option(validation)

    conversion = command("convert", "Convert conversation records")
    conversion.add_argument("input")
    conversion.add_argument("-o", "--output", required=True)
    conversion.add_argument("--from", dest="source_format")
    conversion.add_argument("--to", dest="target_format", required=True)
    conversion.add_argument(
        "--input-container", choices=["auto", "json", "jsonl"], default="auto"
    )
    conversion.add_argument(
        "--output-container", choices=["auto", "json", "jsonl"], default="auto"
    )
    conversion.add_argument("--on-error", choices=["fail", "skip"], default="fail")
    conversion.add_argument("--dry-run", action="store_true")
    conversion.add_argument("--report")
    profile_option(conversion)

    clean = command(
        "clean", "Apply ordered YAML conversation cleaning and filtering rules"
    )
    clean.add_argument("input")
    clean.add_argument("-o", "--output", required=True)
    clean.add_argument("--from", dest="source_format", required=True)
    rule_source = clean.add_mutually_exclusive_group(required=True)
    rule_source.add_argument("--rules", metavar="YAML")
    rule_source.add_argument("--rules-json", help=argparse.SUPPRESS)
    clean.add_argument("--report")
    profile_option(clean)

    scan = command("dedup scan", "Write conversation duplicate candidate groups")
    scan.add_argument("input")
    scan.add_argument("-o", "--output", required=True)
    scan.add_argument("--from", dest="source_format")
    scan.add_argument("--threshold", type=float, default=0.8)
    scan.add_argument(
        "--scope",
        choices=["conversation", "user"],
        default="conversation",
        help="Compare the full conversation or only user messages",
    )
    scan.add_argument("--report")
    profile_option(scan)
    apply = command("dedup apply", "Remove conversation duplicates in reviewed groups")
    apply.add_argument("input")
    apply.add_argument("--groups", required=True)
    apply.add_argument("-o", "--output", required=True)
    apply.add_argument("--from", dest="source_format")
    apply.add_argument("--keep", choices=["first"], required=True)
    apply.add_argument("--report")
    profile_option(apply)

    classify = command("classify", "Label conversations using an external model")
    classify.add_argument("input")
    classify.add_argument("-o", "--output", required=True)
    classify.add_argument("--from", dest="source_format")
    sampling = classify.add_mutually_exclusive_group(required=True)
    sampling.add_argument("--sample", type=int)
    sampling.add_argument("--all", action="store_true")
    classify.add_argument("--seed", type=int, default=42)
    classify.add_argument(
        "--labels", metavar="YAML", help="Custom label names and descriptions"
    )
    classify.add_argument("--taxonomy", dest="labels", help=argparse.SUPPRESS)
    classify.add_argument(
        "--type",
        dest="scheme",
        choices=["subject", "usage"],
        default="subject",
        help="Label by subject or use of the assistant",
    )
    classify.add_argument(
        "--scheme", dest="scheme", choices=["subject", "usage"], help=argparse.SUPPRESS
    )
    classify.add_argument(
        "--base-url",
        default=os.environ.get("DATAUTIL_API_BASE_URL", "https://api.openai.com/v1"),
    )
    classify.add_argument(
        "--model",
        default=os.environ.get("DATAUTIL_MODEL"),
        help="OpenAI-compatible model ID; or set DATAUTIL_MODEL",
    )
    classify.add_argument("--api-key-env", default="DATAUTIL_API_KEY")
    classify.add_argument("--report")
    profile_option(classify)

    stats = command("stats", "Summarize existing label artifacts without model calls")
    stats.add_argument("input")
    stats.add_argument(
        "--by",
        choices=["domain", "usage_topic"],
        help="Label field; inferred when unambiguous",
    )
    stats.add_argument("--report")

    sample = command("sample", "Select a deterministic conversation sample")
    sample.add_argument("input")
    sample.add_argument("-o", "--output", required=True)
    sample.add_argument("--from", dest="source_format")
    sample.add_argument("--count", type=int, required=True)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--report")
    profile_option(sample)

    split = command("split", "Create conversation train and test datasets")
    split.add_argument("input")
    split.add_argument("--train", required=True)
    split.add_argument("--test", required=True)
    split.add_argument("--from", dest="source_format")
    split.add_argument("--groups")
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--report")
    profile_option(split)

    pretrain_run = command(
        "pretrain run", "Map, filter, deduplicate, and mix documents"
    )
    pretrain_run.add_argument("config", metavar="YAML")

    run = command("run", "Run a YAML pipeline")
    run.add_argument("pipeline")
    return templates


class CommandParser(argparse.ArgumentParser):
    def parse_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        values = sys.argv[1:] if args is None else args
        return super().parse_args(normalize_argv(values), namespace)


def _parser() -> argparse.ArgumentParser:
    parser = CommandParser(
        prog="datautil",
        description="Inspect and prepare conversation and pretraining datasets",
        epilog="Flat commands remain compatible aliases. Use GROUP --help for actions.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    root = parser.add_subparsers(required=True, parser_class=argparse.ArgumentParser)
    group_help = {
        "format": "Discover and convert record formats",
        "analyze": "Inspect, validate, summarize, and label datasets",
        "prepare": "Clean, deduplicate, select, and build datasets",
        "pipeline": "Execute YAML workflows",
    }
    groups = {}
    for name, description in group_help.items():
        group = root.add_parser(name, help=description, description=description)
        groups[name] = group.add_subparsers(required=True)
    templates = _argument_templates()
    dedup = groups["prepare"].add_parser(
        "dedup", help="Review and apply conversation duplicates"
    )
    dedup_actions = dedup.add_subparsers(required=True)
    for operation, path in COMMAND_PATHS.items():
        parts = path.split()
        actions = dedup_actions if len(parts) == 3 else groups[parts[0]]
        template = templates[operation]
        leaf = actions.add_parser(
            parts[-1],
            parents=[template],
            help=template.description,
            description=template.description,
        )
        leaf.set_defaults(command=operation.split()[0])
        if operation.startswith("dedup "):
            leaf.set_defaults(dedup_command=operation.split()[1])
        elif operation == "pretrain run":
            leaf.set_defaults(pretrain_command="run")
    return parser
