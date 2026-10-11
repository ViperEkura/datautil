from __future__ import annotations

import argparse
import os

from datautil import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datautil",
        description="Inspect and prepare conversation and pretraining datasets",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def profile_option(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--format-config",
            action="append",
            default=[],
            metavar="YAML",
            help="Load a custom record-format profile; repeatable",
        )

    formats = sub.add_parser("formats", help="List supported record formats")
    profile_option(formats)

    detection = sub.add_parser("detect", help="Detect input structure")
    detection.add_argument("input")
    detection.add_argument("--sample", type=int, default=100)
    detection.add_argument(
        "--input-container", choices=["auto", "json", "jsonl"], default="auto"
    )
    profile_option(detection)

    inspection = sub.add_parser("inspect", help="Summarize records and issues")
    inspection.add_argument("input")
    inspection.add_argument("--sample", type=int, default=1000)
    inspection.add_argument("--from", dest="source_format")
    profile_option(inspection)

    analysis = sub.add_parser(
        "analyze", help="Summarize conversations, thinking, and tool chains"
    )
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

    validation = sub.add_parser(
        "validate", help="Validate source and target compatibility"
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

    conversion = sub.add_parser("convert", help="Convert conversation records")
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

    transform = sub.add_parser(
        "transform", help="Apply ordered YAML cleaning and filtering rules"
    )
    transform.add_argument("input")
    transform.add_argument("-o", "--output", required=True)
    transform.add_argument("--from", dest="source_format", required=True)
    rule_source = transform.add_mutually_exclusive_group(required=True)
    rule_source.add_argument("--rules", metavar="YAML")
    rule_source.add_argument("--rules-json", help=argparse.SUPPRESS)
    transform.add_argument("--report")
    profile_option(transform)

    dedup = sub.add_parser("dedup", help="Find and apply near-duplicate groups")
    dedup_sub = dedup.add_subparsers(dest="dedup_command", required=True)
    scan = dedup_sub.add_parser("scan", help="Write candidate groups")
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
    apply = dedup_sub.add_parser("apply", help="Remove duplicates in reviewed groups")
    apply.add_argument("input")
    apply.add_argument("--groups", required=True)
    apply.add_argument("-o", "--output", required=True)
    apply.add_argument("--from", dest="source_format")
    apply.add_argument("--keep", choices=["first"], required=True)
    apply.add_argument("--report")
    profile_option(apply)

    classify = sub.add_parser(
        "classify", help="Label conversations using an external model"
    )
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

    stats = sub.add_parser("stats", help="Summarize existing label artifacts")
    stats.add_argument("input")
    stats.add_argument(
        "--by",
        choices=["domain", "usage_topic"],
        help="Label field; inferred when unambiguous",
    )
    stats.add_argument("--report")

    sample = sub.add_parser("sample", help="Select a deterministic sample")
    sample.add_argument("input")
    sample.add_argument("-o", "--output", required=True)
    sample.add_argument("--from", dest="source_format")
    sample.add_argument("--count", type=int, required=True)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--report")
    profile_option(sample)

    split = sub.add_parser("split", help="Create train and test datasets")
    split.add_argument("input")
    split.add_argument("--train", required=True)
    split.add_argument("--test", required=True)
    split.add_argument("--from", dest="source_format")
    split.add_argument("--groups")
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--report")
    profile_option(split)

    pretrain = sub.add_parser(
        "pretrain", help="Prepare a document corpus from a YAML recipe"
    )
    pretrain_sub = pretrain.add_subparsers(dest="pretrain_command", required=True)
    pretrain_run = pretrain_sub.add_parser(
        "run", help="Map, filter, deduplicate, and mix documents"
    )
    pretrain_run.add_argument("config", metavar="YAML")

    run = sub.add_parser("run", help="Run a YAML pipeline")
    run.add_argument("pipeline")
    return parser
