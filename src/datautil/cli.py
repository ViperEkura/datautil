from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from typing import Any

from . import __version__
from .analysis import analyze_dataset
from .formats import ADAPTERS
from .io import iter_raw, output_writer
from .model import DataError, Issue
from .operations import (OpenAICompatibleClassifier, convert, dedup_scan,
                         detect, inspect, load_labels, read_records, validate)
from .profiles import load_profiles
from .rules import load_rules, transform_dataset, validate_rules
from .streaming import DiskIndex, dedup_apply_file, replayable_input, sample_file, source_format, split_file
from .terminal import render_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datautil", description="Inspect and prepare conversation datasets")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    def profile_option(command: argparse.ArgumentParser) -> None:
        command.add_argument("--format-config", action="append", default=[], metavar="YAML", help="Load a custom record-format profile; repeatable")

    formats = sub.add_parser("formats", help="List supported record formats")
    profile_option(formats)

    detection = sub.add_parser("detect", help="Detect input structure")
    detection.add_argument("input")
    detection.add_argument("--sample", type=int, default=100)
    detection.add_argument("--input-container", choices=["auto", "json", "jsonl"], default="auto")
    profile_option(detection)

    inspection = sub.add_parser("inspect", help="Summarize records and issues")
    inspection.add_argument("input")
    inspection.add_argument("--sample", type=int, default=1000)
    inspection.add_argument("--from", dest="source_format")
    profile_option(inspection)

    analysis = sub.add_parser("analyze", help="Summarize conversations, thinking, and tool chains")
    analysis.add_argument("input")
    analysis.add_argument("-o", "--output", help="Write per-record JSONL; omit for summary only")
    analysis.add_argument("--from", dest="source_format")
    analysis.add_argument("--report")
    analysis.add_argument("--json", action="store_true", help="Print the full JSON summary instead of ASCII charts")
    profile_option(analysis)

    validation = sub.add_parser("validate", help="Validate source and target compatibility")
    validation.add_argument("input")
    validation.add_argument("--from", dest="source_format")
    validation.add_argument("--to", dest="target_format")
    validation.add_argument("--max-errors", type=int, default=1000, metavar="N", help="Maximum issue details to retain; the full issue count is still reported")
    validation.add_argument("--report")
    profile_option(validation)

    conversion = sub.add_parser("convert", help="Convert conversation records")
    conversion.add_argument("input")
    conversion.add_argument("-o", "--output", required=True)
    conversion.add_argument("--from", dest="source_format")
    conversion.add_argument("--to", dest="target_format", required=True)
    conversion.add_argument("--input-container", choices=["auto", "json", "jsonl"], default="auto")
    conversion.add_argument("--output-container", choices=["auto", "json", "jsonl"], default="auto")
    conversion.add_argument("--on-error", choices=["fail", "skip"], default="fail")
    conversion.add_argument("--dry-run", action="store_true")
    conversion.add_argument("--report")
    profile_option(conversion)

    transform = sub.add_parser("transform", help="Apply ordered YAML cleaning and filtering rules")
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
    scan.add_argument("--scope", choices=["conversation", "user"], default="conversation", help="Compare the full conversation or only user messages")
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

    classify = sub.add_parser("classify", help="Label conversations using an external model")
    classify.add_argument("input")
    classify.add_argument("-o", "--output", required=True)
    classify.add_argument("--from", dest="source_format")
    sampling = classify.add_mutually_exclusive_group(required=True)
    sampling.add_argument("--sample", type=int)
    sampling.add_argument("--all", action="store_true")
    classify.add_argument("--seed", type=int, default=42)
    classify.add_argument("--labels", metavar="YAML", help="Custom label names and descriptions")
    classify.add_argument("--taxonomy", dest="labels", help=argparse.SUPPRESS)
    classify.add_argument("--type", dest="scheme", choices=["subject", "usage"], default="subject", help="Label by subject or use of the assistant")
    classify.add_argument("--scheme", dest="scheme", choices=["subject", "usage"], help=argparse.SUPPRESS)
    classify.add_argument("--base-url", default=os.environ.get("DATAUTIL_API_BASE_URL", "https://api.openai.com/v1"))
    classify.add_argument("--model", default=os.environ.get("DATAUTIL_MODEL"), help="OpenAI-compatible model ID; or set DATAUTIL_MODEL")
    classify.add_argument("--api-key-env", default="DATAUTIL_API_KEY")
    classify.add_argument("--report")
    profile_option(classify)

    stats = sub.add_parser("stats", help="Summarize existing label artifacts")
    stats.add_argument("input")
    stats.add_argument("--by", choices=["domain", "usage_topic"], help="Label field; inferred when unambiguous")
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

    run = sub.add_parser("run", help="Run a YAML pipeline")
    run.add_argument("pipeline")
    return parser


def _input_hash(path: str) -> str | None:
    if path == "-":
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _report(args: argparse.Namespace, result: dict[str, Any], *, to_stdout: bool = False) -> None:
    parameters = {key: value for key, value in vars(args).items() if key not in {"input", "command", "report", "output", "train", "test", "rules_json", "json"} and value is not None}
    outputs = {key: getattr(args, key) for key in ("output", "train", "test") if getattr(args, key, None)}
    result = {"version": __version__, "command": args.command, "input_sha256": _input_hash(args.input) if hasattr(args, "input") else None, "parameters": parameters, "outputs": outputs, **result}
    if getattr(args, "report", None):
        with output_writer(args.report, "jsonl") as writer:
            writer.write(result)
    display = render_analysis(result) if args.command == "analyze" and not args.json else json.dumps(result, ensure_ascii=False, sort_keys=True)
    print(display, file=sys.stdout if to_stdout else sys.stderr)


def _execute(args: argparse.Namespace) -> int:
    if args.command == "formats":
        print(json.dumps({"formats": [{"name": name, "read": True, "write": True, "container": getattr(adapter, "container", "json/jsonl"), "layout": getattr(adapter, "layout", "built-in")} for name, adapter in ADAPTERS.items()]}, ensure_ascii=False))
        return 0
    if args.command == "detect":
        container = args.input_container
        if container == "auto" and len(args.format_config) == 1:
            container = ADAPTERS[load_profiles(args.format_config)[0].name].container
        result = detect(args.input, container, args.sample)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["selected"] else 1
    if args.command == "inspect":
        print(json.dumps(inspect(args.input, args.sample, args.source_format), ensure_ascii=False))
        return 0
    if args.command == "analyze":
        result = analyze_dataset(args.input, args.output, args.source_format)
        _report(args, result, to_stdout=args.output is None)
        return 0
    if args.command == "validate":
        result = validate(args.input, args.source_format, args.target_format, "skip", args.max_errors)
        _report(args, result, to_stdout=True)
        return 1 if result["issue_count"] else 0
    if args.command == "convert":
        result = convert(args.input, args.output, args.source_format, args.target_format, on_error=args.on_error, dry_run=args.dry_run, input_container=args.input_container, output_container=args.output_container)
        _report(args, result)
        return 1 if result["skipped_count"] else 0
    if args.command == "transform":
        try:
            document = json.loads(args.rules_json) if args.rules_json else None
        except json.JSONDecodeError as exc:
            raise DataError("invalid_rules", str(exc)) from exc
        rules = validate_rules(document, "pipeline") if document is not None else load_rules(args.rules)
        result = transform_dataset(args.input, args.output, args.source_format, rules)
        result["rule_count"] = len(rules)
        result["rules_sha256"] = hashlib.sha256(json.dumps(rules, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        _report(args, result)
        return 0
    if args.command == "dedup":
        if args.dedup_command == "scan":
            scan_stats: dict[str, Any] = {}
            with replayable_input(args.input) as replay, tempfile.TemporaryDirectory(prefix="datautil-dedup-ids-") as directory:
                source = source_format(replay, args.source_format)
                index = DiskIndex(directory)
                try:
                    groups = dedup_scan(read_records(replay, source, id_index=index, location_alias=args.input), args.threshold, scope=args.scope, stats=scan_stats)
                finally:
                    index.close()
            with output_writer(args.output, "jsonl") as writer:
                for group in groups:
                    writer.write(group)
            _report(args, {"group_count": len(groups), "output": args.output, "threshold": args.threshold, **scan_stats})
        else:
            _report(args, dedup_apply_file(args.input, args.groups, args.output, args.source_format))
        return 0
    if args.command == "classify":
        if args.sample is not None and args.sample <= 0:
            raise DataError("invalid_argument", "sample must be positive")
        if not args.model:
            raise DataError("missing_model", "Set --model or DATAUTIL_MODEL")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise DataError("missing_api_key", f"Set {args.api_key_env}")
        labels = load_labels(args.labels, args.scheme)
        classifier = OpenAICompatibleClassifier(args.base_url, args.model, api_key)
        labels_hash = hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()
        counts: Counter[str] = Counter()
        issues = []
        issue_count = 0
        sampled_count = 0
        with replayable_input(args.input) as replay, tempfile.TemporaryDirectory(prefix="datautil-classify-") as directory:
            index = DiskIndex(directory)
            try:
                if args.sample is None:
                    chosen_indexes = None
                    records = read_records(replay, args.source_format, id_index=index, location_alias=args.input)
                else:
                    source = source_format(replay, args.source_format)
                    total = sum(1 for _ in read_records(replay, source, id_index=index, location_alias=args.input))
                    chosen_indexes = set(random.Random(args.seed).sample(range(total), min(args.sample, total)))
                    records = read_records(replay, source, check_duplicate_ids=False, location_alias=args.input)
                with output_writer(args.output, "jsonl") as writer:
                    for position, record in enumerate(records):
                        if chosen_indexes is not None and position not in chosen_indexes:
                            continue
                        sampled_count += 1
                        try:
                            label = classifier.classify(record, labels, args.scheme)
                            label_field = "domain" if args.scheme == "subject" else "usage_topic"
                            writer.write({"record_id": record.record_id, label_field: label, "model": args.model, "labels_sha256": labels_hash})
                            counts[label] += 1
                        except DataError as exc:
                            issue_count += 1
                            if len(issues) < 1000:
                                issues.append(Issue(exc.code, str(exc), exc.location, record.record_id).as_dict())
            finally:
                index.close()
        distribution_key = "domains" if args.scheme == "subject" else "usage_topics"
        _report(args, {"sampled_count": sampled_count, "classified_count": sum(counts.values()), distribution_key: dict(counts), "issue_count": issue_count, "issues": issues, "issues_truncated": issue_count > len(issues), "output": args.output})
        return 1 if issue_count else 0
    if args.command == "stats":
        counts: Counter[str] = Counter()
        total = 0
        selected = args.by
        for location, value in iter_raw(args.input):
            if not isinstance(value, dict):
                raise DataError("invalid_labels", "Expected a label object", location)
            if selected is None:
                available = [key for key in ("domain", "usage_topic") if isinstance(value.get(key), str)]
                if len(available) != 1:
                    raise DataError("invalid_labels", "Cannot infer label field; pass --by", location)
                selected = available[0]
            if not isinstance(value.get(selected), str):
                raise DataError("invalid_labels", f"Expected {selected} label", location)
            counts[value[selected]] += 1
            total += 1
        if selected is None:
            raise DataError("invalid_labels", "Cannot infer label field from an empty file; pass --by")
        _report(args, {"by": selected, "count": total, selected + "s": dict(counts), "shares": {key: value / total for key, value in counts.items()} if total else {}}, to_stdout=True)
        return 0
    if args.command == "sample":
        _report(args, sample_file(args.input, args.output, args.source_format, args.count, args.seed))
        return 0
    if args.command == "split":
        _report(args, split_file(args.input, args.train, args.test, args.source_format, args.test_ratio, args.seed, args.groups))
        return 0
    if args.command == "run":
        from .pipeline import PipelineRunner
        PipelineRunner().run(args.pipeline)
        return 0
    raise DataError("invalid_command", "Unknown command")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        for profile in load_profiles(getattr(args, "format_config", [])):
            if profile.name in {"openai-messages", "sharegpt", "alpaca"}:
                raise DataError("invalid_profile", f"Profile name {profile.name!r} is reserved")
            ADAPTERS[profile.name] = profile
        return _execute(args)
    except DataError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc), "location": exc.location}, ensure_ascii=False), file=sys.stderr)
        return 1
    except (OSError, UnicodeError) as exc:
        print(json.dumps({"error": "io_error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
