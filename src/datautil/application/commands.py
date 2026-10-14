from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from datautil import __version__
from datautil.application.conversion import convert, detect
from datautil.application.datasets import RecordFormats, load_dataset_profiles
from datautil.application.results import CommandResult, CommandSpec
from datautil.common.errors import DataError, Issue
from datautil.common.index import DiskIndex
from datautil.common.io import container_for, iter_raw, output_writer
from datautil.common.storage import STORAGE_BACKENDS
from datautil.conversation.analysis import analyze_dataset
from datautil.conversation.classification import OpenAICompatibleClassifier, load_labels
from datautil.conversation.dedup import dedup_apply_file, dedup_scan
from datautil.conversation.formats import use_profiles
from datautil.conversation.reader import read_records, replayable_input, source_format
from datautil.conversation.sampling import sample_file, split_file
from datautil.conversation.transform import (
    load_rules,
    transform_dataset,
    validate_rules,
)
from datautil.conversation.validation import inspect, validate
from datautil.pretrain.runner import run_recipe
from datautil.terminal import render_analysis


def _input_hash(path: str) -> str | None:
    if path == "-":
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_report(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "input",
            "command",
            "report",
            "output",
            "train",
            "test",
            "rules_json",
            "json",
        }
        and value is not None
    }
    outputs = {
        key: getattr(args, key)
        for key in ("output", "train", "test")
        if getattr(args, key, None)
    }
    result = {
        "version": __version__,
        "command": args.command,
        "input_sha256": _input_hash(args.input) if hasattr(args, "input") else None,
        "parameters": parameters,
        "outputs": outputs,
        **summary,
    }
    if getattr(args, "report", None):
        with output_writer(args.report, "jsonl") as writer:
            writer.write(result)
    return result


def _execute(
    args: argparse.Namespace,
    outcome: CommandResult,
    on_step: Callable[[CommandResult], None] | None = None,
    profiles: Any = None,
) -> int:
    formats = RecordFormats(profiles or [])

    def _report(
        args: argparse.Namespace, summary: dict[str, Any], *, to_stdout: bool = False
    ) -> None:
        outcome.summary = make_report(args, summary)
        outcome.messages.append(
            (
                "stdout" if to_stdout else "stderr",
                render_analysis(outcome.summary)
                if args.command == "analyze" and not args.json
                else json.dumps(outcome.summary, ensure_ascii=False, sort_keys=True),
            )
        )

    if args.command == "formats":
        outcome.summary = {
            "storage_backends": list(STORAGE_BACKENDS),
            "formats": [
                {
                    "name": name,
                    "read": True,
                    "write": True,
                    "container": "/".join(STORAGE_BACKENDS),
                    "record_kind": codec.record_kind,
                    "layout": getattr(codec.adapter, "layout", "built-in"),
                }
                for name, codec in formats.codecs.items()
            ],
        }
        outcome.messages.append(
            ("stdout", json.dumps(outcome.summary, ensure_ascii=False))
        )
        return 0
    if args.command == "detect":
        container = args.input_container
        if container == "auto" and len(args.format_config) == 1:
            container = container_for(
                args.input, "auto", getattr(profiles[0], "storage_hint", None)
            )
        result = detect(args.input, formats, container, args.sample)
        outcome.messages.append(("stdout", json.dumps(result, ensure_ascii=False)))
        outcome.summary = result
        return 0 if result["selected"] else 1
    if args.command == "inspect":
        outcome.summary = inspect(args.input, args.sample, args.source_format)
        outcome.messages.append(
            ("stdout", json.dumps(outcome.summary, ensure_ascii=False))
        )
        return 0
    if args.command == "analyze":
        result = analyze_dataset(args.input, args.output, args.source_format)
        _report(args, result, to_stdout=args.output is None)
        return 0
    if args.command == "validate":
        result = validate(
            args.input, args.source_format, args.target_format, "skip", args.max_errors
        )
        _report(args, result, to_stdout=True)
        return 1 if result["issue_count"] else 0
    if args.command == "convert":
        result = convert(
            args.input,
            args.output,
            args.source_format,
            args.target_format,
            formats=formats,
            on_error=args.on_error,
            dry_run=args.dry_run,
            input_container=args.input_container,
            output_container=args.output_container,
        )
        _report(args, result)
        return 1 if result["skipped_count"] else 0
    if args.command == "clean":
        try:
            document = json.loads(args.rules_json) if args.rules_json else None
        except json.JSONDecodeError as exc:
            raise DataError("invalid_rules", str(exc)) from exc
        rules = (
            validate_rules(document, "pipeline")
            if document is not None
            else load_rules(args.rules)
        )
        result = transform_dataset(args.input, args.output, args.source_format, rules)
        result["rule_count"] = len(rules)
        result["rules_sha256"] = hashlib.sha256(
            json.dumps(
                rules, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        _report(args, result)
        return 0
    if args.command == "dedup":
        if args.dedup_command == "scan":
            scan_stats: dict[str, Any] = {}
            with (
                replayable_input(args.input) as replay,
                tempfile.TemporaryDirectory(prefix="datautil-dedup-ids-") as directory,
            ):
                source = source_format(replay, args.source_format)
                index = DiskIndex(directory)
                try:
                    groups = dedup_scan(
                        read_records(
                            replay, source, id_index=index, location_alias=args.input
                        ),
                        args.threshold,
                        scope=args.scope,
                        stats=scan_stats,
                    )
                finally:
                    index.close()
            with output_writer(args.output, "jsonl") as writer:
                for group in groups:
                    writer.write(group)
            _report(
                args,
                {
                    "group_count": len(groups),
                    "output": args.output,
                    "threshold": args.threshold,
                    **scan_stats,
                },
            )
        else:
            _report(
                args,
                dedup_apply_file(
                    args.input, args.groups, args.output, args.source_format
                ),
            )
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
        labels_hash = hashlib.sha256(
            json.dumps(labels, sort_keys=True).encode()
        ).hexdigest()
        counts: Counter[str] = Counter()
        issues = []
        issue_count = 0
        sampled_count = 0
        with (
            classifier,
            replayable_input(args.input) as replay,
            tempfile.TemporaryDirectory(prefix="datautil-classify-") as directory,
        ):
            index = DiskIndex(directory)
            try:
                if args.sample is None:
                    chosen_indexes = None
                    records = read_records(
                        replay,
                        args.source_format,
                        id_index=index,
                        location_alias=args.input,
                    )
                else:
                    source = source_format(replay, args.source_format)
                    total = sum(
                        1
                        for _ in read_records(
                            replay, source, id_index=index, location_alias=args.input
                        )
                    )
                    chosen_indexes = set(
                        random.Random(args.seed).sample(
                            range(total), min(args.sample, total)
                        )
                    )
                    records = read_records(
                        replay,
                        source,
                        check_duplicate_ids=False,
                        location_alias=args.input,
                    )
                with output_writer(args.output, "jsonl") as writer:
                    for position, record in enumerate(records):
                        if (
                            chosen_indexes is not None
                            and position not in chosen_indexes
                        ):
                            continue
                        sampled_count += 1
                        try:
                            label = classifier.classify(record, labels, args.scheme)
                            label_field = (
                                "domain" if args.scheme == "subject" else "usage_topic"
                            )
                            writer.write(
                                {
                                    "record_id": record.record_id,
                                    label_field: label,
                                    "model": args.model,
                                    "labels_sha256": labels_hash,
                                }
                            )
                            counts[label] += 1
                        except DataError as exc:
                            issue_count += 1
                            if len(issues) < 1000:
                                issues.append(
                                    Issue(
                                        exc.code,
                                        str(exc),
                                        exc.location,
                                        record.record_id,
                                    ).as_dict()
                                )
            finally:
                index.close()
        distribution_key = "domains" if args.scheme == "subject" else "usage_topics"
        _report(
            args,
            {
                "sampled_count": sampled_count,
                "classified_count": sum(counts.values()),
                distribution_key: dict(counts),
                "issue_count": issue_count,
                "issues": issues,
                "issues_truncated": issue_count > len(issues),
                "output": args.output,
            },
        )
        return 1 if issue_count else 0
    if args.command == "stats":
        counts: Counter[str] = Counter()
        total = 0
        selected = args.by
        for location, value in iter_raw(args.input):
            if not isinstance(value, dict):
                raise DataError("invalid_labels", "Expected a label object", location)
            if selected is None:
                available = [
                    key
                    for key in ("domain", "usage_topic")
                    if isinstance(value.get(key), str)
                ]
                if len(available) != 1:
                    raise DataError(
                        "invalid_labels",
                        "Cannot infer label field; pass --by",
                        location,
                    )
                selected = available[0]
            if not isinstance(value.get(selected), str):
                raise DataError(
                    "invalid_labels", f"Expected {selected} label", location
                )
            counts[value[selected]] += 1
            total += 1
        if selected is None:
            raise DataError(
                "invalid_labels",
                "Cannot infer label field from an empty file; pass --by",
            )
        _report(
            args,
            {
                "by": selected,
                "count": total,
                selected + "s": dict(counts),
                "shares": {key: value / total for key, value in counts.items()}
                if total
                else {},
            },
            to_stdout=True,
        )
        return 0
    if args.command == "sample":
        _report(
            args,
            sample_file(
                args.input, args.output, args.source_format, args.count, args.seed
            ),
        )
        return 0
    if args.command == "split":
        _report(
            args,
            split_file(
                args.input,
                args.train,
                args.test,
                args.source_format,
                args.test_ratio,
                args.seed,
                args.groups,
            ),
        )
        return 0
    if args.command == "pretrain":
        if getattr(args, "pretrain_command", "run") == "index":
            from datautil.pretrain.manifest_index import run as run_manifest_index

            if (
                args.candidate_cap < 1
                or args.workers < 1
                or args.max_files < 0
                or args.stop_after_groups < 0
                or args.stop_after_tasks < 0
                or not 1 <= args.shards <= 256
            ):
                raise DataError(
                    "invalid_argument", "Index numeric options must be positive or zero"
                )
            args.manifest = Path(args.manifest)
            args.output = Path(args.output)
            try:
                result = run_manifest_index(args)
            except RuntimeError as exc:
                raise DataError("pretrain_index", str(exc)) from exc
            rows = sum(item["rows"] for item in result["categories"].values())
            kept = sum(item["kept"] for item in result["categories"].values())
            outcome.summary = {
                "status": result["status"],
                "rows": rows,
                "kept": kept,
                "output": str(args.output),
                "status_file": str(args.output / "status.json"),
            }
            outcome.messages.append(
                (
                    "stderr",
                    json.dumps(outcome.summary, ensure_ascii=False, sort_keys=True),
                )
            )
            return 0
        result = run_recipe(args.config)
        outcome.summary = result
        outcome.messages.append(
            ("stderr", json.dumps(result, ensure_ascii=False, sort_keys=True))
        )
        return 0
    if args.command == "run":
        from datautil.application.pipeline import PipelineRunner

        pipeline = PipelineRunner().run(args.pipeline, on_step=on_step)
        outcome.datasets.update(pipeline.datasets)
        outcome.reports.update(pipeline.reports)
        outcome.messages.extend(pipeline.messages)
        outcome.summary = {
            "datasets": dict(pipeline.datasets),
            "reports": dict(pipeline.reports),
        }
        return 0
    raise DataError("invalid_command", "Unknown command")


def execute(
    spec: CommandSpec, on_step: Callable[[CommandResult], None] | None = None
) -> CommandResult:
    args = argparse.Namespace(**dict(spec.options))
    args.command = spec.name
    outcome = CommandResult(0, {}, {}, {})
    profiles = load_dataset_profiles(getattr(args, "format_config", []))
    with use_profiles(
        [profile for profile in profiles if profile.record_kind == "conversation"]
    ):
        outcome.exit_code = _execute(args, outcome, on_step, profiles)
    for key in ("output", "train", "test"):
        value = getattr(args, key, None)
        if value and not (args.command == "analyze" and key == "output"):
            outcome.datasets[key] = value
    if getattr(args, "report", None):
        outcome.reports["report"] = args.report
    if args.command == "analyze" and getattr(args, "output", None):
        outcome.reports["output"] = args.output
    return outcome
