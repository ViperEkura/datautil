"""Storage conversion and optional record-format conversion."""

import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from datautil.application.datasets import DatasetReader, DatasetWriter, RecordFormats
from datautil.common.errors import DataError, Issue
from datautil.common.storage import STORAGE_BACKENDS, storage_for


def detect(
    path: str, formats: RecordFormats, storage: str = "auto", sample: int = 100
) -> Dict[str, Any]:
    if sample <= 0:
        raise DataError("invalid_argument", "sample must be positive")
    selected_storage = storage_for(path, storage)
    counts = Counter()
    seen = 0
    for _, raw in STORAGE_BACKENDS[selected_storage].read(path):
        seen += 1
        counts.update(formats.matches(raw))
        if seen >= sample:
            break
    candidates = [
        {
            "format": name,
            "record_kind": codec.record_kind,
            "matches": counts[name],
            "sampled": seen,
            "confidence": counts[name] / seen if seen else 0,
        }
        for name, codec in formats.codecs.items()
    ]
    ranked = sorted(candidates, key=lambda value: (-value["matches"], value["format"]))
    selected = (
        ranked[0]["format"]
        if seen
        and ranked[0]["matches"] == seen
        and (len(ranked) < 2 or ranked[1]["matches"] == 0)
        else None
    )
    return {
        "container": selected_storage,
        "storage": selected_storage,
        "sampled": seen,
        "selected": selected,
        "candidates": ranked,
    }


def convert(
    path: str,
    output: str,
    source: Optional[str],
    target: Optional[str],
    *,
    formats: RecordFormats,
    on_error: str = "fail",
    dry_run: bool = False,
    input_container: str = "auto",
    output_container: str = "auto",
) -> Dict[str, Any]:
    source_codec = formats.get(source) if source else None
    target_codec = formats.get(target) if target else source_codec
    if (
        source_codec
        and target_codec
        and source_codec.record_kind != target_codec.record_kind
    ):
        raise DataError(
            "incompatible_record_kind",
            "An explicit semantic mapping is required between document and conversation records",
            path,
        )
    input_storage = storage_for(
        path,
        input_container,
        getattr(source_codec.adapter, "storage_hint", None) if source_codec else None,
    )
    output_storage = storage_for(
        output,
        output_container,
        getattr(target_codec.adapter, "storage_hint", None) if target_codec else None,
    )
    reader = DatasetReader(STORAGE_BACKENDS[input_storage], formats, source_codec)
    writer = DatasetWriter(STORAGE_BACKENDS[output_storage], target_codec)
    report: Dict[str, Any] = {
        "read_count": 0,
        "written_count": 0,
        "skipped_count": 0,
        "issues": [],
        "input_storage": input_storage,
        "output_storage": output_storage,
    }

    def process(append: Any) -> None:
        for location, raw in reader.rows(
            path, "yield" if on_error == "skip" else "raise"
        ):
            report["read_count"] += 1
            try:
                if isinstance(raw, DataError):
                    raise raw
                if target_codec is None:
                    append(raw)
                else:
                    codec, record = reader.parse(raw, location)
                    if codec.record_kind != target_codec.record_kind:
                        raise DataError(
                            "incompatible_record_kind",
                            "Document and conversation conversion needs an explicit semantic mapping",
                            location,
                        )
                    append(record)
                report["written_count"] += 1
            except DataError as exc:
                report["skipped_count"] += 1
                if len(report["issues"]) < 1000:
                    report["issues"].append(
                        Issue(exc.code, str(exc), exc.location or location).as_dict()
                    )
                if on_error == "fail":
                    raise

    if dry_run:
        with tempfile.TemporaryDirectory(prefix="datautil-convert-check-") as directory:
            with writer.open(str(Path(directory) / "output")) as append:
                process(append)
    else:
        with writer.open(output) as append:
            process(append)
    report["issues_truncated"] = report["skipped_count"] > len(report["issues"])
    return report
