"""Observable thinking and tool-trajectory measurements for conversations."""

from __future__ import annotations

import re
import math
from collections import Counter
from contextlib import nullcontext
from typing import Any

from .io import output_writer
from .formats import ADAPTERS
from .model import DatasetRecord
from .operations import read_records


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ROLES = ("system", "developer", "user", "assistant", "tool")


def _quantile(items: list[tuple[int, int]], proportion: float) -> float:
    total = sum(count for _, count in items)
    position = proportion * (total - 1)

    def at(rank: int) -> int:
        seen = 0
        for value, count in items:
            seen += count
            if rank < seen:
                return value
        raise ValueError("rank exceeds histogram")

    lower = math.floor(position)
    fraction = position - lower
    return at(lower) * (1 - fraction) + at(math.ceil(position)) * fraction


def _histogram(values: Counter[int], *, separate_zero: bool = False, max_bins: int = 12) -> tuple[dict[str, int], str]:
    """Use Freedman-Diaconis widths, Sturges for zero IQR, and a display cap."""
    result = {"0": values[0]} if separate_zero and values[0] else {}
    items = sorted((value, count) for value, count in values.items() if count and (value != 0 or not separate_zero))
    if not items:
        return result, "empty"
    minimum, maximum = items[0][0], items[-1][0]
    if minimum == maximum:
        result[str(minimum)] = sum(count for _, count in items)
        return result, "single-value"
    total = sum(count for _, count in items)
    span = maximum - minimum + 1
    iqr = _quantile(items, 0.75) - _quantile(items, 0.25)
    if iqr > 0:
        width = max(1, math.ceil(2 * iqr / total ** (1 / 3)))
        method = "freedman-diaconis"
    else:
        width = max(1, math.ceil(span / (math.ceil(math.log2(total)) + 1)))
        method = "sturges-fallback"
    width = max(width, math.ceil(span / max_bins))
    bins: Counter[tuple[int, int]] = Counter()
    for value, count in items:
        lower = minimum + (value - minimum) // width * width
        upper = min(maximum, lower + width - 1)
        bins[(lower, upper)] += count
    for (lower, upper), count in sorted(bins.items()):
        result[str(lower) if lower == upper else f"{lower}-{upper}"] = count
    return result, method


def analyze_record(record: DatasetRecord) -> dict[str, Any]:
    thinking_chars = thinking_messages = malformed_thinking_messages = 0
    field_messages = tag_messages = duplicate_sources = 0
    call_ids: list[str] = []
    result_ids: set[str] = set()
    chain_lengths: list[int] = []
    current_chain = max_parallel_calls = 0
    role_counts: Counter[str] = Counter(message.role for message in record.conversation.messages)
    content_chars = sum(len(message.content) for message in record.conversation.messages)

    for message in record.conversation.messages:
        if message.role == "assistant":
            blocks = _THINK_BLOCK.findall(message.content)
            if message.content.count("<think>") != len(blocks) or message.content.count("</think>") != len(blocks):
                malformed_thinking_messages += 1
            if message.thinking is not None and blocks:
                duplicate_sources += 1
            if message.thinking:
                thinking_chars += len(message.thinking)
                thinking_messages += 1
                field_messages += 1
            elif blocks:
                thinking_chars += sum(len(block) for block in blocks)
                thinking_messages += 1
                tag_messages += 1

            if message.tool_calls:
                current_chain += 1
                call_ids.extend(call.id for call in message.tool_calls)
                max_parallel_calls = max(max_parallel_calls, len(message.tool_calls))
                continue

        if message.role == "tool" and message.tool_call_id:
            result_ids.add(message.tool_call_id)
            continue
        if current_chain:
            chain_lengths.append(current_chain)
            current_chain = 0

    if current_chain:
        chain_lengths.append(current_chain)
    sources = [source for source, count in (("field", field_messages), ("tag", tag_messages)) if count]
    return {
        "record_id": record.record_id,
        "message_count": len(record.conversation.messages),
        "role_counts": {role: role_counts[role] for role in _ROLES},
        "content_chars": content_chars,
        "user_chars": sum(len(message.content) for message in record.conversation.messages if message.role == "user"),
        "assistant_chars": sum(len(message.content) for message in record.conversation.messages if message.role == "assistant"),
        "thinking_observed": bool(thinking_messages),
        "thinking_chars": thinking_chars,
        "thinking_messages": thinking_messages,
        "thinking_source": "+".join(sources) if sources else "none",
        "malformed_thinking_messages": malformed_thinking_messages,
        "duplicate_thinking_sources": duplicate_sources,
        "tool_call_count": len(call_ids),
        "tool_rounds": sum(chain_lengths),
        "tool_chain_lengths": chain_lengths,
        "max_tool_chain_length": max(chain_lengths, default=0),
        "max_parallel_calls": max_parallel_calls,
        "unanswered_tool_calls": sum(call_id not in result_ids for call_id in call_ids),
    }


def analyze_dataset(input_path: str, output_path: str | None, source_format: str | None) -> dict[str, Any]:
    observed_format = source_format
    count = thinking_records = thinking_chars = tool_records = tool_calls = unanswered_calls = malformed = 0
    messages_total = content_chars_total = 0
    chain_histogram: Counter[int] = Counter()
    max_chain_histogram: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    message_lengths: Counter[int] = Counter()
    content_lengths: Counter[int] = Counter()
    thinking_lengths: Counter[int] = Counter()
    with (output_writer(output_path, "jsonl") if output_path else nullcontext(None)) as writer:
        for record in read_records(input_path, source_format):
            observed_format = record.source_format
            row = analyze_record(record)
            if writer is not None:
                writer.write(row)
            count += 1
            messages_total += row["message_count"]
            content_chars_total += row["content_chars"]
            roles.update(row["role_counts"])
            message_lengths[row["message_count"]] += 1
            content_lengths[row["content_chars"]] += 1
            thinking_lengths[row["thinking_chars"]] += 1
            thinking_records += int(row["thinking_observed"])
            thinking_chars += row["thinking_chars"]
            tool_records += int(row["tool_call_count"] > 0)
            tool_calls += row["tool_call_count"]
            unanswered_calls += row["unanswered_tool_calls"]
            malformed += row["malformed_thinking_messages"]
            source_counts[row["thinking_source"]] += 1
            max_chain_histogram[row["max_tool_chain_length"]] += 1
            chain_histogram.update(row["tool_chain_lengths"])
    adapter = ADAPTERS.get(observed_format) if observed_format else None
    source_labels: Counter[str] = Counter()
    for role, role_count in roles.items():
        if role_count:
            source_labels[adapter.display_label(role) if adapter else role] += role_count
    message_distribution, message_method = _histogram(message_lengths)
    content_distribution, content_method = _histogram(content_lengths, separate_zero=True)
    thinking_distribution, thinking_method = _histogram(thinking_lengths, separate_zero=True)
    chain_distribution, chain_method = _histogram(max_chain_histogram, separate_zero=True)
    return {
        "source_format": observed_format,
        "source_label_heading": adapter.label_heading if adapter else "Message roles",
        "source_label_counts": dict(source_labels),
        "record_count": count,
        "message_count_total": messages_total,
        "average_messages_per_record": messages_total / count if count else 0,
        "role_counts": {role: roles[role] for role in _ROLES},
        "content_chars_total": content_chars_total,
        "message_count_distribution": message_distribution,
        "content_chars_distribution": content_distribution,
        "thinking_chars_distribution": thinking_distribution,
        "tool_chain_bucket_distribution": chain_distribution,
        "histogram_binning": {"message_count": message_method, "content_chars": content_method, "thinking_chars": thinking_method, "tool_chain": chain_method, "max_bins": 12, "zero_bucket": "separate"},
        "thinking_observed_records": thinking_records,
        "thinking_chars_total": thinking_chars,
        "thinking_source_counts": dict(source_counts),
        "malformed_thinking_messages": malformed,
        "tool_records": tool_records,
        "tool_call_count": tool_calls,
        "unanswered_tool_calls": unanswered_calls,
        "tool_chain_length_distribution": {str(key): value for key, value in sorted(chain_histogram.items())},
        "max_tool_chain_length_distribution": {str(key): value for key, value in sorted(max_chain_histogram.items())},
    }
