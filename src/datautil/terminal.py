"""Plain ASCII charts for conversation analysis in a terminal."""

from __future__ import annotations

from typing import Any


_BAR_WIDTH = 20


def _row(label: str, count: int, total: int) -> str:
    share = count / total if total else 0.0
    filled = min(_BAR_WIDTH, int(share * _BAR_WIDTH + 0.5))
    bar = "#" * filled + "." * (_BAR_WIDTH - filled)
    return f"  {label:<12} [{bar}] {share:6.1%} ({count:,})"


def _distribution(title: str, counts: dict[str, int], total: int) -> list[str]:
    lines = [f"{title} (of {total:,})"]
    lines.extend(_row(label, count, total) for label, count in counts.items() if count)
    return lines


def render_analysis(summary: dict[str, Any]) -> str:
    """Render aggregate counts as percentage bars with explicit denominators."""
    records = summary["record_count"]
    messages = summary["message_count_total"]
    if not records:
        return f"Format: {summary['source_format'] or 'unknown'}\nConversations: 0  No records to display"
    lines = [
        f"Format: {summary['source_format'] or 'unknown'}",
        f"Conversations: {records:,}  Messages: {messages:,}  Average: {summary['average_messages_per_record']:.1f} messages/conversation",
        "",
        *_distribution(summary["source_label_heading"], summary["source_label_counts"], messages),
        "",
        *_distribution("Messages per conversation", summary["message_count_distribution"], records),
        "",
        *_distribution("Text characters per conversation", summary["content_chars_distribution"], records),
        "",
    ]
    coverage = []
    if summary["thinking_observed_records"]:
        coverage.append(_row("thinking", summary["thinking_observed_records"], records))
    if summary["tool_records"]:
        coverage.append(_row("tool use", summary["tool_records"], records))
    if coverage:
        lines.extend(["Coverage (of conversations)", *coverage, ""])
    if summary["thinking_observed_records"]:
        lines.extend(_distribution("Thinking characters per conversation", summary["thinking_chars_distribution"], records))
        lines.append("")
    if summary["tool_records"]:
        lines.extend(_distribution("Longest tool chain in rounds", summary["tool_chain_bucket_distribution"], records))
        lines.extend(["", f"Tool calls: {summary['tool_call_count']:,}  Unanswered: {summary['unanswered_tool_calls']:,}"])
    if summary["malformed_thinking_messages"]:
        lines.append(f"Malformed thinking: {summary['malformed_thinking_messages']:,}")
    return "\n".join(lines).rstrip()
