import io
import json
from collections import Counter
from contextlib import redirect_stdout

from datautil.analysis import _histogram
from datautil.cli import main
from datautil.io import iter_raw

from support import DatautilTestCase, write_jsonl


class AnalysisTests(DatautilTestCase):
    def test_histograms_use_data_driven_bins(self):
        self.assertEqual(_histogram(Counter({7: 3})), ({"7": 3}, "single-value"))
        histogram, method = _histogram(Counter({0: 20, 1: 10, 2: 1, 100: 1}), separate_zero=True)
        self.assertEqual(histogram["0"], 20)
        self.assertEqual(sum(histogram.values()), 32)
        self.assertLessEqual(len(histogram), 13)
        self.assertEqual(method, "sturges-fallback")
        histogram, method = _histogram(Counter({1: 5, 2: 5, 10: 5, 20: 5}))
        self.assertEqual(sum(histogram.values()), 20)
        self.assertEqual(method, "freedman-diaconis")

    def test_analyze_thinking_and_tool_chain(self):
        source = self.root / "source.jsonl"
        detail = self.root / "analysis.jsonl"
        summary = self.root / "summary.jsonl"
        calls = lambda *ids: [{"id": ident, "type": "function", "function": {"name": "lookup", "arguments": "{}"}} for ident in ids]
        write_jsonl(source, [{"id": "trace", "messages": [
            {"role": "user", "content": "Investigate"},
            {"role": "assistant", "content": "", "reasoning_content": "abc", "tool_calls": calls("c1", "c2")},
            {"role": "tool", "tool_call_id": "c1", "content": "one"},
            {"role": "tool", "tool_call_id": "c2", "content": "two"},
            {"role": "assistant", "content": "<think>defg</think>", "tool_calls": calls("c3")},
            {"role": "tool", "tool_call_id": "c3", "content": "three"},
            {"role": "assistant", "content": "Done"},
        ]}])
        self.assertEqual(main(["analyze", str(source), "--from", "openai-messages", "-o", str(detail), "--report", str(summary)]), 0)
        row = next(iter_raw(str(detail)))[1]
        self.assertEqual(row["message_count"], 7)
        self.assertEqual(row["role_counts"], {"system": 0, "developer": 0, "user": 1, "assistant": 3, "tool": 3})
        self.assertEqual((row["thinking_chars"], row["thinking_source"]), (7, "field+tag"))
        self.assertEqual((row["tool_call_count"], row["tool_chain_lengths"], row["max_parallel_calls"]), (3, [2], 2))
        self.assertEqual(row["unanswered_tool_calls"], 0)
        summary_row = next(iter_raw(str(summary)))[1]
        self.assertEqual(summary_row["tool_chain_length_distribution"], {"2": 1})
        self.assertEqual(summary_row["message_count_distribution"], {"7": 1})
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["analyze", str(source)]), 0)
        self.assertIn("Message roles (of 7)", captured.getvalue())
        self.assertIn("14.3% (1)", captured.getvalue())
        self.assertIn("thinking     [####################] 100.0% (1)", captured.getvalue())
        with redirect_stdout(captured := io.StringIO()):
            self.assertEqual(main(["analyze", str(source), "--json"]), 0)
        self.assertEqual(json.loads(captured.getvalue())["tool_call_count"], 3)
        restored = self.root / "restored.jsonl"
        self.assertEqual(main(["convert", str(source), "--from", "openai-messages", "--to", "openai-messages", "-o", str(restored)]), 0)
        self.assertEqual(next(iter_raw(str(restored)))[1]["messages"][1]["reasoning_content"], "abc")
        blocked = self.root / "blocked.jsonl"
        self.assertEqual(main(["convert", str(source), "--from", "openai-messages", "--to", "sharegpt", "-o", str(blocked)]), 1)
        self.assertFalse(blocked.exists())

    def test_analyze_chain_resets_and_missing_results(self):
        source = self.root / "source.jsonl"
        detail = self.root / "analysis.jsonl"
        call = lambda ident: {"id": ident, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        write_jsonl(source, [{"id": "trace", "messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "<think>unfinished", "tool_calls": [call("c1")]},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "", "tool_calls": [call("c2")]},
            {"role": "tool", "tool_call_id": "c2", "content": "result"},
            {"role": "assistant", "content": "Done"},
        ]}])
        self.assertEqual(main(["analyze", str(source), "--from", "openai-messages", "-o", str(detail)]), 0)
        row = next(iter_raw(str(detail)))[1]
        self.assertEqual(row["tool_chain_lengths"], [1, 1])
        self.assertEqual(row["unanswered_tool_calls"], 1)
        self.assertEqual(row["malformed_thinking_messages"], 1)
        self.assertEqual(row["thinking_chars"], 0)

    def test_empty_conversation_summary(self):
        source = self.root / "empty.jsonl"
        source.write_text("", encoding="utf-8")
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["analyze", str(source), "--from", "openai-messages"]), 0)
        self.assertIn("Conversations: 0", captured.getvalue())
        self.assertNotIn("tool use", captured.getvalue())
        with redirect_stdout(captured := io.StringIO()):
            self.assertEqual(main(["analyze", str(source), "--from", "openai-messages", "--json"]), 0)
        self.assertEqual(json.loads(captured.getvalue())["record_count"], 0)

    def test_analyze_percentages_use_explicit_denominators(self):
        source = self.root / "source.jsonl"
        write_jsonl(source, [
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "One"}, {"role": "assistant", "content": "Two"}, {"role": "user", "content": "Three"}, {"role": "assistant", "content": "Four"}]},
        ])
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["analyze", str(source), "--from", "openai-messages"]), 0)
        display = captured.getvalue()
        self.assertIn("Message roles (of 6)", display)
        self.assertIn("50.0% (3)", display)
        self.assertIn("Messages per conversation (of 2)", display)
        self.assertIn("50.0% (1)", display)
        self.assertNotIn("tool use", display)
        self.assertNotIn("Thinking characters", display)

    def test_analyze_uses_source_labels_and_hides_missing_metrics(self):
        examples = [
            ("sharegpt", {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}, "Source roles (of 2)", ("human", "gpt")),
            ("alpaca", {"instruction": "Hi", "input": "", "output": "Hello"}, "Mapped fields (of 2)", ("instruction", "output")),
        ]
        for format_name, record, heading, labels in examples:
            with self.subTest(format=format_name):
                source = self.root / f"{format_name}.jsonl"
                write_jsonl(source, [record])
                captured = io.StringIO()
                with redirect_stdout(captured):
                    self.assertEqual(main(["analyze", str(source), "--from", format_name]), 0)
                display = captured.getvalue()
                self.assertIn(f"Format: {format_name}", display)
                self.assertIn(heading, display)
                for label in labels:
                    self.assertIn(label, display)
                self.assertNotIn("tool use", display)
                self.assertNotIn("tool chain", display)
                self.assertNotIn("thinking", display.lower())
                with redirect_stdout(captured := io.StringIO()):
                    self.assertEqual(main(["analyze", str(source), "--from", format_name, "--json"]), 0)
                report = json.loads(captured.getvalue())
                self.assertEqual(report["source_format"], format_name)
                self.assertEqual(set(report["source_label_counts"]), set(labels))
                self.assertEqual(report["role_counts"]["user"], 1)
