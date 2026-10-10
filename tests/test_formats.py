import io
import json
from contextlib import redirect_stdout

from datautil.cli import main
from datautil.io import iter_raw
from datautil.model import DataError
from datautil.operations import read_records

from support import DatautilTestCase, write_jsonl


class FormatTests(DatautilTestCase):
    def test_convert_round_trip_and_loss_rejection(self):
        source = self.root / "sharegpt.jsonl"
        target = self.root / "openai.jsonl"
        restored = self.root / "restored.jsonl"
        write_jsonl(source, [{"id": "c1", "conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}])
        self.assertEqual(main(["convert", str(source), "-o", str(target), "--to", "openai-messages"]), 0)
        self.assertEqual(main(["convert", str(target), "-o", str(restored), "--to", "sharegpt"]), 0)
        self.assertEqual(next(iter_raw(str(restored)))[1]["conversations"][1]["value"], "Hello")

        tool_source = self.root / "tools.jsonl"
        write_jsonl(tool_source, [{"messages": [{"role": "user", "content": "weather"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "call1", "content": "sunny"}]}])
        blocked = self.root / "blocked.jsonl"
        self.assertEqual(main(["convert", str(tool_source), "-o", str(blocked), "--to", "alpaca"]), 1)
        self.assertFalse(blocked.exists())

    def test_json_array_and_duplicate_ids(self):
        path = self.root / "array.json"
        path.write_text('[{"id":"a","instruction":"Hi","input":"","output":"Hello"},{"id":"b","instruction":"Bye","input":"","output":"Goodbye"}]', encoding="utf-8")
        self.assertEqual(len(list(read_records(str(path), "alpaca"))), 2)
        path.write_text('[{"id":"a","instruction":"Hi","input":"","output":"Hello"},]', encoding="utf-8")
        with self.assertRaises(DataError):
            list(iter_raw(str(path)))
        path.with_suffix(".jsonl").write_text('{"id":"x","instruction":"Hi","input":"","output":"Hello"}\n{"id":"x","instruction":"Bye","input":"","output":"Goodbye"}\n', encoding="utf-8")
        with self.assertRaises(DataError):
            list(read_records(str(path.with_suffix(".jsonl"))))

    def test_validate_summary_prints_to_stdout(self):
        path = self.root / "source.jsonl"
        write_jsonl(path, [{"id": "x", "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}])
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["validate", str(path), "--from", "openai-messages"]), 0)
        self.assertEqual(json.loads(captured.getvalue())["issue_count"], 0)

    def test_clickhouse_json_each_row_profile(self):
        source = self.root / "clickhouse-export.json"
        profile = self.root / "profile.yaml"
        output = self.root / "openai.jsonl"
        source.write_text('{"request_id":"r1","request":{"prompt":"Hi"},"response":{"text":"Hello"},"latency_ms":12}\n', encoding="utf-8")
        profile.write_text("version: 1\nname: ch-test\ncontainer: JSONEachRow\nlayout: pair\nfields:\n  id: request_id\n  prompt: request.prompt\n  completion: response.text\n", encoding="utf-8")
        self.assertEqual(main(["convert", str(source), "-o", str(output), "--format-config", str(profile), "--from", "ch-test", "--to", "openai-messages"]), 0)
        row = next(iter_raw(str(output)))[1]
        self.assertEqual(row["messages"][1]["content"], "Hello")
        self.assertEqual(row["latency_ms"], 12)
        restored = self.root / "restored.json"
        self.assertEqual(main(["convert", str(output), "-o", str(restored), "--format-config", str(profile), "--from", "openai-messages", "--to", "ch-test"]), 0)
        self.assertEqual(next(iter_raw(str(restored), "jsonl"))[1]["response"]["text"], "Hello")
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["analyze", str(source), "--format-config", str(profile), "--from", "ch-test"]), 0)
        self.assertIn("request.prompt", captured.getvalue())
        self.assertIn("response.text", captured.getvalue())
        self.assertNotIn("tool use", captured.getvalue())

    def test_custom_pair_profile_preserves_thinking(self):
        profile = self.root / "profile.yaml"
        source = self.root / "export.jsonl"
        openai = self.root / "openai.jsonl"
        restored = self.root / "restored.jsonl"
        profile.write_text("version: 1\nname: reasoning-pair\ncontainer: JSONEachRow\nlayout: pair\nfields:\n  id: id\n  prompt: prompt\n  completion: answer\n  thinking: reasoning\n", encoding="utf-8")
        write_jsonl(source, [{"id": "r1", "prompt": "Why?", "answer": "Because.", "reasoning": "Let me think"}])
        self.assertEqual(main(["convert", str(source), "-o", str(openai), "--from", "reasoning-pair", "--to", "openai-messages", "--format-config", str(profile)]), 0)
        self.assertEqual(next(iter_raw(str(openai)))[1]["messages"][1]["reasoning_content"], "Let me think")
        self.assertEqual(main(["convert", str(openai), "-o", str(restored), "--from", "openai-messages", "--to", "reasoning-pair", "--format-config", str(profile)]), 0)
        self.assertEqual(next(iter_raw(str(restored)))[1]["reasoning"], "Let me think")

    def test_custom_messages_tools_and_unmapped_fields(self):
        source = self.root / "openai.jsonl"
        output = self.root / "custom.jsonl"
        restored = self.root / "restored.jsonl"
        profile = self.root / "chat.yaml"
        profile.write_text("version: 1\nname: mapped-chat\ncontainer: jsonl\nlayout: messages\nfields:\n  id: conversation_id\n  messages: dialogue.turns\n  role: speaker\n  content: text\n  tool_calls: calls\n  tool_call_id: call_id\nrole_map:\n  human: user\n  bot: assistant\n  function: tool\n", encoding="utf-8")
        write_jsonl(source, [{"id": "t1", "messages": [{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "Sunny"}]}])
        self.assertEqual(main(["convert", str(source), "-o", str(output), "--format-config", str(profile), "--from", "openai-messages", "--to", "mapped-chat"]), 0)
        self.assertEqual(next(iter_raw(str(output)))[1]["dialogue"]["turns"][1]["speaker"], "bot")
        self.assertEqual(main(["convert", str(output), "-o", str(restored), "--format-config", str(profile), "--from", "mapped-chat", "--to", "openai-messages"]), 0)
        self.assertEqual(next(iter_raw(str(restored)))[1]["messages"][2]["tool_call_id"], "c1")
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["analyze", str(output), "--format-config", str(profile), "--from", "mapped-chat"]), 0)
        self.assertIn("Source roles (of 3)", captured.getvalue())
        self.assertIn("function", captured.getvalue())
        self.assertIn("tool use", captured.getvalue())
        bad = next(iter_raw(str(output)))[1]
        bad["dialogue"]["turns"][0]["unmapped"] = 5
        write_jsonl(output, [bad])
        blocked = self.root / "blocked.jsonl"
        self.assertEqual(main(["convert", str(output), "-o", str(blocked), "--format-config", str(profile), "--from", "mapped-chat", "--to", "openai-messages"]), 1)
        self.assertFalse(blocked.exists())

    def test_profile_rejects_overlapping_paths(self):
        from datautil.profiles import load_profiles
        profile = self.root / "invalid.yaml"
        profile.write_text("version: 1\nname: invalid\nlayout: pair\nfields:\n  prompt: request\n  completion: request.text\n", encoding="utf-8")
        with self.assertRaises(DataError):
            load_profiles([str(profile)])
