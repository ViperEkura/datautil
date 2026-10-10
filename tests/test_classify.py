from pathlib import Path
from unittest.mock import patch

from datautil.cli import main
from datautil.io import iter_raw
from datautil.operations import load_labels

from support import DatautilTestCase, write_jsonl


class ClassifyTests(DatautilTestCase):
    def test_classify_and_stats_without_repeated_api(self):
        source = self.root / "source.jsonl"
        labels = self.root / "labels.jsonl"
        write_jsonl(source, [{"id": "x", "messages": [{"role": "user", "content": "Write Python"}, {"role": "assistant", "content": "print(1)"}]}])

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return None
            def read(self, *_): return b'{"choices":[{"message":{"content":"code"}}]}'

        with patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}), patch("urllib.request.urlopen", return_value=Response()) as call:
            self.assertEqual(main(["classify", str(source), "-o", str(labels), "--sample", "1", "--model", "mock"]), 0)
            self.assertEqual(call.call_count, 1)
        with patch("urllib.request.urlopen", side_effect=AssertionError("stats called API")):
            self.assertEqual(main(["stats", str(labels)]), 0)
        with patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}), patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            self.assertEqual(main(["classify", str(source), "-o", str(labels), "--sample", "1", "--model", "mock"]), 1)

    def test_default_labels_match_example(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "subject-labels.yaml"
        self.assertEqual(load_labels(None), load_labels(str(example)))
        self.assertEqual(len(load_labels(None)), 14)
        self.assertIn("other", load_labels(None))

    def test_usage_labels_and_stats(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "usage-labels.yaml"
        self.assertEqual(load_labels(None, "usage"), load_labels(str(example), "usage"))
        source = self.root / "source.jsonl"
        labels = self.root / "usage.jsonl"
        write_jsonl(source, [{"id": "x", "messages": [{"role": "user", "content": "How should I learn Python?"}, {"role": "assistant", "content": "Practice."}]}])

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return None
            def read(self, *_): return b'{"choices":[{"message":{"content":"advice"}}]}'

        with patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}), patch("urllib.request.urlopen", return_value=Response()):
            self.assertEqual(main(["classify", str(source), "-o", str(labels), "--sample", "1", "--model", "mock", "--type", "usage"]), 0)
        row = next(iter_raw(str(labels)))[1]
        self.assertEqual(row["usage_topic"], "advice")
        self.assertIn("labels_sha256", row)
        self.assertNotIn("domain", row)
        self.assertEqual(main(["stats", str(labels)]), 0)

        with patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}), patch("urllib.request.urlopen", return_value=Response()):
            self.assertEqual(main(["classify", str(source), "-o", str(labels), "--sample", "1", "--model", "mock", "--scheme", "usage", "--taxonomy", str(example)]), 0)

    def test_stats_requires_explicit_field_when_ambiguous_or_empty(self):
        path = self.root / "labels.jsonl"
        write_jsonl(path, [{"domain": "code", "usage_topic": "technical"}])
        self.assertEqual(main(["stats", str(path)]), 1)
        self.assertEqual(main(["stats", str(path), "--by", "domain"]), 0)
        path.write_text("", encoding="utf-8")
        self.assertEqual(main(["stats", str(path)]), 1)
        self.assertEqual(main(["stats", str(path), "--by", "domain"]), 0)
