import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from support import DatautilTestCase, write_jsonl

from datautil.cli import main
from datautil.common.io import iter_raw
from datautil.conversation.classification import load_labels


@contextmanager
def classifier_server(label="code", status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, self.headers.get("Authorization")))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if status == 200:
                response = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "mock",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": label},
                        }
                    ],
                }
            else:
                response = {"error": {"message": "server failed", "type": "api_error"}}
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ClassifyTests(DatautilTestCase):
    def test_classify_and_stats_without_repeated_api(self):
        source = self.root / "source.jsonl"
        labels = self.root / "labels.jsonl"
        write_jsonl(
            source,
            [
                {
                    "id": "x",
                    "messages": [
                        {"role": "user", "content": "Write Python"},
                        {"role": "assistant", "content": "print(1)"},
                    ],
                }
            ],
        )
        with (
            classifier_server() as (base_url, requests),
            patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}),
        ):
            args = [
                "classify",
                str(source),
                "-o",
                str(labels),
                "--sample",
                "1",
                "--model",
                "mock",
                "--base-url",
                base_url,
            ]
            self.assertEqual(main(args), 0)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0][0], "/v1/chat/completions")
            self.assertEqual(requests[0][1]["model"], "mock")
            self.assertEqual(requests[0][1]["temperature"], 0)
            self.assertEqual(requests[0][2], "Bearer test")
            self.assertEqual(main(["stats", str(labels)]), 0)
            self.assertEqual(len(requests), 1)

        with (
            classifier_server(status=500) as (base_url, requests),
            patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}),
        ):
            self.assertEqual(
                main(
                    [
                        "classify",
                        str(source),
                        "-o",
                        str(labels),
                        "--sample",
                        "1",
                        "--model",
                        "mock",
                        "--base-url",
                        base_url,
                    ]
                ),
                1,
            )
            self.assertEqual(len(requests), 1)

    def test_default_labels_match_example(self):
        example = (
            Path(__file__).resolve().parents[1] / "examples" / "subject-labels.yaml"
        )
        self.assertEqual(load_labels(None), load_labels(str(example)))
        self.assertEqual(len(load_labels(None)), 14)
        self.assertIn("other", load_labels(None))

    def test_usage_labels_and_stats(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "usage-labels.yaml"
        self.assertEqual(load_labels(None, "usage"), load_labels(str(example), "usage"))
        source = self.root / "source.jsonl"
        labels = self.root / "usage.jsonl"
        write_jsonl(
            source,
            [
                {
                    "id": "x",
                    "messages": [
                        {"role": "user", "content": "How should I learn Python?"},
                        {"role": "assistant", "content": "Practice."},
                    ],
                }
            ],
        )
        with (
            classifier_server("advice") as (base_url, requests),
            patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}),
        ):
            self.assertEqual(
                main(
                    [
                        "classify",
                        str(source),
                        "-o",
                        str(labels),
                        "--sample",
                        "1",
                        "--model",
                        "mock",
                        "--type",
                        "usage",
                        "--base-url",
                        base_url,
                    ]
                ),
                0,
            )
            row = next(iter_raw(str(labels)))[1]
            self.assertEqual(row["usage_topic"], "advice")
            self.assertIn("labels_sha256", row)
            self.assertNotIn("domain", row)
            self.assertEqual(main(["stats", str(labels)]), 0)
            self.assertEqual(
                main(
                    [
                        "classify",
                        str(source),
                        "-o",
                        str(labels),
                        "--sample",
                        "1",
                        "--model",
                        "mock",
                        "--scheme",
                        "usage",
                        "--taxonomy",
                        str(example),
                        "--base-url",
                        base_url,
                    ]
                ),
                0,
            )
            self.assertEqual(len(requests), 2)

    def test_classify_custom_yaml_mapped_input(self):
        source = self.root / "nonstandard.jsonl"
        profile = self.root / "mapping.yaml"
        output = self.root / "labels.jsonl"
        write_jsonl(
            source,
            [
                {
                    "request_id": "custom-1",
                    "request": {"prompt": "Write Python"},
                    "response": {"text": "print(1)"},
                }
            ],
        )
        profile.write_text(
            "version: 1\nname: custom-pair\nlayout: pair\n"
            "fields:\n  id: request_id\n  prompt: request.prompt\n"
            "  completion: response.text\n",
            encoding="utf-8",
        )
        with (
            classifier_server() as (base_url, requests),
            patch.dict("os.environ", {"DATAUTIL_API_KEY": "test"}),
        ):
            self.assertEqual(
                main(
                    [
                        "classify",
                        str(source),
                        "-o",
                        str(output),
                        "--sample",
                        "1",
                        "--model",
                        "mock",
                        "--base-url",
                        base_url,
                        "--format-config",
                        str(profile),
                        "--from",
                        "custom-pair",
                    ]
                ),
                0,
            )
        self.assertEqual(len(requests), 1)
        self.assertIn("Write Python", requests[0][1]["messages"][1]["content"])
        self.assertEqual(next(iter_raw(str(output)))[1]["record_id"], "custom-1")

    def test_stats_requires_explicit_field_when_ambiguous_or_empty(self):
        path = self.root / "labels.jsonl"
        write_jsonl(path, [{"domain": "code", "usage_topic": "technical"}])
        self.assertEqual(main(["stats", str(path)]), 1)
        self.assertEqual(main(["stats", str(path), "--by", "domain"]), 0)
        path.write_text("", encoding="utf-8")
        self.assertEqual(main(["stats", str(path)]), 1)
        self.assertEqual(main(["stats", str(path), "--by", "domain"]), 0)
