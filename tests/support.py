import json
import tempfile
import unittest
from pathlib import Path


def write_jsonl(path, values):
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in values), encoding="utf-8"
    )


class DatautilTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
