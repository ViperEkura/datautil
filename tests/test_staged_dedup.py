import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from support import DatautilTestCase

from datautil.pretrain.manifest_index import (
    code_equal,
    code_tokens,
    content_features,
    extract,
    normalize,
    verify_pair,
)
from datautil.pretrain.manifest_pipeline import EDGE, resolve_edges, run


class StagedDedupTests(DatautilTestCase):
    def prose(self):
        return " ".join(
            f"section{i} gives a detailed explanation of mechanism{i} and example{i}."
            for i in range(60)
        )

    def manifest(self, entries):
        files = []
        for index, (category, texts) in enumerate(entries):
            path = self.root / f"source-{index}.parquet"
            pq.write_table(pa.table({"content": texts}), path, row_group_size=2)
            files.append(
                {
                    "category": category,
                    "output_file": path.name,
                    "bytes": path.stat().st_size,
                }
            )
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"status": "downloaded", "files": files}))
        return path

    def args(self, manifest, output, workers=1, stop=0):
        return SimpleNamespace(
            profile="astrai-100b",
            manifest=manifest,
            output=output,
            workers=workers,
            shards=4,
            candidate_cap=2000,
            max_files=0,
            stop_after_tasks=stop,
            stop_after_groups=0,
        )

    def test_roles_and_answers_are_part_of_exact_text(self):
        self.assertNotEqual(
            extract({"question": "Same question", "answer": "Yes"}),
            extract({"question": "Same question", "answer": "No"}),
        )
        self.assertNotEqual(
            extract({"messages": [{"role": "user", "content": "same"}]}),
            extract({"messages": [{"role": "assistant", "content": "same"}]}),
        )
        self.assertEqual(normalize(" a\t b\n c\u00a0 d ", False), "a b c d")
        self.assertEqual(
            normalize("  a = 1  \r\n    b = 2\n", True), "  a = 1  \r\n    b = 2\n"
        )

    def test_punctuation_does_not_make_unrelated_content_duplicate(self):
        decoration = "-" * 600 + "\n"
        left = decoration * 30 + self.prose()
        right = decoration * 30 + " ".join(
            f"database{i} transaction{i} query{i} index{i} server{i}" for i in range(60)
        )
        self.assertFalse(verify_pair(left, right, "web2_en")[0])
        self.assertFalse(verify_pair(left, right, "code2_php")[0])
        self.assertFalse(verify_pair(left, right, "math2")[0])
        self.assertFalse(
            any(
                set(feature.split()) == {"-"}
                for feature in content_features(left, "web2_en")
            )
        )

    def test_critical_code_and_math_changes_are_retained(self):
        code = "\n".join(f"value_{i} = argument_{i} + {i}" for i in range(100))
        left = code + "\nanswer = 42\n"
        right = code + "\nanswer = 43\n"
        allowed, score, reason = verify_pair(left, right, "code2_py")
        self.assertFalse(allowed)
        self.assertGreater(score, 0.95)
        self.assertEqual(reason, "code_changed")
        self.assertTrue(
            verify_pair(code + "\n# note one", code + "\n# note two", "code2_py")[0]
        )
        left = self.prose() + " The final answer: 42."
        right = self.prose() + " The final answer: 43."
        self.assertFalse(verify_pair(left, right, "math2")[0])
        left = self.prose() + " The final answer: $x-y$."
        right = self.prose() + " The final answer: $x+y$."
        self.assertFalse(verify_pair(left, right, "math2")[0])

    def test_verified_edges_do_not_delete_through_a_chain(self):
        flags = np.ones(3, dtype=bool)
        edges = np.array([(1, 0, 0.99), (2, 1, 0.99)], dtype=EDGE)
        removed = resolve_edges(flags, np.array([0, 3]), edges, [0])
        self.assertEqual(flags.tolist(), [True, False, True])
        self.assertEqual(removed["lose"].tolist(), [1])

    def test_code_indentation_newlines_and_directives_are_preserved(self):
        python = "\n".join(f"value_{i} = argument_{i} + {i}" for i in range(100))
        a = (
            python
            + "\ndef choose(flag):\n    if flag:\n        return 1\n    return 2\n"
        )
        b = (
            python
            + "\ndef choose(flag):\n    if flag:\n        return 1\n        return 2\n"
        )
        self.assertFalse(verify_pair(a, b, "code2_py")[0])
        javascript = "\n".join(
            f"const value_{i} = argument_{i} + {i};" for i in range(100)
        )
        a = javascript + "\nfunction choose(){return value_0;}\n"
        b = javascript + "\nfunction choose(){return\nvalue_0;}\n"
        self.assertFalse(verify_pair(a, b, "code2_js")[0])
        cpp = "\n".join(f"int value_{i} = argument_{i} + {i};" for i in range(100))
        self.assertFalse(
            verify_pair(
                cpp + "\n#define FLAG 1\n", cpp + "\n#define FLAG 2\n", "code2_cpp"
            )[0]
        )

    def test_staged_outputs_are_deterministic_and_auditable(self):
        text = self.prose()
        manifest = self.manifest(
            [
                ("web2_en", [text, text.upper(), "short", text + " Synthetic anchor."]),
                ("web3_en_QA-Synthetic", [text + " Synthetic anchor."]),
                ("web2_en", [text.upper(), text.lower()]),
            ]
        )
        serial = self.root / "serial"
        partial = run(self.args(manifest, serial, stop=1))
        self.assertEqual(partial["status"], "partial")
        result = run(self.args(manifest, serial))
        parallel = self.root / "parallel"
        other = run(self.args(manifest, parallel, workers=2))
        self.assertEqual(result, other)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["categories"]["web3_en_QA-Synthetic"]["kept"], 1)
        self.assertEqual(result["categories"]["web2_en"]["exact"], 3)
        self.assertEqual(result["categories"]["web2_en"]["near"], 1)
        self.assertEqual(result["categories"]["web2_en"]["filtered"], 1)
        self.assertEqual(result["categories"]["web2_en"]["kept"], 1)
        self.assertFalse((serial / "dedup.sqlite").exists())
        for file in (serial / "keep").glob("*.u64"):
            self.assertEqual(
                file.read_bytes(), (parallel / "keep" / file.name).read_bytes()
            )
        samples = json.loads((serial / "audit/near_samples.json").read_text())
        self.assertEqual(len(samples["web2_en"]), 1)
        self.assertTrue((serial / "audit/candidate_samples.json").is_file())
        self.assertEqual(run(self.args(manifest, serial)), result)
        with self.assertRaisesRegex(RuntimeError, "different manifest"):
            changed = self.args(manifest, serial)
            changed.candidate_cap = 10
            run(changed)

    def test_near_candidates_stay_within_language_domains(self):
        text = self.prose()
        manifest = self.manifest([("web2_en", [text]), ("web2_zh", [text.upper()])])
        result = run(self.args(manifest, self.root / "domains"))
        self.assertEqual(sum(x["kept"] for x in result["categories"].values()), 2)
        self.assertEqual(sum(x["near"] for x in result["categories"].values()), 0)

    def test_truncated_checkpoint_is_not_silently_reused(self):
        manifest = self.manifest([("math2", [self.prose()])])
        output = self.root / "corrupt"
        run(self.args(manifest, output, stop=1))
        binary = next((output / "prepare/0000").glob("part-*.bin"))
        binary.write_bytes(binary.read_bytes()[:-1])
        with self.assertRaisesRegex(RuntimeError, "Incomplete committed"):
            run(self.args(manifest, output))
        self.assertFalse((output / "manifest.json").exists())

    def test_row_block_verification_is_merged_into_final_output(self):
        text = self.prose()
        manifest = self.manifest(
            [("web2_en", [text] + ["short"] * 1023 + [text.upper()])]
        )
        result = run(self.args(manifest, self.root / "row-blocks", workers=2))
        self.assertEqual(result["categories"]["web2_en"]["near"], 1)
        self.assertEqual(result["categories"]["web2_en"]["filtered"], 1023)
        self.assertEqual(
            np.fromfile(self.root / "row-blocks/keep/0000.u64", dtype="<u8").tolist(),
            [0],
        )

    def test_streaming_code_comparison_matches_complete_tokens(self):
        for category, texts in [
            ("code2_py", ["x = 1\n", "x = 2\n", "x = 1\n# comment", "x = '\n"]),
            (
                "code2_php",
                ["<?php echo 1;", "<?php echo 2;", "<?php echo 1; //comment"],
            ),
            ("code2_js", ["return 1;", "return\n1;", "return 1; // comment"]),
        ]:
            for left in texts:
                for right in texts:
                    a, b = code_tokens(left, category), code_tokens(right, category)
                    self.assertEqual(
                        code_equal(left, right, category), bool(a) and a == b
                    )

    def test_exact_code_preserves_whitespace_inside_string_literals(self):
        prefix = "\n".join(f"value_{i} = argument_{i} + {i}" for i in range(50))
        left = prefix + '\nmessage = """first line  \nsecond line"""\n'
        right = prefix + '\nmessage = """first line\nsecond line"""\n'
        manifest = self.manifest([("code2_py", [left, right])])
        result = run(self.args(manifest, self.root / "string-whitespace"))
        self.assertEqual(result["categories"]["code2_py"]["kept"], 2)
        self.assertEqual(result["categories"]["code2_py"]["exact"], 0)
        self.assertEqual(result["categories"]["code2_py"]["near"], 0)
