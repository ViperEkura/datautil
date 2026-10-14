"""Stable document adapters for the staged, verified Parquet dedup pipeline."""

import json
import re
from functools import lru_cache

TEXT_FIELDS = (
    "full_content",
    "content",
    "text",
    "document",
    "body",
    "article",
    "markdown",
    "source_text",
    "code",
)
FALLBACK_FIELDS = (
    "title",
    "problem",
    "question",
    "task",
    "instruction",
    "analysis",
    "solution",
    "answer",
    "response",
    "completion",
    "output",
    "test",
)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORD = re.compile(r"\w+", re.UNICODE)
NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
FORMULA = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
SCHEMA_VERSION = "4"


def extract(row):
    full = row.get("full_content")
    if isinstance(full, str) and full and not full.isspace():
        return full
    messages = row.get("messages")
    if isinstance(messages, list):
        values = [
            {
                key: message[key]
                for key in ("role", "name", "content", "text", "tool_calls")
                if key in message
            }
            for message in messages
            if isinstance(message, dict)
        ]
        if values:
            return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    structured = any(
        isinstance(row.get(key), str) and row[key].strip()
        for key in (
            "analysis",
            "solution",
            "answer",
            "response",
            "completion",
            "output",
        )
    )
    if structured:
        values = [
            (key, row[key])
            for key in TEXT_FIELDS + FALLBACK_FIELDS
            if isinstance(row.get(key), str) and row[key] and not row[key].isspace()
        ]
        if values:
            return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    for key in TEXT_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value and not value.isspace():
            return value
    values = [
        (key, row[key])
        for key in FALLBACK_FIELDS
        if isinstance(row.get(key), str) and row[key] and not row[key].isspace()
    ]
    return (
        json.dumps(values, ensure_ascii=False, separators=(",", ":")) if values else ""
    )


def normalize(text, is_code):
    # Whitespace inside literals and line continuations can change code behavior.
    if is_code:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return " ".join(text.split())


def near_group(category):
    if tier(category) == 3:
        return None
    if category.startswith("web"):
        return "web:" + category.split("_", 1)[1]
    if category.startswith("code"):
        return "code:" + category.split("_", 1)[1]
    if category.startswith("math"):
        return "math"
    return None


def content_features(text, category):
    if category.startswith("web"):
        text = text.casefold()
    if category.endswith("_zh"):
        chars = "".join(WORD.findall(text))
        return {chars[i : i + 20] for i in range(max(0, len(chars) - 19))}
    words = WORD.findall(text)
    return {" ".join(words[i : i + 5]) for i in range(max(0, len(words) - 4))}


@lru_cache(maxsize=64)
def cached_features(text, category):
    return content_features(text, category)


@lru_cache(maxsize=16)
def code_lexer(category):
    from pygments.lexers import get_lexer_by_name

    language = category.split("_", 1)[1]
    language = {
        "cs": "csharp",
        "js": "javascript",
        "py": "python",
        "rb": "ruby",
        "sh": "bash",
        "r": "r",
    }.get(language, language)
    options = {"startinline": True} if language == "php" else {}
    return get_lexer_by_name(language, **options)


@lru_cache(maxsize=128)
def cached_code_tokens(text, category):
    return uncached_code_tokens(text, category)


def code_tokens(text, category):
    if len(text) <= 100000:
        return cached_code_tokens(text, category)
    return uncached_code_tokens(text, category)


def code_token_stream(text, category):
    if category.endswith("_py"):
        import io
        import tokenize

        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.ERRORTOKEN:
                    raise ValueError("Invalid code token")
                if token.type not in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.ENDMARKER,
                ):
                    yield (token.type, token.string)
            return
        except (tokenize.TokenError, IndentationError, SyntaxError):
            raise ValueError("Invalid Python tokens")
    from pygments.token import Comment, Error, Text

    previous = None
    for kind, value in code_lexer(category).get_tokens(text):
        if kind in Error:
            raise ValueError("Invalid code token")
        ignored = (
            kind in Comment
            and kind not in Comment.Preproc
            and kind not in Comment.PreprocFile
        )
        if ignored or (kind in Text and value.isspace()):
            if "\n" in value and previous != ("linebreak", "\n"):
                previous = ("linebreak", "\n")
                yield previous
            continue
        if value:
            previous = (str(kind), value)
            yield previous


def uncached_code_tokens(text, category):
    try:
        return tuple(code_token_stream(text, category))
    except ValueError:
        return None


def code_equal(left, right, category):
    """Stop lexing as soon as a meaningful token differs."""
    from itertools import zip_longest

    sentinel = object()
    seen = False
    try:
        for a, b in zip_longest(
            code_token_stream(left, category),
            code_token_stream(right, category),
            fillvalue=sentinel,
        ):
            if a != b:
                return False
            seen = True
    except ValueError:
        return False
    return seen


def math_key(text):
    critical = re.findall(
        r"\b\w*\d\w*\b|\b[^\W\d_]\b|\\[a-zA-Z]+|[=<>+\-*/^]|[≤≥≠±√∑∫]", text
    )
    answers = re.findall(
        r"(?:final\s+answer|answer|result)\s*[:=]\s*([^\n]{1,200})", text, re.IGNORECASE
    )
    return (
        tuple(NUMBER.findall(text)),
        tuple(critical),
        tuple("".join(value.split()) for value in FORMULA.findall(text)),
        tuple(answers),
    )


def verify_pair(left, right, category):
    if not left or not right:
        return False, 0.0, "empty"
    minimum_ratio = 0.8 if category.startswith("web") else 0.9
    if min(len(left), len(right)) / max(len(left), len(right)) < minimum_ratio:
        return False, 0.0, "length"
    feature_left = cached_features if len(left) <= 100000 else content_features
    feature_right = cached_features if len(right) <= 100000 else content_features
    a, b = feature_left(left, category), feature_right(right, category)
    if min(len(a), len(b)) < 20:
        return False, 0.0, "short"
    intersection = len(a & b)
    score = intersection / (len(a) + len(b) - intersection)
    threshold = 0.9 if category.startswith("web") else 0.95
    if score < threshold:
        return False, score, "content"
    if category.startswith("code"):
        if not code_equal(left, right, category):
            return False, score, "code_changed"
    if category.startswith("math") and math_key(left) != math_key(right):
        return False, score, "math_changed"
    return True, score, "verified"


def atomic_json(path, value):
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def ordered_files(manifest):
    return sorted(
        manifest["files"],
        key=lambda item: (
            0 if item["category"].startswith(("web3_", "code3_", "math3_")) else 1,
            item["category"],
            item["output_file"],
        ),
    )


def tier(category):
    match = re.match(r"^(?:web|code|math)([123])(?:_|$)", category)
    if match is None:
        raise RuntimeError(f"Unknown pretraining tier: {category}")
    return int(match.group(1))


def processing_order(files):
    return sorted(
        enumerate(files), key=lambda pair: (3 - tier(pair[1]["category"]), pair[0])
    )


def run(args):
    from datautil.pretrain.manifest_pipeline import run as run_pipeline

    return run_pipeline(args)


def main():
    from datautil.pretrain.manifest_pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
