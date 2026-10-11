# datautil

CLI for inspecting and preparing conversation datasets. The command is `datautil`; the current wheel distribution name is `datautil-llm-dev`.

## Install

```bash
python -m pip install .
datautil --help
```

Python 3.10+ is required. Runtime dependencies are `PyYAML` and `datasketch`.

See the [usage guide](docs/usage.md) for command examples, the [design document](docs/design.md) for architecture and planned changes, and [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.

## Commands

```bash
datautil formats
datautil detect conversations.jsonl
datautil inspect conversations.jsonl --sample 1000
datautil analyze conversations.jsonl
datautil analyze conversations.jsonl -o analysis.jsonl --report analysis-summary.jsonl
datautil validate conversations.jsonl --from sharegpt --to openai-messages
datautil convert conversations.jsonl -o training.jsonl --from sharegpt --to openai-messages
datautil transform training.jsonl -o cleaned.jsonl --from openai-messages --rules examples/clean-rules.yaml
datautil dedup scan training.jsonl -o groups.jsonl --threshold 0.8
datautil dedup scan training.jsonl -o prompt-groups.jsonl --scope user
datautil dedup apply training.jsonl --groups groups.jsonl -o unique.jsonl --keep first
datautil classify unique.jsonl --sample 1000 -o labels.jsonl --model YOUR_MODEL
datautil classify unique.jsonl --sample 1000 -o usage-labels.jsonl --model YOUR_MODEL --type usage
datautil stats labels.jsonl
datautil stats usage-labels.jsonl
datautil sample unique.jsonl -o sample.jsonl --count 100 --seed 42
datautil split unique.jsonl --train train.jsonl --test test.jsonl --groups groups.jsonl --seed 42
datautil run pipeline.yaml
```

Classification calls an external OpenAI-compatible `/chat/completions` endpoint. `--model` is the model ID served by that endpoint (for example, the name supplied by your provider), not a model bundled with `datautil`. Set `DATAUTIL_API_KEY`; set `DATAUTIL_MODEL` to omit `--model` on each run. `DATAUTIL_API_BASE_URL` selects a compatible endpoint. Only `classify` makes model calls and incurs provider costs. The command requires `--sample N` or `--all`. By default, `--type subject` assigns exactly one primary subject label per conversation:

| Label | Scope |
|---|---|
| `code` | Programming and software engineering |
| `math` | Mathematics, statistics, and formal reasoning |
| `science` | Natural sciences and research |
| `technology` | Hardware, networks, devices, and non-programming IT |
| `health` | Medicine, health, and clinical care |
| `law` | Legal questions and compliance |
| `finance` | Personal finance, accounting, and investing |
| `business` | Management, marketing, and commerce |
| `humanities` | History, philosophy, religion, and arts |
| `social_science` | Politics, economics, psychology, and society |
| `education` | Teaching and learning methods |
| `writing` | Writing, editing, translation, and creative text |
| `general` | Everyday assistance without a specialized subject |
| `other` | Insufficient context, equally mixed subjects, or no match |

For subject-specific tutoring or writing, use the subject label (`code`, `health`, etc.). Use `education` or `writing` when the activity itself is the focus. [`examples/subject-labels.yaml`](examples/subject-labels.yaml) is an optional YAML file listing each label and its meaning. You do not need to pass a file for the built-in labels. To change them, copy and edit the file, then pass it with `--labels`:

```yaml
code: Programming and software engineering
math: Mathematics
other: Everything else
```

`--type usage` labels the purpose of each conversation:

| Label | Purpose |
|---|---|
| `writing` | Write or edit text |
| `advice` | Give tailored guidance |
| `technical` | Help with code, math, or data |
| `media` | Create or examine media |
| `information` | Find factual information |
| `personal` | Chat, reflect, or role-play |
| `other` | Unclear or none of these |

The labels appear under `usage_topic`; `datautil stats usage-labels.jsonl` infers that field automatically. [`examples/usage-labels.yaml`](examples/usage-labels.yaml) contains the editable label list; it is optional. Use `--by` if a file contains both label fields or is empty. A coding tutorial can be `code` under `--type subject` and `advice` under `--type usage`.

## Conversation statistics

`datautil analyze INPUT` prints ASCII percentage charts to stdout. It shows labels from the input format: OpenAI message roles, ShareGPT `human`/`gpt`, Alpaca `instruction`/`output`, or field paths and roles from a YAML profile. Roles and mapped fields use total messages as the denominator; conversation length, text length, thinking, and tool use use total conversations. Thinking and tool sections appear only when those features are observed; empty datasets show no charts. Add `-o analysis.jsonl` to write one row per complete conversation; the charts then go to stderr. Use `--json` to print the complete machine-readable summary, or `--report summary.jsonl` to save it independently of the terminal view. JSON keeps canonical `role_counts` alongside `source_label_counts` and `source_format`.

Histogram intervals are computed from each dataset using the Freedman–Diaconis rule (interquartile range and sample size). A zero interquartile range uses Sturges' rule; zero is shown separately for text and tool-chain lengths. Terminal charts cap nonzero intervals at 12, and the JSON summary records each interval and the method used. Since intervals depend on the dataset, compare their numeric boundaries before comparing charts from different datasets.

The same command reads assistant `reasoning_content` fields and complete `<think>...</think>` blocks. When both occur in one message, the field takes precedence to avoid double counting. It reports thinking presence, Unicode character count, source, malformed tags, and tool-call metrics.

Tool-chain length is the number of consecutive assistant tool-call **rounds** before a final assistant answer or a new user turn; tool result messages do not end the chain. Multiple calls issued in one assistant message count as one round and contribute to `max_parallel_calls`. The report includes per-chain and per-record maximum length distributions, total calls, and calls without a matching tool result.

For OpenAI messages, `reasoning_content` is preserved on conversion to the same format. A custom message or pair profile can map a source field with `fields.thinking`; exporting to a format without a thinking field fails rather than dropping it. Inline `<think>` blocks remain part of `content` during conversion.

Pipeline YAML lists steps in order. Use `@step_id` as the input for a previous step's output:

```yaml
steps:
  - id: converted
    command: convert
    input: source.jsonl
    output: converted.jsonl
    from: sharegpt
    to: openai-messages
  - id: candidates
    command: dedup scan
    input: "@converted"
    output: groups.jsonl
    threshold: 0.8
  - id: split
    command: split
    input: "@converted"
    from: openai-messages
    groups: groups.jsonl
    train: train.jsonl
    test: test.jsonl
```

`dedup scan` first finds exact matches of normalized text with SHA-256, then uses `datasketch` MinHash LSH to find near-match candidates, which are checked with exact 5-character-shingle Jaccard similarity. At `--threshold 1`, exact shingle sets are compared directly. `--scope conversation` (default) compares the whole conversation, including roles, text, tool names, and arguments; `--scope user` compares only user messages. Records without usable user text remain singletons in user scope and are counted in the report. Review candidate groups before `dedup apply`; `pairs[].match` identifies `exact` or `near` edges. LSH may miss near matches, and rescanning can change groups because its candidate index differs from earlier versions. Older group files still work with `dedup apply` and `split --groups`, which keeps each group wholly in one partition. The output format follows its filename extension: `.jsonl` and `.ndjson` are line-delimited; `.json` is an array. Conversion fails by default if the destination cannot represent the source data.

For large inputs, `sample`, `split`, `classify`, and `dedup apply` stream records and use temporary SQLite indexes where needed. `dedup scan` stores normalized text and shingles on disk, but its MinHash LSH index and matched groups still grow with input size. These commands need temporary disk space. A fixed seed gives repeatable results; `split` assignments can differ from versions that shuffled all records in memory. `dedup apply` includes at most 1,000 decisions in its summary and sets `decisions_truncated` when more were made.

`validate` continues after malformed JSONL rows, reports duplicate record IDs, and retains at most 1,000 issue details by default. Use `--max-errors N` to change the detail limit; `issue_count` always counts all detected issues. The limit controls reporting, not how many rows are checked.

## Custom format profiles

ClickHouse's `JSONEachRow` describes the **file container** (one JSON object per line). The columns inside each row are dataset-specific. A YAML profile maps those columns to a conversation record; it does not connect to ClickHouse or run SQL. See [`examples/clickhouse-inference.yaml`](examples/clickhouse-inference.yaml):

```yaml
version: 1
name: clickhouse-inference
container: JSONEachRow
layout: pair
fields:
  id: request_id
  system: request.system_prompt
  prompt: request.prompt
  completion: response.text
```

For a ClickHouse export saved as `inference.json` despite containing one object per line:

```bash
datautil convert inference.json -o conversations.jsonl \
  --format-config examples/clickhouse-inference.yaml \
  --from clickhouse-inference --to openai-messages
```

Use `layout: messages` for a nested message array. Map `fields.messages` from the root; `role`, `content`, and optional tool fields are paths relative to each message. `role_map` maps source role names to canonical roles and must be one-to-one. See [`examples/custom-chat.yaml`](examples/custom-chat.yaml). Use `--format-config` repeatedly to load multiple profiles, including a custom output format:

```bash
datautil convert conversations.jsonl -o custom.jsonl \
  --format-config examples/custom-chat.yaml \
  --from openai-messages --to custom-chat
```

Only dotted object-key paths are supported, such as `request.prompt`; arrays and expressions are not evaluated. Top-level unmapped fields are preserved as metadata. Unmapped fields within a message cause an error unless explicitly listed under `ignore_message_fields` in the profile. Tool calls require `tool_calls` and `tool_call_id` mappings; otherwise conversion fails rather than dropping them.

## Configurable processing

`datautil transform` applies a versioned YAML rule list in order. Supported operations are `trim_content`, `collapse_whitespace`, `replace_text`, `drop_empty_messages`, and `keep_if`. `keep_if` combines `min_messages`, `max_messages`, `min_chars`, `max_chars`, `roles_all`, and `roles_any` with AND. Records failing a condition are dropped and counted by rule; remaining records are validated before output. See [`examples/clean-rules.yaml`](examples/clean-rules.yaml).

For a complete workflow, [`examples/prepare-pipeline.yaml`](examples/prepare-pipeline.yaml) places rules inline before a deterministic sample step:

```bash
datautil run examples/prepare-pipeline.yaml
```

Pipeline paths resolve relative to the YAML file, and `@step_id` refers to an earlier output. A transform step can use either inline `rules: [...]` or `rules: path/to/rules.yaml`. Only built-in operations are allowed; YAML cannot invoke arbitrary code or shell commands.

Reports are JSONL files with one summary object. Exit codes: 0 success, 1 data/operation error, 2 CLI argument error, 3 I/O error.
Read-only summaries from `detect`, `inspect`, `validate`, and `stats` print JSON to stdout. `analyze` prints ASCII charts by default and JSON with `--json`. Commands that write a dataset or per-record artifact print their summary to stderr.
