# datautil usage guide

This guide covers the current CLI. Run examples from the repository root and replace the input filenames with your own datasets. The repository does not include datasets.

## Install

Python 3.10 or newer is required. In the development checkout on 5090x8:

```bash
cd ~/kxq/datautil
source .venv/bin/activate
python -m pip install -e .
datautil --version
datautil --help
```

Use `python -m pip install .` in a new environment. The current distribution name is `datautil-llm-dev`; the installed command is `datautil`.

## Detect, inspect, and validate

```bash
datautil formats
datautil detect openai.jsonl
datautil inspect openai.jsonl --sample 1000
datautil validate openai.jsonl --from openai-messages
datautil validate openai-tools.jsonl --from openai-messages --to sharegpt
```

`detect` identifies a likely format. `inspect` summarizes fields, roles, and issues. `validate` checks source records; `--to` also checks whether the target format can represent them. The last command reports `lossy_conversion` if the input has tool calls or thinking that ShareGPT cannot represent. Configurable `--checks` rules are planned but are not available yet; see the [design document](design.md).

Validation continues after malformed JSONL rows and detects duplicate record IDs across the file. It retains 1,000 issue details by default; `--max-errors N` changes that limit while `issue_count` keeps the full count. A duplicate ID issue includes the first occurrence's location. A malformed JSON array may still stop the scan because its record boundary cannot always be recovered.

## Convert formats

```bash
datautil convert sharegpt.jsonl -o /tmp/sharegpt-openai.jsonl --from sharegpt --to openai-messages
datautil convert alpaca.jsonl -o /tmp/alpaca-openai.jsonl --from alpaca --to openai-messages
```

Built-in formats are `openai-messages`, `sharegpt`, and `alpaca`. Conversion fails by default when the target cannot represent source fields. `.jsonl` and `.ndjson` outputs contain one record per line; `.json` outputs contain a JSON array.

Use `--format-config` for custom record layouts. The ClickHouse example reads an exported `JSONEachRow` file; it does not connect to a database:

```bash
datautil validate clickhouse-inference.jsonl \
  --format-config examples/clickhouse-inference.yaml \
  --from clickhouse-inference --to openai-messages

datautil convert clickhouse-inference.jsonl -o /tmp/clickhouse-openai.jsonl \
  --format-config examples/clickhouse-inference.yaml \
  --from clickhouse-inference --to openai-messages
```

`examples/custom-chat.yaml` demonstrates nested message fields and role mapping. `--format-config` maps fields; it does not define data quality rules.

## Analyze conversations

```bash
datautil analyze openai-tools.jsonl
datautil analyze openai-tools.jsonl --json
datautil analyze openai-tools.jsonl -o /tmp/analysis.jsonl --report /tmp/analysis-summary.jsonl
```

By default, `analyze` prints ASCII percentage charts. Role percentages use the number of messages as their denominator; thinking and tool-use coverage use the number of conversations. `-o` writes one row per conversation, `--report` saves the summary, and `--json` prints the summary as JSON. Text length is the sum of all message `content` character counts in a conversation.

## Find and apply duplicates

```bash
datautil dedup scan openai.jsonl -o /tmp/groups.jsonl --scope conversation
cat /tmp/groups.jsonl
datautil dedup apply openai.jsonl --groups /tmp/groups.jsonl -o /tmp/unique.jsonl --keep first
```

Use `scan` to generate candidate groups, review them, and then use `apply`. The default `--scope conversation` compares complete conversations. `--scope user` compares only user messages, so the same prompt with different answers can enter one group. Scanning finds normalized exact matches first, then uses MinHash LSH to propose near matches and exact Jaccard similarity to confirm them. LSH can still miss near duplicates.

## Classify and summarize domains

Existing labels can be summarized without calling a model:

```bash
datautil stats labels.jsonl --by domain
datautil stats labels.jsonl --by usage_topic
```

In the JSON output, `domains` or `usage_topics` contains counts and `shares` contains proportions. The current release does not provide an ASCII chart for domain statistics.

To label your own data, configure an endpoint compatible with OpenAI `/chat/completions`:

```bash
export DATAUTIL_API_KEY='your-key'
export DATAUTIL_API_BASE_URL='https://your-provider.example/v1'

datautil classify input.jsonl --from openai-messages --type subject \
  --sample 1000 --seed 42 --model YOUR_MODEL -o labels.jsonl
datautil stats labels.jsonl --by domain
```

`classify` requires `--sample N` or `--all`. It sends selected conversations to the configured endpoint and may incur provider charges. `--type subject` assigns a domain; `--type usage` assigns a use category. Pass `--labels` to load custom labels from YAML.

Classification processes selected records one at a time. Its report retains at most 1,000 failure details; `issue_count` gives the full count and `issues_truncated` indicates whether details were capped.

## Transform, sample, split, and run a pipeline

```bash
datautil transform openai.jsonl -o /tmp/cleaned.jsonl \
  --from openai-messages --rules examples/clean-rules.yaml
datautil sample openai.jsonl -o /tmp/sample.jsonl --count 5 --seed 42
datautil split openai.jsonl \
  --train /tmp/train.jsonl --test /tmp/test.jsonl --test-ratio 0.1 --seed 42
```

`transform` applies YAML cleaning and filtering rules in order. `sample` and `split` use a seed for reproducibility. `split --groups /tmp/groups.jsonl` keeps each candidate group in one partition.

For large files, selection uses temporary SQLite indexes rather than retaining every conversation in memory. Ensure the system temporary directory has free space. `sample` keeps the previous seeded selection order; `split` remains deterministic but its assignments can differ from older versions that shuffled all records in memory. `dedup apply` also streams its output and caps the `decisions` list in its report at 1,000 entries. `removed_count` counts unique records removed; `decision_count` counts all group decisions.

`run` executes pipeline YAML. `@step_id` refers to an earlier step's output. `examples/prepare-pipeline.yaml` shows the configuration shape; replace its `source.jsonl` path with your input before running it:

```bash
datautil run path/to/pipeline.yaml
```

## Reports and exit codes

Read-only results from `detect`, `inspect`, `validate`, and `stats` go to standard output. Commands that write files normally send their summaries to standard error. `--report` writes one JSONL summary with the input fingerprint, parameters, counts, issues, and output paths. Exit codes are `0` for success, `1` for a data or operation error, `2` for a CLI argument error, and `3` for an I/O error.
