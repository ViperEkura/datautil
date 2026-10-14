# datautil design

This document describes the current architecture and planned extensions. Status is stated per feature; a class in a diagram does not by itself imply a public Python interface.

## CLI command system

The CLI exposes four groups based on operation: `format`, `analyze`,
`prepare`, and `pipeline`. Flat spellings remain compatible aliases.
The implemented command tree is:

```text
datautil
├── format
│   ├── list
│   ├── detect INPUT
│   └── convert INPUT -o OUTPUT --to FORMAT
├── analyze
│   ├── inspect INPUT
│   ├── validate INPUT [--to FORMAT]
│   ├── summary INPUT [--json]
│   ├── labels LABELS.jsonl [--by domain|usage_topic]
│   └── classify INPUT (--sample N | --all) -o LABELS.jsonl
├── prepare
│   ├── clean INPUT -o OUTPUT --from FORMAT --rules RULES.yaml
│   ├── dedup
│   │   ├── scan INPUT -o GROUPS.jsonl
│   │   └── apply INPUT --groups GROUPS.jsonl -o OUTPUT --keep first
│   ├── sample INPUT -o OUTPUT --count N
│   ├── split INPUT --train TRAIN --test TEST
│   └── build RECIPE.yaml
└── pipeline
    └── run PIPELINE.yaml
```

Conversation operations accept the built-in OpenAI messages, ShareGPT,
and Alpaca adapters or YAML profiles. `prepare build` accepts document
JSONL and Parquet from its recipe. Standalone document conversion,
inspection, validation, and reviewable document/paragraph dedup remain
planned, as do `analyze score`, `analyze contamination scan`, and
`pipeline validate`. Help registers only implemented actions.

### Routing and compatibility

`application/routing.py` maps user-facing command paths to stable
application operation names. `application/parser.py` defines each
operation's options once, then attaches the template under its group.
Parsing returns the existing operation name to `CommandSpec`, so CLI
and pipeline execution share the same handlers and report schemas.
For example, both `analyze summary` and the flat `analyze INPUT`
produce the existing `command: analyze` report. No route-specific
fields are added to report parameters.

Flat aliases are translated before argument parsing. `clean` maps
to `prepare clean`, `stats` to `analyze labels`, `pretrain run`
to `prepare build`, and `run` to `pipeline run`. The removed
`transform` spelling is not restored. `analyze --help` shows the
group; `analyze summary --help` shows the operation. A bare filename
matching an analyze subcommand can be passed as a path such as
`./labels` when using the legacy flat syntax.

Pipeline YAML accepts grouped command strings such as
`command: prepare clean` alongside previous spellings such as
`command: clean`. Both resolve through the same routing table.
The existing step allowlist remains in force, so nested pipeline runs
and corpus builds are not accepted as generic pipeline steps.
Inline cleaning rules and dataset references retain their behavior;
summary output remains a report artifact instead of a dataset.
`prepare build` uses its own recipe workflow.

### Command contracts and future format support

Use `INPUT` for datasets, `-o/--output` for the primary artifact,
`--report` for JSONL summaries, `--from` and `--to` for formats,
and `--seed` for reproducible selection. `--format-config` maps fields;
`--rules` changes or filters records; planned `--checks` enforces
quality constraints. Read-only results go to stdout; file-producing
commands send summaries to stderr. `--json` selects machine-readable
output where ASCII charts are default. Exit codes remain 0 success,
1 data or operation error, 2 usage error, and 3 I/O error. Classification
requires `--sample N` or `--all` before external model calls.

As document adapters expand, an adapter should declare record kind and
operation capabilities. Shared commands should dispatch through those
capabilities rather than introduce separate `chat` and `corpus`
command families. Unsupported combinations should fail before output
replacement. Document/paragraph and conversation/user dedup scopes
must keep distinct semantics and versioned artifacts.

The [CLI Guidelines](https://clig.dev/) recommend consistent subcommand
names, flags, help, and examples. The [GitHub CLI manual](https://cli.github.com/manual/gh_pr)
shows two-level command families. The [uv CLI reference](https://docs.astral.sh/uv/reference/cli/)
shows direct commands alongside nested families. These are interaction
references, not project dependencies.

## Implemented package architecture

The distribution keeps the standard `src/datautil` layout. Dependencies flow from the
CLI and application layer into the conversation or pretraining domain, and from
those domains into `common`:

```text
src/datautil/
  cli.py, __main__.py       command-line presentation
  application/              shared parser, CommandSpec/CommandResult, dispatch, pipeline
  conversation/             conversation model, format profiles, validation,
                            analysis, transformation, dedup, classification, sampling
  pretrain/                 independent document model, JSONL/Parquet sources,
                            quality, dedup, recipe validation, corpus runner
  common/                   DataError/Issue, JSONL and atomic output, disk index
```

`Conversation` and `DocumentRecord` are separate models with no shared record
base class. The format registry is scoped to one command execution. YAML profiles
cannot leak into later CLI or pipeline steps. JSONL and Parquet document sources
implement the same reading protocol; the pretraining runner chooses one from the
recipe and performs quality checks, document dedup, paragraph dedup, post-dedup
quality recheck, mixing, and writing. Dedup index updates for one candidate
document commit or roll back together.

CLI arguments and pipeline steps become `CommandSpec(name, options)`; the
application dispatcher returns `CommandResult(exit_code, summary, datasets,
reports)`. The CLI renders messages and exit codes. The pipeline resolves
relative paths and `@step_id` references before calling the dispatcher, and
stops on the first nonzero status. `prepare build` remains independent of the
generic YAML pipeline.

The stable Python exports are `Conversation`, `DatasetRecord`, `Message`,
`ToolCall`, `YamlFormatAdapter`, `load_profiles`, and `__version__` from
`datautil`. Internal module paths moved in this refactor; callers should use
the CLI or those package exports. Generated `build/`, `dist/`, and
`*.egg-info` are excluded from source migration.

## Pretraining document format (P0)

`record_id` identifies the source document and stays stable when text is
cleaned. `source_location` points to the original file and row. Optional URL,
crawl time, language, license, and source-specific fields stay in `metadata`
without invented values. Current reports record input and recipe fingerprints;
explicit adapter versions and paragraph character offsets remain planned.

The first interchange format is UTF-8 document JSONL, one complete document per line:

```json
{"id":"source-0001","text":"First paragraph.\n\nSecond paragraph.","metadata":{"source":"web","url":"https://example.org/page"}}
```

Preserve embedded newlines in `text`; a physical JSONL line is one document, not one paragraph. `prepare build` reads generic JSONL and Parquet rows and maps their fields with YAML, including C4-style exports. PyArrow is a default dependency. Plain `.txt`, compressed shards, and a standalone document conversion command remain planned. Token IDs, tokenizer version, sequence packing, and training splits belong to a separate training export, since they depend on a particular model recipe. Do not put perplexity or dedup decisions into the source document body; write keyed score and decision artifacts.

The initial implementation round-trips text and JSON-compatible metadata through document JSONL or Parquet, derives IDs when absent, streams input batches, and stores corpus state in temporary SQLite files. Tests cover mixed JSONL/Parquet sources, metadata mapping, duplicate removal, and deterministic output. Remaining acceptance work includes broader malformed-row coverage, explicit cross-kind conversion diagnostics, and large-corpus disk and memory benchmarks.

## Pretraining quality and deduplication (P0)

The first `gopher` and `c4` presets in `pretrain/quality.py` are explicitly opt-in English subsets, not exact reproductions of the original pipelines. They report rejection checks with measured values and thresholds; `quality.language: en` is required. Generic character limits work without a language preset. Full rule coverage, per-threshold overrides, versioned preset semantics, per-rule pass counts, and a labeled-sample benchmark remain planned. English-specific heuristics must not be silently applied to non-English or code corpora. See the [Gopher paper](https://arxiv.org/abs/2112.11446) and [T5/C4 paper](https://www.jmlr.org/papers/v21/20-074.html).

`prepare build` performs document-level exact and near deduplication, then removes repeated paragraphs split at blank lines. Its MinHash band buckets and candidate shingle sets live in temporary SQLite files; each LSH candidate still receives exact Jaccard verification. Candidate overflow fails explicitly. The optional decisions artifact records kept IDs, similarity, and paragraph index, and the operation applies these decisions immediately. A separate scan/review/apply workflow, character-span annotations, benchmarked peak memory and temporary disk use, and comparison with conversation dedup remain planned. Source order sets duplicate precedence; the subsequent document-count mix is computed from surviving documents.

## Local scoring and benchmark protection

P1 perplexity scoring uses an AstrAI-trained small LM served locally on the eight-RTX-5090 host. The CLI streams documents in batches to the serving endpoint and writes `record_id`, token count, total negative log-likelihood, mean negative log-likelihood, perplexity, model and tokenizer identifiers, context window, stride, and failures to a separate artifact. There is no third-party API charge, but GPU time and serving capacity still need budgeting. Define truncation and sliding-window behavior, retry semantics, and deterministic ordering before using scores as filters. Calibrate score distributions by language and source; a global perplexity cutoff would mix unlike populations.

Benchmark decontamination should precede evaluation-driven recipe selection. Build versioned protected references for MMLU, HumanEval, and GSM8K from the exact evaluation snapshots in use. Normalize prose and code separately, index configurable word/token n-grams, retrieve candidates, and verify overlap spans against source documents before quarantine or removal. Report benchmark, split, reference version, normalization, n, matched span, document ID, and decision; keep train and evaluation references separate. Common short phrases are not sufficient evidence by themselves, and n-gram matching does not catch paraphrases or translated copies. Compare evaluation results on full and clean subsets and do not treat a contaminated score as a clean capability estimate. The [GPT-3 paper](https://arxiv.org/abs/2005.14165) documents an n-gram overlap analysis; [rephrasing research](https://arxiv.org/abs/2311.04850) describes a limitation of surface matching.

## Validation layers

| Layer | CLI | Responsibility | Status |
|---|---|---|---|
| Detection | `datautil format detect INPUT` | Identify likely container and record format; detection does not establish validity | Available |
| Structural validation | `datautil analyze validate INPUT --from FORMAT` | Parse records and check required fields, types, roles, nonempty messages, tool result references, and duplicate record IDs | Available |
| Quality checks | `datautil analyze validate INPUT --from FORMAT --checks checks.yaml` | Apply configurable record and dataset constraints | Planned |
| Export compatibility | `datautil analyze validate INPUT --from FORMAT --to FORMAT` | Check whether the target adapter can serialize each record without loss | Available |

No separate `--mode` switch is needed. `--checks` adds quality checks and `--to` adds target compatibility checks to the same validation pass.

`--format-config` defines where fields are located in a custom input or output format. It is distinct from `--checks`, which defines acceptable data values and relationships. `clean --rules` remains a data modification operation; validation must not modify records.

## Planned checks file

```yaml
version: 1
checks:
  unique_record_id: true
  allowed_roles: [system, user, assistant, tool]
  message_count:
    min: 2
    max: 50
  content_chars:
    max_per_message: 20000
  require_user_message: true
  tool_results: matched
```

Version 1 should accept only documented keys and valid value types. A bad checks file is a configuration error and stops before reading data. Checks should have stable codes so reports can be compared across runs. The built-in structural checks always run; the file adds constraints rather than disabling basic validation.

## Planned execution and report behavior

1. Load and validate the format profile and checks file before opening the input.
2. Read each JSONL record, reporting its line number. A malformed line becomes a validation issue; later lines are still checked. For a JSON array, report the array record index. An explicit partial-scan marker for unrecoverable array errors remains planned.
3. Parse the record, run structural checks, then quality checks. Run target serialization only for records that parsed successfully.
4. Run dataset-wide checks such as duplicate record IDs while reading. Keep the first location for each ID so the issue can point to both occurrences.
5. Aggregate counts by issue code and check name. `--max-errors N` already limits retained issue details without silently reporting success; per-check aggregation remains planned.

The existing `--report` remains a JSONL summary artifact. A planned issue entry has `code`, `location`, `record_id` when known, `check` when a configurable check produced it, and `message`. The summary includes `read_count`, `valid_count`, `issue_count`, `issues`, and `truncated`. An issue returns exit code 1; a malformed CLI argument returns 2; an I/O failure returns 3. JSON output to stdout remains the default for `validate`.

## Other operations

- `analyze` measures conversation length, text length, observed thinking, tool calls, and role distributions. Histogram intervals are chosen from the input data.
- `dedup scan` finds normalized exact matches, then uses MinHash LSH to propose near matches and exact Jaccard to confirm candidates. `dedup apply` consumes reviewed groups.
- `classify` calls an external model and writes per-record labels. `stats` consumes existing labels without model calls. A terminal percentage view for `stats --by domain` is planned; JSON output must remain available for scripts.
- `sample` and `split` use a seed. `split --groups` keeps grouped records in one partition.
- `run` executes configured steps in order and passes prior outputs by `@step_id`.

## Prioritized roadmap

The user-set priority is pretraining data support. Each item should be reviewable and tested before the next depends on it. Ordinary feature commits do not require a version bump; a release can bundle several completed items.

| Priority | Work | Acceptance |
|---|---|---|
| P0.1 | Document records, JSONL/Parquet mapping, and deterministic document-count mixing: initial implementation available | Add broader format/error coverage and large-corpus benchmarks |
| P0.2 | Opt-in Gopher/C4 rule subsets: initial implementation available | Complete and version the presets, expose threshold overrides, and audit pass/reject distributions |
| P0.3 | Document and paragraph dedup with disk-backed LSH buckets: initial implementation available | Add a separate review/apply flow, stable span offsets, and measured disk/memory limits |
| P0.4 | Add MMLU, HumanEval, and GSM8K n-gram overlap screening before evaluation-driven recipe choices | Reports name the exact benchmark snapshot and matched spans; clean-subset scores are distinguishable from full-set scores |
| P1 | Score documents with the locally served AstrAI small LM | Score artifact records model/tokenizer and token accounting; retries and failures are explicit; no external API is required |

The earlier validation plan remains in the [validation layers](#validation-layers) section, but its remaining JSON-array partial-scan marker and configurable `--checks` are now backlog work. ASCII label statistics, classifier retry support, a Python adapter extension boundary, direct ClickHouse reads, and release readiness also remain backlog items. Keep the existing large-file benchmarks as a baseline for the document work. Update the design and usage guides whenever behavior or report fields change.

### Large-file benchmark (phase 4, partial)

The first implementation pass streams records for `sample`, `split`, `classify`, `analyze`, `clean`, and `dedup apply`. Temporary SQLite files hold IDs, group membership, or selection decisions. `dedup scan` stores normalized text and shingles in SQLite while retaining the `datasketch` MinHash LSH index and match graph in memory. Temporary disk use therefore grows with input size, and dedup scanning is not constant-memory.

Measured on 5090x8 with Python 3.12 and `/usr/bin/time`, using 100,000 synthetic records in an 18 MB JSONL file (5,000 records for the candidate-heavy scan):

| Command | Old peak RSS | New peak RSS | Old time | New time | Result |
|---|---:|---:|---:|---:|---|
| `sample --count 1000` | 127 MiB | 28 MiB | 1.77 s | 2.63 s | Output byte-identical |
| `split --test-ratio 0.1` | 147 MiB | 31 MiB | 1.42 s | 4.51 s | 90,000 train / 10,000 test; deterministic assignment changed |
| `dedup apply` with 10,000 pairs | 140 MiB | 29 MiB | 2.28 s | 3.92 s | Dataset output byte-identical |
| `analyze --json` | 36 MiB | 28 MiB | 2.36 s | 2.70 s | Output byte-identical |
| `clean --rules` | 36 MiB | 28 MiB | 2.98 s | 3.24 s | Output byte-identical |
| `dedup scan` on 5,000 similar records | 240 MiB | 176 MiB | 25.08 s | 49.90 s | Group file byte-identical |

These are local measurements, not performance guarantees. The 5,000-record scan generated over three million LSH candidates; its cost is highly dependent on input similarity. Further work should benchmark less repetitive datasets, measure temporary disk use, and evaluate a disk-backed candidate index without changing the exact-match and Jaccard checks.

A separate split of the 100,000-record file with 10,000 two-record groups used 31 MiB peak RSS and 4.66 seconds. It produced 10,000 test records and no group crossed partitions. Replaying stdin through a temporary file preserves the original `-:line` source location, so generated record IDs remain deterministic.


### Pretraining document benchmark (synthetic)

Run `.venv/bin/python benchmarks/pretrain_corpus.py --rows N` on 5090x8 (Python 3.12) to reproduce this workload. The generator writes three roughly 60-word paragraphs per document, makes every twentieth document an exact copy of its predecessor, and reuses a middle paragraph in every tenth newly generated document. The recipe uses document near-dedup at 0.8, paragraph exact dedup at 1.0, `quality.min_chars: 200`, and selects half the input count. These are synthetic measurements, not throughput or duplicate-rate estimates for a real corpus.

| Input documents | Input size | Output documents | Elapsed | Peak RSS | Observed peak temp files | LSH candidates |
|---|---:|---:|---:|---:|---:|---:|
| 20,000 | 29.9 MiB | 10,000 | 75.2 s | 130.1 MiB | 383.7 MiB | 124,004 |
| 50,000 | 74.8 MiB | 25,000 | 239.6 s | 129.4 MiB | 965.0 MiB | 750,323 |

The benchmark polls temporary file sizes every 0.1 seconds, so the reported disk peak is an observed lower bound. Both runs had no shortfall; they removed 1,000 and 2,500 repeated paragraphs, respectively. Temporary files used about 13 times the JSONL input size. The 2.5-fold increase in rows took 3.2 times as long, with six times as many LSH candidates; candidate-heavy or longer real documents may behave differently. Benchmark real source samples and reserve temporary space before a production run.
