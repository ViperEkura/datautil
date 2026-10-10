# Contributing to datautil

## Commit messages

Use this commit message format:

```text
type: short imperative description

- describe the behavior or design change
- mention important compatibility or migration details
- state the verification performed when relevant
```

- Use one of `fix`, `feat`, `chore`, `docs`, `refactor`, `perf`, `test`, `style`, `ci`, `build`, or `revert`.
- Keep the subject around 50 characters, in English, without a scope or trailing period.
- Use an imperative verb after the colon, such as `add`, `fix`, or `document`.
- Add body bullets for changes that need explanation; keep each bullet on one line.
- For measured performance claims, add a `Benchmark:` section with the environment, method, and before/after values.

Example:

```text
feat: add YAML processing rules

- apply cleaning and filtering rules in configuration order
- allow inline rules in pipeline steps
- verify source and installed-wheel tests
```

Keep commits focused. Do not change the package version in ordinary feature, fix, documentation, or refactor commits.

## Version and release policy

`pyproject.toml` and `src/datautil/__init__.py` must carry the same version. Bump them together only for an intentional release after a batch of changes is ready. The release commit uses `chore: bump version to X.Y.Z` and lists the release checks in its body. Do not bump the version for every commit or every wheel built locally. Until a release is requested, keep the current version even when features are added.

## Before submitting

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -q
```

For packaging changes, also build a wheel, install it in a clean virtual environment, and check `datautil --version` and `datautil --help`. Include relevant test or benchmark results in the commit body when they clarify the change.

Avoid rewriting published history during routine development. If the repository owner explicitly requests a history rewrite, verify the remote head first and use `--force-with-lease`.
