"""CLI spellings mapped to stable application operations."""

from typing import Dict, List, Sequence

from datautil.common.errors import DataError

COMMAND_PATHS: Dict[str, str] = {
    "formats": "format list",
    "detect": "format detect",
    "convert": "format convert",
    "inspect": "analyze inspect",
    "validate": "analyze validate",
    "analyze": "analyze summary",
    "stats": "analyze labels",
    "classify": "analyze classify",
    "clean": "prepare clean",
    "dedup scan": "prepare dedup scan",
    "dedup apply": "prepare dedup apply",
    "sample": "prepare sample",
    "split": "prepare split",
    "pretrain run": "prepare build",
    "run": "pipeline run",
}

_OPERATIONS = {path: operation for operation, path in COMMAND_PATHS.items()}
_ANALYZE_ACTIONS = {
    path.split()[1] for path in _OPERATIONS if path.startswith("analyze ")
}


def resolve_operation(command: str) -> str:
    """Accept either a grouped command path or its legacy pipeline spelling."""
    if command in _OPERATIONS:
        return _OPERATIONS[command]
    if command in COMMAND_PATHS:
        return command
    raise DataError("invalid_command", "Unknown command: " + command)


def normalize_argv(argv: Sequence[str]) -> List[str]:
    """Translate flat CLI aliases before argparse reads the four-group tree."""
    args = list(argv)
    if not args:
        return args
    name = args[0]
    if name == "analyze":
        if len(args) > 1 and args[1] in _ANALYZE_ACTIONS | {
            "-h",
            "--help",
            "score",
            "contamination",
        }:
            return args
        return ["analyze", "summary"] + args[1:]
    if name == "dedup":
        return ["prepare", "dedup"] + args[1:]
    if name == "pretrain":
        if len(args) > 1 and args[1] == "run":
            return ["prepare", "build"] + args[2:]
        if len(args) > 1 and args[1] in {"-h", "--help"}:
            return ["prepare", "build"] + args[1:]
        return args
    if name in COMMAND_PATHS:
        return COMMAND_PATHS[name].split() + args[1:]
    return args
