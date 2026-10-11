"""Command-line presentation for the datautil application."""

from __future__ import annotations

import json
import sys

from datautil.application.commands import execute
from datautil.application.parser import _parser
from datautil.application.results import CommandResult, CommandSpec
from datautil.common.errors import DataError


def _print_result(result: CommandResult) -> None:
    for stream, message in result.messages:
        print(message, file=sys.stdout if stream == "stdout" else sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute(CommandSpec(args.command, vars(args)), on_step=_print_result)
        _print_result(result)
        return result.exit_code
    except DataError as exc:
        print(
            json.dumps(
                {"error": exc.code, "message": str(exc), "location": exc.location},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    except (OSError, UnicodeError) as exc:
        print(
            json.dumps({"error": "io_error", "message": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
