#!/usr/bin/env python
"""Sync the TypeDB adapter's vendored copy of ``linkml.generators.typedb.naming``.

Pass ``--check`` to fail if the copy is stale. Delete this script once a linkml release ships the module.
"""

import argparse
import sys
from pathlib import Path

VENDORED = Path(__file__).resolve().parent.parent / "src" / "linkml_store" / "api" / "stores" / "typedb" / "naming.py"

MARKER = "# >>> BEGIN VENDORED CONTENT >>>"

HEADER = f"""# ---------------------------------------------------------------------------
# VENDORED FILE - DO NOT EDIT BY HAND
#
# Verbatim copy of linkml.generators.typedb.naming, which no linkml release ships yet.
# Update with `just sync-typedb-naming`; delete once a release includes the module.
# ---------------------------------------------------------------------------
{MARKER}
"""


def upstream_source() -> str:
    """Return the source text of linkml's naming module."""
    try:
        from linkml.generators.typedb import naming
    except ImportError:
        sys.exit(
            "linkml is not importable, so there is nothing to sync against.\n"
            "Install it (e.g. 'uv pip install -e ../linkml/packages/linkml') and retry."
        )
    return Path(naming.__file__).read_text()


def vendored_body() -> str:
    """Return the vendored copy's content below the marker, or '' if absent."""
    if not VENDORED.exists():
        return ""
    text = VENDORED.read_text()
    if MARKER not in text:
        return text
    return text.split(MARKER + "\n", 1)[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the copy is stale")
    args = parser.parse_args()

    upstream = upstream_source()
    current = vendored_body()

    if current == upstream:
        print(f"up to date: {VENDORED.relative_to(Path.cwd())}")
        return 0

    if args.check:
        print(
            f"STALE: {VENDORED.relative_to(Path.cwd())} differs from "
            "linkml.generators.typedb.naming.\nRun: just sync-typedb-naming",
            file=sys.stderr,
        )
        return 1

    VENDORED.write_text(HEADER + upstream)
    verb = "created" if not current else "updated"
    print(f"{verb}: {VENDORED.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
