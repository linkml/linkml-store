#!/usr/bin/env python
"""Sync the vendored copy of linkml's TypeDB naming rules.

The TypeDB adapter has to choose exactly the labels that ``gen-typedb`` wrote into the
schema it loads into. Those rules live in ``linkml.generators.typedb.naming``, which is
not yet in any linkml release, so a verbatim copy is vendored into the adapter and this
script keeps it current.

Usage::

    python scripts/sync_typedb_naming.py           # update the vendored copy
    python scripts/sync_typedb_naming.py --check   # fail if it is stale

``--check`` is what CI and the equality test use. Both require ``linkml`` to be
importable; without it there is nothing to compare against and the script says so
rather than pretending the copy is current.

Once a linkml release carries the module, delete the vendored file and this script and
import ``linkml.generators.typedb.naming`` directly.
"""

import argparse
import sys
from pathlib import Path

VENDORED = Path(__file__).resolve().parent.parent / "src" / "linkml_store" / "api" / "stores" / "typedb" / "naming.py"

MARKER = "# >>> BEGIN VENDORED CONTENT >>>"

HEADER = f"""# ---------------------------------------------------------------------------
# VENDORED FILE - DO NOT EDIT BY HAND
#
# Verbatim copy of linkml.generators.typedb.naming.
#
# The adapter must choose exactly the labels gen-typedb wrote into the schema it
# loads into, so these rules cannot be reimplemented here. They are vendored rather
# than imported because the module is not in any linkml release yet, and uv sources
# do not survive into published wheel metadata.
#
# To update:   just sync-typedb-naming
# To verify:   python scripts/sync_typedb_naming.py --check
#
# Everything below the marker is byte-identical to upstream, enforced by
# tests/test_api/test_typedb_adapter.py::test_vendored_naming_matches_upstream
# whenever linkml is installed.
#
# When a linkml release includes the module, delete this file and import
# linkml.generators.typedb.naming directly.
# ---------------------------------------------------------------------------
{MARKER}
"""


def upstream_source() -> str:
    """Return the source text of linkml's naming module.

    :raises SystemExit: if linkml is not importable
    """
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
