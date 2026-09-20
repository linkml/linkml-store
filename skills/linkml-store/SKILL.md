---
name: linkml-store
description: Import, query, validate, index, and search structured data with LinkML Store. Use for local DuckDB collections, LinkML-aware YAML/JSON/TSV data workflows, backend configuration, and distinguishing exact queries from ranked similarity search.
---

# Work with LinkML Store

Use explicit database and collection selections. Inspect existing configuration
and data before changing a persistent store; `insert` appends by default and
rerunning an import can add duplicates.

## Select the runtime and backend

Examples use `uvx linkml-store`; use `uv run linkml-store` in its checkout.
Skill installation does not install the Python package or backend services.
Check `uvx linkml-store --help` and subcommand help against the installed release.

Use DuckDB for a local workflow unless the task calls for another backend.
Reuse the project's configuration when appropriate. The CLI otherwise discovers
`linkml.yaml` or `~/.linkml.yaml`; an explicit `-C` avoids relying on those defaults.
Put global options (`-C`, `-d`, `-c`, `-S`) before the subcommand.

For a new local collection, create `store-config.yaml` in the working directory:

```yaml
databases:
  local:
    handle: duckdb:///research.duckdb
```

Use an unused database path for a new example. For existing work, verify the
configured database and collection rather than creating a similarly named one.
Some backend features require package extras or a running service; do not switch
backends silently when a dependency or connection is missing.

## Import and query

Given `people.json` containing a list of records with `name` and `occupation`:

```bash
uvx linkml-store -C store-config.yaml -d local -c persons insert people.json
uvx linkml-store -C store-config.yaml -d local -c persons query \
  --where 'occupation: Scientist' --limit 20 --output-type json
```

`--where` is YAML; quote it as one shell argument. Use `--select '[name, occupation]'`
to project fields. For nested input documents, inspect `insert --help` for
`--json-select-query` so the intended objects become rows. Do not use `--replace`,
`store`, or `drop` as a workaround for an import or query error.

For SQL-capable backends, use explicit read-only SQL when answering questions:

```bash
uvx linkml-store -C store-config.yaml -d local query \
  --sql 'SELECT occupation, COUNT(*) AS n FROM persons GROUP BY occupation' \
  --output-type json
```

Do not combine `--sql` with `--where` or `--select`. In SQL mode put limits in the
SQL itself; the collection-query `--limit` option is not applied to raw SQL.
Avoid interpolating untrusted text into SQL.

## Validate against the intended schema

```bash
uvx linkml-store -C store-config.yaml -d local schema --output-type yaml
uvx --from 'linkml-store[validation]' linkml-store \
  -C store-config.yaml -d local -S schema.yaml validate \
  --output-type json --output validation.json
```

Validation requires the `validation` extra (the `linkml` package). In a checkout,
use `uv run --extra validation linkml-store` for this command.

An inferred schema describes the imported data; use the supplied domain schema
for meaningful constraint checks. Database validation is the default even when
`-c` is supplied. Add `-c persons` before `validate --collection-only` when only
that collection should be checked. Referential-integrity checks are enabled by
default. Read the validation results: this command can exit successfully while
returning validation errors, so exit 0 alone is not a clean validation result.

## Index and search when requested

For a small local similarity-search example, use the simple trigram index.
It is a demonstration index, not a production search recommendation:

```bash
uvx linkml-store -C store-config.yaml -d local -c persons index --index-type simple
uvx linkml-store -C store-config.yaml -d local -c persons search Scientist \
  --index-type simple --limit 5 --output-type json
```

Indexing writes additional data. Rebuild the index after changing its underlying
collection when needed. LLM indexes can require optional dependencies, model
downloads, or provider configuration; use them only when the task calls for that
kind of search. Ranked scores indicate similarity, not exact filter matches or
validated scientific associations.

Return the selected backend/database/collection, import counts when applicable,
query or search criteria, result path/count, and validation findings. Keep
changes to existing data within the requested operation.

See the [CLI tutorial](https://linkml.io/linkml-store/tutorials/Command-Line-Tutorial.html)
for configuration and backend-specific workflows.
