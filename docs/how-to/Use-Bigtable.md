# Using Google Cloud Bigtable with LinkML-Store

## Overview

LinkML-Store can store and query collections in [Cloud Bigtable](https://cloud.google.com/bigtable),
Google's wide-column key-value store.

| LinkML-Store | Bigtable |
|--------------|----------|
| Database     | Instance |
| Collection   | Table    |
| Object       | Row      |
| Field        | Column (in one column family, `d` by default) |
| Identifier   | Row key  |

## Installation

```bash
pip install 'linkml-store[bigtable]'
```

## Connecting

Handles have the form `bigtable://<project>/<instance>`:

```bash
linkml-store -d bigtable://my-project/my-instance -c persons insert persons.json
linkml-store -d bigtable://my-project/my-instance -c persons query -w 'occupation: Builder'
```

Optional query parameters:

- `column_family`: the column family that object fields are stored in (default `d`)
- `app_profile_id`: the [app profile](https://cloud.google.com/bigtable/docs/app-profiles) to use for reads and writes

If the project is omitted (`bigtable:///my-instance`), it is taken from the environment.
Credentials come from [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials),
e.g. `gcloud auth application-default login`.

The instance must already exist. Tables are created when you first insert into a collection.
The adapter uses the Bigtable admin API to list, create, truncate and delete tables, so the
credentials need those permissions as well as read/write access to data.

### Using the emulator

For local development, run the [Bigtable emulator](https://cloud.google.com/bigtable/docs/emulator)
and point the client at it:

```bash
gcloud beta emulators bigtable start --host-port=localhost:8086
export BIGTABLE_EMULATOR_HOST=localhost:8086
linkml-store -d bigtable://test-project/test-instance -c persons insert persons.json
```

The emulator accepts any project and instance name. The handle `bigtable` on its own is short for
`bigtable://test-project/test-instance`.

## Python usage

```python
from linkml_store import Client

client = Client()
db = client.attach_database("bigtable://my-project/my-instance", alias="bt")
persons = db.create_collection("Person", alias="persons")
persons.insert([
    {"id": "P1", "name": "Alice", "occupation": "Architect", "skills": ["design"]},
    {"id": "P2", "name": "Bob", "occupation": "Builder"},
])
qr = persons.find({"occupation": "Builder"})
facets = persons.query_facets(facet_columns=["occupation"])
```

## How data is stored

- Each top-level field of an object is a column in the column family. Values are JSON-encoded,
  so types, nested objects and lists are preserved.
- `None` values are not stored. Fields that are null come back missing.
- Only the latest version of each cell is kept (the column family is created with a max-versions GC rule of 1).
- If the collection's class has an identifier attribute in the schema, its value is the row key.
  Inserting an object with an existing identifier replaces that row. Without an identifier, row keys are random,
  so inserting the same object twice creates two rows.

Declare the identifier in the schema **before** loading data. Row keys are fixed when rows are written,
so rows loaded without an identifier cannot be looked up by it later.

### Existing tables

Tables written by other tools can be read too. Cells in the configured column family become top-level fields.
Cells in other column families are nested under the family name, e.g. `{"name": "x", "stats": {"views": 10}}`.
Values that are not JSON are returned as strings, or as hex strings if they are not valid UTF-8.

## Query performance

Bigtable has no secondary indexes. Its only fast access path is by row key.

- Where clauses that constrain the identifier, e.g. `{"id": "P1"}` or `{"id": {"$in": ["P1", "P2"]}}`,
  are answered with direct row lookups.
- All other filtering, sorting and faceting is done client-side over a full table scan.
- `find` with no filter counts rows with a keys-only scan and reads only the rows it returns.
- `find_iter` streams rows in a single scan without counting, so prefer it for large tables.
- `upsert` is fast when `filter_fields` is the identifier. Otherwise each object needs a scan.

## Dropping data

`db.drop()` deletes **every table in the instance**, not just the ones linkml-store created.
The instance itself is not deleted. `collection.delete_where({})` truncates a single table.
