# Using Ibis with LinkML-Store

## Overview

LinkML-Store now supports [Ibis](https://ibis-project.org/) as a backend adapter. Ibis is a Python library that provides a unified interface to query multiple database backends using a consistent API. By integrating Ibis, LinkML-Store can now work with a wide variety of SQL and analytical databases through a single abstraction layer.

## Supported Backends

Through Ibis, LinkML-Store can connect to:

- **DuckDB** (recommended for analytics)
- **PostgreSQL**
- **SQLite**
- **BigQuery**
- **Snowflake**
- **MySQL**
- **ClickHouse**
- **Polars**
- And many more...

For a complete list of supported backends, see the [Ibis documentation](https://ibis-project.org/backends/).

## Installation

To use Ibis with LinkML-Store, install with the `ibis` extra:

```bash
pip install 'linkml-store[ibis]'
```

For specific backends, you may need additional dependencies. For example:

```bash
# For PostgreSQL
pip install 'ibis-framework[postgres]'

# For BigQuery
pip install 'ibis-framework[bigquery]'

# For multiple backends
pip install 'ibis-framework[duckdb,postgres,sqlite]'
```

## Connection Strings

Ibis connections use the format: `ibis+<backend>://<connection_details>`

### Examples

**DuckDB (in-memory):**
```python
handle = "ibis+duckdb:///:memory:"
# Or use the short form:
handle = "ibis://"
```

**DuckDB (file-based):**
```python
handle = "ibis+duckdb:///path/to/database.duckdb"
# Or short form:
handle = "ibis:///path/to/database.duckdb"
```

**PostgreSQL:**
```python
handle = "ibis+postgres://username:password@localhost:5432/database"
```

**SQLite:**
```python
handle = "ibis+sqlite:///path/to/database.sqlite"
```

**BigQuery:**
```python
handle = "ibis+bigquery://project_id/dataset_id"
```

## Python API Usage

### Basic Example

```python
from linkml_store import Client

# Create a client
client = Client()

# Attach an Ibis database (using DuckDB backend)
db = client.attach_database("ibis+duckdb:///:memory:", alias="mydb")

# Create a collection
persons = db.create_collection("Person")

# Insert data
persons.insert([
    {"id": "P1", "name": "Alice", "age": 30},
    {"id": "P2", "name": "Bob", "age": 25},
    {"id": "P3", "name": "Charlie", "age": 35},
])

# Query data
results = persons.find({"age": 30})
print(results)
# [{"id": "P1", "name": "Alice", "age": 30}]

# Use LinkML Query API
from linkml_store.api.queries import Query

query = Query(
    where_clause={"age": 30},
    sort_by=["name"],
    limit=10
)
result = persons.query(query)
print(result.rows)
```

### Using PostgreSQL

```python
from linkml_store import Client

client = Client()

# Connect to PostgreSQL
db = client.attach_database(
    "ibis+postgres://user:password@localhost:5432/mydb",
    alias="pgdb"
)

# Create and populate a collection
collection = db.create_collection("Customer")
collection.insert([
    {"id": 1, "name": "ACME Corp", "revenue": 50000},
    {"id": 2, "name": "Tech Inc", "revenue": 75000},
])

# Query with aggregation
results = collection.find()
print(f"Total customers: {len(results)}")
```

### Using BigQuery

```python
from linkml_store import Client

client = Client()

# Connect to BigQuery
db = client.attach_database(
    "ibis+bigquery://my-project/my-dataset",
    alias="bqdb"
)

# Work with existing tables
collection = db.get_collection("my_existing_table")
results = collection.peek(limit=5)
print(results)
```

## Command Line Usage

```bash
# Using in-memory DuckDB via Ibis
linkml-store -d "ibis+duckdb:///:memory:" -c persons insert data.json

# Using PostgreSQL
linkml-store -d "ibis+postgres://user:pass@localhost/db" -c persons query

# Using file-based DuckDB
linkml-store -d "ibis+duckdb:///mydata.duckdb" -c persons validate
```

## Advanced Features

### Schema Introspection

Ibis backends support automatic schema introspection:

```python
# Connect to existing database
db = client.attach_database("ibis+postgres://user:pass@host/db", alias="db")

# Induce LinkML schema from database structure
schema_view = db.induce_schema_view()
print(schema_view.all_classes())
```

### Querying with Filters

```python
from linkml_store.api.queries import Query

# Complex queries
query = Query(
    where_clause={"age": {"$gt": 25}},  # Age greater than 25
    select_cols=["name", "age"],        # Select specific columns
    sort_by=["age"],                     # Sort by age
    limit=10,                            # Limit results
    offset=5                             # Skip first 5
)

results = collection.query(query)
```

### Working with DataFrames

```python
# Query and get results as pandas DataFrame
result = collection.query(Query(limit=100))
df = result.rows_dataframe

# Analyze with pandas
print(df.describe())
print(df.groupby("age").size())
```

## Benefits of Using Ibis

1. **Unified Interface**: Write once, run on multiple database backends
2. **Performance**: Ibis optimizes queries for each backend
3. **Flexibility**: Switch between backends without changing code
4. **Rich Ecosystem**: Leverage Ibis's powerful query capabilities
5. **Type Safety**: Benefit from Ibis's type system and query validation

## Comparison with Direct Backend Access

### Direct DuckDB
```python
db = client.attach_database("duckdb:///:memory:", alias="db")
```

### Via Ibis
```python
db = client.attach_database("ibis+duckdb:///:memory:", alias="db")
```

**Key Differences:**
- Ibis provides a consistent API across all backends
- You can switch backends by changing the connection string
- Ibis offers additional query optimization and features
- Direct backends may have backend-specific optimizations

## When to Use Ibis

**Use Ibis when:**
- You need to support multiple database backends
- You want a consistent query interface
- You're working with analytical/OLAP databases
- You need advanced query capabilities

**Use direct backends when:**
- You're committed to a single backend
- You need backend-specific features
- You want minimal dependencies

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'ibis'`:
```bash
pip install 'linkml-store[ibis]'
```

### Backend-Specific Issues

For backend-specific errors, ensure you've installed the required extras:
```bash
pip install 'ibis-framework[<backend>]'
```

### Connection Issues

Verify your connection string format matches the backend requirements. See the [Ibis documentation](https://ibis-project.org/backends/) for backend-specific connection details.

## Resources

- [Ibis Documentation](https://ibis-project.org/)
- [Ibis GitHub](https://github.com/ibis-project/ibis)
- [Supported Backends](https://ibis-project.org/backends/)
- [LinkML Store Documentation](https://linkml.io/linkml-store/)

## Example Use Cases

### Data Migration

Use Ibis to migrate data between different databases:

```python
# Source: PostgreSQL
source_db = client.attach_database("ibis+postgres://host/sourcedb", alias="source")
source_data = source_db.get_collection("customers").find()

# Target: BigQuery
target_db = client.attach_database("ibis+bigquery://project/dataset", alias="target")
target_collection = target_db.create_collection("customers")
target_collection.insert(source_data)
```

### Multi-Backend Analytics

Query data from multiple backends:

```python
# Get data from PostgreSQL
pg_db = client.attach_database("ibis+postgres://host/db", alias="pg")
transactions = pg_db.get_collection("transactions").find()

# Analyze in DuckDB (optimized for analytics)
duckdb_db = client.attach_database("ibis+duckdb:///:memory:", alias="analytics")
analytics = duckdb_db.create_collection("transactions_analytics")
analytics.insert(transactions)

# Run analytical queries
result = analytics.query(Query(
    select_cols=["customer_id", "SUM(amount) as total"],
    group_by=["customer_id"]
))
```
