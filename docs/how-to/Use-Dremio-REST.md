# Using Dremio REST API with LinkML-Store

## Overview

LinkML-Store provides a `dremio-rest` adapter for connecting to [Dremio](https://www.dremio.com/) data lakehouse instances using the REST API v3. This is useful when the Arrow Flight SQL port (32010) is not accessible, such as when Dremio is behind a firewall or Cloudflare Access.

## When to Use This Adapter

Use `dremio-rest://` when:
- Dremio is behind Cloudflare Access or similar proxy
- The Arrow Flight port (32010) is blocked
- You need to connect via HTTPS (port 443)

Use `dremio://` (Flight SQL) when:
- You have direct network access to Dremio
- Port 32010 is accessible
- You need maximum query performance

## Installation

The Dremio REST adapter is included in the base linkml-store installation:

```bash
pip install linkml-store
```

## Connection String Format

```
dremio-rest://[username:password@]host[:port][?params]
```

### Examples

**Basic connection:**
```python
handle = "dremio-rest://lakehouse.example.com"
```

**With credentials in URL:**
```python
handle = "dremio-rest://user:pass@lakehouse.example.com"
```

**With default schema:**
```python
handle = "dremio-rest://lakehouse.example.com?schema=gold.tables"
```

**Disable SSL verification (for testing):**
```python
handle = "dremio-rest://localhost:9047?verify_ssl=false"
```

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema` | None | Default schema for unqualified table names |
| `verify_ssl` | `true` | Whether to verify SSL certificates |
| `username_env` | `DREMIO_USER` | Environment variable for username |
| `password_env` | `DREMIO_PASSWORD` | Environment variable for password |
| `cf_token_env` | `CF_AUTHORIZATION` | Environment variable for Cloudflare Access token |

## Environment Variables

The adapter reads credentials from environment variables by default:

| Variable | Description |
|----------|-------------|
| `DREMIO_USER` | Dremio username |
| `DREMIO_PASSWORD` | Dremio password |
| `CF_AUTHORIZATION` | Cloudflare Access token (if behind Cloudflare) |

### Getting the Cloudflare Access Token

If your Dremio instance is behind Cloudflare Access:

1. Open Dremio in your browser
2. Open Developer Tools (F12) → Application/Storage → Cookies
3. Copy the value of the `CF_Authorization` cookie
4. Set it as an environment variable:
   ```bash
   export CF_AUTHORIZATION="your_token_here"
   ```

## Python API Usage

### Basic Example

```python
from linkml_store import Client

# Create client and attach database
client = Client()
db = client.attach_database("dremio-rest://lakehouse.example.com", alias="dremio")

# Query a table using its full path
study = db.get_collection('"gold-db-2 postgresql".gold.study')

# Find all rows (with limit)
result = study.find({}, limit=10)
print(f"Found {len(result.rows)} rows")

# Find with filter
result = study.find({"is_public": "Yes"}, limit=10)
for row in result.rows:
    print(row["gold_id"], row["study_name"])
```

### Using MongoDB-Style Query Operators

```python
# Greater than
result = study.find({"year": {"$gt": 2020}})

# IN operator
result = study.find({"ecosystem": {"$in": ["Environmental", "Host-associated"]}})

# LIKE (case-sensitive)
result = study.find({"study_name": {"$like": "%methane%"}})

# ILIKE (case-insensitive)
result = study.find({"study_name": {"$ilike": "%Methane%"}})

# Combined conditions (AND)
result = study.find({
    "is_public": "Yes",
    "metagenomic": "Yes",
    "ecosystem": {"$in": ["Environmental"]}
}, limit=20)
```

### Supported Operators

| Operator | SQL Equivalent | Example |
|----------|---------------|---------|
| `$gt` | `>` | `{"age": {"$gt": 30}}` |
| `$gte` | `>=` | `{"age": {"$gte": 30}}` |
| `$lt` | `<` | `{"age": {"$lt": 30}}` |
| `$lte` | `<=` | `{"age": {"$lte": 30}}` |
| `$ne` | `!=` or `IS NOT NULL` | `{"status": {"$ne": "deleted"}}` |
| `$in` | `IN (...)` | `{"status": {"$in": ["a", "b"]}}` |
| `$nin` | `NOT IN (...)` | `{"status": {"$nin": ["deleted"]}}` |
| `$like` | `LIKE` | `{"name": {"$like": "%test%"}}` |
| `$ilike` | `LOWER() LIKE LOWER()` | `{"name": {"$ilike": "%Test%"}}` |
| `$regex` | `REGEXP_LIKE` | `{"name": {"$regex": "^test.*"}}` |

### Using with Environment Variables

```python
import os
from dotenv import load_dotenv
from linkml_store import Client

# Load credentials from .env file
load_dotenv()

client = Client()
db = client.attach_database("dremio-rest://lakehouse.jgi.lbl.gov", alias="jgi")

# Credentials are automatically read from DREMIO_USER, DREMIO_PASSWORD
collection = db.get_collection('"gold-db-2 postgresql".gold.study')
result = collection.find({"is_public": "Yes"}, limit=5)
```

## Command Line Usage

### Basic Query

```bash
# Set credentials
export DREMIO_USER=myuser
export DREMIO_PASSWORD=mypass

# Query with limit
linkml-store -d 'dremio-rest://lakehouse.example.com' \
  -c '"schema".table' \
  query -l 10

# Query with filter
linkml-store -d 'dremio-rest://lakehouse.example.com' \
  -c '"gold-db-2 postgresql".gold.study' \
  query -w 'is_public: Yes' -l 10

# Output as table
linkml-store -d 'dremio-rest://lakehouse.example.com' \
  -c '"gold-db-2 postgresql".gold.study' \
  query -w 'is_public: Yes' -l 10 -O table
```

### Case-Insensitive Search

```bash
# Using $ilike for case-insensitive search
linkml-store -d 'dremio-rest://lakehouse.jgi.lbl.gov' \
  -c '"gold-db-2 postgresql".gold.study' \
  query -w 'study_name: {$ilike: "%methane%"}' -l 10
```

### Using with dotenv

Create a wrapper script to load environment variables:

```bash
#!/bin/bash
# dremio-query.sh
set -a; source ~/.dremio.env; set +a
linkml-store -d 'dremio-rest://lakehouse.jgi.lbl.gov' "$@"
```

Then use it:

```bash
./dremio-query.sh -c '"gold-db-2 postgresql".gold.study' query -l 5
```

## Table Naming

Dremio uses a hierarchical namespace for tables. Fully qualified table names may include:

- **Source**: The data source name (e.g., `"gold-db-2 postgresql"`)
- **Schema/Space**: The schema or space name (e.g., `gold`)
- **Table**: The table name (e.g., `study`)

Full path example: `"gold-db-2 postgresql".gold.study`

When specifying table names:
- Use double quotes around names with special characters or spaces
- The full path can be specified directly in `get_collection()`
- Or set a default schema in the connection string

```python
# Full path
collection = db.get_collection('"gold-db-2 postgresql".gold.study')

# With default schema
db = client.attach_database(
    'dremio-rest://lakehouse.example.com?schema="gold-db-2 postgresql".gold',
    alias="dremio"
)
collection = db.get_collection('study')  # Uses default schema
```

## Performance Considerations

1. **Use specific table paths**: Don't rely on table discovery - specify exact paths
2. **Add LIMIT**: Always use `limit` parameter to avoid fetching too many rows
3. **Filter on server**: Use WHERE clauses to filter data on the server side
4. **Avoid `search`**: The semantic search command loads all data locally - use `query` with `$like`/`$ilike` instead

## Comparison: REST vs Flight SQL

| Feature | `dremio-rest://` | `dremio://` |
|---------|------------------|-------------|
| Protocol | HTTPS REST API | Arrow Flight SQL |
| Port | 443 (default) | 32010 |
| Works behind proxy | Yes | No |
| Performance | Good | Better |
| Pagination | Automatic | Native |

## Troubleshooting

### Authentication Errors

```
ConnectionError: Dremio authentication failed: 401
```

- Check `DREMIO_USER` and `DREMIO_PASSWORD` are set correctly
- If behind Cloudflare, ensure `CF_AUTHORIZATION` is set and not expired

### SSL Certificate Errors

```
SSLError: certificate verify failed
```

For testing only, disable SSL verification:
```python
db = client.attach_database("dremio-rest://localhost?verify_ssl=false", alias="test")
```

### Slow Startup

If the adapter is slow to start, it may be scanning for all tables. The adapter now skips this by default. If you need to list all tables:

```python
db.discover_collections()  # Explicitly scan for tables
print(db.list_collection_names())
```

### Query Syntax Errors

Ensure table names with special characters are properly quoted:
```python
# Correct
collection = db.get_collection('"gold-db-2 postgresql".gold.study')

# Wrong - missing quotes
collection = db.get_collection('gold-db-2 postgresql.gold.study')
```

## Resources

- [Dremio REST API Documentation](https://docs.dremio.com/current/reference/api/)
- [Dremio SQL Reference](https://docs.dremio.com/current/reference/sql/)
- [LinkML-Store Documentation](https://linkml.io/linkml-store/)
