# Index Configuration Reference

Complete reference for configuring indexes in LinkML-Store.

## Indexer Base Class

All indexers inherit from the `Indexer` base class.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique name for the index within a collection |
| `text_template` | `str` | `None` | Template for converting objects to text |
| `text_template_syntax` | `TemplateSyntaxEnum` | `f_string` | Template syntax type |
| `index_attributes` | `List[str]` | `None` | Attributes to include in indexing |

### Methods

#### `index(obj: dict, **kwargs) -> Optional[VECTOR_TYPE]`

Index a single object.

**Parameters:**
- `obj`: Dictionary object to index
- `**kwargs`: Additional implementation-specific parameters

**Returns:**
- Vector representation or None if object cannot be indexed

#### `search(query: str, collection: Collection, **kwargs) -> List[Tuple[float, dict, VECTOR_TYPE]]`

Search the index.

**Parameters:**
- `query`: Search query string
- `collection`: Collection to search
- `limit`: Maximum results (default: 10)
- `include_similarity`: Include similarity scores (default: False)
- `mmr_lambda`: Lambda for MMR diversity (0-1, default: 0.5)
- `initial_matches`: Number of candidates for MMR (default: 20)

**Returns:**
- List of tuples: (similarity_score, object, vector)

#### `index_collection(collection: Collection) -> Collection`

Create or update index for entire collection.

**Parameters:**
- `collection`: Collection to index

**Returns:**
- Index collection containing vectors

#### `text_to_vectors(text: str) -> VECTOR_TYPE`

Abstract method to convert text to vectors.

**Parameters:**
- `text`: Text to vectorize

**Returns:**
- Vector representation

**Must be implemented by subclasses.**

## LLMIndexer

LLM-based semantic indexing using embedding models.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique index name |
| `text_template` | `str` | `None` | Object to text template |
| `text_template_syntax` | `TemplateSyntaxEnum` | `f_string` | Template syntax |
| `index_attributes` | `List[str]` | `None` | Attributes to index |
| `model_name` | `str` | `"text-embedding-ada-002"` | Embedding model name |
| `cached_embeddings_database` | `str` | `None` | Path to cache database |
| `embedding_dimensions` | `int` | `None` | Vector dimensions (auto-detected) |

### Supported Models

Models available through the `llm` library:

#### OpenAI Models
| Model | Dimensions | Context | Cost/1M tokens |
|-------|------------|---------|----------------|
| `text-embedding-ada-002` | 1536 | 8192 | $0.10 |
| `text-embedding-3-small` | 1536 | 8192 | $0.02 |
| `text-embedding-3-large` | 3072 | 8192 | $0.13 |



### Cache Configuration

The cache database stores embeddings to avoid repeated API calls.

#### Cache Schema

```sql
CREATE TABLE embeddings (
    text TEXT,
    model_id TEXT,
    embedding BLOB,  -- Serialized vector
    created_at TIMESTAMP,
    PRIMARY KEY (text, model_id)
);
```

#### Cache Management

```python
# Custom cache location
indexer = LLMIndexer(
    name="semantic",
    cached_embeddings_database="/var/cache/embeddings.db"
)

# Shared cache
SHARED_CACHE = "./shared_embeddings.db"
indexer1 = LLMIndexer(cached_embeddings_database=SHARED_CACHE)
indexer2 = LLMIndexer(cached_embeddings_database=SHARED_CACHE)

# No cache (not recommended)
indexer = LLMIndexer(
    name="nocache",
    cached_embeddings_database=":memory:"  # In-memory only
)
```

### Token Management

Automatic handling of token limits:

```python
# Tokens are counted using tiktoken
# If text exceeds limit:
# 1. Text is chunked into ~1000 char segments
# 2. Chunks are concatenated up to token limit
# 3. Remaining text is truncated

indexer = LLMIndexer(
    name="semantic",
    model_name="text-embedding-ada-002"  # 8192 token limit
)
```

### Methods

#### `_cached_embed_text(text: str) -> List[float]`

Get embedding with caching.

**Parameters:**
- `text`: Text to embed

**Returns:**
- Embedding vector as list of floats

**Behavior:**
1. Check cache for text+model combination
2. If found: return cached embedding
3. If not: call API, cache result, return embedding

## SimpleIndexer

Trigram-based indexer for testing and simple use cases.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique index name |
| `text_template` | `str` | `None` | Object to text template |
| `text_template_syntax` | `TemplateSyntaxEnum` | `f_string` | Template syntax |
| `index_attributes` | `List[str]` | `None` | Attributes to index |
| `dimensions` | `int` | `1000` | Vector dimensions |

### Algorithm

```python
def text_to_vectors(text: str) -> List[float]:
    # Extract character trigrams
    trigrams = [text[i:i+3] for i in range(len(text)-2)]

    # Hash trigrams to vector positions
    vector = [0.0] * dimensions
    for trigram in trigrams:
        position = hash(trigram) % dimensions
        vector[position] += 1.0

    # Normalize
    norm = sum(v**2 for v in vector) ** 0.5
    return [v/norm for v in vector]
```

### Use Cases

- Testing without API keys
- Fast approximate matching
- Baseline comparisons
- Development environments

## Template Configuration

### Template Syntax Options

#### F-String Templates

Default Python f-string syntax:

```python
indexer = LLMIndexer(
    text_template="{name} is a {profession} in {city}",
    text_template_syntax=TemplateSyntaxEnum.f_string
)
```

**Features:**
- Simple variable substitution
- No logic or conditionals
- Fast processing

#### Jinja2 Templates

Full Jinja2 template engine:

```python
indexer = LLMIndexer(
    text_template="""
    {{name}}
    {% if profession %}
    Profession: {{profession}}
    {% endif %}
    {% for skill in skills %}
    - {{skill}}
    {% endfor %}
    """,
    text_template_syntax=TemplateSyntaxEnum.jinja2
)
```

**Features:**
- Conditionals and loops
- Filters and functions
- Complex formatting

### Template Variables

Available variables in templates:

- All object attributes
- Special variables:
  - `_id`: Object ID
  - `_collection`: Collection name
  - `_timestamp`: Index timestamp

### Index Attributes

Control which attributes are indexed:

```python
# Index only specific fields
indexer = LLMIndexer(
    index_attributes=["title", "abstract", "keywords"]
)

# Exclude certain fields
class SelectiveIndexer(LLMIndexer):
    def get_index_attributes(self, obj):
        # Dynamic attribute selection
        return [k for k in obj.keys()
                if not k.startswith("_")
                and k not in ["id", "created_at"]]
```

## Search Configuration

### Search Algorithms

#### Cosine Similarity

Standard vector similarity:

```python
results = collection.search(
    query="search terms",
    algorithm="cosine",  # Default
    limit=10
)
```

**Formula:**
```
similarity = dot(v1, v2) / (norm(v1) * norm(v2))
```

#### Maximal Marginal Relevance (MMR)

Diversity-aware search:

```python
results = collection.search(
    query="search terms",
    algorithm="mmr",
    mmr_lambda=0.5,  # Balance parameter
    initial_matches=20  # Candidate pool size
)
```

**Parameters:**
- `mmr_lambda`: 0.0-1.0 (0=max diversity, 1=max relevance)
- `initial_matches`: Number of candidates to consider

**Formula:**
```
MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected))
```

### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `int` | `10` | Maximum results |
| `include_similarity` | `bool` | `False` | Include scores in results |
| `algorithm` | `str` | `"cosine"` | Search algorithm |
| `mmr_lambda` | `float` | `0.5` | MMR diversity parameter |
| `initial_matches` | `int` | `20` | MMR candidate pool |
| `threshold` | `float` | `None` | Minimum similarity score |

## Collection Integration

### Attaching Indexers

```python
# Single indexer
collection.attach_indexer(indexer)

# Multiple indexers
collection.attach_indexer(semantic_indexer)
collection.attach_indexer(keyword_indexer)
```

### Index Naming

Index collections follow the pattern:
```
internal__index__{collection_name}__{index_name}
```

Example:
```
internal__index__documents__semantic
internal__index__documents__keywords
```

### Index Operations

```python
# List indexes
indexes = collection.list_indexes()

# Reindex collection
collection.reindex(index_name="semantic")

# Remove index
collection.detach_indexer("semantic")

# Clear index
collection.clear_index("semantic")
```

## CLI Configuration

### Index Creation

```bash
# Basic index
linkml-store index -t llm

# Named index
linkml-store index -t llm --name my_index

# Custom template
linkml-store index -t llm \
    --template "{title}: {content}" \
    --attributes title,content

# Custom model
linkml-store index -t llm \
    --model text-embedding-3-large

# With cache
linkml-store index -t llm \
    --cache-db /var/cache/embeddings.db
```

### Search Options

```bash
# Basic search
linkml-store search "query terms"

# Specify index
linkml-store search "query terms" --index-name semantic

# MMR search
linkml-store search "query terms" \
    --algorithm mmr \
    --mmr-lambda 0.7 \
    --initial-matches 50

# With threshold
linkml-store search "query terms" \
    --threshold 0.8 \
    --limit 20
```

## Environment Variables

Configure indexers via environment:

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `LLM_USER_PATH` | LLM library config | `~/.config/llm` |
| `LINKML_STORE_CACHE_DIR` | Default cache directory | `/var/cache/linkml` |
| `LINKML_STORE_DEFAULT_MODEL` | Default embedding model | `text-embedding-3-small` |

## Performance Tuning

### Batch Sizes

```python
# Configure batch processing
class BatchedIndexer(LLMIndexer):
    batch_size = 100  # Process 100 items at once

    def index_collection(self, collection):
        # Custom batching logic
        for batch in self.get_batches(collection, self.batch_size):
            self.index_batch(batch)
```

### Parallel Processing

```python
import concurrent.futures

def parallel_index(collection, indexer, workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for obj in collection.find():
            future = executor.submit(indexer.index, obj)
            futures.append(future)

        # Wait for completion
        concurrent.futures.wait(futures)
```

### Memory Management

```python
# Stream large collections
def stream_index(collection, indexer, chunk_size=1000):
    cursor = collection.find_iter()  # Iterator, not list

    while True:
        chunk = list(itertools.islice(cursor, chunk_size))
        if not chunk:
            break

        for obj in chunk:
            indexer.index(obj)

        # Clear memory
        del chunk
        gc.collect()
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `APIKeyError` | Missing API key | Set `OPENAI_API_KEY` |
| `RateLimitError` | Too many requests | Add delays or use cache |
| `TokenLimitError` | Text too long | Use shorter templates |
| `ModelNotFoundError` | Invalid model name | Check available models |
| `CacheError` | Cache corruption | Delete and rebuild cache |

### Error Recovery

```python
class ResilientIndexer(LLMIndexer):
    def index(self, obj, retry_count=3):
        for attempt in range(retry_count):
            try:
                return super().index(obj)
            except RateLimitError:
                time.sleep(2 ** attempt)  # Exponential backoff
            except TokenLimitError:
                # Truncate and retry
                obj = self.truncate_object(obj)

        # Final fallback
        return self.fallback_index(obj)
```

## Debugging

### Enable Debug Logging

```python
import logging

# Enable debug logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("linkml_store.index")
logger.setLevel(logging.DEBUG)

# See cache hits/misses
# See API calls
# See vector operations
```

### Index Inspection

```python
def inspect_index(collection, index_name):
    """Debug index contents."""
    index_col = collection._database.get_collection(
        f"internal__index__{collection.name}__{index_name}"
    )

    sample = index_col.find_one()
    print(f"Sample entry: {sample}")
    print(f"Vector dimensions: {len(sample['__index__'])}")
    print(f"Text representation: {sample.get('__index_text__')}")
    print(f"Total indexed: {index_col.count()}")
```

## Examples

### Complete Configuration Example

```python
from linkml_store import Client
from linkml_store.index.implementations import LLMIndexer
from linkml_store.index.indexer import TemplateSyntaxEnum

# Full configuration
indexer = LLMIndexer(
    name="production_index",
    text_template="""
    Title: {{title}}
    {% if abstract %}
    Abstract: {{abstract}}
    {% endif %}
    Keywords: {{keywords|join(', ')}}
    """,
    text_template_syntax=TemplateSyntaxEnum.jinja2,
    index_attributes=["title", "abstract", "keywords"],
    model_name="text-embedding-3-large",
    cached_embeddings_database="/var/cache/prod_embeddings.db",
    embedding_dimensions=3072
)

# Attach to collection
client = Client("duckdb:///prod.db")
collection = client.get_collection("documents")
collection.attach_indexer(indexer)

# Configure search
results = collection.search(
    "machine learning applications",
    index_name="production_index",
    limit=20,
    include_similarity=True,
    algorithm="mmr",
    mmr_lambda=0.7,
    initial_matches=50,
    threshold=0.6
)
```