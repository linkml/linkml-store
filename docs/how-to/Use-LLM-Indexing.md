# How to Use LLM Indexing for Semantic Search

This guide shows you how to set up and use LLM-based indexing for semantic search in LinkML-Store.

## Prerequisites

- LinkML-Store installed with LLM extras: `pip install linkml-store[llm]`
- OpenAI API key (or alternative LLM provider configured)
- A LinkML-Store database with data

## Setting Up LLM Indexing

### Step 1: Configure Your API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or configure an alternative provider through the `llm` library:

```bash
llm keys set openai
# Enter your OpenAI API key when prompted
```

### Step 2: Create an LLM Index via CLI

The simplest way to add an LLM index to your collection:

```bash
# Create an index with default settings
linkml-store -d mydb.ddb -c persons index -t llm

# Create a named index
linkml-store -d mydb.ddb -c persons index -t llm --name semantic_index
```

TODO: `--name` not supported yet

### Step 3: Create an LLM Index via Python

For more control, use the Python API:

```python
from linkml_store import Client
from linkml_store.index.implementations.llm_indexer import LLMIndexer

# Connect to database
client = Client("mydb.ddb")
collection = client.get_collection("persons")

# Create and attach indexer
indexer = LLMIndexer(
    name="semantic",
    text_template="{name} works as {profession} and lives in {city}",
    index_attributes=["name", "profession", "city"]
)
collection.attach_indexer(indexer)
```

## Performing Semantic Search

### Basic Search

Search for semantically similar items:

```bash
# CLI search
linkml-store -d mydb.ddb -c persons search "people in healthcare"

# Python search
results = collection.search("people in healthcare", limit=5)
for result in results:
    print(f"Score: {result['score']:.3f} - {result['name']}")
```

### Search with Specific Index

When you have multiple indexes:

```python
# Use a specific index
results = collection.search(
    "medical professionals",
    index_name="semantic",
    limit=10
)
```

## Advanced Configuration

### Custom Text Templates

Control how objects are converted to text for indexing:

```python
# Simple template with specific fields
indexer = LLMIndexer(
    text_template="{title}: {description}",
    index_attributes=["title", "description"]
)

# Jinja2 template with conditional logic
from linkml_store.index.indexer import TemplateSyntaxEnum

indexer = LLMIndexer(
    text_template="""
    {{name}}
    {% if skills %}
    Skills: {{skills|join(', ')}}
    {% endif %}
    {% if experience_years %}
    Experience: {{experience_years}} years
    {% endif %}
    """,
    text_template_syntax=TemplateSyntaxEnum.jinja2
)
```

### Using Different Embedding Models

Change the embedding model from the default:

```python
# Use a different OpenAI model
indexer = LLMIndexer(
    model_name="text-embedding-3-large",
    embedding_dimensions=3072  # Large model has more dimensions
)

# Use a local model via llm
indexer = LLMIndexer(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Configuring the Cache

The LLM indexer caches embeddings to avoid repeated API calls:

```python
# Use a custom cache location
indexer = LLMIndexer(
    cached_embeddings_database="./cache/embeddings.db"
)

# Share cache across multiple indexers
shared_cache = "./shared_embeddings.db"
indexer1 = LLMIndexer(cached_embeddings_database=shared_cache)
indexer2 = LLMIndexer(cached_embeddings_database=shared_cache)
```

TODO - add docs on how to do this on the CLI

### Token Limit Management

TODO - document how large documents are truncated

## Search Strategies

### Similarity Search

Basic cosine similarity search:

```python
# Find most similar items
results = collection.search(
    "renewable energy experts",
    limit=10,
    index_name="semantic"
)
```

### MMR (Maximal Marginal Relevance) Search

Reduce redundancy in results:

```python
# Get diverse results
results = collection.search(
    "data scientists",
    limit=10,
    include_similarity=True,
    algorithm="mmr",
    lambda_mult=0.5  # Balance relevance and diversity
)

# lambda_mult values:
# 1.0 = pure relevance (like regular search)
# 0.5 = balanced relevance and diversity
# 0.0 = maximum diversity
```

## Working with Search Results

### Understanding Result Format

Search results include scores and metadata:

```python
results = collection.search("machine learning")
for result in results:
    # Access the score
    similarity_score = result.get("__similarity__", 0)

    # Access the original object
    person = result
    print(f"{person['name']} - Score: {similarity_score:.3f}")

    # The indexed text used for search
    indexed_text = result.get("__indexed_text__", "")
```

### Filtering and Post-Processing

Combine semantic search with filters:

```python
# Search with additional constraints
results = collection.query(
    where={"city": "London"},  # Filter first
    index_name="semantic",
    search_term="software engineers"  # Then semantic search
)
```

## Performance Optimization

### Batch Indexing

Index multiple items efficiently:

```python
# Add many items at once - indexing happens in batch
items = [
    {"name": "Alice", "profession": "Doctor"},
    {"name": "Bob", "profession": "Nurse"},
    # ... many more
]
collection.insert_many(items)  # Indexed automatically if indexer attached
```

### Monitoring Cache Performance

Check if the cache is being utilized:

```python
import logging
logging.basicConfig(level=logging.INFO)

# The indexer logs cache hits/misses
indexer = LLMIndexer(name="semantic")
# Look for: "Cache hit for text: ..." or "Cache miss, fetching embedding..."
```

### Incremental Indexing

Only index new or updated items:

```python
# Items are indexed automatically on insert/update
collection.insert({"name": "Charlie", "profession": "Developer"})
# Charlie is now searchable

# Update triggers reindexing
collection.update({"name": "Charlie"}, {"profession": "Senior Developer"})
# Charlie's index is updated
```

## Troubleshooting

### Common Issues

**Issue: "No API key found"**
```bash
# Solution: Set your API key
export OPENAI_API_KEY="sk-..."
```

**Issue: "Rate limit exceeded"**
```python
# Solution: Add delays or use caching
import time

for item in items:
    collection.insert(item)
    time.sleep(0.1)  # Rate limiting
```

**Issue: "Token limit exceeded"**
```python
# Solution: Use shorter templates or preprocess text
indexer = LLMIndexer(
    text_template="{name} {profession}",  # Shorter template
    index_attributes=["name", "profession"]  # Fewer attributes
)
```

### Debugging Search Results

Understand why certain results are returned:

```python
# Enable debug mode to see indexed text
results = collection.search("query", include_debug=True)
for r in results:
    print(f"Indexed as: {r.get('__index_text__')}")
    print(f"Score: {r.get('__similarity__')}")
```

## Best Practices

### 1. Choose Appropriate Text Templates

Create templates that capture the semantic meaning:

```python
# Good: Captures semantic relationships
template = "{product_name} is a {category} used for {purpose}"

# Less effective: Just concatenates fields
template = "{field1} {field2} {field3}"
```

### 2. Use Index Attributes Wisely

Only index fields that contribute to semantic meaning:

```python
# Good: Meaningful fields
index_attributes = ["description", "skills", "experience"]

# Avoid: IDs and timestamps
index_attributes = ["id", "created_at", "uuid"]  # Not useful for semantic search
```

### 3. Leverage Caching

Always use caching in production:

```python
# Reuse cache across sessions
indexer = LLMIndexer(
    cached_embeddings_database="/var/cache/linkml/embeddings.db"
)
```

### 4. Monitor Costs

Track API usage to manage costs:

```python
# Log embedding requests
import logging
logging.getLogger("linkml_store.index").setLevel(logging.DEBUG)

# Consider using smaller models for testing
dev_indexer = LLMIndexer(model_name="text-embedding-3-small")
prod_indexer = LLMIndexer(model_name="text-embedding-3-large")
```

## Integration Examples

### With RAG Systems

Use indexed data for retrieval-augmented generation:

```python
from linkml_store.inference import get_inference_engine

# Index your knowledge base
knowledge_collection = client.get_collection("knowledge")
knowledge_collection.attach_indexer(LLMIndexer(name="rag_index"))

# Use in RAG
engine = get_inference_engine("rag", collection=knowledge_collection)
result = engine.derive({"question": "What is machine learning?"})
```

### With Data Pipelines

Integrate indexing into data processing:

```python
# ETL pipeline with automatic indexing
def process_documents(documents):
    collection = client.get_collection("documents")
    collection.attach_indexer(LLMIndexer(
        text_template="{title} {abstract} {content}"
    ))

    for doc in documents:
        # Process document
        doc = clean_text(doc)
        doc = extract_metadata(doc)

        # Insert triggers indexing
        collection.insert(doc)

    return collection
```

## Next Steps

- Learn about [Multi-Index Strategies](./Use-Multi-Index-Strategies.md) for hybrid search
- Explore [RAG Inference](./Perform-RAG-Inference.ipynb) using indexed data
- See the [Index Configuration Reference](../reference/index-configuration.md) for all options