# Understanding LinkML-Store's Indexing Architecture

## Overview

LinkML-Store implements a **composable indexing system** that decouples indexing logic from storage backends. This design philosophy means you can use advanced indexing techniques like LLM embeddings with any storage backend - you don't need a dedicated vector database to perform semantic search.

## Core Concepts

### Indexing as an Orthogonal Concern

Traditional database systems tightly couple indexing with storage. For example:
- Elasticsearch bundles Lucene indexing with document storage
- Vector databases like Pinecone combine embedding storage with similarity search
- SQL databases integrate B-tree indexes with table storage

LinkML-Store takes a different approach: **indexing is a composable layer** that can be added to any collection, regardless of the underlying storage mechanism. This means:

1. You can add LLM-based semantic search to a DuckDB database
2. You can use multiple index types on the same collection
3. You can switch storage backends without changing your indexing strategy

### The Index Abstraction

At its core, an index in LinkML-Store consists of three components:

1. **Text Extraction**: Converting structured data to searchable text
2. **Vectorization**: Transforming text into numerical vectors
3. **Similarity Search**: Finding relevant items based on vector similarity

```
Object → Text → Vector → Search Results
```

## Architecture Deep Dive

### The Indexer Base Class

The `Indexer` base class provides the fundamental abstraction for all indexing operations. It handles:

#### Text Template System

Objects are converted to text using configurable templates. This allows flexible representation of your data for indexing:

```python
# Simple template
template = "{name} :: {profession}"

# Jinja2 template with logic
template = """
{{name}}{% if profession %} works as a {{profession}}{% endif %}
{% if skills %}with skills in {{skills|join(', ')}}{% endif %}
"""
```

The template system supports:
- F-string syntax for simple substitutions
- Jinja2 for complex logic and formatting
- Attribute filtering via `index_attributes`

#### Vector Storage

Indexes are stored in shadow collections following the naming pattern:
```
internal__index__{collection_name}__{index_name}
```

These shadow collections contain:
- The original object data
- The extracted text representation
- The computed vector in the `__index__` field

### LLM Indexer Implementation

The `LLMIndexer` is the primary implementation for semantic search, leveraging large language models for text embedding.

#### Embedding Generation

The LLM indexer uses the `llm` library to generate embeddings:

1. **Model Selection**: Defaults to OpenAI's `text-embedding-ada-002`, but supports any model available through the llm library
2. **Token Management**: Uses `tiktoken` to count tokens and intelligently truncate text to fit model limits
3. **Batch Processing**: Efficiently processes multiple texts in batches for better performance

#### The Caching Layer

One of the most sophisticated features is the **persistent embedding cache**:

```python
# Cache structure
{
    "text": "the input text",
    "model_id": "ada-002",
    "embedding": [0.1, 0.2, ...],  # the vector
    "created_at": "2024-01-01T00:00:00Z"
}
```

The cache:
- Stores embeddings in a separate DuckDB database
- Deduplicates by text + model combination
- Significantly reduces API calls and costs
- Persists across sessions

**Cache Flow**:
1. Check if text+model exists in cache
2. If yes: retrieve cached embedding
3. If no: call API, store in cache, return embedding

#### Text Processing Pipeline

For large texts, the LLM indexer implements sophisticated processing:

1. **Initial Check**: Count tokens using tiktoken
2. **Truncation Strategy**:
   - If under limit: use full text
   - If over limit: chunk into ~1000 character segments
   - Use `render_formatted_text` for intelligent truncation
3. **Embedding**: Generate vector for processed text
4. **Storage**: Save to index collection

### Search Mechanisms

LinkML-Store provides two primary search mechanisms:

#### Cosine Similarity Search

The basic search algorithm:

1. Convert query text to vector using same model
2. Compute cosine similarity with all indexed vectors
3. Return top-N results by similarity score

```python
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

#### Maximal Marginal Relevance (MMR)

MMR balances relevance with diversity to avoid redundant results:

```python
MMR(d) = λ * similarity(query, d) - (1-λ) * max(similarity(d, selected))
```

Where:
- `λ` (lambda_mult) controls the relevance/diversity trade-off
- Higher λ = more relevance-focused
- Lower λ = more diversity-focused

The algorithm:
1. Find the most similar document
2. For each remaining document, compute MMR score
3. Select document with highest MMR
4. Repeat until N documents selected

## Integration with Collections

### Attaching Indexes

Collections can have multiple indexes attached:

```python
# Create different index types
llm_index = LLMIndexer(name="semantic")
simple_index = SimpleIndexer(name="trigram")

# Attach to collection
collection.attach_indexer(llm_index)
collection.attach_indexer(simple_index)

# Use specific index for search
semantic_results = collection.search("query", index_name="semantic")
trigram_results = collection.search("query", index_name="trigram")
```

### Index Lifecycle

1. **Creation**: Index created when first attached to collection
2. **Population**: Documents indexed on insert/update
3. **Maintenance**: Index updated automatically with collection changes
4. **Deletion**: Index removed when detached or collection deleted

## RAG Integration

The indexing system is deeply integrated with Retrieval-Augmented Generation (RAG) workflows:

### Training Phase

1. **Automatic Indexing**: Training data automatically indexed with LLM embeddings
2. **Vector Storage**: Embeddings stored alongside training examples
3. **Model Persistence**: Index becomes part of the trained model

### Inference Phase

1. **Query Processing**: Input converted to embedding
2. **Context Retrieval**: Similar examples retrieved via index
3. **Prompt Assembly**: Retrieved examples used as few-shot context
4. **Generation**: LLM generates output based on retrieved context

The index serves as the "memory" for the RAG system, enabling:
- Efficient retrieval of relevant examples
- Scaling to large training sets
- Domain-specific knowledge injection

## Performance Considerations

### Caching Strategy

The caching system is critical for performance:

- **Cache Hits**: Near-instant retrieval (microseconds)
- **Cache Misses**: API call required (100-500ms)
- **Cache Growth**: Linear with unique texts
- **Cache Management**: Currently no automatic expiration

### Optimization Techniques

1. **Batch Embedding**: Process multiple texts in single API calls
2. **Lazy Indexing**: Index only when needed for search
3. **Incremental Updates**: Only index new/changed documents
4. **Vector Compression**: Store vectors in efficient formats

### Scalability

The indexing system scales across multiple dimensions:

- **Document Count**: Shadow collections scale with backend capabilities
- **Vector Dimensions**: Typically 1536 for ada-002, stored efficiently
- **Query Performance**: O(n) for cosine similarity, optimizable with approximations
- **Multiple Indexes**: Linear overhead per index

## Backend Compatibility

### Native Integration

Some backends have native support that LinkML-Store leverages:

- **ChromaDB**: Direct vector storage and similarity search
- **Solr**: Native text indexing via Lucene
- **MongoDB**: Text indexes and aggregation pipelines

### Emulated Support

For backends without native support, LinkML-Store provides full emulation:

- **DuckDB**: Vectors stored as arrays, similarity computed in Python
- **Filesystem**: Index stored in separate JSON files
- **SQL Databases**: Vectors in BLOB/JSON columns

The abstraction ensures consistent behavior across all backends.

## Future Directions

### Planned Enhancements

1. **Approximate Nearest Neighbor**: FAISS/Annoy integration for faster search
2. **Hybrid Search**: Combine semantic and keyword search
3. **Custom Embeddings**: Support for domain-specific models
4. **Index Versioning**: Track index versions for reproducibility
5. **Distributed Indexing**: Scale across multiple machines

### Extensibility Points

The architecture supports custom indexers through:

1. Subclassing `Indexer` base class
2. Implementing `text_to_vectors()` method
3. Optional: Override search algorithms
4. Optional: Custom caching strategies

This enables specialized indexes for:
- Genomic sequences
- Chemical structures
- Time series data
- Graph embeddings

## Conclusion

LinkML-Store's indexing architecture provides a powerful, flexible foundation for modern data management. By decoupling indexing from storage and providing sophisticated LLM integration, it enables semantic search capabilities across any data backend without requiring specialized infrastructure.