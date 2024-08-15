# linkml-store

An AI-ready data management and integration platform. LinkML-Store
provides an abstraction layer over multiple different backends
(including DuckDB, MongoDB, Neo4j, and local filesystems), allowing for
common query, index, and storage operations.

For full documentation, see [https://linkml.io/linkml-store/](https://linkml.io/linkml-store/)

See [these slides](https://docs.google.com/presentation/d/e/2PACX-1vSgtWUNUW0qNO_ZhMAGQ6fYhlXZJjBNMYT0OiZz8DDx8oj7iG9KofRs6SeaMXBBOICGknoyMG2zaHnm/embed?start=false&loop=false&delayms=3000) for a high level overview.

__Warning__ LinkML-Store is still undergoing changes and refactoring,
APIs and command line options are subject to change!

## Quick Start

Install, add data, query it:

```
pip install linkml-store[all]
linkml-store -d duckdb:///db/my.db -c persons insert data/*.json
linkml-store -d duckdb:///db/my.db -c persons query -w "occupation: Bricklayer"
```

Index it, search it:

```
linkml-store -d duckdb:///db/my.db -c persons index -t llm
linkml-store -d duckdb:///db/my.db -c persons search "all persons employed in construction"
```

Validate it:

```
linkml-store -d duckdb:///db/my.db -c persons validate
```

## Basic usage

* [Command Line](https://linkml.io/linkml-store/tutorials/Command-Line-Tutorial.html)
* [Python](https://linkml.io/linkml-store/tutorials/Python-Tutorial.html)
* API
* Streamlit applications

## The CRUDSI pattern

Most database APIs implement the **CRUD** pattern: Create, Read, Update, Delete.
LinkML-Store adds **Search** and **Inference** to this pattern, making it **CRUDSI**.

The notion of "Search" and "Inference" is intended to be flexible and extensible,
including:

* Search
   * Traditional keyword search
   * Search using LLM Vector embeddings (*without* a dedicated vector database)
   * Pluggable specialized search, e.g. genomic sequence (not yet implemented)
* Inference (encompassing  *validation*, *repair*, and inference of missing data)
   * Classic rule-based inference
   * Inference using LLM Retrieval Augmented Generation (RAG)
   * Statistical/ML inference

## Features

### Multiple Adapters

LinkML-Store is designed to work with multiple backends, giving a common abstraction layer

* [MongoDB](https://linkml.io/linkml-store/how-to/Use-MongoDB.html)
* [DuckDB](https://linkml.io/linkml-store/tutorials/Python-Tutorial.html)
* [Solr](https://linkml.io/linkml-store/how-to/Query-Solr-using-CLI.html)
* [Neo4j](https://linkml.io/linkml-store/how-to/Use-Neo4j.html)

* Filesystem

Coming soon: any RDBMS, any triplestore, Neo4J, HDF5-based stores, ChromaDB/Vector dbs ...

The intent is to give a union of all features of each backend. For
example, analytic faceted queries are provided for *all* backends, not
just Solr.

### Composable indexes

Many backends come with their own indexing and search
schemes. Classically this was Lucene-based indexes, now it is semantic
search using LLM embeddings.

LinkML store treats indexing as an orthogonal concern - you can
compose different indexing schemes with different backends. You don't
need to have a vector database to run embedding search!

See [How to Use-Semantic-Search](https://linkml.io/linkml-store/how-to/Use-Semantic-Search.html)

### Use with LLMs

TODO - docs

### Validation

LinkML-Store is backed by [LinkML](https://linkml.io), which allows
for powerful expressive structural and semantic constraints.

See [Indexing JSON](https://linkml.io/linkml-store/how-to/Index-Phenopackets.html)

and [Referential Integrity](https://linkml.io/linkml-store/how-to/Check-Referential-Integrity.html)

## Web API

There is a preliminary API following HATEOAS principles implemented using FastAPI.

To start you should first create a config file, e.g. `db/conf.yaml`:

Then run:

```
export LINKML_STORE_CONFIG=./db/conf.yaml
make api
```

The API returns links as well as data objects, it's recommended to use a Chrome plugin for JSON viewing
for exploring the API. TODO: add docs here.

The main endpoints are:

* `http://localhost:8000/` - the root of the API
* `http://localhost:8000/pages/` - browse the API via HTML
* `http://localhost:8000/docs` - the Swagger UI

## Streamlit app

```
make app
```

## Background

See [these slides](https://docs.google.com/presentation/d/e/2PACX-1vSgtWUNUW0qNO_ZhMAGQ6fYhlXZJjBNMYT0OiZz8DDx8oj7iG9KofRs6SeaMXBBOICGknoyMG2zaHnm/embed?start=false&loop=false&delayms=3000) for more details

