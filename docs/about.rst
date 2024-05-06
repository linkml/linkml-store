.. about

About
=========================================================

LinkML-Store is an early effort to provide a unifying storage layer
over multiple different backends, unified via LinkML schemas.

Quickstart
----------

See the :ref:`tutorials`

Data Model
----------

* A :class:`.Client` provides a top-level interface over one or more databases.
* A :class:`.Database` consists of one or more possibly heterogeneous collections.
* A :class:`.Collection` is a set of objects of a similar type.

Adapters
--------

The current backends supported are:

- :py:mod:`DuckDB<linkml_store.api.stores.duckdb>`
- :py:mod:`MongoDB<linkml_store.api.stores.mongodb>`
- :py:mod:`Solr<linkml_store.api.stores.solr>`
- :py:mod:`ChromaDB<linkml_store.api.stores.chromadb>` (pre-alpha)
- :py:mod:`HDF5<linkml_store.api.stores.mdf5>` (pre-alpha)

Indexing
--------

This frameworks also allows *composable indexes*. Currently two indexers are supported:

- :py:mod:`SimpleIndexer<linkml_store.index.implementations.simple_indexer>` Simple native trigram method
- :py:mod:`LLMIndexer<linkml_store.index.implementations.llm_indexer>` LLM text embedding
