.. _about:

About
=========================================================

LinkML-Store is an early effort to provide a unifying storage layer
over multiple different backends, unified via LinkML schemas.

The overall goals are to provide:

* Make it easier to work with data in different forms (tabular, JSON, columnar, RDF)
* Expressive validation at scale, including full referential integrity validation
* Ability to mix and match different backends (e.g. DuckDB, MongoDB, Solr, ChromaDB, HDF5)
* Composability of different search indexes, including LLM textual embeddings
* LAMP-like stack for LinkML

Installation
------------

At this stage we recommend installing all extras:

.. code-block:: bash

    pip install "linkml-store[all]"

Minimal implementation:

.. code-block:: bash

    pip install "linkml-store"

For developers working on the linkml-store codebase, we recommend checking out the repo, and then
installing all packages via the `make` command:

.. code-block:: bash

    git clone <URL>
    cd linkml-store
    make install

Quickstart
----------

See the :ref:`tutorials`

Data Model
----------

* A :class:`.Client` provides a top-level interface over one or more databases.
* A :class:`.Database` consists of one or more possibly heterogeneous collections.
* A :class:`.Collection` is a queryable set of objects of a similar type.
* A :class:`.Indexer` creates indexes over collections to enable efficient searching.

Adapters
--------

The current backends are:

- :py:mod:`DuckDB<linkml_store.api.stores.duckdb>` (supports read and write)
- :py:mod:`MongoDB<linkml_store.api.stores.mongodb>` (supports read and write)
- :py:mod:`Solr<linkml_store.api.stores.solr>` (write not yet supported)
- :py:mod:`ChromaDB<linkml_store.api.stores.chromadb>` (pre-alpha)
- :py:mod:`HDF5<linkml_store.api.stores.mdf5>` (pre-alpha)


Indexing
--------

This frameworks also allows *composable indexes*. Currently two indexers are supported:

- :py:mod:`SimpleIndexer<linkml_store.index.implementations.simple_indexer>` Simple native trigram method
- :py:mod:`LLMIndexer<linkml_store.index.implementations.llm_indexer>` LLM text embedding

Metadata and Configuration
--------------------------

- :py:mod:`ClientConfig<linkml_store.api.config.ClientConfig>` provides a structure for configuring the client
