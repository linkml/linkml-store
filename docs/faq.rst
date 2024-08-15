.. _faq:

Frequently Asked Questions
==========================

General
-------

What is this project?
~~~~~~~~~~~~~~~~~~~~~~

linkml-store is a data management solution that provides a common interface to multiple backends,
including DuckDB, MongoDB, Neo4J, and Solr.
It is designed to make it easier to work with data in different forms (tabular, JSON, columnar, RDF),
provide expressive validation at scale, and enable the ability to mix and match different backends.

For a high-level overview, see `These slides <https://docs.google.com/presentation/d/e/2PACX-1vSgtWUNUW0qNO_ZhMAGQ6fYhlXZJjBNMYT0OiZz8DDx8oj7iG9KofRs6SeaMXBBOICGknoyMG2zaHnm/embed?start=false&loop=false&delayms=3000>`_.


Is this a database engine?
~~~~~~~~~~~~~~~~~~~~~~~~~~

No, linkml-store is not a database engine in itself. It is designed to be used *in combination*
with your favorite database engines.

Do I need to know LinkML to use this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, you do not need to know LinkML to use linkml-store. In fact you can use linkml-store in
"YOLO mode" where you don't even specify a schema (a schema will be induced as far as possible).

However, for serious applications we recommend you always provide a LinkML schema for your
different datasets.

For more information on LinkML, see the `LinkML documentation <https://linkml.io/linkml/>`_.

Can I use the command line?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, linkml-store provides a command line interface.

See the `Command Line Tutorial <https://linkml.io/linkml-store/tutorials/Command-Line-Tutorial.html>`_ for examples.

All commands can be used via the base ``linkml-store`` command:

.. code-block:: bash

    linkml-store --help

Note some command line options may change in future until this package is 1.0.0

Can I use the Python API?
~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, linkml-store provides a Python API.

See the `Python Tutorial <https://linkml.io/linkml-store/tutorials/Python-Tutorial.html>`_ for examples.

Example:

.. code-block:: python

    from linkml_store import Client

    client = Client()
    db = client.attach_database("duckdb")
    collection = db.attach_collection("my_collection")
    collection.insert({"name": "Alice", "age": 42})
    result = collection.find({"name": "Alice"})


Can I use a web API?
~~~~~~~~~~~~~~~~~~~~

Yes, you can stand up a web API.

To start you should first create a config file, e.g. ``db/conf.yaml``:

Then run:

.. code-block:: bash

    export LINKML_STORE_CONFIG=./db/conf.yaml
    make api

Can I use a web UI?
~~~~~~~~~~~~~~~~~~~~

We provide a *very rudimentary* web UI. To start you should first create a config file, e.g. ``db/conf.yaml``:

Then run:

.. code-block:: bash

    export LINKML_STORE_CONFIG=./db/conf.yaml
    make app


What is CRUDSI?
~~~~~~~~~~~~~~~

CRUDSI is our not particularly serious name for the design pattern that linkml-store follows.

Many database engines and database solutions implement a CRUD layer:

* Create
* Read
* Update
* Delete

linkml-store adds two more operations:

* Search
* Inference


Is this an AI/Machine Learning/LLM/Vector database platform?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

linkml-store is first and foremost a *data management* platform. However,
we do provide optional integrations to AI and ML tooling. In particular, you can plug and
play different solutions for implementing search indexes, including LLM textual embeddings.

Additionally, we believe that robust data management using rich and expressive semantic
schemas (in combination with the database engine of your choice) is the key to
making data **AI-ready**.

Is linkml-store production ready?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

linkml-store is currently not as mature as the core LinkML products. Be warned that
the API and command line options may change. However, things may be moving fast,
and you are invited to check back in here later!

Are there tutorials?
~~~~~~~~~~~~~~~~~~~~

See :ref:`tutorials`

Installation
-------

How do I install linkml-store?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install "linkml-store[all]"

This installs both necessary and optional dependencies. We recommend this for now.

As a developer, how do I install linkml-store?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out the repo, and like all linkml projects, use Poetry:

.. code-block:: bash

    git clone <URL>
    cd linkml-store
    make install

Backend Integrations
------------

What is a database integration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This framework provides different integrations (aka adapters or implementations) that can hook into
your favorite backend database (if your database engine is not supported, please be patient - or
consider contributing one as a PR!)

Does linkml-store support DuckDB?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, linkml-store supports DuckDB as a backend. DuckDB is a modern columnar in-memory database

See the :ref:`tutorial <tutorials>` for examples.

Note that currently for DuckDB we bypass the `standard linkml to SQL to relational mapping <https://linkml.io/linkml/generators/sqltable.html>`_ step,
and instead use DuckDB more like a data frame store. Nested objects and lists are stored directly
(using DuckDB's json integrations behind the scenes), rather than fully normalized.

Does linkml-store support MongoDB?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, linkml-store supports MongoDB as a backend. MongoDB is a popular NoSQL database.

See the `MongoDB how-to guide <https://linkml.io/linkml-store/how-to/Use-MongoDB.html>`_ for examples.

Does linkml-store support Neo4J?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, linkml-store supports Neo4J as a backend. Neo4J is a popular graph database.

See the `Neo4J how-to guide <https://linkml.io/linkml-store/how-to/Use-Neo4J.html>`_ for examples.

Does linkml-store support Solr?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently we provide only read support for Solr. We are working on write support.

See the `Solr how-to guide <https://linkml.io/linkml-store/how-to/Query-Solr-using-CLI.html>`_ for examples.

Can I use linkml-store with my favorite triplestore?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not yet! This is a surprising omission given LinkML's roots in the semantic web community. However,
this is planned soon, so check back later.

Data model
----------

What is the data model in linkml-store?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

linkml-store has a simple data model:

* A :class:`.Client` provides a top-level interface over one or more databases.
* A :class:`.Database` consists of one or more possibly heterogeneous collections.
* A :class:`.Collection` is a queryable set of objects of a similar type.

Search
------

Can I use LLM vector embeddings for search?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, you can use LLM vector embeddings for search. This is an optional feature.

See `How to use semantic search <https://linkml.io/linkml-store/how-to/Use-Semantic-Search.html>`_ for examples.

Do I need to use an LLM for search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, but currently other options are limited. You can use a naive tripartite index, or if your backend
supports search out the box (e.g. Solr) then linkml-store should directly wire into this.

Validation
----------

Does linkml-store provide validation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, linkml-store provides expressive validation using the LinkML framework.

Note that currently validation primarily leverages json-schema integrations, but the intent is to
provide validation integrations directly with underlying backend stores.

Does linkml-store provide referential integrity validation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `Check Referential Integrity <https://linkml.io/linkml-store/how-to/Check-Referential-Integrity.html>`_ for examples.

Inference
---------

What is inference in linkml-store?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a very flexible notion of inference. It can encompass:

* Statistical or Machine Learning (ML) inference, e.g. via supervised learning
* Ontological inference, e.g. via reasoning over an ontology
* Rule-based or procedural inference
* LLM-based inference

How do I do standard ML inference?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently we provide integrations to scikit-learn, but only expose DecisionTree classifiers for now.
Remember, linkml-store is not a full fledged ML platform; you should use packages like XGBoost, PyTorch,
or scikit-learn directly for more complex ML tasks.

See `Predict Missing Data <https://linkml.io/linkml-store/how-to/Predict-Missing-Data.html>`_ for examples.

See also the `Command Line Tutorial <https://linkml.io/linkml-store/tutorials/Command-Line-Tutorial.html>`_ for
a simple example.

How do I do LLM inference?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the `Command Line Tutorial <https://linkml.io/linkml-store/tutorials/Command-Line-Tutorial.html>`_ (see
the final section) for an example.

How do I do rule-based inference?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check back later for tutorials. For now, you can read about:

- the `LinkML expression language <https://linkml.io/linkml/schemas/expression-language.html>`_
- `Rules in LinkML <https://linkml.io/linkml/schemas/advanced.html#rules>`_

In future we will provide bindings for rule engines, datalog engines, and OWL reasoners.

Contributing
------------

How do I contribute to linkml-store?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We welcome contributions! Please see the `LinkML contributing guide <https://linkml.io/linkml/contributing.html>`_.