"""
Adapter for MongoDB document store.

Handles have the form: ``mongodb://<host>:<port>/<database>``

To use this, you must have the `pymongo` extra installed.

.. code-block:: bash

    pip install linkml-store[mongodb]

or

.. code-block:: bash

    pip install linkml-store[all]
"""

from linkml_store.api.stores.mongodb.mongodb_collection import MongoDBCollection
from linkml_store.api.stores.mongodb.mongodb_database import MongoDBDatabase

__all__ = [
    "MongoDBCollection",
    "MongoDBDatabase",
]
