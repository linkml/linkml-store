"""
Adapter for Google Cloud Bigtable.

Handles have the form: ``bigtable://<project>/<instance>``

- A LinkML-Store :class:`.Database` corresponds to a Bigtable *instance*
- A LinkML-Store :class:`.Collection` corresponds to a Bigtable *table*
- Each object is stored as one row. Each top-level field of the object is stored
  as a column (qualifier) in a single column family (``d`` by default), with
  the value JSON-encoded. ``None`` values are not stored.
- If the collection has an identifier attribute (from the schema), its value is
  used as the row key; otherwise a random row key is generated.

The instance must already exist; tables are created on demand.

To use the local emulator, set ``BIGTABLE_EMULATOR_HOST``, e.g.

.. code-block:: bash

    gcloud beta emulators bigtable start --host-port=localhost:8086
    export BIGTABLE_EMULATOR_HOST=localhost:8086

To use this, you must have the `bigtable` extra installed.

.. code-block:: bash

    pip install linkml-store[bigtable]

or

.. code-block:: bash

    pip install linkml-store[all]
"""

from linkml_store.api.stores.bigtable.bigtable_collection import BigtableCollection
from linkml_store.api.stores.bigtable.bigtable_database import BigtableDatabase

__all__ = [
    "BigtableCollection",
    "BigtableDatabase",
]
