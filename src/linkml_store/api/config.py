from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from linkml_store.graphs.graph_map import EdgeProjection, NodeProjection


class ConfiguredBaseModel(BaseModel, extra="forbid"):
    """
    Base class for all configuration models.
    """

    pass


class DerivationConfiguration(ConfiguredBaseModel):
    """
    Configuration for a derivation
    """

    database: Optional[str] = None
    collection: Optional[str] = None
    mappings: Optional[Dict[str, Any]] = None
    where: Optional[Dict[str, Any]] = None


class CollectionSource(ConfiguredBaseModel):
    """
    Metadata about a source
    """

    url: Optional[str] = None
    """Remote URL to fetch data from"""

    local_path: Optional[str] = None
    """Local path to fetch data from"""

    source_location: Optional[str] = None

    refresh_interval_days: Optional[float] = None
    """How often to refresh the data, in days"""

    expected_type: Optional[str] = None
    """The expected type of the data, e.g list"""

    format: Optional[str] = None
    """The format of the data, e.g., json, yaml, csv"""

    compression: Optional[str] = None
    """The compression of the data, e.g., tgz, gzip, zip"""

    select_query: Optional[str] = None
    """A jsonpath query to preprocess the objects with"""

    arguments: Optional[Dict[str, Any]] = None
    """Optional arguments to pass to the source"""


class CollectionConfig(ConfiguredBaseModel):
    """
    Configuration for a collection
    """

    alias: Optional[str] = Field(
        default=None,
        description="An optional alias for the collection",
    )
    type: Optional[str] = Field(
        default=None,
        description="The type of object in the collection. TODO; use this instead of name",
    )
    additional_properties: Optional[Dict] = Field(
        default=None,
        description="Optional metadata for the collection",
    )
    attributes: Optional[Dict[str, Dict]] = Field(
        default=None,
        description="Optional attributes for the collection, following LinkML schema",
    )
    indexers: Optional[Dict[str, Dict]] = Field(
        default=None,
        description="Optional configuration for indexers",
    )
    hidden: Optional[bool] = Field(
        default=False,
        description="Whether the collection is hidden",
    )
    is_prepopulated: Optional[bool] = Field(
        default=False,
        description="Whether the collection is prepopulated",
    )
    source: Optional[CollectionSource] = Field(
        default=None,
        description="Source for the collection",
    )
    derived_from: Optional[List[DerivationConfiguration]] = Field(
        default=None,
        description="LinkML-Map derivations",
    )
    page_size: Optional[int] = Field(default=None, description="Suggested page size (items per page) in apps and APIs")
    graph_projection: Optional[Union[EdgeProjection, NodeProjection]] = Field(
        default=None,
        description="Optional graph projection configuration",
    )
    validate_modifications: Optional[bool] = Field(
        default=False,
        description="Whether to validate inserts, updates, and deletes",
    )


class DatabaseConfig(ConfiguredBaseModel):
    """
    Configuration for a database
    """

    handle: str = Field(
        default="duckdb:///:memory:",
        description="The database handle, e.g., 'duckdb:///:memory:' or 'mongodb://localhost:27017'",
    )
    alias: Optional[str] = Field(
        default=None,
        description="An optional alias for the database",
    )
    schema_location: Optional[str] = Field(
        default=None,
        description="The location of the schema file, either a path on disk or URL",
    )
    schema_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The LinkML schema as a dictionary",
    )
    collections: Optional[Dict[str, CollectionConfig]] = Field(
        default={},
        description="A dictionary of collection configurations",
    )
    recreate_if_exists: bool = Field(
        default=False,
        description="Whether to recreate the database if it already exists",
    )
    collection_type_slot: Optional[str] = Field(
        default=None,
        description=(
            "For databases that combine multiple collections into a single space, this field"
            "specifies the field that contains the collection type. An example of this is a Solr"
            "index that does not use cores for collections, and instead uses a single global"
            "document space; if this has a field 'document_type', then this field should be set"
        ),
    )
    searchable_slots: Optional[List[str]] = Field(
        default=None,
        description="Optional configuration for search fields",
    )
    ensure_referential_integrity: bool = Field(
        default=False,
        description="Whether to ensure referential integrity",
    )
    source: Optional[CollectionSource] = Field(
        default=None,
        description="Source for the database",
    )


class ClientConfig(ConfiguredBaseModel):
    """
    Configuration for a client
    """

    handle: Optional[str] = Field(
        default=None,
        description="The client handle",
    )
    databases: Dict[str, DatabaseConfig] = Field(
        default={},
        description="A dictionary of database configurations",
    )
    default_database: Optional[str] = Field(
        default=None,
        description="The default database",
    )
    schema_path: Optional[str] = Field(
        default=None,
        description="The path to the LinkML schema file",
    )
    base_dir: Optional[str] = Field(
        default=None,
        description="The base directory for the client",
    )
