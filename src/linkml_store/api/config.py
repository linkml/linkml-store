from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CollectionConfig(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="An optional name for the collection",
    )
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
    source_location: Optional[str] = Field(
        default=None,
        description="Filesystem or remote URL that stores the data",
    )


class DatabaseConfig(BaseModel):
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


class ClientConfig(BaseModel):
    handle: Optional[str] = Field(
        default=None,
        description="The client handle",
    )
    databases: Dict[str, DatabaseConfig] = Field(
        default={},
        description="A dictionary of database configurations",
    )
    schema_path: Optional[str] = Field(
        default=None,
        description="The path to the LinkML schema file",
    )
    base_dir: Optional[str] = Field(
        default=None,
        description="The base directory for the client",
    )
