from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CollectionConfig(BaseModel):
    alias: Optional[str] = Field(
        default=None,
        description="An optional alias for the collection",
    )
    type: Optional[str] = Field(
        default=None,
        description="The type of object in the collection",
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Optional metadata for the collection",
    )
    attributes: Optional[Dict] = Field(
        default=None,
        description="Optional attributes for the collection, following LinkML schema",
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
    schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The LinkML schema as a dictionary",
    )
    collections: Dict[str, CollectionConfig] = Field(
        default={},
        description="A dictionary of collection configurations",
    )
    recreate_if_exists: bool = Field(
        default=False,
        description="Whether to recreate the database if it already exists",
    )


class ClientConfig(BaseModel):
    databases: Dict[str, DatabaseConfig] = Field(
        default={},
        description="A dictionary of database configurations",
    )
    schema_path: Optional[Path] = Field(
        default=None,
        description="The path to the LinkML schema file",
    )

    class Config:
        arbitrary_types_allowed = True
