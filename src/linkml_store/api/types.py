from typing import TypeVar

DatabaseType = TypeVar("DatabaseType", bound="Database")  # noqa: F821
CollectionType = TypeVar("CollectionType", bound="Collection")  # noqa: F821
