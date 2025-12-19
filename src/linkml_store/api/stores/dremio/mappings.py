"""Type mappings between LinkML types and Dremio/Arrow types."""

import pyarrow as pa

# Mapping from LinkML types to PyArrow types
LINKML_TO_ARROW = {
    "string": pa.string(),
    "integer": pa.int64(),
    "float": pa.float64(),
    "boolean": pa.bool_(),
    "date": pa.date32(),
    "datetime": pa.timestamp("us"),
    "decimal": pa.decimal128(38, 10),
    "Any": pa.string(),  # Fallback to string for Any type
}

# Mapping from Arrow types to LinkML types
ARROW_TO_LINKML = {
    pa.string(): "string",
    pa.utf8(): "string",
    pa.large_string(): "string",
    pa.int8(): "integer",
    pa.int16(): "integer",
    pa.int32(): "integer",
    pa.int64(): "integer",
    pa.uint8(): "integer",
    pa.uint16(): "integer",
    pa.uint32(): "integer",
    pa.uint64(): "integer",
    pa.float16(): "float",
    pa.float32(): "float",
    pa.float64(): "float",
    pa.bool_(): "boolean",
    pa.date32(): "date",
    pa.date64(): "date",
}

# Mapping from Dremio SQL type names to LinkML types
DREMIO_SQL_TO_LINKML = {
    "VARCHAR": "string",
    "CHAR": "string",
    "BIGINT": "integer",
    "INTEGER": "integer",
    "INT": "integer",
    "SMALLINT": "integer",
    "TINYINT": "integer",
    "BOOLEAN": "boolean",
    "DOUBLE": "float",
    "FLOAT": "float",
    "DECIMAL": "float",
    "DATE": "date",
    "TIMESTAMP": "datetime",
    "TIME": "string",
    "BINARY": "string",
    "VARBINARY": "string",
    "LIST": "string",  # Complex types mapped to string
    "STRUCT": "string",
    "MAP": "string",
}


def get_arrow_type(linkml_type: str) -> pa.DataType:
    """Convert a LinkML type to a PyArrow type.

    Args:
        linkml_type: The LinkML type name.

    Returns:
        The corresponding PyArrow data type.
    """
    return LINKML_TO_ARROW.get(linkml_type, pa.string())


def get_linkml_type_from_arrow(arrow_type: pa.DataType) -> str:
    """Convert a PyArrow type to a LinkML type.

    Args:
        arrow_type: The PyArrow data type.

    Returns:
        The corresponding LinkML type name.
    """
    # Handle parameterized types by checking base type
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_integer(arrow_type):
        return "integer"
    if pa.types.is_floating(arrow_type):
        return "float"
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "datetime"
    if pa.types.is_decimal(arrow_type):
        return "float"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return "string"  # Complex types as string
    if pa.types.is_struct(arrow_type):
        return "string"
    if pa.types.is_map(arrow_type):
        return "string"

    return "string"  # Default fallback
