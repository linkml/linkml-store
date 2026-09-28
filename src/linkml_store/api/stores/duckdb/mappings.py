import sqlalchemy as sqla

# LinkML float and double are 64-bit, and integer is unbounded, so the narrower
# FLOAT (32-bit) and INTEGER (32-bit) would round floats and refuse integers
# above 2**31 - 1. A boolean with no entry here would be stored as text.
TMAP = {
    "string": sqla.String,
    "integer": sqla.BigInteger,
    "float": sqla.Double,
    "double": sqla.Double,
    "boolean": sqla.Boolean,
    "linkml:Any": sqla.JSON,
}
