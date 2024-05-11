import sqlalchemy as sqla

TMAP = {
    "string": sqla.String,
    "integer": sqla.Integer,
    "float": sqla.Float,
    "linkml:Any": sqla.JSON,
}
