import sqlalchemy as sqla

TMAP = {
    "string": sqla.String,
    "integer": sqla.Integer,
    "linkml:Any": sqla.JSON,
}
