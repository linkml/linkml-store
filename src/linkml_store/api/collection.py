import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Type, Union

import numpy as np
from linkml_runtime.linkml_model import ClassDefinition, SlotDefinition
from linkml_runtime.linkml_model.meta import ArrayExpression
from pydantic import BaseModel

from linkml_store.api.queries import Query, QueryResult
from linkml_store.index.index import Index

if TYPE_CHECKING:
    from linkml_store.api.database import Database

logger = logging.getLogger(__name__)

OBJECT = Union[Dict[str, Any], BaseModel, Type]

DEFAULT_FACET_LIMIT = 100
IDENTIFIER = str
FIELD_NAME = str


@dataclass
class Collection:
    """
    A collection is an organized set of objects of the same or similar type.

    - For relational databases, a collection is typically a table
    - For document databases such as MongoDB, a collection is the native type
    - For a file system, a collection could be a single tabular file such as Parquet or CSV
    """

    name: str
    parent: Optional["Database"] = None
    _indexes: Optional[Dict[str, Index]] = None
    hidden: Optional[bool] = False

    def add(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Add one or more objects to the collection

        :param objs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        """
        Delete one or more objects from the collection

        :param objs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def delete_where(self, where: Optional[Dict[str, Any]] = None, **kwargs) -> int:
        """
        Delete objects that match a query

        :param where:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def update(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Update one or more objects in the collection

        :param objs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def _create_query(self, **kwargs) -> Query:
        return Query(from_table=self.name, **kwargs)

    def query(self, query: Query, **kwargs) -> QueryResult:
        """
        Run a query against the collection

        :param query:
        :param kwargs:
        :return:
        """
        return self.parent.query(query, **kwargs)

    def query_facets(
        self, where: Optional[Dict] = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[str, Dict[str, int]]:
        """
        Run a query to get facet counts for one or more columns.

        This function takes a database connection, a Query object, and a list of column names.
        It generates and executes a facet count query for each specified column and returns
        the results as a dictionary where the keys are the column names and the values are
        pandas DataFrames containing the facet counts.

        The facet count query is generated by modifying the original query's WHERE clause
        to exclude conditions directly related to the facet column. This allows for counting
        the occurrences of each unique value in the facet column while still applying the
        other filtering conditions.

        :param con: A DuckDB database connection.
        :param query: A Query object representing the base query.
        :param facet_columns: A list of column names to get facet counts for.
        :param facet_limit:
        :return: A dictionary where keys are column names and values are pandas DataFrames
                 containing the facet counts for each unique value in the respective column.
        """
        raise NotImplementedError

    def get(self, ids: Optional[IDENTIFIER], **kwargs) -> QueryResult:
        id_field = self.identifier_field
        q = self._create_query(where_clause={id_field: ids})
        return self.query(q, **kwargs)

    def find(self, where: Optional[Any] = None, **kwargs) -> QueryResult:
        query = self._create_query(where_clause=where)
        return self.query(query, **kwargs)

    def search(
        self,
        query: str,
        where: Optional[Any] = None,
        index_name: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Search the collection using a full-text search index.

        :param query:
        :param where:
        :param index_name:
        :param limit:
        :param kwargs:
        :return:
        """
        if index_name is None:
            if len(self._indexes) == 1:
                index_name = list(self._indexes.keys())[0]
            else:
                raise ValueError("Multiple indexes found. Please specify an index name.")
        ix_coll = self.parent.get_collection(self._index_collection_name(index_name))
        ix = self._indexes.get(index_name)
        if not ix:
            raise ValueError(f"No index named {index_name}")
        qr = ix_coll.find(where=where, limit=-1, **kwargs)
        index_col = ix.index_field
        vector_pairs = [(row, np.array(row[index_col], dtype=float)) for row in qr.rows]
        results = ix.search(query, vector_pairs, limit=limit)
        for r in results:
            del r[1][index_col]
        new_qr = QueryResult(num_rows=len(results))
        new_qr.ranked_rows = results
        return new_qr

    def attach_index(self, index: Index, auto_index=True, **kwargs):
        """
        Attach an index to the collection.

        :param index:
        :param auto_index: Automatically index all objects in the collection
        :param kwargs:
        :return:
        """
        index_name = index.name
        if not index_name:
            raise ValueError("Index must have a name")
        if not self._indexes:
            self._indexes = {}
        self._indexes[index_name] = index
        if auto_index:
            all_objs = self.find(limit=-1).rows
            self.index_objects(all_objs, index_name, **kwargs)

    def _index_collection_name(self, index_name: str) -> str:
        return f"index__{self.name}_{index_name}"

    def index_objects(self, objs: List[OBJECT], index_name: str, **kwargs):
        """
        Index a list of objects

        :param objs:
        :param index_name:
        :param kwargs:
        :return:
        """
        ix = self._indexes.get(index_name)
        if not ix:
            raise ValueError(f"No index named {index_name}")
        ix_coll = self.parent.get_collection(self._index_collection_name(index_name), create_if_not_exists=True)
        vectors = [list(float(e) for e in v) for v in ix.objects_to_vectors(objs)]
        objects_with_ix = []
        index_col = ix.index_field
        for obj, vector in zip(objs, vectors):
            # TODO: id field
            objects_with_ix.append({**obj, **{index_col: vector}})
        ix_coll.add(objects_with_ix, **kwargs)

    def peek(self, limit: Optional[int] = None) -> QueryResult:
        q = self._create_query()
        return self.query(q, limit=limit)

    def class_definition(self) -> Optional[ClassDefinition]:
        """
        Return the class definition for the collection.

        :return:
        """
        sv = self.parent.schema_view
        if sv:
            return sv.get_class(self.name)
        return None

    def identifier_attribute_name(self) -> Optional[str]:
        """
        Return the name of the identifier attribute for the collection.

        :return: The name of the identifier attribute, if one exists.
        """
        cd = self.class_definition()
        if cd:
            for att in cd.attributes.values():
                if att.identifier:
                    return att.name
        return None

    def induce_class_definition_from_objects(self, objs: List[OBJECT], max_sample_size=10) -> ClassDefinition:
        """
        Induce a class definition from a list of objects.

        This uses a heuristic procedure to infer the class definition from a list of objects.
        In general it is recommended you explicitly provide a schema.

        :param objs:
        :param max_sample_size:
        :return:
        """
        cd = ClassDefinition(self.name)
        keys = defaultdict(list)
        for obj in objs[0:max_sample_size]:
            if isinstance(obj, BaseModel):
                obj = obj.model_dump()
            if not isinstance(obj, dict):
                logger.warning(f"Skipping non-dict object: {obj}")
                continue
            for k, v in obj.items():
                keys[k].append(v)
        for k, vs in keys.items():
            multivalueds = []
            inlineds = []
            rngs = []
            exact_dimensions_list = []
            for v in vs:
                if v is None:
                    continue
                if isinstance(v, np.ndarray):
                    rngs.append("float")
                    exact_dimensions_list.append(v.shape)
                    break
                if isinstance(v, list):
                    v = v[0]
                    multivalueds.append(True)
                elif isinstance(v, dict):
                    v = list(v.values())[0]
                    multivalueds.append(True)
                else:
                    multivalueds.append(False)
                if not v:
                    continue
                if isinstance(v, str):
                    rng = "string"
                elif isinstance(v, bool):
                    rng = "boolean"
                elif isinstance(v, int):
                    rng = "integer"
                elif isinstance(v, float):
                    rng = "float"
                elif isinstance(v, dict):
                    rng = None
                    inlineds.append(True)
                else:
                    # raise ValueError(f"No mappings for {type(v)} // v={v}")
                    rng = None
                    inlineds.append(False)
                rngs.append(rng)
            multivalued = any(multivalueds)
            inlined = any(inlineds)
            if multivalued and False in multivalueds:
                raise ValueError(f"Mixed list non list: {vs} // inferred= {multivalueds}")
            # if not rngs:
            #    raise AssertionError(f"Empty rngs for {k} = {vs}")
            rng = rngs[0] if rngs else None
            for other_rng in rngs:
                if rng != other_rng:
                    raise ValueError(f"Conflict: {rng} != {other_rng} for {vs}")
            cd.attributes[k] = SlotDefinition(k, range=rng, multivalued=multivalued, inlined=inlined)
            if exact_dimensions_list:
                array_expr = ArrayExpression(exact_number_dimensions=len(exact_dimensions_list[0]))
                cd.attributes[k].array = array_expr
        sv = self.parent.schema_view
        sv.schema.classes[self.name] = cd
        sv.set_modified()
        return cd

    def import_data(self, location: Union[Path, str, TextIO], **kwargs):
        """
        Import data from a file or stream

        :param location:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def export_data(self, location: Union[Path, str, TextIO], **kwargs):
        """
        Export data to a file or stream

        :param location:
        :param kwargs:
        :return:
        """
        raise NotImplementedError
