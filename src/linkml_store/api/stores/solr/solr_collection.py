# solr_collection.py

import logging
from copy import copy
from typing import Any, Dict, List, Optional, Union

import requests

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT
from linkml_store.api.queries import Query, QueryResult

logger = logging.getLogger(__name__)


class SolrCollection(Collection):

    @property
    def _collection_base(self) -> str:
        if self.parent.use_cores:
            base_url = f"{self.parent.base_url}/{self.alias}"
        else:
            base_url = self.parent.base_url
        return base_url

    def search(
        self,
        query: str,
        where: Optional[Any] = None,
        index_name: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> QueryResult:
        if index_name is None:
            index_name = "edismax"
        qfs = self.parent.metadata.searchable_slots
        if not qfs:
            raise ValueError("No searchable slots configured for Solr collection")
        solr_query = self._build_solr_query(where, search_term=query, extra={"defType": index_name, "qf": qfs})
        logger.info(f"Querying Solr collection {self.alias} with query: {solr_query}")

        response = requests.get(f"{self._collection_base}/select", params=solr_query)
        response.raise_for_status()

        data = response.json()
        num_rows = data["response"]["numFound"]
        rows = data["response"]["docs"]
        ranked_rows = [(1.0, row) for row in rows]
        return QueryResult(query=where, search_term=query, num_rows=num_rows, rows=rows, ranked_rows=ranked_rows)

    def query(self, query: Query, **kwargs) -> QueryResult:
        solr_query = self._build_solr_query(query)
        logger.info(f"Querying Solr collection {self.alias} with query: {solr_query}")

        response = requests.get(f"{self._collection_base}/select", params=solr_query)
        response.raise_for_status()

        data = response.json()
        num_rows = data["response"]["numFound"]
        rows = data["response"]["docs"]

        return QueryResult(query=query, num_rows=num_rows, rows=rows)

    def query_facets(
        self, where: Optional[Dict] = None, facet_columns: List[str] = None, facet_limit=DEFAULT_FACET_LIMIT, **kwargs
    ) -> Dict[str, Dict[str, int]]:
        solr_query = self._build_solr_query(where)
        solr_query["facet"] = "true"
        solr_query["facet.field"] = facet_columns
        solr_query["facet.limit"] = facet_limit

        logger.info(f"Querying Solr collection {self.alias} for facets with query: {solr_query}")

        response = requests.get(f"{self._collection_base}/select", params=solr_query)
        response.raise_for_status()

        data = response.json()
        facet_counts = data["facet_counts"]["facet_fields"]

        results = {}
        for facet_field, counts in facet_counts.items():
            results[facet_field] = list(zip(counts[::2], counts[1::2]))

        return results

    def _build_solr_query(
        self, query: Union[Query, Dict], search_term="*:*", extra: Optional[Dict] = None
    ) -> Dict[str, Any]:
        solr_query = {}
        if query is None:
            query = {}

        if isinstance(query, Query):
            where = query.where_clause
            solr_query["fq"] = self._build_solr_where_clause(where)

            if query.select_cols:
                solr_query["fl"] = ",".join(query.select_cols)

            if query.limit:
                solr_query["rows"] = query.limit

            if query.offset:
                solr_query["start"] = query.offset

        elif isinstance(query, dict):
            solr_query["fq"] = self._build_solr_where_clause(query)

        solr_query["wt"] = "json"
        if "q" not in solr_query:
            solr_query["q"] = search_term
        if extra:
            solr_query.update(extra)
        logger.info(f"Built Solr query: {solr_query}")
        return solr_query

    def _build_solr_where_clause(self, where_clause: Dict) -> str:
        if where_clause is None:
            where_clause = {}
        conditions = []
        if self.parent.metadata.collection_type_slot:
            where_clause = copy(where_clause)
            where_clause[self.parent.metadata.collection_type_slot] = self.alias
        for field, value in where_clause.items():
            if not isinstance(value, (list, tuple)):
                value = [value]
            value = [f'"{v}"' if isinstance(v, str) else str(v) for v in value]
            if len(value) > 1:
                conditions.append(f"{field}:({' '.join(value)})")
            else:
                conditions.append(f"{field}:{value[0]}")

        return " AND ".join(conditions)
