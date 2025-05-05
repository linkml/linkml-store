# solr_collection.py

import logging
from copy import copy
from typing import Any, Dict, List, Optional, Union, Tuple

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
        logger.debug(f"Response: {data}")
        num_rows = data["response"]["numFound"]
        rows = data["response"]["docs"]

        return QueryResult(query=query, num_rows=num_rows, rows=rows)

    def query_facets(
        self,
        where: Optional[Dict] = None,
        facet_columns: List[Union[str, Tuple[str, ...]]] = None,
        facet_limit=DEFAULT_FACET_LIMIT,
        facet_min_count: int = 1,
        **kwargs,
    ) -> Dict[Union[str, Tuple[str, ...]], List[Tuple[Any, int]]]:
        """
        Query facet counts for fields or field combinations.
        
        :param where: Filter conditions
        :param facet_columns: List of fields to facet on. Elements can be:
                            - Simple strings for single field facets
                            - Tuples of strings for field combinations (pivot facets)
        :param facet_limit: Maximum number of facet values to return
        :param facet_min_count: Minimum count for facet values to be included
        :return: Dictionary mapping fields or field tuples to lists of (value, count) tuples
        """
        solr_query = self._build_solr_query(where)
        
        # Separate single fields and tuple fields
        single_fields = []
        tuple_fields = []
        
        if facet_columns:
            for field in facet_columns:
                if isinstance(field, str):
                    single_fields.append(field)
                elif isinstance(field, tuple):
                    tuple_fields.append(field)
        
        # Process regular facets
        results = {}
        if single_fields:
            solr_query["facet"] = "true"
            solr_query["facet.field"] = single_fields
            solr_query["facet.limit"] = facet_limit
            solr_query["facet.mincount"] = facet_min_count
            
            logger.info(f"Querying Solr collection {self.alias} for facets with query: {solr_query}")
            response = requests.get(f"{self._collection_base}/select", params=solr_query)
            response.raise_for_status()
            
            data = response.json()
            facet_counts = data["facet_counts"]["facet_fields"]
            
            for facet_field, counts in facet_counts.items():
                results[facet_field] = list(zip(counts[::2], counts[1::2]))
        
        # Process pivot facets for tuple fields
        if tuple_fields:
            # TODO: Add a warning if Solr < 4.0, when this was introduced
            for field_tuple in tuple_fields:
                # Create a query for this specific field tuple
                pivot_query = self._build_solr_query(where)
                pivot_query["facet"] = "true"
                
                # Create pivot facet
                field_str = ','.join(field_tuple)
                pivot_query["facet.pivot"] = field_str
                pivot_query["facet.pivot.mincount"] = facet_min_count
                pivot_query["facet.limit"] = facet_limit
                
                logger.info(f"Querying Solr collection {self.alias} for pivot facets with query: {pivot_query}")
                response = requests.get(f"{self._collection_base}/select", params=pivot_query)
                response.raise_for_status()
                
                data = response.json()
                pivot_facets = data.get("facet_counts", {}).get("facet_pivot", {})
                
                # Process pivot facets into the same format as MongoDB results
                field_str = ','.join(field_tuple)
                pivot_data = pivot_facets.get(field_str, [])
                
                # Build a list of tuples (field values, count)
                pivot_results = []
                self._process_pivot_facets(pivot_data, [], pivot_results, field_tuple)
                
                results[field_tuple] = pivot_results
        
        return results
        
    def _process_pivot_facets(self, pivot_data, current_values, results, field_tuple):
        """
        Recursively process pivot facet results to extract combinations of field values.
        
        :param pivot_data: The pivot facet data from Solr
        :param current_values: The current path of values in the recursion
        :param results: The result list to populate
        :param field_tuple: The original field tuple for reference
        """
        for item in pivot_data:
            # Add the current field value
            value = item.get("value")
            count = item.get("count", 0)
            
            # Update the current path with this value
            values = current_values + [value]
            
            # If we have all the fields from the tuple, add a result
            if len(values) == len(field_tuple):
                # Create a tuple of values corresponding to the field tuple
                results.append((tuple(values), count))
            
            # Process child pivot fields recursively
            pivot = item.get("pivot", [])
            if pivot and len(values) < len(field_tuple):
                self._process_pivot_facets(pivot, values, results, field_tuple)

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
