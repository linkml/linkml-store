import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from neo4j import Driver, Session

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.graphs.graph_map import EdgeProjection, GraphProjection, NodeProjection

logger = logging.getLogger(__name__)


class DeletePolicy(Enum):
    CASCADE = "cascade"
    ERROR = "error"
    STUB = "stub"


class Neo4jCollection(Collection):
    """
    Adapter for collections in a Neo4j database.
    """

    # _graph_projection: Optional[GraphProjection] = None
    delete_policy: DeletePolicy = DeletePolicy.CASCADE

    @property
    def driver(self) -> Driver:
        return self.parent.driver

    def session(self) -> Session:
        return self.parent.session()

    def _check_if_initialized(self) -> bool:
        with self.session() as session:
            result = session.run("MATCH (n) RETURN count(n) > 0 as exists")
            return result.single()["exists"]

    @property
    def graph_projection(self) -> GraphProjection:
        return self.metadata.graph_projection

    @property
    def node_projection(self) -> Optional[NodeProjection]:
        return self.metadata.graph_projection if isinstance(self.graph_projection, NodeProjection) else None

    @property
    def edge_projection(self) -> Optional[EdgeProjection]:
        return self.metadata.graph_projection if isinstance(self.graph_projection, EdgeProjection) else None

    @property
    def is_edge_collection(self) -> bool:
        return isinstance(self.graph_projection, EdgeProjection)

    @property
    def category_labels_attribute(self) -> str:
        np = self.node_projection
        category_labels_attribute = None
        if np:
            category_labels_attribute = np.category_labels_attribute
        if not category_labels_attribute:
            category_labels_attribute = "category"
        return category_labels_attribute

    @property
    def identifier_attribute(self) -> str:
        gp = self.graph_projection
        id_attribute = None
        if gp:
            id_attribute = gp.identifier_attribute
        if not id_attribute:
            id_attribute = "id"
        return id_attribute

    def _node_pattern(self, obj: Optional[OBJECT] = None, node_var="n") -> str:
        obj = {} if obj is None else obj
        category_labels_attribute = self.category_labels_attribute
        categories = obj.get(category_labels_attribute or "category", [])
        if not isinstance(categories, list):
            categories = [categories]
        cstr = (":" + ":".join(categories)) if categories else ""
        return f"{node_var}{cstr}"

    @property
    def is_node_collection(self) -> bool:
        return not self.is_edge_collection

    def set_is_edge_collection(self, force=False):
        if self.is_edge_collection:
            return
        if self.graph_projection and not force:
            raise ValueError("Cannot reassign without force=True")
        self.metadata.graph_projection = EdgeProjection()

    def set_is_node_collection(self, force=False):
        if self.is_node_collection:
            return
        if self.graph_projection and not force:
            raise ValueError("Cannot reassign without force=True")
        self.metadata.graph_projection = NodeProjection()

    def _prop_clause(
        self, obj: OBJECT, node_var: Optional[str] = None, exclude_attributes: Optional[List[str]] = None
    ) -> str:
        if exclude_attributes is None:
            exclude_attributes = [self.category_labels_attribute]
        node_prefix = node_var + "." if node_var else ""
        terms = [f"{node_prefix}{k}: ${k}" for k in obj.keys() if k not in exclude_attributes]
        return ", ".join(terms)

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        if not isinstance(objs, list):
            objs = [objs]
        self._pre_insert_hook(objs)

        with self.session() as session:
            for obj in objs:
                query = self._create_insert_cypher_query(obj)
                session.run(query, **obj)

        self._post_insert_hook(objs)

    def _create_insert_cypher_query(self, obj: OBJECT) -> str:
        id_attribute = self.identifier_attribute
        if not self.is_edge_collection:
            logger.debug(f"Inserting node: {obj}")
            category_labels_attribute = self.category_labels_attribute
            node_pattern = self._node_pattern(obj)
            props = self._prop_clause(obj, exclude_attributes=[id_attribute, category_labels_attribute])
            return f"CREATE ({node_pattern} {{{id_attribute}: ${id_attribute}, {props}}})"
        else:
            logger.debug(f"Inserting edge: {obj}")
            ep = self.edge_projection
            if ep.predicate_attribute not in obj:
                raise ValueError(f"Predicate attribute {ep.predicate_attribute} not found in edge {obj}.")
            if ep.subject_attribute not in obj:
                raise ValueError(f"Subject attribute {ep.subject_attribute} not found in edge {obj}.")
            if ep.object_attribute not in obj:
                raise ValueError(f"Object attribute {ep.object_attribute} not found in edge {obj}.")
            pred = obj[ep.predicate_attribute]
            # check if nodes present; if not, make dangling stubs
            # TODO: decide on how this should be handled in validation if some fields are required
            for node_id in [obj[ep.subject_attribute], obj[ep.object_attribute]]:
                check_query = (
                    f"MATCH (n {{{ep.identifier_attribute}: ${ep.identifier_attribute}}}) RETURN count(n) as count"
                )
                with self.session() as session:
                    result = session.run(check_query, **{ep.identifier_attribute: node_id})
                    if result.single()["count"] == 0:
                        if self.delete_policy == DeletePolicy.STUB:
                            stub_query = f"CREATE (n {{{ep.identifier_attribute}: ${ep.identifier_attribute}}})"
                            session.run(stub_query, **{ep.identifier_attribute: node_id})
                        else:
                            raise ValueError(f"Node with identifier {node_id} not found in the database.")
            edge_props = self._prop_clause(
                obj, exclude_attributes=[ep.subject_attribute, ep.predicate_attribute, ep.object_attribute]
            )
            return f"""
            MATCH (s {{{id_attribute}: ${ep.subject_attribute}}}), (o {{{id_attribute}: ${ep.object_attribute}}})
            CREATE (s)-[r:{pred} {{{edge_props}}}]->(o)
            """

    def _prop_clause(self, obj: OBJECT, exclude_attributes: List[str] = None, node_var: Optional[str] = None) -> str:
        if exclude_attributes is None:
            exclude_attributes = []
        node_prefix = f"{node_var}." if node_var else ""
        terms = [f"{node_prefix}{k}: ${k}" for k in obj.keys() if k not in exclude_attributes]
        return ", ".join(terms)

    def query(self, query: Query, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs) -> QueryResult:
        cypher_query = self._build_cypher_query(query, limit, offset)
        ca = self.category_labels_attribute
        with self.session() as session:
            result = session.run(cypher_query, query.where_clause)
            if self.is_edge_collection:
                rows = [self._edge_to_dict(record) for record in result]
            else:

                def node_to_dict(n) -> dict:
                    d = dict(n.items())
                    if ca:
                        labels = list(n.labels)
                        if labels:
                            d[ca] = labels[0]
                    return d

                rows = [node_to_dict(record["n"]) for record in result]

        # count_query = self._build_count_query(query, is_count=True)
        count_query = self._build_cypher_query(query, is_count=True)
        with self.session() as session:
            count = session.run(count_query, query.where_clause).single()["count"]

        return QueryResult(query=query, num_rows=count, rows=rows)

    def _build_cypher_query(
        self, query: Query, limit: Optional[int] = None, offset: Optional[int] = None, is_count=False
    ) -> str:
        if self.is_edge_collection:
            ep = self.edge_projection
            ia = ep.identifier_attribute
            sa = ep.subject_attribute
            pa = ep.predicate_attribute
            oa = ep.object_attribute
            wc = query.where_clause or {}
            rq = "r"
            pred = wc.get(pa, None)
            if pred:
                rq = f"r:{pred}"
            sq = "s"
            subj = wc.get(sa, None)
            if subj:
                sq = f"s {{{ia}: '{subj}'}}"
            oq = "o"
            obj = wc.get(oa, None)
            if obj:
                oq = f"o {{{ia}: '{obj}'}}"
            where = {k: v for k, v in wc.items() if k not in [sa, pa, oa]}
            cypher_query = f"""
            MATCH ({sq})-[{rq}]->({oq})
            {self._build_where_clause(where, 'r')}
            """
            if is_count:
                cypher_query += """
                RETURN count(r) as count
                """
            else:
                cypher_query += f"""
                RETURN r, type(r) as predicate, s.{ia} as subject, o.{ia} as object
                """
        else:
            node_pattern = self._node_pattern(query.where_clause)
            cypher_query = f"""
            MATCH ({node_pattern})
            {self._build_where_clause(query.where_clause)}
            """
            if is_count:
                cypher_query += """
                RETURN count(n) as count
                """
            else:
                cypher_query += """
                RETURN n
                """

        if not is_count:
            if limit and limit >= 0:
                cypher_query += f" LIMIT {limit}"
            if offset and offset >= 0:
                cypher_query += f" SKIP {offset}"

        return cypher_query

    def _build_where_clause(self, where_clause: Dict[str, Any], prefix: str = "n") -> str:
        conditions = []
        if where_clause is None:
            return ""
        for key, value in where_clause.items():
            if key == self.category_labels_attribute:
                continue
            if isinstance(value, str):
                conditions.append(f"{prefix}.{key} = '{value}'")
            else:
                conditions.append(f"{prefix}.{key} = {value}")

        return "WHERE " + " AND ".join(conditions) if conditions else ""

    def _edge_to_dict(self, record: Dict) -> Dict[str, Any]:
        r = record["r"]
        ep = self.edge_projection
        return {
            ep.subject_attribute: record["subject"],
            ep.predicate_attribute: record["predicate"],
            ep.object_attribute: record["object"],
            **dict(r.items()),
        }

    def query_facets(
        self,
        where: Dict = None,
        facet_columns: List[Union[str, Tuple[str, ...]]] = None,
        facet_limit=DEFAULT_FACET_LIMIT,
        **kwargs,
    ) -> Dict[Union[str, Tuple[str, ...]], List[Tuple[Any, int]]]:
        results = {}
        if not facet_columns:
            facet_columns = list(self.class_definition().attributes.keys())

        category_labels_attribute = self.category_labels_attribute
        with self.session() as session:
            for col in facet_columns:
                where_clause = self._build_where_clause(where) if where else ""
                if col == category_labels_attribute:
                    # Handle faceting on labels
                    query = f"""
                    MATCH (n)
                    {where_clause}
                    WITH labels(n) AS nodeLabels, count(*) as count
                    UNWIND nodeLabels AS label
                    WITH label, count
                    ORDER BY count DESC, label
                    LIMIT {facet_limit}
                    RETURN label as value, count
                    """
                else:
                    query = f"""
                    MATCH (n)
                    {where_clause}
                    WITH n.{col} as value, count(*) as count
                    WITH value, count
                    ORDER BY count DESC
                    LIMIT {facet_limit}
                    RETURN value, count
                    """
                result = session.run(query)
                results[col] = [(record["value"], record["count"]) for record in result]

        return results

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        delete_policy = self.delete_policy
        if not isinstance(objs, list):
            objs = [objs]

        deleted_nodes = 0
        deleted_relationships = 0
        identifier_attribute = self.identifier_attribute

        with self.session() as session:
            for obj in objs:
                node_pattern = self._node_pattern(obj)
                id_value = obj[identifier_attribute]
                where_clause = f"{{{identifier_attribute}: $id}}"
                dn, dr = self._execute_delete(session, node_pattern, where_clause, delete_policy, id=id_value)
                deleted_nodes += dn
                deleted_relationships += dr

        return deleted_nodes

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> int:
        delete_policy = self.delete_policy
        where_clause = self._build_where_clause(where) if where else ""
        node_pattern = self._node_pattern(where)

        with self.session() as session:
            deleted_nodes, deleted_relationships = self._execute_delete(
                session, node_pattern, where_clause, delete_policy
            )

        if deleted_nodes == 0 and not missing_ok:
            raise ValueError(f"No nodes found for {where}")

        return deleted_nodes

    def _execute_delete(
        self, session, node_pattern: str, where_clause: str, delete_policy: DeletePolicy, **params
    ) -> Tuple[int, int]:
        deleted_relationships = 0
        deleted_nodes = 0

        if delete_policy == DeletePolicy.ERROR:
            check_query = f"MATCH ({node_pattern} {where_clause})-[r]-() RETURN count(r) as rel_count"
            result = session.run(check_query, **params)
            if result.single()["rel_count"] > 0:
                raise ValueError("Nodes with existing relationships found and cannot be deleted.")

        if delete_policy == DeletePolicy.CASCADE:
            rel_query = f"MATCH ({node_pattern} {where_clause})-[r]-() DELETE r"
            result = session.run(rel_query, **params)
            deleted_relationships = result.consume().counters.relationships_deleted

        if delete_policy in [DeletePolicy.CASCADE, DeletePolicy.ERROR]:
            node_query = f"MATCH ({node_pattern} {where_clause}) DELETE n"
            result = session.run(node_query, **params)
            deleted_nodes = result.consume().counters.nodes_deleted
        elif delete_policy == DeletePolicy.STUB:
            stub_query = f"MATCH ({node_pattern} {where_clause}) SET n.deleted = true RETURN count(n) as stub_count"
            result = session.run(stub_query, **params)
            deleted_nodes = result.single()["stub_count"]

        return deleted_nodes, deleted_relationships

    def update(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        if not isinstance(objs, list):
            objs = [objs]

        updated_count = 0
        with self.session() as session:
            for obj in objs:
                query = self._create_update_cypher_query(obj)
                result = session.run(query, **obj)
                updated_count += result.consume().counters.properties_set

        return updated_count

    def _create_update_cypher_query(self, obj: OBJECT) -> str:
        id_attribute = self.identifier_attribute
        category_labels_attribute = self.category_labels_attribute

        # Prepare SET clause
        set_items = [f"n.{k} = ${k}" for k in obj.keys() if k not in [id_attribute, category_labels_attribute]]
        set_clause = ", ".join(set_items)

        # Prepare labels update
        labels_to_add = []
        # labels_to_remove = []
        if category_labels_attribute in obj:
            new_labels = (
                obj[category_labels_attribute]
                if isinstance(obj[category_labels_attribute], list)
                else [obj[category_labels_attribute]]
            )
            labels_to_add = [f":{label}" for label in new_labels]
            # labels_to_remove = [":Label" for _ in new_labels]  # Placeholder for labels to remove

        # Construct the query
        query = f"MATCH (n {{{id_attribute}: ${id_attribute}}})\n"
        # f labels_to_remove:
        #    query += f"REMOVE n{' '.join(labels_to_remove)}\n"
        if labels_to_add:
            query += f"SET n{' '.join(labels_to_add)}\n"
            # f"REMOVE n{' '.join(labels_to_remove)}' if labels_to_remove else ''}"
            # f"{f'SET n{' '.join(labels_to_add)}' if labels_to_add else ''}"
        query += f"SET {set_clause}\n"
        query += "RETURN n"
        print(query)
        return query
