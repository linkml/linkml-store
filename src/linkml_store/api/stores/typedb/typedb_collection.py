"""Collection-level operations for the TypeDB adapter."""

from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.api.stores.typedb.naming import resolve_typedb_value_type, typedb_name
from linkml_store.api.stores.typedb.typeql_insert import (
    DANGLING_ERROR,
    MODE_ENTITIES,
    MODE_FULL,
    TypeQLInserter,
    plan_inserts,
    typeql_literal,
)

logger = logging.getLogger(__name__)

#: Mongo-style operators to TypeQL; ``$in``, ``$contains``, ``$like`` and ``$ne`` are handled separately.
COMPARATORS = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}
SUPPORTED_OPERATORS = sorted([*COMPARATORS, "$in", "$contains", "$like", "$ne"])


def like_to_regex(pattern: str) -> str:
    r"""Translate a SQL ``LIKE`` pattern to an anchored regex for TypeQL's ``like``.

    >>> like_to_regex("a%b_c.")
    '^a.*b.c\\.$'
    """
    parts = [".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern]
    return "^" + "".join(parts) + "$"


class TypeDBCollection(Collection):
    """Adapter for a single LinkML class stored in a TypeDB database."""

    #: Default "error": an unmatched reference makes TypeQL silently insert nothing.
    on_dangling_reference: str = DANGLING_ERROR

    #: Pre-fetch existing ids to validate references (one query per referenced class).
    check_references: bool = True

    @property
    def inserter(self) -> TypeQLInserter:
        """Return a planner bound to this database's schema, rebuilt only when that or the policy changes."""
        sv = self.parent.schema_view
        cached = self.__dict__.get("_inserter")
        if cached is None or cached.sv is not sv or cached.on_dangling_reference != self.on_dangling_reference:
            cached = TypeQLInserter(sv, on_dangling_reference=self.on_dangling_reference)
            self.__dict__["_inserter"] = cached
        return cached

    def _check_if_initialized(self) -> bool:
        """Return True if the target entity type holds at least one instance."""
        label = typedb_name(self.target_class_name)
        rows = self._read(f"match $x isa {label}; reduce $n = count;")
        return bool(rows and rows[0].get("n", 0) > 0)

    def insert(self, objs: Union[OBJECT, List[OBJECT]], mode: str = MODE_FULL, **kwargs):
        """Insert objects into TypeDB.

        The batch runs in a single write transaction, then commits.

        :param objs: one object or a list of them
        :param mode: ``full`` by default. Cyclic datasets cannot be loaded in one pass,
            since a relation can only link existing instances; load those as ``entities``
            then ``references``.
        :raises DanglingReferenceError: under the ``error`` policy, if a reference
            resolves to nothing
        """
        if not isinstance(objs, list):
            objs = [objs]
        objs = [self._as_dict(o) for o in objs]
        self._pre_insert_hook(objs)

        class_name = self.target_class_name
        check = self.check_references and mode != MODE_ENTITIES
        known_ids = self._known_ids(class_name, objs) if check else None
        if known_ids is not None:
            # The batch's own objects count as present, under their own class.
            id_slot = self.inserter.identifier_slot(class_name)
            if id_slot:
                own = {str(o[id_slot]) for o in objs if o.get(id_slot) is not None}
                known_ids.setdefault(class_name, set()).update(own)
        queries = plan_inserts(
            self.parent.schema_view,
            class_name,
            objs,
            on_dangling_reference=self.on_dangling_reference,
            known_ids=known_ids,
            mode=mode,
        )

        self._run_writes([q.to_typeql() for q in queries])
        self._post_insert_hook(objs)

    def upsert(
        self,
        objs: Union[OBJECT, List[OBJECT]],
        filter_fields: Optional[List[str]] = None,
        update_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        """Insert objects that are absent and update those already present.

        Matching is by ``filter_fields``, defaulting to the class's identifier slot,
        which is the natural key in a generated TypeDB schema.

        :param objs: one object or a list of them
        :param filter_fields: slots identifying an existing object; defaults to the
            identifier slot
        :param update_fields: scalar slots to write; defaults to every scalar slot
            present on the object
        :raises ValueError: if an object carries none of the filter fields

        Updates and inserts commit together, inside any open :meth:`batch_write`.

        .. note::
           Relations are written only when an object is newly inserted. Updating an
           existing object's references would mean deleting and recreating relations,
           which is not implemented.
        """
        if not isinstance(objs, list):
            objs = [objs]
        objs = [self._as_dict(o) for o in objs]
        class_name = self.target_class_name
        id_slot = self.inserter.identifier_slot(class_name)

        if not filter_fields:
            if not id_slot:
                raise ValueError(f"{class_name!r} has no identifier slot, so upsert needs explicit filter_fields.")
            filter_fields = [id_slot]

        all_criteria = []
        for obj in objs:
            criteria = {f: obj[f] for f in filter_fields if obj.get(f) is not None}
            if not criteria:
                raise ValueError(f"Object is missing every filter field {filter_fields}; cannot upsert it.")
            all_criteria.append(criteria)

        ctx = nullcontext() if self.parent._active_tx is not None else self.parent.batch_write()
        with ctx:
            # Matching on the identifier alone needs one query, not one per object.
            existing = self._existing_ids(class_name, id_slot) if filter_fields == [id_slot] else None
            to_insert: List[dict] = []
            updates: List[str] = []
            for obj, criteria in zip(objs, all_criteria):
                if existing is not None:
                    present = str(criteria[id_slot]) in existing
                else:
                    present = self.find(criteria).num_rows > 0
                if present:
                    updates.extend(self._update_queries(obj, criteria, update_fields))
                else:
                    to_insert.append(obj)

            self._run_writes(updates)
            if updates:
                logger.info("Updated %d existing object(s) in %s", len(objs) - len(to_insert), class_name)
            if to_insert:
                self.insert(to_insert)

    def _existing_ids(self, class_name: str, id_slot: str) -> set:
        """Return the identifiers of every stored instance of ``class_name``."""
        id_label = self.inserter.attr_names.get(id_slot, typedb_name(id_slot))
        rows = self._read(f"match $x isa {typedb_name(class_name)}, has {id_label} $i; select $i;")
        return {str(row["i"]) for row in rows if row.get("i") is not None}

    def _run_writes(self, queries: List[str]) -> None:
        """Run write queries in the open batch, or else in one committed transaction."""
        from typedb.driver import TransactionType

        if not queries:
            return
        batch = self.parent._active_tx
        if batch is not None:
            for query in queries:
                batch.query(query).resolve()
            return
        with self.parent.driver.transaction(self.parent._db_name, TransactionType.WRITE) as tx:
            for query in queries:
                tx.query(query).resolve()
            tx.commit()

    def _update_queries(self, obj: dict, criteria: dict, update_fields: Optional[List[str]]) -> List[str]:
        """Build the TypeQL needed to bring one existing object in line with ``obj``."""
        class_name = self.target_class_name
        sv = self.parent.schema_view
        all_classes = set(sv.all_classes().keys())

        match_lines = [f"$x isa {typedb_name(class_name)};"]
        for field, value in criteria.items():
            label = self.inserter.attr_names.get(field, typedb_name(field))
            value_type = resolve_typedb_value_type(sv, self._slot_range(field))
            match_lines.append(f"$x has {label} == {typeql_literal(value, value_type)};")
        match_block = "match" + "".join(f"\n  {line}" for line in match_lines)

        queries: List[str] = []
        for slot in sv.class_induced_slots(class_name):
            if slot.range in all_classes:
                continue  # relations are insert-only; see upsert's note
            if slot.name in criteria or obj.get(slot.name) is None:
                continue
            if update_fields and slot.name not in update_fields:
                continue
            label = self.inserter.attr_names.get(slot.name, typedb_name(slot.name))
            value_type = resolve_typedb_value_type(sv, slot.range)
            value = obj[slot.name]
            if slot.multivalued:
                values = value if isinstance(value, list) else [value]
                # `update` rejects multivalued attributes, so replace the whole set.
                queries.append(f"{match_block}\n  $x has {label} $old;\ndelete $old of $x;")
                literals = "".join(f"\n  $x has {label} {typeql_literal(v, value_type)};" for v in values)
                queries.append(f"{match_block}\ninsert{literals}")
            else:
                item = value[0] if isinstance(value, list) else value
                queries.append(f"{match_block}\nupdate $x has {label} {typeql_literal(item, value_type)};")
        return queries

    @staticmethod
    def _as_dict(obj: Any) -> dict:
        """Coerce a pydantic model or LinkML object to a plain dict."""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if v is not None}
        raise TypeError(f"Cannot insert object of type {type(obj)!r} into TypeDB")

    def _known_ids(self, class_name: str, objs: List[dict]) -> Dict[str, set]:
        """Collect identifiers already present in the database, grouped by class.

        :param class_name: the batch's target class
        :param objs: the batch, scanned to find which classes it references
        :return: class name to the identifiers of its instances
        """
        inserter = self.inserter
        sv = self.parent.schema_view
        referenced: set[str] = set()
        for slot in sv.class_induced_slots(class_name):
            if slot.range in sv.all_classes():
                referenced.add(slot.range)
                referenced.update(sv.class_descendants(slot.range))

        known: Dict[str, set] = {}
        for target in sorted(referenced):
            id_slot = inserter.identifier_slot(target)
            if id_slot is None:
                continue
            class_def = sv.get_class(target)
            if class_def is not None and class_def.mixin:
                continue  # mixins aren't emitted as TypeDB types
            label = typedb_name(target)
            id_label = inserter.attr_names.get(id_slot, typedb_name(id_slot))
            try:
                rows = self._read(f"match $x isa {label}, has {id_label} $i; select $i;")
            except Exception as exc:
                # [INF2]: no such type, e.g. an abstract class the generator dropped.
                if "[INF2]" not in str(exc):
                    raise
                logger.debug("No TypeDB type for %r; skipping it in the reference check", target)
                continue
            known[target] = {str(row["i"]) for row in rows if row.get("i") is not None}
        return known

    def _read(self, typeql: str) -> List[dict]:
        """Run a read query and return its rows as dicts of concept values.

        :param typeql: a complete TypeQL read query
        :return: one dict per answer, keyed by variable name without the ``$``
        """
        from typedb.driver import TransactionType

        batch = self.parent._active_tx
        if batch is not None:
            # Read through the batch so uncommitted writes are visible.
            return self._rows(batch.query(typeql).resolve())

        with self.parent.driver.transaction(self.parent._db_name, TransactionType.READ) as tx:
            return self._rows(tx.query(typeql).resolve())

    @staticmethod
    def _rows(answers) -> List[dict]:
        """Turn driver answers into dicts, whether concept rows or fetch documents."""
        rows = []
        for answer in answers:
            if not hasattr(answer, "column_names"):
                # A `fetch` stage yields JSON documents rather than concept rows.
                rows.append(dict(answer))
                continue
            row = {}
            for var in answer.column_names():
                concept = answer.get(var)
                # Covers attributes, reduce results and types.
                value = concept.try_get_value()
                row[var] = value if value is not None else concept
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ reading

    def query(self, query: Query, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs) -> QueryResult:
        """Run a query against this collection.

        A separate ``reduce ... count`` gives the total number of matches, because
        :meth:`Collection.find_iter` pages on ``num_rows`` while ``rows`` holds only the
        current page.

        :param query: the query to run; ``where_clause`` is a dict of slot constraints
        :param limit: maximum rows in this page
        :param offset: rows to skip
        :return: the matching objects, with ``num_rows`` as the total match count
        :raises NotImplementedError: for an unsupported where-clause operator
        """
        self._pre_query_hook(query)
        class_name = self.target_class_name
        match_block = self._match_block(query.where_clause)

        count_rows = self._read(f"{match_block}\nreduce $n = count;")
        total = int(count_rows[0]["n"]) if count_rows else 0

        pipeline = [match_block]
        sort_label = self._sort_label(class_name)
        if sort_label:
            # Stable order needed for offset paging.
            pipeline.append("sort $sort_key asc;")
        if offset:
            pipeline.append(f"offset {offset};")
        # Negative limit means unlimited (group_by passes -1).
        if limit is not None and limit >= 0:
            pipeline.append(f"limit {limit};")
        pipeline.append(self._fetch_block(class_name, query.select_cols))

        rows = [self._row_to_object(row, class_name) for row in self._read("\n".join(pipeline))]
        return QueryResult(query=query, num_rows=total, rows=rows)

    def _sort_label(self, class_name: str) -> Optional[str]:
        """Return the attribute label to sort by, or None if the class has no identifier."""
        id_slot = self.inserter.identifier_slot(class_name)
        if not id_slot:
            return None
        return self.inserter.attr_names.get(id_slot, typedb_name(id_slot))

    def _match_block(self, where: Optional[Any]) -> str:
        """Build the ``match`` block for this collection plus a where clause.

        :param where: ``None``/``{}`` for everything, else slot name to constraint,
            where a constraint is a scalar or an operator dict
        :raises NotImplementedError: for a raw string/list where clause
        """
        if isinstance(where, (str, list)):
            raise NotImplementedError(
                "The TypeDB adapter does not accept a raw where clause; SQL text has no "
                "TypeQL equivalent. Pass a dict of slot constraints instead."
            )

        class_name = self.target_class_name
        lines = [f"$x isa {typedb_name(class_name)};"]
        sort_label = self._sort_label(class_name)
        if sort_label:
            lines.append(f"$x has {sort_label} $sort_key;")

        for slot_name, constraint in (where or {}).items():
            lines.extend(self._constraint_patterns(slot_name, constraint))
        return "match\n" + "\n".join(f"  {line}" for line in lines)

    def _constraint_patterns(self, slot_name: str, constraint: Any) -> List[str]:
        """Translate one where-clause entry into TypeQL patterns.

        :raises NotImplementedError: for any other operator
        """
        label = self.inserter.attr_names.get(slot_name, typedb_name(slot_name))
        value_type = resolve_typedb_value_type(self.parent.schema_view, self._slot_range(slot_name))

        if not isinstance(constraint, dict):
            return [f"$x has {label} == {typeql_literal(constraint, value_type)};"]

        unsupported = [op for op in constraint if op not in SUPPORTED_OPERATORS]
        if unsupported:
            raise NotImplementedError(
                f"The TypeDB adapter does not support {sorted(unsupported)} for slot "
                f"{slot_name!r}; supported operators are {SUPPORTED_OPERATORS}."
            )

        var = f"$v_{label}"
        patterns = [f"$x has {label} {var};"] if any(op != "$ne" for op in constraint) else []
        for op, value in constraint.items():
            if op == "$ne":
                patterns.append(f"not {{ $x has {label} == {typeql_literal(value, value_type)}; }};")
            elif op == "$in":
                options = " or ".join(f"{{ {var} == {typeql_literal(v, value_type)}; }}" for v in value)
                patterns.append(f"{options};")
            elif op == "$contains":
                # A multivalued slot is repeated `has`, so membership is equality.
                patterns.append(f"{var} == {typeql_literal(value, value_type)};")
            elif op == "$like":
                patterns.append(f"{var} like {typeql_literal(like_to_regex(str(value)), 'string')};")
            else:
                patterns.append(f"{var} {COMPARATORS[op]} {typeql_literal(value, value_type)};")
        return patterns

    def _slot_range(self, slot_name: str) -> Optional[str]:
        """Return the declared range of ``slot_name`` on this collection's class."""
        sv = self.parent.schema_view
        for slot in sv.class_induced_slots(self.target_class_name):
            if slot.name == slot_name:
                return slot.range
        return None

    def _reference_slots(self, class_name: str) -> list:
        """Return the object-valued slots of ``class_name``, which become relations."""
        sv = self.parent.schema_view
        all_classes = set(sv.all_classes().keys())
        return [s for s in sv.class_induced_slots(class_name) if s.range in all_classes]

    def _fetch_block(self, class_name: str, select_cols: Optional[List[str]] = None) -> str:
        """Build the ``fetch`` block: all attributes, plus one subquery per reference slot.

        Referenced objects are returned as their identifiers, matching how they are
        supplied on insert. The *base* relation and role labels are used.
        """
        parts = ['  "_attributes": { $x.* }']
        inserter = self.inserter
        for slot in self._reference_slots(class_name):
            if select_cols and slot.name not in select_cols:
                continue
            relation, owning_role, played_role = inserter._relation_for(class_name, slot.name, slot.range)
            target_id_slot = inserter.identifier_slot(slot.range)
            subject = f"match $r_{slot.name} isa {relation}, links ({owning_role}: $x, {played_role}: $y_{slot.name});"
            if target_id_slot:
                # Referenced by identifier on the way in, so return the identifier.
                id_label = inserter.attr_names.get(target_id_slot, typedb_name(target_id_slot))
                body = f'fetch {{ "id": $y_{slot.name}.{id_label} }};'
            else:
                # No identifier: return attributes inline.
                body = f"fetch {{ $y_{slot.name}.* }};"
            parts.append(f'  "{slot.name}": [ {subject} {body} ]')
        return "fetch {\n" + ",\n".join(parts) + "\n};"

    def query_facets(
        self,
        where: Optional[Dict] = None,
        facet_columns: Optional[List[str]] = None,
        facet_limit: int = DEFAULT_FACET_LIMIT,
        **kwargs,
    ) -> Dict[str, List[Tuple[Any, int]]]:
        """Return value counts for one or more slots, computed in the database.

        Uses TypeQL's ``reduce ... groupby``.

        :param where: optional constraints, in the same form :meth:`query` accepts
        :param facet_columns: slots to facet on; defaults to every scalar slot
        :param facet_limit: maximum distinct values per slot, highest counts first
        :return: slot name to a list of ``(value, count)`` pairs
        :raises NotImplementedError: for a compound (tuple) facet
        """
        if facet_limit is None:
            facet_limit = DEFAULT_FACET_LIMIT
        class_name = self.target_class_name
        sv = self.parent.schema_view
        all_classes = set(sv.all_classes().keys())

        if not facet_columns:
            facet_columns = [s.name for s in sv.class_induced_slots(class_name) if s.range not in all_classes]

        results: Dict[str, List[Tuple[Any, int]]] = {}
        for col in facet_columns:
            if isinstance(col, tuple):
                raise NotImplementedError(f"Compound facets are not supported by the TypeDB adapter yet; got {col!r}.")
            label = self.inserter.attr_names.get(col, typedb_name(col))
            pipeline = [
                self._match_block(where),
                f"  $x has {label} $fv;",
                "reduce $n = count groupby $fv;",
                "sort $n desc;",
                f"limit {facet_limit};",
            ]
            rows = self._read("\n".join(pipeline))
            results[col] = [(row.get("fv"), int(row["n"])) for row in rows if row.get("n") is not None]
        return results

    def _row_to_object(self, row: dict, class_name: str) -> dict:
        """Turn one fetched row into a LinkML-shaped object.

        TypeDB attribute labels are mapped back to LinkML slot names.
        """
        label_to_slot = {v: k for k, v in self.inserter.attr_names.items()}
        obj: dict = {}
        for label, value in (row.get("_attributes") or {}).items():
            obj[label_to_slot.get(label, label)] = value
        ref_slots = {s.name: s for s in self._reference_slots(class_name)}
        for key, value in row.items():
            if key == "_attributes" or not value:
                continue
            slot = ref_slots.get(key)
            if slot is None:
                continue
            if self.inserter.identifier_slot(slot.range):
                entries = [e["id"] for e in value if e.get("id") is not None]
            else:
                # Nested inlined objects: map their labels back to slot names too.
                entries = [{label_to_slot.get(k, k): v for k, v in entry.items()} for entry in value]
            if not entries:
                continue
            obj[key] = entries if slot.multivalued else entries[0]
        return obj

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> Optional[int]:
        """Delete objects matching the given field values.

        Each object contributes one ``match``/``delete`` pair and the whole batch runs in
        a single write transaction. Where an object carries its identifier, that alone
        is matched, since it is the class ``@key``. Otherwise every scalar field is matched.

        :param objs: one object or a list of them
        :return: the number of instances actually deleted
        :raises ValueError: for an object with no identifier and no scalar fields to
            match on, which would otherwise delete the whole collection
        """
        if not isinstance(objs, list):
            objs = [objs]
        class_name = self.target_class_name
        id_slot = self.inserter.identifier_slot(class_name)
        queries = []
        for obj in objs:
            if id_slot and obj.get(id_slot) is not None:
                where: Dict[str, Any] = {id_slot: obj[id_slot]}
            else:
                where = {
                    k: v
                    for k, v in obj.items()
                    if resolve_typedb_value_type(self.parent.schema_view, self._slot_range(k)) is not None
                }
                if not where:
                    raise ValueError(
                        f"Cannot delete from {class_name!r}: the object has no identifier and no "
                        f"scalar fields to match on, so the delete would be unbounded. Object keys: "
                        f"{sorted(obj)}"
                    )
            queries.append(f"{self._match_block(where)}\ndelete $x;")
        return self._run_deletes(queries)

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok: bool = True, **kwargs) -> Optional[int]:
        """Delete every object matching a where clause.

        :param where: ``None``/``{}`` deletes the whole collection; otherwise slot name
            to constraint, accepting the same operators as :meth:`query`
        :param missing_ok: when False, raise if nothing matched
        :return: the number of instances actually deleted
        :raises ValueError: if nothing matched and ``missing_ok`` is False
        """
        logger.info("Deleting from %s where: %s", self.target_class_name, where)
        deleted = self._run_deletes([f"{self._match_block(where)}\ndelete $x;"])
        if not deleted and not missing_ok:
            raise ValueError(f"No instances found for {where}")
        return deleted

    def _run_deletes(self, queries: List[str]) -> int:
        """Run delete queries in one transaction and count the instances removed.

        :param queries: complete ``match``/``delete`` queries
        :return: total instances deleted
        """
        from typedb.driver import TransactionType

        if not queries:
            return 0
        batch = self.parent._active_tx
        if batch is not None:
            return sum(len(list(batch.query(q).resolve())) for q in queries)
        with self.parent.driver.transaction(self.parent._db_name, TransactionType.WRITE) as tx:
            deleted = sum(len(list(tx.query(q).resolve())) for q in queries)
            tx.commit()
        return deleted
