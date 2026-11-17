import itertools
import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud.bigtable import column_family as bt_column_family
from google.cloud.bigtable import row_filters
from google.cloud.bigtable.row import DirectRow
from google.cloud.bigtable.row_data import PartialRowData
from google.cloud.bigtable.row_set import RowSet
from google.cloud.bigtable.table import Table
from linkml_runtime.linkml_model import SlotDefinition
from pydantic import BaseModel

from linkml_store.api import Collection
from linkml_store.api.collection import DEFAULT_FACET_LIMIT, OBJECT
from linkml_store.api.queries import Query, QueryResult
from linkml_store.utils.object_utils import object_path_get
from linkml_store.utils.query_utils import mongo_query_to_match_function, path_values

logger = logging.getLogger(__name__)

# Bigtable allows at most 100,000 mutations per MutateRows request; stay well under it.
MAX_MUTATIONS_PER_BATCH = 50_000
MAX_ROWS_PER_BATCH = 1_000


def _encode_value(value: Any) -> bytes:
    return json.dumps(value, default=str).encode("utf-8")


def _decode_value(value: bytes) -> Any:
    """
    Decode a cell value.

    Values written by this adapter are JSON. Cells written by other tools may not be,
    in which case the raw UTF-8 string is returned (or a hex string for non-UTF-8 bytes).

    >>> _decode_value(b'{"a": [1, 2]}')
    {'a': [1, 2]}
    >>> _decode_value(b'plain text')
    'plain text'
    >>> _decode_value(b'\\xff\\x00')
    'ff00'
    """
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return value.hex()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _get_path(obj: Dict[str, Any], path: str) -> Any:
    for part in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def _hashable(value: Any) -> Any:
    """A hashable key for a JSON value, used to count facet values."""
    if isinstance(value, (dict, list)):
        return ("json", json.dumps(value, sort_keys=True, default=str))
    return value


def _facet_values(obj: Dict[str, Any], path: str) -> List[Any]:
    """
    Values of a field used for faceting.

    Paths traverse lists, and multivalued fields contribute one count per element.

    >>> _facet_values({"a": [1, 2], "b": None, "c": "x"}, "a")
    [1, 2]
    >>> _facet_values({"a": [1, 2], "b": None, "c": "x"}, "b")
    []
    >>> _facet_values({"a": [1, 2], "b": None, "c": "x"}, "c")
    ['x']
    >>> _facet_values({"rels": [{"type": "friend"}, {"type": "sibling"}]}, "rels.type")
    ['friend', 'sibling']
    """
    values = []
    for v in path_values(obj, path.split(".")):
        values.extend(v if isinstance(v, list) else [v])
    return [v for v in values if v is not None]


def _row_key_value(row_key: bytes, slot: SlotDefinition) -> Any:
    """
    The identifier value for a row key, converted to the identifier slot's range.

    >>> _row_key_value(b"42", SlotDefinition("id", range="integer"))
    42
    >>> _row_key_value(b"P1", SlotDefinition("id"))
    'P1'
    """
    text = row_key.decode("utf-8", errors="replace")
    converters = {"integer": int, "float": float, "double": float, "decimal": float}
    converter = converters.get(slot.range)
    if converter:
        try:
            return converter(text)
        except ValueError:
            pass
    return text


def _sort_key(value: Any) -> Tuple[int, int, Any]:
    """
    Total ordering over mixed JSON values: numbers, then strings, then everything else; None last.

    >>> sorted([None, "b", 2, "a", 1.5], key=_sort_key)
    [1.5, 2, 'a', 'b', None]
    """
    if value is None:
        return (1, 0, 0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, 0, value)
    if isinstance(value, str):
        return (0, 1, value)
    return (0, 2, json.dumps(value, sort_keys=True, default=str))


def _parse_sort_spec(spec: str) -> Tuple[str, bool]:
    """
    Parse a sort specification into (field, descending).

    >>> _parse_sort_spec("name")
    ('name', False)
    >>> _parse_sort_spec("-age")
    ('age', True)
    >>> _parse_sort_spec("age DESC")
    ('age', True)
    """
    spec = spec.strip()
    if spec.startswith("-"):
        return spec[1:], True
    parts = spec.split()
    if len(parts) == 2 and parts[1].upper() in ("ASC", "DESC"):
        return parts[0], parts[1].upper() == "DESC"
    return spec, False


class BigtableCollection(Collection):
    """
    Adapter for tables in a Google Cloud Bigtable instance.

    Each object is one row. Top-level fields are columns in a single column family,
    with JSON-encoded values. If the collection has an identifier attribute, its value
    is the row key, and where clauses that constrain the identifier (by equality or ``$in``)
    are answered by direct row lookups. All other filtering is done client-side over a scan.

    Cells in other column families (e.g. in tables not created by linkml-store) are
    returned nested under the column family name.

    .. note::

        You should not use or manipulate this class directly.
        Instead, use the general :class:`linkml_store.api.Collection`
    """

    _table_ready: bool = False
    _table_known_to_exist: bool = False
    _discovered: bool = False
    """True if this collection was found as an existing table rather than created by the caller."""

    @property
    def native_table(self) -> Table:
        return self.parent.native_instance.table(self.alias, app_profile_id=self.parent.app_profile_id)

    @property
    def column_family(self) -> str:
        return self.parent.column_family

    # -- table management --

    def _table_exists(self) -> bool:
        if not self._table_known_to_exist:
            self._table_known_to_exist = self.native_table.exists()
        return self._table_known_to_exist

    def _check_if_initialized(self) -> bool:
        return self._table_exists()

    def _ensure_table(self):
        """Create the table and column family if they do not exist."""
        if self._table_ready:
            return
        table = self.native_table
        gc_rule = bt_column_family.MaxVersionsGCRule(1)
        if not table.exists():
            logger.info(f"Creating Bigtable table {self.alias}")
            try:
                table.create(column_families={self.column_family: gc_rule})
            except AlreadyExists:
                pass
        if self.column_family not in table.list_column_families():
            logger.info(f"Creating column family {self.column_family} in {self.alias}")
            try:
                table.column_family(self.column_family, gc_rule=gc_rule).create()
            except AlreadyExists:
                pass
        self._table_ready = True
        self._table_known_to_exist = True

    def _reset_table_state(self):
        self._table_ready = False
        self._table_known_to_exist = False

    # -- encoding --

    def _identifier_slot(self) -> Optional[SlotDefinition]:
        """
        The identifier slot, if one is declared in a schema that has already been set or induced.

        Unlike :meth:`identifier_attribute_name`, this never triggers schema induction,
        which would scan the table and fix the induced schema to the collections that exist now.
        """
        sv = self.parent._schema_view
        if not sv:
            return None
        cls = sv.get_class(self.target_class_name)
        if not cls:
            return None
        for slot in sv.class_induced_slots(cls.name):
            if slot.identifier:
                return slot
        return None

    def _identifier_field(self) -> Optional[str]:
        slot = self._identifier_slot()
        return slot.name if slot else None

    @staticmethod
    def _encode_row_key(value: Any) -> bytes:
        key = str(value).encode("utf-8")
        if not key:
            raise ValueError("Bigtable row keys must not be empty")
        return key

    def _new_row_key(self, obj: Dict[str, Any], pk: Optional[str]) -> bytes:
        if pk:
            if obj.get(pk) is None:
                raise ValueError(f"Identifier {pk} not found in object {obj}")
            return self._encode_row_key(obj[pk])
        return uuid.uuid4().hex.encode("utf-8")

    @staticmethod
    def _as_dict(obj: OBJECT, exclude_unset=False) -> Dict[str, Any]:
        """
        Convert an object to a dictionary.

        :param exclude_unset: for Pydantic objects, keep only fields that were explicitly set
            (including those explicitly set to None), rather than dropping all None values
        """
        if isinstance(obj, BaseModel):
            if exclude_unset:
                return obj.model_dump(exclude_unset=True)
            return obj.model_dump(exclude_none=True)
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Cannot store object of type {type(obj)} in Bigtable: {obj}")

    def _decode_row(self, row: PartialRowData, id_slot: Optional[SlotDefinition] = None) -> Dict[str, Any]:
        """
        Decode a row into an object.

        :param id_slot: the identifier slot; if the row has no cell for it, the row key is used
        """
        obj: Dict[str, Any] = {}
        # process the default family first, so other families cannot overwrite top-level fields
        families = sorted(row.cells.items(), key=lambda fc: fc[0] != self.column_family)
        for family, columns in families:
            if family == self.column_family:
                target = obj
            else:
                target = obj.setdefault(family, {})
                if not isinstance(target, dict):
                    logger.warning(f"Column family {family} clashes with a field of the same name; skipping")
                    continue
            for qualifier, cells in columns.items():
                if cells:
                    target[qualifier.decode("utf-8")] = _decode_value(cells[0].value)
        if id_slot and obj.get(id_slot.name) is None:
            # e.g. tables written by other tools, where the identifier is only in the row key
            obj[id_slot.name] = _row_key_value(row.row_key, id_slot)
        return obj

    def _set_cells(self, row: DirectRow, obj: Dict[str, Any], timestamp: datetime, delete_nulls=False) -> int:
        """Add one SetCell (or DeleteFromColumn) mutation per field; return the number of mutations."""
        n = 0
        for k, v in obj.items():
            qualifier = k.encode("utf-8")
            if v is None:
                if delete_nulls:
                    row.delete_cell(self.column_family, qualifier)
                    n += 1
                continue
            row.set_cell(self.column_family, qualifier, _encode_value(v), timestamp=timestamp)
            n += 1
        return n

    # -- low-level reads and writes --

    def _read_rows(
        self,
        row_keys: Optional[List[bytes]] = None,
        limit: Optional[int] = None,
        keys_only=False,
        start_key: Optional[bytes] = None,
    ) -> Iterator[PartialRowData]:
        """
        Stream rows from the table.

        :param row_keys: if not None, read only these rows (an empty list reads nothing)
        :param limit: maximum number of rows to read
        :param keys_only: strip cell values, for counting or finding row keys
        :param start_key: read from this row key (inclusive); only used when row_keys is None
        """
        if limit == 0:
            return
        row_set = None
        if row_keys is not None:
            if not row_keys:
                return
            row_set = RowSet()
            for key in row_keys:
                row_set.add_row_key(key)
        if keys_only:
            filter_ = row_filters.RowFilterChain(
                filters=[row_filters.CellsRowLimitFilter(1), row_filters.StripValueTransformerFilter(True)]
            )
        else:
            filter_ = row_filters.CellsColumnLimitFilter(1)
        try:
            yield from self.native_table.read_rows(
                start_key=None if row_set else start_key, row_set=row_set, limit=limit, filter_=filter_
            )
        except NotFound:
            # the table was deleted (e.g. by another client) since we checked it exists
            logger.info(f"Bigtable table {self.alias} not found")
            self._reset_table_state()

    def _count_rows(self) -> int:
        return sum(1 for _ in self._read_rows(keys_only=True))

    def _mutate(self, rows: List[Tuple[DirectRow, int]]):
        """Apply row mutations in batches; raise if any row fails."""
        batch: List[DirectRow] = []
        batch_mutations = 0

        def flush():
            if not batch:
                return
            statuses = self.native_table.mutate_rows(batch)
            errors = [(r.row_key, s.message) for r, s in zip(batch, statuses) if s.code != 0]
            if errors:
                raise RuntimeError(f"Failed to write {len(errors)} rows to Bigtable table {self.alias}: {errors[:5]}")
            batch.clear()

        for row, n_mutations in rows:
            if batch and (len(batch) >= MAX_ROWS_PER_BATCH or batch_mutations + n_mutations > MAX_MUTATIONS_PER_BATCH):
                flush()
                batch_mutations = 0
            batch.append(row)
            batch_mutations += n_mutations
        flush()

    def _delete_row_keys(self, row_keys: List[bytes]):
        rows = []
        for key in row_keys:
            row = self.native_table.direct_row(key)
            row.delete()
            rows.append((row, 1))
        self._mutate(rows)

    # -- query planning --

    @staticmethod
    def _where_dict(where: Any) -> Dict[str, Any]:
        if where is None:
            return {}
        if not isinstance(where, dict):
            raise NotImplementedError(f"Bigtable adapter only supports dictionary where clauses; got {where!r}")
        return where

    def _row_keys_for_where(self, where: Dict[str, Any], pk: Optional[str]) -> Optional[List[bytes]]:
        """
        Row keys that a where clause restricts the query to, or None if a full scan is needed.

        Only constraints on the identifier field (equality, ``$eq`` or ``$in``) can be pushed down,
        because the identifier is the row key.
        """
        if not pk or pk not in where:
            return None
        cond = where[pk]
        if isinstance(cond, dict):
            if "$eq" in cond:
                values = [cond["$eq"]]
            elif isinstance(cond.get("$in"), list):
                values = cond["$in"]
            else:
                return None
        elif isinstance(cond, list):
            return None
        else:
            values = [cond]
        keys = []
        for v in values:
            if v is None or v == "":
                continue
            key = self._encode_row_key(v)
            if key not in keys:
                keys.append(key)
        return keys

    def _iter_matching(self, where: Dict[str, Any]) -> Iterator[Tuple[bytes, Dict[str, Any]]]:
        """Yield (row_key, object) for every row matching a where clause."""
        id_slot = self._identifier_slot()
        row_keys = self._row_keys_for_where(where, id_slot.name if id_slot else None)
        matcher = mongo_query_to_match_function(where)
        for row in self._read_rows(row_keys):
            obj = self._decode_row(row, id_slot)
            if matcher(obj):
                yield row.row_key, obj

    def _exact_matches(self, criteria: Dict[str, Any]) -> List[Tuple[bytes, Dict[str, Any]]]:
        """(row_key, object) for objects whose fields are exactly equal to every value in criteria."""
        id_slot = self._identifier_slot()
        pk = id_slot.name if id_slot else None
        keys = self._row_keys_for_where(criteria, pk) if pk in criteria else None
        matched = []
        for row in self._read_rows(keys):
            obj = self._decode_row(row, id_slot)
            if all(obj.get(k) == v for k, v in criteria.items()):
                matched.append((row.row_key, obj))
        return matched

    @staticmethod
    def _project(obj: Dict[str, Any], select_cols: Optional[List[str]]) -> Dict[str, Any]:
        if not select_cols:
            return obj
        result = {}
        for col in select_cols:
            if "." in col or "[" in col:
                try:
                    result[col] = object_path_get(obj, col)
                except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                    result[col] = None
            elif col in obj:
                result[col] = obj[col]
        return result

    # -- Collection API --

    def insert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Insert objects.

        If the collection has an identifier attribute, an object with the same identifier
        as an existing row replaces that row.
        """
        if not isinstance(objs, list):
            objs = [objs]
        if not objs:
            return
        objs = [self._as_dict(obj) for obj in objs]
        self._pre_insert_hook(objs, **kwargs)
        self._ensure_table()
        pk = self._identifier_field()
        timestamp = datetime.now(timezone.utc)
        rows = []
        for obj in objs:
            row = self.native_table.direct_row(self._new_row_key(obj, pk))
            n = 0
            if pk:
                # clear any existing row with this key so stale fields do not survive
                row.delete()
                n += 1
            n += self._set_cells(row, obj, timestamp)
            if n:
                rows.append((row, n))
            else:
                logger.warning(f"Skipping object with no non-null fields: {obj}")
        self._mutate(rows)
        self._post_insert_hook(objs, **kwargs)

    def upsert(
        self,
        objs: Union[OBJECT, List[OBJECT]],
        filter_fields: List[str],
        update_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Insert objects, or update the rows that match them on ``filter_fields``.

        Matching is efficient when ``filter_fields`` is the identifier attribute;
        otherwise each object requires a table scan.

        Fields set to None on matched rows are removed. An upsert cannot change an identifier,
        because the identifier is the row key; an object that matches no rows is inserted,
        replacing any existing row with the same identifier.

        :param objs: The object(s) to insert or update.
        :param filter_fields: Field names used to match existing rows.
        :param update_fields: Field names to write on matched rows. If None, all fields are written.
        """
        if not isinstance(objs, list):
            objs = [objs]
        objs = [self._as_dict(obj, exclude_unset=True) for obj in objs]
        self._ensure_table()
        pk = self._identifier_field()
        inserted = []
        n_updated = 0
        for obj in objs:
            criteria = {f: obj[f] for f in filter_fields if f in obj}
            if not criteria:
                raise ValueError("At least one valid filter field must be present in each object.")
            timestamp = datetime.now(timezone.utc)
            existing = self._exact_matches(criteria)
            if existing:
                fields = update_fields if update_fields is not None else list(obj.keys())
                updates = {k: obj[k] for k in fields if k in obj}
                rows = []
                for key, current in existing:
                    if pk and pk in updates and updates[pk] != current.get(pk):
                        raise ValueError(
                            f"Cannot change identifier {pk} from {current.get(pk)!r} to {updates[pk]!r}: "
                            "it is the Bigtable row key"
                        )
                    merged = {k: v for k, v in {**current, **updates}.items() if v is not None}
                    self._pre_insert_hook([merged], **kwargs)
                    row = self.native_table.direct_row(key)
                    n = self._set_cells(row, updates, timestamp, delete_nulls=True)
                    if n:
                        rows.append((row, n))
                self._mutate(rows)
                n_updated += len(rows)
            else:
                self._pre_insert_hook([obj], **kwargs)
                row = self.native_table.direct_row(self._new_row_key(obj, pk))
                n = 0
                if pk:
                    # same as insert: replace any existing row with this identifier
                    row.delete()
                    n += 1
                n += self._set_cells(row, obj, timestamp)
                if n:
                    self._mutate([(row, n)])
                    inserted.append(obj)
        if inserted:
            self._post_insert_hook(inserted, **kwargs)
        elif n_updated:
            self._post_modification_hook(**kwargs)

    def query(self, query: Query, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs) -> QueryResult:
        self._pre_query_hook(query)
        if limit is None:
            limit = query.limit
        if limit is not None and limit < 0:
            limit = None
        if offset is None:
            offset = query.offset
        offset = max(offset or 0, 0)
        where = self._where_dict(query.where_clause)
        if not self._table_exists():
            return QueryResult(query=query, num_rows=0, rows=[], offset=offset)

        if not where and not query.sort_by:
            # No filtering: a keys-only scan gives the count and the first row key of the page,
            # then only the rows on the page are read.
            count = 0
            start_key = None
            for r in self._read_rows(keys_only=True):
                if count == offset:
                    start_key = r.row_key
                count += 1
            rows = []
            if start_key is not None:
                id_slot = self._identifier_slot()
                rows = [self._decode_row(r, id_slot) for r in self._read_rows(start_key=start_key, limit=limit)]
        else:
            matched = (obj for _, obj in self._iter_matching(where))
            if query.sort_by:
                matched_list = list(matched)
                for spec in reversed(query.sort_by):
                    field, descending = _parse_sort_spec(spec)
                    matched_list.sort(key=lambda o: _sort_key(_get_path(o, field)), reverse=descending)
                    # missing values go last in either direction
                    matched_list.sort(key=lambda o: _get_path(o, field) is None)
                count = len(matched_list)
                end = None if limit is None else offset + limit
                rows = matched_list[offset:end]
            else:
                rows = []
                count = 0
                for obj in matched:
                    if count >= offset and (limit is None or len(rows) < limit):
                        rows.append(obj)
                    count += 1
        rows = [self._project(row, query.select_cols) for row in rows]
        return QueryResult(query=query, num_rows=count, rows=rows, offset=offset)

    def find_iter(self, where: Optional[Any] = None, page_size: int = 100, **kwargs) -> Iterator[OBJECT]:
        """
        Iterate over matching objects, streaming rows from Bigtable.

        Unlike the base implementation this makes a single streaming read and never counts rows.
        ``page_size`` is accepted for compatibility; Bigtable streams rows itself.
        """
        if page_size < 1:
            raise ValueError(f"Invalid page size: {page_size}")
        self._pre_query_hook()
        where = self._where_dict(where)
        if not self._table_exists():
            return
        select_cols = kwargs.get("select_cols")
        for _, obj in self._iter_matching(where):
            yield self._project(obj, select_cols)

    def query_facets(
        self,
        where: Optional[Dict] = None,
        facet_columns: Optional[List[Union[str, Tuple[str, ...]]]] = None,
        facet_limit=DEFAULT_FACET_LIMIT,
        **kwargs,
    ) -> Dict[Union[str, Tuple[str, ...]], List[Tuple[Any, int]]]:
        """
        Count distinct values of each facet column, client-side over a single scan.

        Multivalued fields contribute one count per element, and dotted paths traverse lists.
        A tuple of columns counts combinations of values.
        """
        if facet_limit is None:
            facet_limit = DEFAULT_FACET_LIMIT
        if not facet_columns:
            cd = self.class_definition()
            facet_columns = [s.name for s in self.parent.schema_view.class_induced_slots(cd.name)] if cd else []
        counters: Dict[Union[str, Tuple[str, ...]], Counter] = {col: Counter() for col in facet_columns}
        # facet values are counted by a hashable key; the original value is what is returned
        originals: Dict[Any, Any] = {}
        if facet_columns and self._table_exists():
            for _, obj in self._iter_matching(self._where_dict(where)):
                for col in facet_columns:
                    if isinstance(col, tuple):
                        for combo in itertools.product(*[_facet_values(obj, c) for c in col]):
                            key = tuple(_hashable(v) for v in combo)
                            originals.setdefault(key, combo)
                            counters[col][key] += 1
                    else:
                        for v in _facet_values(obj, col):
                            key = _hashable(v)
                            originals.setdefault(key, v)
                            counters[col][key] += 1
        n = facet_limit if facet_limit > 0 else None
        return {col: [(originals[k], c) for k, c in counter.most_common(n)] for col, counter in counters.items()}

    def delete(self, objs: Union[OBJECT, List[OBJECT]], **kwargs) -> int:
        """
        Delete objects.

        Objects with an identifier are deleted by row key; otherwise a row is deleted
        if all of the object's fields are equal to the row's.
        """
        if not isinstance(objs, list):
            objs = [objs]
        objs = [self._as_dict(obj) for obj in objs]
        if not objs or not self._table_exists():
            return 0
        pk = self._identifier_field()
        candidate_keys = []
        by_value = []
        for obj in objs:
            if pk and obj.get(pk) is not None:
                candidate_keys.append(self._encode_row_key(obj[pk]))
            else:
                by_value.append(obj)
        keys = set()
        if candidate_keys:
            # only count rows that actually exist
            candidate_keys = list(dict.fromkeys(candidate_keys))
            keys.update(r.row_key for r in self._read_rows(candidate_keys, keys_only=True))
        if by_value:
            id_slot = self._identifier_slot()
            for row in self._read_rows():
                row_obj = self._decode_row(row, id_slot)
                if any(all(row_obj.get(k) == v for k, v in obj.items()) for obj in by_value):
                    keys.add(row.row_key)
        self._delete_row_keys(sorted(keys))
        self._post_delete_hook()
        return len(keys)

    def delete_where(self, where: Optional[Dict[str, Any]] = None, missing_ok=True, **kwargs) -> int:
        logger.info(f"Deleting from {self.alias} where: {where}")
        where = self._where_dict(where)
        if not self._table_exists():
            if not missing_ok:
                raise ValueError(f"Table {self.alias} does not exist")
            return 0
        if not where:
            n = self._count_rows()
            if n:
                self.native_table.truncate()
        else:
            keys = [key for key, _ in self._iter_matching(where)]
            self._delete_row_keys(keys)
            n = len(keys)
        if n == 0 and not missing_ok:
            raise ValueError(f"No rows found for {where}")
        self._post_delete_hook()
        return n
