"""Turn LinkML instance data into TypeQL insert queries, one ``match``/``insert`` per object.

Needs no server. Labels come from the vendored ``naming`` module so they match ``gen-typedb``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from linkml_runtime.utils.schemaview import SchemaView

from linkml_store.api.stores.typedb.naming import (
    build_class_range_narrowings,
    build_name_maps,
    build_relationship_role_info,
    build_role_names,
    is_relationship_class,
    resolve_typedb_value_type,
    slot_ranges,
    typedb_name,
    typeql_string,
)

logger = logging.getLogger(__name__)

#: How to handle a reference to an identifier that neither the batch nor the database holds.
DANGLING_ERROR = "error"
DANGLING_SKIP = "skip"
DANGLING_STUB = "stub"

_DANGLING_POLICIES = (DANGLING_ERROR, DANGLING_SKIP, DANGLING_STUB)

#: What a plan emits; cyclic data loads in two passes, ``MODE_ENTITIES`` then ``MODE_REFERENCES``.
MODE_FULL = "full"
MODE_ENTITIES = "entities"
MODE_REFERENCES = "references"
_MODES = (MODE_FULL, MODE_ENTITIES, MODE_REFERENCES)

_BARE_LITERAL = re.compile(r"[\w:+\-./ ]+")


@dataclass(frozen=True)
class DanglingReference:
    """A reference that resolves to nothing."""

    slot: str
    identifier: str
    target_class: str
    source_id: Optional[str] = None


class DanglingReferenceError(ValueError):
    """Lists every dangling reference under the ``error`` policy; planning has written nothing."""

    def __init__(self, message: str, violations: list[DanglingReference]):
        super().__init__(message)
        self.violations = violations


class StubNotPossibleError(ValueError):
    """The ``stub`` policy can't stub a class with required slots."""


@dataclass
class InsertQuery:
    """A TypeQL insert query, held as clause lines."""

    source_class: str
    match: list[str] = field(default_factory=list)
    insert: list[str] = field(default_factory=list)
    skipped_references: list[tuple[str, str]] = field(default_factory=list)
    #: identifier -> variable, for stubs already created in this query
    stub_vars: dict[str, str] = field(default_factory=dict, repr=False)
    #: (class, identifier) -> variable, for keyed entities already bound in this query
    keyed_vars: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)

    def to_typeql(self) -> str:
        """Render the query as TypeQL."""
        parts = []
        if self.match:
            parts.append("match\n" + "\n".join(f"  {line}" for line in self.match))
        parts.append("insert\n" + "\n".join(f"  {line}" for line in self.insert))
        return "\n".join(parts) + "\n"


def typeql_literal(
    value: Any,
    value_type: Optional[str],
    losses: Optional[list] = None,
    label: Optional[str] = None,
) -> str:
    """Render ``value`` as a TypeQL literal of ``value_type``, recording dropped UTC offsets in ``losses``.

    >>> typeql_literal("ab\\"c", "string")
    '"ab\\\\"c"'
    >>> typeql_literal(True, "boolean")
    'true'
    >>> typeql_literal(3, "integer")
    '3'
    >>> typeql_literal("2024-01-02T03:04:05", "datetime")
    '2024-01-02T03:04:05'
    """
    if value_type == "boolean":
        if isinstance(value, str):
            return "true" if value.lower() in ("true", "1", "yes") else "false"
        return "true" if value else "false"
    # Coerce numbers so a string can't inject TypeQL.
    if value_type == "integer":
        return str(int(value))
    if value_type == "double":
        return str(float(value))
    if value_type == "decimal":
        return f"{Decimal(str(value))}dec"
    if value_type == "datetime-tz":
        return _bare(value.isoformat() if isinstance(value, datetime) else str(value))
    if value_type == "datetime":
        return _bare(_naive_datetime(value, losses, label))
    if value_type in ("date", "duration"):
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return _bare(str(value))
    return typeql_string(str(value))


def _bare(text: str) -> str:
    """Return ``text`` as an unquoted literal, rejecting anything that could inject TypeQL."""
    if not _BARE_LITERAL.fullmatch(text):
        raise ValueError(f"Not a valid TypeQL temporal literal: {text!r}")
    return text


def _local_name(identifier: str) -> str:
    """Return the local part of a CURIE or URI.

    >>> _local_name("prov:Association")
    'Association'
    >>> _local_name("http://www.w3.org/ns/prov#Association")
    'Association'
    >>> _local_name("https://example.org/mini/Study")
    'Study'
    """
    return identifier.rsplit("#", 1)[-1].rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _naive_datetime(value: Any, losses: Optional[list] = None, label: Optional[str] = None) -> str:
    """Convert ``value`` to naive UTC, since TypeDB ``datetime`` rejects offsets ([DVL15])."""
    parsed = value
    if not isinstance(parsed, datetime):
        text = str(parsed)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text

    if parsed.tzinfo is None:
        return parsed.isoformat()

    offset = parsed.utcoffset()
    parsed_utc = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if offset and losses is not None:
        losses.append((label, str(value), _format_offset(offset)))
    return parsed_utc.isoformat()


def _format_offset(offset: timedelta) -> str:
    """Render a UTC offset as ``+HH:MM`` / ``-HH:MM``."""
    total = int(offset.total_seconds())
    sign = "-" if total < 0 else "+"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


class TypeQLInserter:
    """Plans TypeQL inserts for instance data conforming to a LinkML schema."""

    def __init__(
        self,
        schema_view: SchemaView,
        on_dangling_reference: str = DANGLING_ERROR,
        mode: str = MODE_FULL,
    ):
        if mode not in _MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {_MODES}")
        self.mode = mode
        if on_dangling_reference not in _DANGLING_POLICIES:
            raise ValueError(
                f"Unknown dangling-reference policy {on_dangling_reference!r}; expected one of {_DANGLING_POLICIES}"
            )
        self.sv = schema_view
        self.on_dangling_reference = on_dangling_reference

        self._all_classes = set(schema_view.all_classes().keys())
        self._slot_ranges = slot_ranges(schema_view)
        self.attr_names, self.rel_names = build_name_maps(schema_view, self._slot_ranges)
        self.role_names = build_role_names(schema_view, self.rel_names)
        relationship_classes = {cn for cn in self._all_classes if is_relationship_class(schema_view, cn)}
        self.relationship_classes = relationship_classes
        #: (class, slot) -> role info, for classes emitted as TypeDB relations
        self.role_info = build_relationship_role_info(schema_view, relationship_classes)
        narrowings = build_class_range_narrowings(schema_view, relationship_classes, self.rel_names, self.role_names)
        self._narrowings = {(n.narrowing_class, n.slot_name): n for n in narrowings}
        self._class_by_uri = self._build_class_uri_index()
        #: identifiers in the database before this batch
        self.existing_ids: set[str] = set()
        #: class -> identifiers, to catch ids stored under the wrong class
        self.known_by_class: Optional[dict] = None
        #: identifiers this batch has inserted
        self.created_ids: set[str] = set()
        #: ``(label, original, offset)`` for each discarded non-zero UTC offset
        self.offset_losses: list[tuple[Optional[str], str, str]] = []
        #: ``(slot, identifier)`` for every reference dropped under the ``skip`` policy
        self.skipped_references: list[tuple[str, str]] = []
        #: ``(slot, identifier)`` for every placeholder created under the ``stub`` policy
        self.stubbed_references: list[tuple[str, str]] = []
        #: every unresolvable reference found under the ``error`` policy
        self.violations: list[DanglingReference] = []

    def _build_class_uri_index(self) -> dict[str, str]:
        """Index classes by every spelling a ``type`` field might use (name, CURIE, URI)."""
        index: dict[str, str] = {}
        # Sorted for deterministic ties.
        for class_name in sorted(self._all_classes):
            class_def = self.sv.get_class(class_name)
            index[class_name] = class_name
            index.setdefault(typedb_name(class_name), class_name)
            # Index derived URIs too (Biolink leaves class_uri unset).
            try:
                derived = self.sv.get_uri(class_name)
            except Exception:  # pragma: no cover - defensive, some elements have no URI
                derived = None
            for key in (class_def.class_uri, getattr(class_def, "definition_uri", None), derived):
                if not key:
                    continue
                key = str(key)
                index.setdefault(key, class_name)
                index.setdefault(_local_name(key), class_name)
                expanded = self._expand(key)
                if expanded:
                    index.setdefault(expanded, class_name)
        return index

    def _expand(self, curie: str) -> Optional[str]:
        """Expand a CURIE using the schema's prefixes, or return None."""
        if ":" not in curie or curie.startswith("http"):
            return None
        prefix, local = curie.split(":", 1)
        namespaces = self.sv.namespaces()
        if prefix not in namespaces:
            return None
        return f"{namespaces[prefix]}{local}"

    def resolve_class(self, declared: Any, fallback: str) -> str:
        """Resolve a ``type``-field value to a LinkML class name, else ``fallback``."""
        if not declared or not isinstance(declared, str):
            return fallback
        resolved = self._class_by_uri.get(declared)
        if resolved:
            return resolved
        return self._class_by_uri.get(_local_name(declared), fallback)

    def identifier_slot(self, class_name: str) -> Optional[str]:
        """Return ``class_name``'s identifier slot, if it has one."""
        for slot in self.sv.class_induced_slots(class_name):
            if slot.identifier or slot.key:
                return slot.name
        return None

    def required_slots(self, class_name: str) -> list[str]:
        """Return ``class_name``'s required slots, excluding its identifier."""
        id_slot = self.identifier_slot(class_name)
        return [slot.name for slot in self.sv.class_induced_slots(class_name) if slot.required and slot.name != id_slot]

    def _relation_for(self, owner_class: str, slot_name: str, target_class: str) -> tuple[str, str, str]:
        """Return ``(relation, owning_role, played_role)`` linking ``owner_class`` to ``target_class``."""
        # Subclasses inherit narrowings from the class that introduces them.
        narrowing = next(
            (
                n
                for anc in self.sv.class_ancestors(owner_class)
                if (n := self._narrowings.get((anc, slot_name))) is not None
            ),
            None,
        )
        owning_role, played_role = self.role_names.get(slot_name, (typedb_name(slot_name), typedb_name(target_class)))
        if narrowing is not None:
            return narrowing.sub_relation, narrowing.owning_role, narrowing.narrowed_played_role
        return self.rel_names.get(slot_name, typedb_name(slot_name)), owning_role, played_role

    def plan_object(
        self,
        obj: dict,
        class_name: str,
        counter: "_Counter",
        known_ids: Optional[set[str]] = None,
    ) -> InsertQuery:
        """Plan the insert for one top-level object and everything inlined in it."""
        resolved = self.resolve_class(obj.get("type"), class_name)
        query = InsertQuery(source_class=resolved)
        if resolved in self.relationship_classes:
            # Relations link existing players, so only the references pass creates them.
            if self.mode != MODE_ENTITIES:
                self._emit_relation_instance(obj, resolved, query, counter, known_ids)
            return query
        self._emit_entity(obj, resolved, query, counter, known_ids, is_root=True)
        return query

    def _emit_entity(
        self,
        obj: dict,
        class_name: str,
        query: InsertQuery,
        counter: "_Counter",
        known_ids: Optional[set[str]],
        is_root: bool = False,
    ) -> Optional[str]:
        """Emit the insert clauses for ``obj`` and return its variable.

        The planner creates each keyed nested object once and matches repeats, to avoid ``[CNT9]``.
        """
        id_slot = self.identifier_slot(class_name)

        if self.mode == MODE_REFERENCES and is_root:
            # The entities pass created it; bind it.
            if not id_slot or obj.get(id_slot) is None:
                return None
            var = f"$x{next(counter)}"
            id_label = self.attr_names.get(id_slot, typedb_name(id_slot))
            query.match.append(
                f"{var} isa {typedb_name(class_name)}, "
                f"has {id_label} {self._id_literal(class_name, id_slot, obj[id_slot])};"
            )
            for slot in self.sv.class_induced_slots(class_name):
                if (
                    slot.name in obj
                    and obj[slot.name] is not None
                    and (self._slot_ranges.get(slot.name) in self._all_classes or slot.range in self._all_classes)
                ):
                    self._emit_links(obj, class_name, slot.name, obj[slot.name], var, query, counter, known_ids)
            return var

        identifier = None
        if not is_root:
            if id_slot and obj.get(id_slot) is not None:
                identifier = str(obj[id_slot])

        if identifier is not None:
            key = (class_name, identifier)
            if key in query.keyed_vars:
                return query.keyed_vars[key]
            if identifier in self.existing_ids or identifier in self.created_ids:
                var = f"$k{next(counter)}"
                id_label = self.attr_names.get(id_slot, typedb_name(id_slot))
                query.match.append(
                    f"{var} isa {typedb_name(class_name)}, "
                    f"has {id_label} {self._id_literal(class_name, id_slot, identifier)};"
                )
                query.keyed_vars[key] = var
                return var

        var = f"$e{next(counter)}"
        if identifier is not None:
            query.keyed_vars[(class_name, identifier)] = var
            self.created_ids.add(identifier)
        attr_terms: list[str] = []
        deferred: list[tuple[str, Any]] = []

        for slot in self.sv.class_induced_slots(class_name):
            if slot.name not in obj or obj[slot.name] is None:
                continue
            value = obj[slot.name]
            if self._slot_ranges.get(slot.name) in self._all_classes or slot.range in self._all_classes:
                deferred.append((slot.name, value))
                continue
            value_type = resolve_typedb_value_type(self.sv, slot.range)
            label = self.attr_names.get(slot.name, typedb_name(slot.name))
            values = value if isinstance(value, list) else [value]
            for item in values:
                literal = typeql_literal(item, value_type, self.offset_losses, label)
                attr_terms.append(f"has {label} {literal}")

        terms = ", ".join([f"isa {typedb_name(class_name)}"] + attr_terms)
        query.insert.append(f"{var} {terms};")

        for slot_name, value in deferred:
            self._emit_links(obj, class_name, slot_name, value, var, query, counter, known_ids)
        return var

    def _emit_relation_instance(
        self,
        obj: dict,
        class_name: str,
        query: InsertQuery,
        counter: "_Counter",
        known_ids: Optional[set[str]],
    ) -> Optional[str]:
        """Emit an instance of a relationship class, or return None if it fills no role."""
        role_terms: list[str] = []
        for slot in self.sv.class_induced_slots(class_name):
            if slot.name not in obj or obj[slot.name] is None:
                continue
            if slot.range not in self._all_classes:
                continue
            info = self.role_info.get((class_name, slot.name))
            if info is None:
                continue
            value = obj[slot.name]
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, dict):
                    item_class = self.resolve_class(item.get("type"), slot.range)
                    var = self._emit_entity(item, item_class, query, counter, known_ids)
                else:
                    var = self._emit_reference(
                        str(item),
                        slot.name,
                        slot.range,
                        query,
                        counter,
                        known_ids,
                        _owner_id(obj, class_name, self),
                    )
                if var is not None:
                    role_terms.append(f"{info.role_name}: {var}")

        if not role_terms:
            return None

        attr_terms = []
        for slot in self.sv.class_induced_slots(class_name):
            if slot.name not in obj or obj[slot.name] is None:
                continue
            if slot.range in self._all_classes:
                continue
            value_type = resolve_typedb_value_type(self.sv, slot.range)
            label = self.attr_names.get(slot.name, typedb_name(slot.name))
            value = obj[slot.name]
            for item in value if isinstance(value, list) else [value]:
                literal = typeql_literal(item, value_type, self.offset_losses, label)
                attr_terms.append(f"has {label} {literal}")

        var = f"$r{next(counter)}"
        terms = [f"isa {typedb_name(class_name)}", f"links ({', '.join(role_terms)})", *attr_terms]
        query.insert.append(f"{var} {', '.join(terms)};")
        return var

    def _emit_links(
        self,
        owner: dict,
        owner_class: str,
        slot_name: str,
        value: Any,
        owner_var: str,
        query: InsertQuery,
        counter: "_Counter",
        known_ids: Optional[set[str]],
    ) -> None:
        """Emit relation inserts for one object-valued slot."""
        induced = self.sv.induced_slot(slot_name, owner_class)
        target_class = induced.range
        items = value if isinstance(value, list) else [value]

        for item in items:
            # The entities pass emits inlined children; the references pass emits by-id links.
            if self.mode == MODE_ENTITIES and not isinstance(item, dict):
                continue
            if self.mode == MODE_REFERENCES and isinstance(item, dict):
                continue
            if isinstance(item, dict):
                item_class = self.resolve_class(item.get("type"), target_class)
                target_var = self._emit_entity(item, item_class, query, counter, known_ids)
            else:
                target_var = self._emit_reference(
                    str(item), slot_name, target_class, query, counter, known_ids, _owner_id(owner, owner_class, self)
                )
                if target_var is None:
                    continue
                item_class = target_class

            relation, owning_role, played_role = self._relation_for(owner_class, slot_name, item_class)
            query.insert.append(f"({owning_role}: {owner_var}, {played_role}: {target_var}) isa {relation};")

    def _reference_exists(self, identifier: str, target_class: str, known_ids) -> bool:
        """Return True if ``identifier`` exists as ``target_class`` or a subtype of it."""
        if self.known_by_class is not None:
            allowed = {target_class} | set(self.sv.class_descendants(target_class))
            return any(identifier in self.known_by_class.get(cls, ()) for cls in allowed)
        return known_ids is None or identifier in known_ids

    def _id_literal(self, class_name: str, id_slot: str, value) -> str:
        """Format an identifier using its declared value type (some ids are integers)."""
        slot = self.sv.induced_slot(id_slot, class_name)
        value_type = resolve_typedb_value_type(self.sv, slot.range) or "string"
        return typeql_literal(value, value_type)

    def _emit_reference(
        self,
        identifier: str,
        slot_name: str,
        target_class: str,
        query: InsertQuery,
        counter: "_Counter",
        known_ids: Optional[set[str]],
        source_id: Optional[str] = None,
    ) -> Optional[str]:
        """Bind an existing object by identifier, applying the dangling-reference policy."""
        if known_ids is not None and not self._reference_exists(identifier, target_class, known_ids):
            if self.on_dangling_reference == DANGLING_ERROR:
                # Collect every violation; plan_inserts raises once with them all.
                self.violations.append(
                    DanglingReference(
                        slot=slot_name,
                        identifier=identifier,
                        target_class=target_class,
                        source_id=source_id,
                    )
                )
                return None
            if self.on_dangling_reference == DANGLING_SKIP:
                # Debug only; plan_inserts logs one summary per load.
                logger.debug("Skipping dangling reference %s -> %s", slot_name, identifier)
                query.skipped_references.append((slot_name, identifier))
                self.skipped_references.append((slot_name, identifier))
                return None
            var = self._emit_stub(identifier, target_class, query, counter)
            self.stubbed_references.append((slot_name, identifier))
            # Later queries must match the stub, not recreate it ([CNT9] on the @key).
            known_ids.add(identifier)
            return var

        id_slot = self.identifier_slot(target_class)
        if id_slot is None:
            raise ValueError(
                f"Slot {slot_name!r} references {identifier!r} by identifier, but its range "
                f"{target_class!r} has no identifier slot to match on."
            )
        var = f"$m{next(counter)}"
        id_label = self.attr_names.get(id_slot, typedb_name(id_slot))
        query.match.append(
            f"{var} isa {typedb_name(target_class)}, "
            f"has {id_label} {self._id_literal(target_class, id_slot, identifier)};"
        )
        return var

    def _emit_stub(self, identifier: str, target_class: str, query: InsertQuery, counter: "_Counter") -> str:
        """Create a placeholder entity holding only the identifier."""
        if identifier in query.stub_vars:
            return query.stub_vars[identifier]
        id_slot = self.identifier_slot(target_class)
        if id_slot is None:
            raise ValueError(f"Cannot stub {target_class!r}: it has no identifier slot.")
        required = self.required_slots(target_class)
        if required:
            raise StubNotPossibleError(
                f"Cannot stub {target_class!r} for reference {identifier!r}: the class "
                f"requires {sorted(required)}, and a stub carries only its identifier, so "
                f"the commit would fail with [CNT5]. Use on_dangling_reference="
                f"{DANGLING_SKIP!r}, or insert the referenced objects first."
            )
        var = f"$s{next(counter)}"
        query.stub_vars[identifier] = var
        id_label = self.attr_names.get(id_slot, typedb_name(id_slot))
        query.insert.append(
            f"{var} isa {typedb_name(target_class)}, "
            f"has {id_label} {self._id_literal(target_class, id_slot, identifier)};"
        )
        return var


def _owner_id(obj: dict, class_name: str, inserter: "TypeQLInserter") -> Optional[str]:
    """Return ``obj``'s identifier, for error messages."""
    id_slot = inserter.identifier_slot(class_name)
    value = obj.get(id_slot) if id_slot else None
    return None if value is None else str(value)


def plan_inserts(
    schema_view: SchemaView,
    class_name: str,
    objs: list[dict],
    on_dangling_reference: str = DANGLING_ERROR,
    known_ids: Optional[set[str]] = None,
    mode: str = MODE_FULL,
) -> list[InsertQuery]:
    """Plan the insert queries for a batch of objects of one class.

    :param known_ids: existing identifiers, as a set or a class -> ids mapping; ``None`` skips the check
    :param mode: ``full``, or ``entities`` then ``references`` for cyclic data
    """
    inserter = TypeQLInserter(schema_view, on_dangling_reference=on_dangling_reference, mode=mode)
    if isinstance(known_ids, Mapping):
        inserter.known_by_class = {cls: set(ids) for cls, ids in known_ids.items()}
        known_ids = {i for ids in known_ids.values() for i in ids}
    inserter.existing_ids = set(known_ids or ())

    effective_ids: Optional[set[str]] = None
    if known_ids is not None:
        effective_ids = set(known_ids)
        id_slot = inserter.identifier_slot(class_name)
        if id_slot:
            effective_ids.update(str(o[id_slot]) for o in objs if o.get(id_slot) is not None)

    counter = _Counter()
    queries = [inserter.plan_object(obj, class_name, counter, effective_ids) for obj in objs]
    # Drop match-only queries (references pass, no by-id links).
    queries = [q for q in queries if q.insert]

    def _summarise(records: list[tuple[str, str]]) -> tuple[int, int, str, str]:
        slots = sorted({slot for slot, _ in records})
        ids = sorted({identifier for _, identifier in records})
        return len(records), len(ids), ", ".join(slots), ids[0]

    if inserter.violations:
        slots = sorted({v.slot for v in inserter.violations})
        missing = sorted({v.identifier for v in inserter.violations})
        sources = {v.source_id for v in inserter.violations if v.source_id}
        shown = ", ".join(repr(m) for m in missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise DanglingReferenceError(
            f"{len(inserter.violations)} unresolvable reference(s) to {len(missing)} missing "
            f"object(s), from {len(sources) or len(objs)} of {len(objs)} object(s) being "
            f"inserted. Slots: {', '.join(slots)}. Missing: {shown}{more}. "
            f"Nothing was written. Insert the referenced objects first, or set "
            f"on_dangling_reference to {DANGLING_SKIP!r} to drop these links or "
            f"{DANGLING_STUB!r} to create placeholders.",
            inserter.violations,
        )

    if inserter.skipped_references:
        count, distinct, slots, example = _summarise(inserter.skipped_references)
        logger.warning(
            "Skipped %d dangling reference(s) to %d missing object(s) across %d object(s) "
            "being inserted (slots: %s; e.g. %r). Those links were dropped. Set "
            "on_dangling_reference to %r to fail instead, or insert the targets first.",
            count,
            distinct,
            len(objs),
            slots,
            example,
            DANGLING_ERROR,
        )

    if inserter.stubbed_references:
        count, distinct, slots, example = _summarise(inserter.stubbed_references)
        logger.warning(
            "Created %d placeholder(s) for %d missing object(s) across %d object(s) being "
            "inserted (slots: %s; e.g. %r). Each placeholder carries only its identifier.",
            count,
            distinct,
            len(objs),
            slots,
            example,
        )

    if inserter.offset_losses:
        labels = sorted({label for label, _, _ in inserter.offset_losses if label})
        first_label, first_value, first_offset = inserter.offset_losses[0]
        logger.warning(
            "Discarded a non-zero UTC offset from %d datetime value(s) across %d attribute(s) "
            "(%s); TypeDB 'datetime' attributes cannot store an offset, so values were "
            "converted to UTC. The instant is preserved; the local offset is not. "
            "First: %s=%s (offset %s). Regenerate the schema with a 'datetime-tz' mapping "
            "if the offset must be kept.",
            len(inserter.offset_losses),
            len(labels),
            ", ".join(labels),
            first_label,
            first_value,
            first_offset,
        )
    return queries


class _Counter:
    """Variable-suffix counter shared across one batch."""

    def __init__(self) -> None:
        self._n = -1

    def __iter__(self) -> "_Counter":
        return self

    def __next__(self) -> int:
        self._n += 1
        return self._n
