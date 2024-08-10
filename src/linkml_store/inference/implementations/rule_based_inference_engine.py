import logging
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

import yaml
from linkml_map.utils.eval_utils import eval_expr
from linkml_runtime import SchemaView
from linkml_runtime.linkml_model.meta import ClassRule, AnonymousClassExpression
from pydantic import BaseModel

from linkml_store.api.collection import OBJECT, Collection
from linkml_store.inference.inference_config import InferenceConfig, Inference, LLMConfig
from linkml_store.inference.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


def expression_matches(ce: AnonymousClassExpression, object: OBJECT) -> bool:
    """
    Check if a class expression matches an object.

    :param ce: The class expression
    :param object: The object
    :return: True if the class expression matches the object
    """
    if ce.any_of:
        if not any(expression_matches(subce, object) for subce in ce.any_of):
            return False
    if ce.all_of:
        if not all(expression_matches(subce, object) for subce in ce.all_of):
            return False
    if ce.none_of:
        if any(expression_matches(subce, object) for subce in ce.none_of):
            return False
    if ce.slot_conditions:
        for slot in ce.slot_conditions.values():
            slot_name = slot.name
            v = object.get(slot_name, None)
            if slot.equals_string is not None:
                if slot.equals_string != str(v):
                    return False
            if slot.equals_integer is not None:
                if slot.equals_integer != v:
                    return False
            if slot.equals_expression is not None:
                eval_v = eval_expr(slot.equals_expression, **object)
                if v != eval_v:
                    return False
    return True



def apply_rule(rule: ClassRule, object: OBJECT):
    """
    Apply a rule to an object.

    Mutates the object

    :param rule: The rule to apply
    :param object: The object to apply the rule to
    """
    for condition in rule.preconditions:
        if expression_matches(condition, object):
            for postcondition in rule.postconditions:
                all_of = [x for x in postcondition.all_of] + [postcondition]
                for pc in all_of:
                    sc = pc.slot_condition
                    if sc:
                        if sc.equals_string:
                            object[sc.name] = sc.equals_string
                        if sc.equals_integer:
                            object[sc.name] = sc.equals_integer
                        if sc.equals_expression:
                            object[sc.name] = eval_expr(sc.equals_expression, **object)
    return object



@dataclass
class RuleBasedInferenceEngine(InferenceEngine):
    """
    TODO

    """
    attribute_rules: Optional[Dict[str, List[Rule]]] = None
    class_rules: Optional[List[ClassRule]] = None
    slot_rules: Optional[Dict[str, List[ClassRule]]] = None
    slot_expressions: Optional[Dict[str, str]] = None


    def initialize_model(self, **kwargs):
        td = self.training_data
        collection: Collection = td.collection
        cd = collection.class_definition()
        sv: SchemaView = collection.parent.schema_view
        class_rules = cd.rules
        if class_rules:
            self.class_rules = class_rules
        for slot in sv.class_induced_slots(cd.name):
            if slot.equals_expression:
                self.slot_expressions[slot.name] = slot.equals_expression

    def derive(self, object: OBJECT) -> Optional[Inference]:
        object = copy(object)
        if self.class_rules:
            for rule in self.class_rules:
                apply_rule(rule, object)
        if self.slot_expressions:
            for slot, expr in self.slot_expressions.items():
                v = eval_expr(expr, **object)
                if v is not None:
                    object[slot] = v
        return Inference(object=object)



































