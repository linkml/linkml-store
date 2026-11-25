import pytest

from linkml_store.inference import get_inference_engine
from linkml_store.inference.implementations.rule_based_inference_engine import RuleBasedInferenceEngine
from tests import OUTPUT_DIR

MODEL_PATH = OUTPUT_DIR / "model.rulebased.yaml"


def test_inference_basic():
    """
    Test basic inference using a rule-based engine.

    :return:
    """
    ie = get_inference_engine("rulebased")
    assert isinstance(ie, RuleBasedInferenceEngine)
    ie.slot_expressions = {
        "age_in_months": "age * 12",
    }
    inf = ie.derive({"age": 1})
    assert inf.predicted_object["age_in_months"] == 12
    ie.save_model(MODEL_PATH)
    ie2 = RuleBasedInferenceEngine.load_model(MODEL_PATH)
    inf = ie2.derive({"age": 1})
    assert inf.predicted_object["age_in_months"] == 12


@pytest.mark.skip(reason="requires linkml_runtime changes")
def test_inference_nested():
    ie = RuleBasedInferenceEngine()
    ie.slot_expressions = {
        "age_in_months": "person.age * 12",
    }
    inf = ie.derive({"person": {"age": 1}})
    assert inf.predicted_object["age_in_months"] == 12


RULES_SCHEMA = """
id: http://example.org/rules
name: rules
imports:
  - linkml:types
default_range: string
classes:
  Person:
    attributes:
      name: {}
      age:
        range: integer
      age_in_months:
        range: integer
        equals_expression: "{age} * 12"
      category: {}
      vote: {}
    rules:
      - preconditions:
          slot_conditions:
            category:
              equals_string: adult
        postconditions:
          slot_conditions:
            vote:
              equals_string: "yes"
"""


@pytest.fixture
def rules_collection():
    from linkml_runtime import SchemaView

    from linkml_store import Client

    db = Client().attach_database("duckdb", alias="test")
    db.set_schema_view(SchemaView(RULES_SCHEMA))
    collection = db.create_collection("Person", alias="persons")
    collection.insert([{"name": "Alice", "age": 30, "category": "adult"}])
    return collection


@pytest.mark.parametrize(
    "obj,expected",
    [
        ({"name": "Bob", "age": 2, "category": "adult"}, {"age_in_months": 24, "vote": "yes"}),
        ({"name": "Bob", "age": 2, "category": "child"}, {"age_in_months": 24, "vote": None}),
    ],
)
def test_rules_and_expressions_from_schema(rules_collection, obj, expected):
    """Class rules and slot expressions in the schema are applied by the engine."""
    from linkml_store.inference.inference_config import InferenceConfig

    ie = get_inference_engine("rulebased", config=InferenceConfig())
    ie.load_and_split_data(rules_collection, split=(1.0, 0.0))
    ie.initialize_model()
    assert ie.slot_expressions == {"age_in_months": "{age} * 12"}
    predicted = ie.derive(obj).predicted_object
    for k, v in expected.items():
        assert predicted.get(k) == v
