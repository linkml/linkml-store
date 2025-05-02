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
