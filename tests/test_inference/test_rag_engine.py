import logging

import pytest

from linkml_store.api.client import Client
from linkml_store.inference import InferenceConfig, get_inference_engine
from linkml_store.inference.implementations.rag_inference_engine import RAGInferenceEngine
from linkml_store.inference.implementations.rule_based_inference_engine import RuleBasedInferenceEngine
from linkml_store.inference.inference_engine import InferenceEngine, ModelSerialization
from linkml_store.utils.format_utils import Format

from tests import INPUT_DIR, OUTPUT_DIR
from tests.test_inference import check_accuracy2

MODEL_FILE_PATH = OUTPUT_DIR / "model.yaml"
RULE_BASED_MODEL_FILE_PATH = OUTPUT_DIR / "sklean-export.rulebase.yaml"

logger = logging.getLogger(__name__)


def roundtrip(ie: InferenceEngine) -> RAGInferenceEngine:
    ie.save_model(MODEL_FILE_PATH)
    ie2 = ie.load_model(MODEL_FILE_PATH)
    assert isinstance(ie2, RAGInferenceEngine)
    return ie2


def make_rule_based(ie: InferenceEngine) -> RuleBasedInferenceEngine:
    ie.export_model(RULE_BASED_MODEL_FILE_PATH, model_serialization=ModelSerialization.RULE_BASED)
    ie2 = RuleBasedInferenceEngine.load_model(RULE_BASED_MODEL_FILE_PATH)
    return ie2


@pytest.mark.integration
def test_inference_nested():
    client = Client()
    db = client.attach_database("duckdb", alias="test")
    db.import_database(INPUT_DIR / "nested-target.yaml", Format.YAML, collection_name="test")
    assert db.list_collection_names() == ["test"]
    collection = db.get_collection("test")
    features = ["paper.abstract"]
    targets = ["triples.subject", "triples.predicate", "triples.object"]
    config = InferenceConfig(target_attributes=targets, feature_attributes=features)
    ie = get_inference_engine("rag", config=config)
    assert isinstance(ie.config, InferenceConfig)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    assert isinstance(ie, RAGInferenceEngine)
    result = ie.derive({"paper": {"abstract": "a precedes b, and b precedes c"}})
    assert result
    obj = result.predicted_object
    assert obj
    print(obj)
    assert "triples" in obj
    assert isinstance(obj["triples"], list)
    # TODO: fuzzy matches for complex objects - we don't expect precise matches
    check_accuracy2(ie, targets, threshold=0.5, features=features)
