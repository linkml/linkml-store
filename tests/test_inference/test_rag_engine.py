import logging

import pytest
from linkml_store.inference import InferenceConfig, get_inference_engine
from linkml_store.inference.implementations.rag_inference_engine import RAGInferenceEngine
from linkml_store.inference.implementations.rule_based_inference_engine import RuleBasedInferenceEngine
from linkml_store.inference.inference_engine import InferenceEngine, ModelSerialization
from linkml_store.utils.format_utils import Format

from tests import INPUT_DIR, OUTPUT_DIR
from tests.test_api.test_api import SCHEMES, create_client
from tests.test_inference import check_accuracy2

MODEL_FILE_PATH = OUTPUT_DIR / "model.llm.yaml"
RULE_BASED_MODEL_FILE_PATH = OUTPUT_DIR / "sklean-export.rulebase.yaml"

logger = logging.getLogger(__name__)


def roundtrip(ie: InferenceEngine) -> RAGInferenceEngine:
    """
    Save and reload an inference engine.

    :param ie:
    :return:
    """
    ie.save_model(MODEL_FILE_PATH)
    ie2 = ie.load_model(MODEL_FILE_PATH)
    assert isinstance(ie2, RAGInferenceEngine)
    return ie2


def make_rule_based(ie: InferenceEngine) -> RuleBasedInferenceEngine:
    ie.export_model(RULE_BASED_MODEL_FILE_PATH, model_serialization=ModelSerialization.RULE_BASED)
    ie2 = RuleBasedInferenceEngine.load_model(RULE_BASED_MODEL_FILE_PATH)
    return ie2


@pytest.mark.integration
@pytest.mark.parametrize("handle", SCHEMES)
def test_inference_nested(handle):
    """
    Test inference on a nested collection.

    The dataset here is for a simple relation-extraction type task, where fake paper
    abstracts are coupled with lists of triples.

    .. code-block:: yaml

      paper:
        abstract: eric likes cheese, dave likes football
      triples:
        - subject: eric
          predicate: likes
          object: cheese
        - subject: dave
          predicate: likes
          object: football

    :return:
    """
    client = create_client(handle)
    db = client.get_database()
    # client = Client()
    # db = client.attach_database("duckdb", alias="test")
    db.import_database(INPUT_DIR / "nested-target.yaml", Format.YAML, collection_name="test_rag")
    # assert db.list_collection_names() == ["test_rag"]
    collection = db.get_collection("test_rag")
    # We will "train" using the abstracts;
    # For RAG we don't actually train, but instead use the features for generating
    # in-context examples.
    features = ["paper.abstract"]
    # We will predict the triples
    targets = ["triples.subject", "triples.predicate", "triples.object"]
    config = InferenceConfig(target_attributes=targets, feature_attributes=features)
    ie = get_inference_engine("rag", config=config)
    assert isinstance(ie.config, InferenceConfig)
    # split into test and training;
    # for RAG the "training" set is the set used as the RAG database
    ie.load_and_split_data(collection)
    ie.initialize_model()
    assert isinstance(ie, RAGInferenceEngine)
    result = ie.derive({"paper": {"abstract": "a precedes b, and b precedes c"}})
    assert result
    obj = result.predicted_object
    assert obj
    assert "triples" in obj
    assert isinstance(obj["triples"], list)
    assert any(t for t in obj["triples"] if t["subject"] == "a" and t["object"] == "b")
    # TODO: fuzzy matches for complex objects - we don't expect precise matches
    check_accuracy2(ie, targets, threshold=0.33, features=features)
    ie2 = roundtrip(ie)
    check_accuracy2(ie2, targets, threshold=0.33, features=features, test_data=ie.testing_data.as_dataframe())
    # test leakage (using training data for testing) - we expect this to raise an error
    with pytest.raises(ValueError):
        check_accuracy2(ie2, targets, threshold=0.33, features=features, test_data=ie.training_data.as_dataframe())
    # re-split and load
    ie = get_inference_engine("rag", config=config)
    ie.load_and_split_data(collection)
    ie.initialize_model()
    # check_accuracy2(ie2, targets, threshold=0.33, features=features, test_data=ie.testing_data.as_dataframe())
