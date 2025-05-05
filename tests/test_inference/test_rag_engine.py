import logging

import pytest
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.utils.schema_builder import SchemaBuilder

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
    # don't enforce a strict match for now
    assert any(t for t in obj["triples"] if t["subject"] == "a" and t["object"] == "b")
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
    # TODO: check why roundtrip doesn't clear the cache
    # check_accuracy2(ie2, targets, threshold=0.33, features=features, test_data=ie.testing_data.as_dataframe())


@pytest.mark.integration
@pytest.mark.parametrize("handle", SCHEMES)
def test_with_validation(handle):
    """
    Test RAG inference in validation mode.

    In validation mode, the results of the RAG inference are validated using the LinkML schema.
    If it fails, the error is presented to the LLM on a second iteration.

    We test this using a simple extraction schema, where we have training examples that pair
    texts with extracted relationships/triples of the subject-predicate-object form.

    We will attempt to foil the engin with a deliberately hard to guess enumeration permissible value
    for the predicate ("played_a_leading_role_in").

    this value is not present in the training set, and we do not present the schema ahead of time,
    so we do not expect the LLM to succeed on the first iteration, in which it will make up a predicate.
    This will fail validation, but the validation error includes the actual permissible values, so
    we expect this to succeed the second time.
    """
    client = create_client(handle)
    db = client.get_database()
    db.import_database(INPUT_DIR / "nested-target.yaml", Format.YAML, collection_name="test_rag")
    collection = db.get_collection("test_rag")
    collection.metadata.type = "Extraction"
    features = ["paper.abstract"]
    targets = ["triples.subject", "triples.predicate", "triples.object"]
    config = InferenceConfig(target_attributes=targets, feature_attributes=features)
    ie = get_inference_engine("rag", config=config)
    assert isinstance(ie.config, InferenceConfig)
    ie.config.validate_results = True
    # tr_coll = ie.training_data.collection
    sb = SchemaBuilder()
    sb.add_class("Triple", ["subject", "predicate", "object"])
    sb.add_class("Paper", ["abstract"])
    sb.add_class("Extraction", ["triples", "paper"])
    sb.add_slot("triples", multivalued=True, inlined_as_list=True, range="Triple", replace_if_present=True)
    sb.add_defaults()
    schema = sb.schema
    sv = SchemaView(schema)
    # print(yaml_dumper.dumps(sv.schema))
    collection.parent.set_schema_view(sv)
    assert collection.target_class_name == "Extraction"
    cd = collection.class_definition()
    assert cd.name == "Extraction"
    assert cd.slots
    print(yaml_dumper.dumps(sv.schema))
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
    # don't enforce a strict match for now
    assert any(t for t in obj["triples"] if t["subject"] == "a" and t["object"] == "b")
    check_accuracy2(ie, targets, threshold=0.33, features=features)
    # now we will attempt to foil the engine by restricting the schema
    sb.add_enum("PredicateType", ["likes", "has_part", "is_a", "created", "consumed", "played_a_leading_role_in"])
    sb.add_slot("predicate", range="PredicateType", replace_if_present=True)
    sv = SchemaView(sb.schema)
    collection.parent.set_schema_view(sv)
    errs = list(
        collection.iter_validate_collection([{"triples": [{"subject": "a", "predicate": "unknown", "object": "b"}]}])
    )
    assert len(errs) == 1
    result = ie.derive({"paper": {"abstract": "Mark Hamill played a starring role in the movie Star Wars"}})
    assert result
    obj = result.predicted_object
    assert obj
    print(obj)
    assert any(t for t in obj["triples"] if t["predicate"] == "played_a_leading_role_in")
    # highly unlikely to solve this on the first go, because it requires out of band knowledge
    # (note that in future this unit test could conceivably be used in training models, in which case
    # it will need to be modified to a different hard-to-guess predicate)
    assert result.iterations > 1
