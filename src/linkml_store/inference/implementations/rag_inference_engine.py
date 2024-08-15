import logging
from dataclasses import dataclass
from typing import Any, Optional

import yaml
from llm import get_key

from linkml_store.api.collection import OBJECT, Collection
from linkml_store.inference.inference_config import Inference, InferenceConfig, LLMConfig
from linkml_store.inference.inference_engine import InferenceEngine
from linkml_store.utils.object_utils import select_nested

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a {llm_config.role}, your task is to inference the YAML
object output given the YAML object input. I will provide you
with a collection of examples that will provide guidance both
on the desired structure of the response, as well as the kind
of content.

You should return ONLY valid YAML in your response.
"""


# def select_object(obj: OBJECT, key_paths: List[str]) -> OBJECT:
# return {k: obj.get(k, None) for k in keys}
# return {k: object_path_get(obj, k, None) for k in key_paths}


@dataclass
class RAGInferenceEngine(InferenceEngine):
    """
    AI Retrieval Augmented Generation (RAG) based predictor.


    >>> from linkml_store.api.client import Client
    >>> from linkml_store.utils.format_utils import Format
    >>> from linkml_store.inference.inference_config import LLMConfig
    >>> client = Client()
    >>> db = client.attach_database("duckdb", alias="test")
    >>> db.import_database("tests/input/countries/countries.jsonl", Format.JSONL, collection_name="countries")
    >>> db.list_collection_names()
    ['countries']
    >>> collection = db.get_collection("countries")
    >>> features = ["name"]
    >>> targets = ["code", "capital", "continent", "languages"]
    >>> llm_config = LLMConfig(model_name="gpt-4o-mini",)
    >>> config = InferenceConfig(target_attributes=targets, feature_attributes=features, llm_config=llm_config)
    >>> ie = RAGInferenceEngine(config=config)
    >>> ie.load_and_split_data(collection)
    >>> ie.initialize_model()
    >>> prediction = ie.derive({"name": "Uruguay"})
    >>> prediction.predicted_object
    {'capital': 'Montevideo', 'code': 'UY', 'continent': 'South America', 'languages': ['Spanish']}

    """

    classifier: Any = None
    encoders: dict = None
    _model: "llm.Model" = None  # noqa: F821

    rag_collection: Collection = None

    def __post_init__(self):
        if not self.config:
            self.config = InferenceConfig()
        if not self.config.llm_config:
            self.config.llm_config = LLMConfig()

    @property
    def model(self) -> "llm.Model":  # noqa: F821
        import llm

        if self._model is None:
            self._model = llm.get_model(self.config.llm_config.model_name)
            if self._model.needs_key:
                key = get_key(None, key_alias=self._model.needs_key)
                self._model.key = key

        return self._model

    def initialize_model(self, **kwargs):
        rag_collection = self.training_data.collection
        rag_collection.attach_indexer("llm", auto_index=False)
        self.rag_collection = rag_collection

    def object_to_text(self, object: OBJECT) -> str:
        return yaml.dump(object)

    def derive(self, object: OBJECT) -> Optional[Inference]:
        import llm
        from tiktoken import encoding_for_model

        from linkml_store.utils.llm_utils import get_token_limit, render_formatted_text

        model: llm.Model = self.model
        model_name = self.config.llm_config.model_name
        feature_attributes = self.config.feature_attributes
        target_attributes = self.config.target_attributes
        num_examples = self.config.llm_config.number_of_few_shot_examples or 5
        query_text = self.object_to_text(object)
        if not self.rag_collection.indexers:
            raise ValueError("RAG collection must have an indexer attached")
        rs = self.rag_collection.search(query_text, limit=num_examples, index_name="llm")
        examples = rs.rows
        if not examples:
            raise ValueError(f"No examples found for {query_text}; size = {self.rag_collection.size()}")
        prompt_clauses = []
        for example in examples:
            # input_obj = {k: example.get(k, None) for k in feature_attributes}
            input_obj = select_nested(example, feature_attributes)
            # output_obj = {k: example.get(k, None) for k in target_attributes}
            output_obj = select_nested(example, target_attributes)
            prompt_clause = (
                "---\nExample:\n"
                f"## INPUT:\n{self.object_to_text(input_obj)}\n"
                f"## OUTPUT:\n{self.object_to_text(output_obj)}\n"
            )
            prompt_clauses.append(prompt_clause)
        # query_obj = {k: object.get(k, None) for k in feature_attributes}
        query_obj = select_nested(object, feature_attributes)
        query_text = self.object_to_text(query_obj)
        prompt_end = "---\nQuery:\n" f"## INPUT:\n{query_text}\n" "## OUTPUT:\n"
        system_prompt = SYSTEM_PROMPT.format(llm_config=self.config.llm_config)

        def make_text(texts):
            return "\n".join(prompt_clauses) + prompt_end

        try:
            encoding = encoding_for_model(model_name)
        except KeyError:
            encoding = encoding_for_model("gpt-4")
        token_limit = get_token_limit(model_name)
        prompt = render_formatted_text(make_text, prompt_clauses, encoding, token_limit)
        logger.info(f"Prompt: {prompt}")
        response = model.prompt(prompt, system_prompt)
        yaml_str = response.text()
        logger.info(f"Response: {yaml_str}")
        try:
            predicted_object = yaml.safe_load(yaml_str)
            return Inference(predicted_object=predicted_object)
        except yaml.parser.ParserError as e:
            logger.error(f"Error parsing response: {yaml_str}\n{e}")
            return None
