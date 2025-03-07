import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, TextIO, Union

import yaml
from llm import get_key
from pydantic import BaseModel

from linkml_store.api.collection import OBJECT, Collection
from linkml_store.inference.inference_config import Inference, InferenceConfig, LLMConfig
from linkml_store.inference.inference_engine import InferenceEngine, ModelSerialization
from linkml_store.utils.object_utils import select_nested

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
DEFAULT_NUM_EXAMPLES = 20
DEFAULT_MMR_RELEVANCE_FACTOR = 0.8

SYSTEM_PROMPT = """
You are a {llm_config.role}, your task is to infer the YAML
object output given the YAML object input. I will provide you
with a collection of examples that will provide guidance both
on the desired structure of the response, as well as the kind
of content.

You should return ONLY valid YAML in your response.
"""


class TrainedModel(BaseModel, extra="forbid"):
    rag_collection_rows: List[OBJECT]
    index_rows: List[OBJECT]
    config: Optional[InferenceConfig] = None


class RAGInference(Inference):
    iterations: int = 0


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

    The "model" can be saved for later use:

    >>> ie.export_model("tests/output/countries.rag_model.json")

    Note in this case the model is not the underlying LLM, but the "RAG Model" which is the vectorized
    representation of training set objects.

    """

    _model: "llm.Model" = None  # noqa: F821

    rag_collection: Collection = None

    PERSIST_COLS: ClassVar[List[str]] = [
        "config",
    ]

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
        logger.info(f"Initializing model {self.model}")
        if self.training_data:
            rag_collection = self.training_data.collection
            rag_collection.attach_indexer("llm", auto_index=False)
            self.rag_collection = rag_collection

    def object_to_text(self, object: OBJECT) -> str:
        return yaml.dump(object)

    def derive(
        self, object: OBJECT, iteration=0, additional_prompt_texts: Optional[List[str]] = None
    ) -> Optional[RAGInference]:
        import llm
        from tiktoken import encoding_for_model

        from linkml_store.utils.llm_utils import get_token_limit, render_formatted_text

        model: llm.Model = self.model
        model_name = self.config.llm_config.model_name
        feature_attributes = self.config.feature_attributes
        target_attributes = self.config.target_attributes
        num_examples = self.config.llm_config.number_of_few_shot_examples or DEFAULT_NUM_EXAMPLES
        query_text = self.object_to_text(object)
        mmr_relevance_factor = DEFAULT_MMR_RELEVANCE_FACTOR
        if not self.rag_collection:
            # TODO: zero-shot mode
            examples = []
        else:
            if not self.rag_collection.indexers:
                raise ValueError("RAG collection must have an indexer attached")
            logger.info(f"Searching {self.rag_collection.alias} for examples for: {query_text}")
            rs = self.rag_collection.search(
                query_text, limit=num_examples, index_name="llm", mmr_relevance_factor=mmr_relevance_factor
            )
            examples = rs.rows
            logger.info(f"Found {len(examples)} examples")
            if not examples:
                raise ValueError(f"No examples found for {query_text}; size = {self.rag_collection.size()}")
        prompt_clauses = []
        this_feature_attributes = feature_attributes
        if not this_feature_attributes:
            this_feature_attributes = list(set(object.keys()) - set(target_attributes))
        query_obj = select_nested(object, this_feature_attributes)
        query_text = self.object_to_text(query_obj)
        for example in examples:
            this_feature_attributes = feature_attributes
            if not this_feature_attributes:
                this_feature_attributes = list(set(example.keys()) - set(target_attributes))
            if not this_feature_attributes:
                raise ValueError(f"No feature attributes found in example {example}")
            input_obj = select_nested(example, this_feature_attributes)
            input_obj_text = self.object_to_text(input_obj)
            if input_obj_text == query_text:
                continue
                # raise ValueError(
                #    f"Query object {query_text} is the same as example object {input_obj_text}\n"
                #    "This indicates possible test data leakage\n."
                #    "TODO: allow an option that allows user to treat this as a basic lookup\n"
                # )
            output_obj = select_nested(example, target_attributes)
            prompt_clause = (
                "---\nExample:\n" f"## INPUT:\n{input_obj_text}\n" f"## OUTPUT:\n{self.object_to_text(output_obj)}\n"
            )
            prompt_clauses.append(prompt_clause)

        system_prompt = SYSTEM_PROMPT.format(llm_config=self.config.llm_config)
        system_prompt += "\n".join(additional_prompt_texts or [])
        prompt_end = "---\nQuery:\n" f"## INPUT:\n{query_text}\n" "## OUTPUT:\n"

        def make_text(texts: List[str]):
            return "\n".join(texts) + prompt_end

        try:
            encoding = encoding_for_model(model_name)
        except KeyError:
            encoding = encoding_for_model("gpt-4")
        token_limit = get_token_limit(model_name)
        prompt = render_formatted_text(
            make_text, values=prompt_clauses, encoding=encoding, token_limit=token_limit, additional_text=system_prompt
        )
        logger.info(f"Prompt: {prompt}")
        response = model.prompt(prompt, system=system_prompt)
        yaml_str = response.text()
        logger.info(f"Response: {yaml_str}")
        predicted_object = self._parse_yaml_payload(yaml_str, strict=True)
        if self.config.validate_results:
            base_collection = self.training_data.base_collection
            errs = list(base_collection.iter_validate_collection([predicted_object]))
            if errs:
                print(f"{iteration} // FAILED TO VALIDATE: {yaml_str}")
                print(f"PARSED: {predicted_object}")
                print(f"ERRORS: {errs}")
                if iteration > MAX_ITERATIONS:
                    raise ValueError(f"Validation errors: {errs}")
                extra_texts = [
                    "Make sure results conform to the schema. Previously you provided:\n",
                    yaml_str,
                    "\nThis was invalid.\n",
                    "Validation errors:\n",
                ] + [self.object_to_text(e) for e in errs]
                return self.derive(object, iteration=iteration + 1, additional_prompt_texts=extra_texts)
        return RAGInference(predicted_object=predicted_object, iterations=iteration + 1, query=object)

    def _parse_yaml_payload(self, yaml_str: str, strict=False) -> Optional[OBJECT]:
        if "```" in yaml_str:
            yaml_str = yaml_str.split("```")[1].strip()
            if yaml_str.startswith("yaml"):
                yaml_str = yaml_str[4:].strip()
        try:
            return yaml.safe_load(yaml_str)
        except Exception as e:
            if strict:
                raise e
            logger.error(f"Error parsing YAML: {yaml_str}\n{e}")
            return None

    def export_model(
        self, output: Optional[Union[str, Path, TextIO]], model_serialization: ModelSerialization = None, **kwargs
    ):
        self.save_model(output)

    def save_model(self, output: Union[str, Path]) -> None:
        """
        Save the trained model and related data to a file.

        :param output: Path to save the model
        """

        # trigger index
        _qr = self.rag_collection.search("*", limit=1)
        assert len(_qr.ranked_rows) > 0

        rows = self.rag_collection.find(limit=-1).rows

        indexers = self.rag_collection.indexers
        assert len(indexers) == 1
        ix = self.rag_collection.indexers["llm"]
        ix_coll = self.rag_collection.parent.get_collection(self.rag_collection.get_index_collection_name(ix))

        ix_rows = ix_coll.find(limit=-1).rows
        assert len(ix_rows) > 0
        tm = TrainedModel(rag_collection_rows=rows, index_rows=ix_rows, config=self.config)
        # tm = TrainedModel(rag_collection_rows=rows, index_rows=ix_rows)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(tm.model_dump(), f)

    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> "RAGInferenceEngine":
        """
        Load a trained model and related data from a file.

        :param file_path: Path to the saved model
        :return: SklearnInferenceEngine instance with loaded model
        """
        with open(file_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
        tm = TrainedModel(**model_data)
        from linkml_store.api import Client

        client = Client()
        db = client.attach_database("duckdb", alias="training")
        db.store({"data": tm.rag_collection_rows})
        collection = db.get_collection("data")
        ix = collection.attach_indexer("llm", auto_index=False)
        assert ix.name
        ix_coll_name = collection.get_index_collection_name(ix)
        assert ix_coll_name
        ix_coll = db.get_collection(ix_coll_name, create_if_not_exists=True)
        ix_coll.insert(tm.index_rows)
        ie = cls(config=tm.config)
        ie.rag_collection = collection
        return ie
