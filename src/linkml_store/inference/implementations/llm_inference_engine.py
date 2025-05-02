import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, TextIO, Union

import yaml
from llm import get_key
from pydantic import BaseModel

from linkml_store.api.collection import OBJECT
from linkml_store.inference.inference_config import Inference, InferenceConfig, LLMConfig
from linkml_store.inference.inference_engine import InferenceEngine, ModelSerialization
from linkml_store.utils.llm_utils import parse_yaml_payload

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
DEFAULT_NUM_EXAMPLES = 20

SYSTEM_PROMPT = """
Your task is to inference the complete YAML
object output given the YAML object input. I will provide you
with contextual information, including the schema,
to help with the inference. You can use the following

You should return ONLY valid YAML in your response.
"""


class TrainedModel(BaseModel, extra="forbid"):
    index_rows: List[OBJECT]
    config: Optional[InferenceConfig] = None


class LLMInference(Inference):
    iterations: int = 0


@dataclass
class LLMInferenceEngine(InferenceEngine):
    """
    LLM based predictor.

    Unlike the RAG predictor this performs few-shot inference
    """

    _model: "llm.Model" = None  # noqa: F821

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

    def object_to_text(self, object: OBJECT) -> str:
        return yaml.dump(object)

    def _schema_str(self) -> str:
        db = self.training_data.base_collection.parent
        from linkml_runtime.dumpers import json_dumper

        schema_dict = json_dumper.to_dict(db.schema_view.schema)
        return yaml.dump(schema_dict)

    def derive(
        self, object: OBJECT, iteration=0, additional_prompt_texts: Optional[List[str]] = None
    ) -> Optional[LLMInference]:
        import llm

        model: llm.Model = self.model
        # model_name = self.config.llm_config.model_name
        # feature_attributes = self.config.feature_attributes
        target_attributes = self.config.target_attributes
        query_text = self.object_to_text(object)

        if not target_attributes:
            target_attributes = [k for k, v in object.items() if v is None or v == ""]
        # if not feature_attributes:
        #    feature_attributes = [k for k, v in object.items() if v is not None and v != ""]

        system_prompt = SYSTEM_PROMPT.format(llm_config=self.config.llm_config)

        system_prompt += "\n## SCHEMA:\n\n" + self._schema_str()

        stub = ", ".join([f"{k}: ..." for k in target_attributes])
        stub = "{" + stub + "}"
        prompt = (
            "Provide a YAML object of the form"
            "```yaml\n"
            f"{stub}\n"
            "```\n"
            "---\nQuery:\n"
            f"## INCOMPLETE OBJECT:\n{query_text}\n"
            "## OUTPUT:\n"
        )
        logger.info(f"Prompt: {prompt}")
        response = model.prompt(prompt, system=system_prompt)
        yaml_str = response.text()
        logger.info(f"Response: {yaml_str}")
        predicted_object = parse_yaml_payload(yaml_str, strict=True)
        predicted_object = {**object, **predicted_object}
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
        return LLMInference(predicted_object=predicted_object, iterations=iteration + 1, query=object)

    def export_model(
        self, output: Optional[Union[str, Path, TextIO]], model_serialization: ModelSerialization = None, **kwargs
    ):
        self.save_model(output)

    def save_model(self, output: Union[str, Path]) -> None:
        """
        Save the trained model and related data to a file.

        :param output: Path to save the model
        """
        raise NotImplementedError("Does not make sense for this engine")

    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> "LLMInferenceEngine":
        raise NotImplementedError("Does not make sense for this engine")
