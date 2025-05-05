import logging
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from linkml_store.api.collection import OBJECT
from linkml_store.utils.format_utils import Format, load_objects

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel, extra="forbid"):
    """
    Configuration for the LLM indexer.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "gpt-4o-mini"
    token_limit: Optional[int] = None
    number_of_few_shot_examples: Optional[int] = None
    role: str = "Domain Expert"
    cached_embeddings_database: Optional[str] = None
    cached_embeddings_collection: Optional[str] = None
    text_template: Optional[str] = None
    text_template_syntax: Optional[str] = None


class InferenceConfig(BaseModel, extra="forbid"):
    """
    Configuration for inference engines.
    """

    target_attributes: Optional[List[str]] = None
    feature_attributes: Optional[List[str]] = None
    train_test_split: Optional[Tuple[float, float]] = None
    llm_config: Optional[LLMConfig] = None
    random_seed: Optional[int] = None
    validate_results: Optional[bool] = None

    @classmethod
    def from_file(cls, file_path: str, format: Optional[Format] = None) -> "InferenceConfig":
        """
        Load an inference config from a file.

        :param file_path: Path to the file.
        :param format: Format of the file (YAML is recommended).
        :return: InferenceConfig
        """
        if format and format.is_xsv():
            logger.warning("XSV format is not recommended for inference config files")
        objs = load_objects(file_path, format=format)
        if len(objs) != 1:
            raise ValueError(f"Expected 1 object, got {len(objs)}")
        return cls(**objs[0])


class Inference(BaseModel, extra="forbid"):
    """
    Result of an inference derivation.
    """

    query: Optional[OBJECT] = Field(default=None, description="The query object.")
    predicted_object: OBJECT = Field(..., description="The predicted object.")
    confidence: Optional[float] = Field(default=None, description="The confidence of the prediction.", le=1.0, ge=0.0)
    explanation: Optional[Any] = Field(default=None, description="Explanation of the prediction.")
