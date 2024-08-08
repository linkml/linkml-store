from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from linkml_store.api.collection import OBJECT


class LLMConfig(BaseModel, extra="forbid"):

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "gpt-4o-mini"
    token_limit: Optional[int] = None
    number_of_few_shot_examples: Optional[int] = None
    role: str = "Domain Expert"


class InferenceConfig(BaseModel, extra="forbid"):
    target_attributes: Optional[List[str]] = None
    feature_attributes: Optional[List[str]] = None
    llm_config: Optional[LLMConfig] = None


class Inference(BaseModel, extra="forbid"):
    predicted_object: OBJECT
    confidence: Optional[float] = None