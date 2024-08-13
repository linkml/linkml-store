import importlib
import inspect
import os
from typing import Dict, Type

from linkml_store.inference.inference_config import InferenceConfig
from linkml_store.inference.inference_engine import InferenceEngine
from linkml_store.utils.object_utils import object_path_update


class InferenceEngineRegistry:
    def __init__(self):
        self.engines: Dict[str, Type[InferenceEngine]] = {}

    def register(self, name: str, engine_class: Type[InferenceEngine]):
        self.engines[name] = engine_class

    def get_engine_class(self, name: str) -> Type[InferenceEngine]:
        if name not in self.engines:
            raise ValueError(f"Unknown inference engine type: {name}" f"Known engines: {list(self.engines.keys())}")
        return self.engines[name]

    def create_engine(self, engine_type: str, config: InferenceConfig = None, **kwargs) -> InferenceEngine:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if ":" in engine_type:
            engine_type, conf_args = engine_type.split(":", 1)
            if config is None:
                config = InferenceConfig()
            for arg in conf_args.split(","):
                k, v = arg.split("=")
                config = object_path_update(config, k, v)

        engine_class = self.get_engine_class(engine_type)
        kwargs["predictor_type"] = engine_type
        return engine_class(config=config, **kwargs)

    @classmethod
    def load_engines(cls, package_path: str):
        registry = cls()
        package_dir = os.path.dirname(importlib.import_module(package_path).__file__)
        for filename in os.listdir(package_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"{package_path}.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, InferenceEngine) and obj != InferenceEngine:
                            engine_name = name.lower().replace("inferenceengine", "")
                            registry.register(engine_name, obj)
                except ImportError as e:
                    print(f"Error importing {module_name}: {e}")
        return registry


# Initialize the registry
registry = InferenceEngineRegistry.load_engines("linkml_store.inference.implementations")


# Function to get an inference engine (can be used as before)
def get_inference_engine(engine_type: str, config: InferenceConfig = None, **kwargs) -> InferenceEngine:
    """
    Get an inference engine.

    >>> from linkml_store.inference import get_inference_engine
    >>> ie = get_inference_engine('sklearn')
    >>> type(ie)
    <class 'linkml_store.inference.implementations.sklearn_inference_engine.SklearnInferenceEngine'>

    :param engine_type:
    :param config:
    :param kwargs:
    :return:
    """
    return registry.create_engine(engine_type, config, **kwargs)
