"""
predictor package.
"""

from typing import Type

from linkml_store.predict.predictor import Predictor
from linkml_store.predict.predictor_config import PredictorConfig
from linkml_store.predict.rag_predictor import RAGPredictor
from linkml_store.predict.sklearn_predictor import SklearnPredictor
from linkml_store.utils.object_utils import object_path_update

PREDICTOR_CLASSES = {
    "sklearn": SklearnPredictor,
    "rag": RAGPredictor,
}


def get_predictor_class(name: str) -> Type[Predictor]:
    """
    Get an predictor class by name.

    :param name: the name of the predictor (simple, llm, ...)
    :return: the predictor class
    """
    if name not in PREDICTOR_CLASSES:
        raise ValueError(f"Unknown predictor class: {name}")
    return PREDICTOR_CLASSES[name]


def get_predictor(predictor_type: str, config: PredictorConfig=None, **kwargs) -> Predictor:
    """
    Get a predictor by name.

    >>> sklearn_predictor = get_predictor("sklearn")
    >>> rag_predictor = get_predictor("rag")

    :param name: the name of the predictor (sklearn, rag, ...)
    :param kwargs: additional arguments to pass to the predictor
    :return: the predictor
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if ":" in predictor_type:
        predictor_type, conf_args = predictor_type.split(":", 1)
        if config is None:
            config = PredictorConfig()
        for arg in conf_args.split(","):
            k, v = arg.split("=")
            config = object_path_update(config, k, v)
    cls = get_predictor_class(predictor_type)
    kwargs["predictor_type"] = predictor_type
    predictor_obj = cls(config=config, **kwargs)
    return predictor_obj
