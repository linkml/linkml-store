import logging
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)

MODEL_TOKEN_MAPPING = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-16k-0613": 16385,
    "gpt-3.5-turbo-instruct": 4096,
    "text-ada-001": 2049,
    "ada": 2049,
    "ada-002": 8192,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
    "claude": 200_000,
    "llama-3": 200_000,
}


def render_formatted_text(
    render_func: Callable,
    values: List[str],
    encoding: "tiktoken.Encoding",
    token_limit: int,
    additional_text: Optional[str] = None,
) -> str:
    """
    Render a formatted text string with a given object, encoding, and token limit.

    >>> from tiktoken import encoding_for_model
    >>> encoding = encoding_for_model("gpt-4o-mini")
    >>> names = ["Alice", "Bob", "DoctorHippopotamusMcHippopotamusFace"]
    >>> f = lambda x: f"Hello, {' '.join(x)}!"
    >>> render_formatted_text(f, names, encoding, 4096)
    'Hello, Alice Bob DoctorHippopotamusMcHippopotamusFace!'
    >>> render_formatted_text(f, names, encoding, 5)
    'Hello, Alice Bob!'

    :param render_func: Rendering function
    :param values: Values to render
    :param encoding: Encoding
    :param token_limit: Token limit
    :param additional_text: Additional text to consider
    :return:
    """
    text = render_func(values)
    if additional_text:
        token_limit -= len(encoding.encode(additional_text))
    text_length = len(encoding.encode(text))
    logger.debug(f"Encoding length: {text_length} (original: {len(text)})")
    if text_length <= token_limit:
        return text
    if not values:
        raise ValueError(f"Cannot fit text into token limit: {text_length} > {token_limit}")
    # remove last element and try again
    return render_formatted_text(render_func, values[0:-1], encoding=encoding, token_limit=token_limit)


def get_token_limit(model_name: str) -> int:
    """
    Estimate the token limit for a model.

    >>> get_token_limit("gpt-4o-mini")
    128000

    also works with nested names:

    >>> get_token_limit("my/claude-opus")
    200000


    :param model_name: Model name
    :return: Estimated token limit
    """
    # sort MODEL_TOKEN_MAPPING by key length to ensure that the longest model names are checked first
    for model, token_limit in sorted(MODEL_TOKEN_MAPPING.items(), key=lambda x: len(x[0]), reverse=True):
        if model in model_name:
            return token_limit
    return 4096


def parse_yaml_payload(yaml_str: str, strict=False) -> Optional[dict]:
    import yaml

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
