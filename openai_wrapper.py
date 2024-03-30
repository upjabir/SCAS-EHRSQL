import logging
import os
import random
import time
from typing import Any, Dict, List, Tuple, Union

import openai
import tiktoken
from pydantic import BaseModel, ConfigDict, Field, FieldValidationInfo, field_validator
from loguru import logger


#logger = logging.getLogger(f"rq.worker.{__name__}")

class OpenaiModelParams(BaseModel):
    text_model: List[str] = [
        "engine",
        "prompt",
        "temperature",
        "max_tokens",
        "top_p",
        "n",
        "stream",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "logprobs",
        "echo",
        "best_of",
        "request_timeout",
    ]
    chat_model: List[str] = [
        "engine",
        "messages",
        "temperature",
        "max_tokens",
        "top_p",
        "n",
        "stream",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "request_timeout",
    ]
    embed_model: List[str] = ["engine", "input", "request_timeout"]

    @classmethod
    def params(cls, model_name: str):
        if model_name == "text":
            return cls.model_fields["text_model"].default
        elif model_name == "chat":
            return cls.model_fields["chat_model"].default
        elif model_name == "embed":
            return cls.model_fields["embed_model"].default


class LlmChatResponse(BaseModel):
    text: str
    response_time: float
    usage: Dict
    cost: float


class LlmEmbedResponse(BaseModel):
    vectors: List
    response_time: float
    usage: Dict
    cost: float


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (
        openai.error.RateLimitError,
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.ServiceUnavailableError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                logger.info(f"Openai Retrying: {num_retries} with delay of {delay}")
                result = func(*args, **kwargs)
                logger.info("request successful")
                return result
            # Retry on specified errors
            except errors as e:
                logger.info(f"Openai Error: {e}")
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def completion_with_retry(llm, **kwargs):
    @retry_with_exponential_backoff
    def completions_with_backoff(**kwargs):
        return llm.client.create(**kwargs)

    return completions_with_backoff(**kwargs)


# OpenAI utils
def get_usd_cost(model, usage):
    """
    Calculate the USD cost of a given model and usage.
    Args:
        model (str): The name of the model being used.
        usage (dict): A dictionary containing usage information.
    Returns:
        float: The USD cost of the given model and usage.
    """
    prompt_tokens = usage["prompt_tokens"]
    output_tokens = usage["total_tokens"] - prompt_tokens
    prompt_price = {
        "gpt35turbo0613": 0.002,
        "gpt35turbo0613_16k":0.002,
        "gpt35turbo": 0.002,
        "doc_embedding": 0.0004,
    }.get(model)
    output_price = {"text-embedding-ada-002": 0, "gpt35turbo": 0.0015, "gpt35turbo0613": 0.0015,"gpt35turbo0613_16k":0.0015}.get(
        model, prompt_price
    )
    return (prompt_tokens / 1000 * prompt_price) + (output_tokens / 1000 * output_price)


class BaseOpenaiModel(BaseModel):
    engine: str
    temperature: int
    openai.api_type = os.environ.get("OPENAI_API_TYPE")
    openai.api_base = os.environ.get("OPENAI_API_BASE")
    openai.api_version = os.environ.get("OPENAI_API_VERSION")
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    max_retries: int = 3
    llm_config: Dict = Field(default=None, validate_default=True)
    encoder: tiktoken.core.Encoding = Field(default=None, validate_default=True)
    max_tokens: int = Field(default=None, validate_default=True)
    request_timeout: int
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @field_validator("llm_config", mode="before")
    def set_llm_config(cls, val, info: FieldValidationInfo) -> Dict[str, Any]:
        if val is None:
            return {"engine": info.data.get("engine"), "temperature": info.data.get("temperature")}

    @field_validator("encoder", mode="before")
    def set_encoder(cls, val, info: FieldValidationInfo) -> tiktoken.core.Encoding:
        if val is None:
            try:
                enc = tiktoken.encoding_for_model(info.data.get("engine"))
            except KeyError:
                enc = tiktoken.encoding_for_model("text-davinci-003")
            return enc

    @field_validator("max_tokens", mode="before")
    def set_max_token(cls, val, info: FieldValidationInfo) -> int:
        if val is None:
            return cls.get_model_max_tokens(info.data.get("engine"))

    @classmethod
    def get_model_max_tokens(cls, model):
        model_max_tokens = {
            os.environ.get("OPENAI_MODEL"): 4096,
            os.environ.get("EMBEDDING_MODEL"): 8191,
            "gpt35turbo0613_16k":8192
        }
        return model_max_tokens.get(model, 4096)

    def token_count(self, text):
        return len(self.encoder.encode(text))

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0301"):
        """Returns the number of tokens used by a list of messages."""
        if messages:
            if model == "gpt-35-turbo":
                print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
                return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
            elif model == "gpt-4":
                print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
                return self.num_tokens_from_messages(messages, model="gpt-4-0314")
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif model == "gpt-4-0314":
                tokens_per_message = 3
                tokens_per_name = 1
            else:
                raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(self.encoder.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def update_llm_config(self, parameters):
        for key, val in parameters.items():
            if key in self.llm_params:
                self.llm_config[key] = val


class TextModel(BaseOpenaiModel):
    llm_params: list = OpenaiModelParams.params("text")
    client: Any = openai.Completion

    def complete(self, message, **kwargs):
        parameters = dict(
            prompt=message,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            request_timeout=kwargs.pop("request_timeout", self.request_timeout),
            **kwargs,
        )
        self.update_llm_config(parameters)
        start_time = time.time()
        resp = completion_with_retry(self, **self.llm_config)
        resp = LlmChatResponse(
            response_time=time.time() - start_time,
            text=resp["choices"][0]["text"],
            usage=dict(resp["usage"]),
            cost=get_usd_cost(self.engine, resp["usage"]),
        )
        logger.info(f"response: {resp}")
        return resp


class ChatModel(BaseOpenaiModel):
    llm_params: list = OpenaiModelParams.params("chat")
    client: Any = openai.ChatCompletion

    def complete(self, message, **kwargs):
        parameters = dict(
            messages=message,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            request_timeout=kwargs.pop("request_timeout", self.request_timeout),
            **kwargs,
        )
        self.update_llm_config(parameters)
        start_time = time.time()
        
        resp = completion_with_retry(self, **self.llm_config)
        resp = LlmChatResponse(
            response_time=time.time() - start_time,
            text=resp["choices"][0]["message"]["content"],
            usage=dict(resp["usage"]),
            cost=get_usd_cost(self.engine, resp["usage"]),
        )
        logger.info(f"response: {resp}")
        return resp


class EmbeddingModel(BaseOpenaiModel):
    llm_params: list = OpenaiModelParams.params("embed")
    client: Any = openai.Embedding

    def embed(self, text, only_vector=False, **kwargs):
        out = self.embedder([text], **kwargs)
        if only_vector:
            return out.vectors
        return out

    def embedder(self, texts, **kwargs):
        parameters = dict(input=texts, request_timeout=kwargs.pop("request_timeout", self.request_timeout))
        self.update_llm_config(parameters)
        start_time = time.time()
        resp = completion_with_retry(self, **self.llm_config)

        return LlmEmbedResponse(
            response_time=time.time() - start_time,
            vectors=[item["embedding"] for item in resp["data"]][0],
            usage=dict(resp["usage"]),
            cost=get_usd_cost(self.engine, resp["usage"]),
        )

    def embed_documnet(self, documents):
        embeddings = []
        for idx in range(len(documents)):
            text = getattr(documents[idx], "page_content")
            embeddings.append(self.embed(text).vectors)
        # logger.info("Get Embedder Response")
        return embeddings


