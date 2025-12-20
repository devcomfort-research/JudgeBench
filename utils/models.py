# A collection of APIs to use for generating model responses
# OpenAI/Anthropic models are supported via their respective APIs below
# Other models are routed to LocalAPI, which assumes a VLLM instance running on localhost:8000
# Additional APIs (e.g., Google, Together, etc.) may need to be added
# Only asyncronous apis are supported
# Non-asnyc requests can be handled, but chat() should still be async, so include something like await asyncio.sleep(0)

from abc import ABC, abstractmethod
from typing import List, Dict

import litellm
import backoff


class ChatAPI(ABC):
    """Abstract base class for model chat APIs."""

    @abstractmethod
    def __init__(self, model: str, api_base: str = None, api_key: str = None):
        """Initialize the ChatAPI with a model name and optional API configuration.

        Parameters
        ----------
        model : str
            The name of the model to use.
        api_base : str, optional
            The base URL for the API.
        api_key : str, optional
            The API key for authentication.
        """
        pass

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a list of messages to the model and get a response.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The conversation history as a list of message objects.
        **kwargs
            Additional arguments for the chat completion request.

        Returns
        -------
        str
            The model's response text.
        """
        pass


class LiteLLMAPI(ChatAPI):
    """API implementation using LiteLLM to support various model providers."""

    def __init__(self, model: str, api_base: str = None, api_key: str = None):
        """Initialize the LiteLLM API client.

        Parameters
        ----------
        model : str
            The name of the model to use.
        api_base : str, optional
            The base URL for the API (e.g., for local vLLM or Ollama).
        api_key : str, optional
            The API key for authentication.

        Raises
        ------
        ValueError
            If the model name is empty or None.
        """
        if not model:
            raise ValueError(
                "Model name must be provided and follow LiteLLM rules (e.g., 'gpt-4', 'anthropic/claude-3-opus')."
            )
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        # LiteLLM uses environment variables for configuration (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
        # but also supports passing api_base/api_key directly in completion calls.

    @backoff.on_exception(backoff.fibo, (Exception), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request via LiteLLM.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The list of messages for the conversation.
        **kwargs
            Additional parameters for the LiteLLM completion call.

        Returns
        -------
        str
            The generated response content.
        """
        if self.api_base:
            kwargs.setdefault("api_base", self.api_base)
        if self.api_key:
            kwargs.setdefault("api_key", self.api_key)

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    @backoff.on_exception(backoff.fibo, (Exception), max_tries=5, max_value=30)
    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a raw completion request via LiteLLM.

        Parameters
        ----------
        prompt : str
            The raw prompt to complete.
        **kwargs
            Additional parameters for the completion.

        Returns
        -------
        str
            The generated completion text.
        """
        if self.api_base:
            kwargs.setdefault("api_base", self.api_base)
        if self.api_key:
            kwargs.setdefault("api_key", self.api_key)

        response = await litellm.atext_completion(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )
        return response.choices[0].text


def get_chat_api_from_model(
    model: str, api_base: str = None, api_key: str = None
) -> ChatAPI:
    """Factory function to get the LiteLLMAPI for any given model.

    Parameters
    ----------
    model : str
        The model identifier.
    api_base : str, optional
        The base URL for the API.
    api_key : str, optional
        The API key for authentication.

    Returns
    -------
    ChatAPI
        An instance of LiteLLMAPI.
    """
    return LiteLLMAPI(model, api_base=api_base, api_key=api_key)
