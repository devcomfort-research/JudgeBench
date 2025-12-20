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
    def __init__(self, model: str):
        """Initialize the ChatAPI with a model name.

        Parameters
        ----------
        model : str
            The name of the model to use.
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

    def __init__(self, model: str):
        """Initialize the LiteLLM API client.

        Parameters
        ----------
        model : str
            The name of the model to use.

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
        # LiteLLM uses environment variables for configuration (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

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
        model_name = self.model

        response = await litellm.acompletion(
            model=model_name,
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
        model_name = self.model

        response = await litellm.atext_completion(
            model=model_name,
            prompt=prompt,
            **kwargs,
        )
        return response.choices[0].text


def get_chat_api_from_model(model: str) -> ChatAPI:
    """Factory function to get the LiteLLMAPI for any given model.

    Parameters
    ----------
    model : str
        The model identifier.

    Returns
    -------
    ChatAPI
        An instance of LiteLLMAPI.
    """
    return LiteLLMAPI(model)
