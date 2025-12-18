# A collection of APIs to use for generating model responses
# OpenAI/Anthropic models are supported via their respective APIs below
# Other models are routed to LocalAPI, which assumes a VLLM instance running on localhost:8000
# Additional APIs (e.g., Google, Together, etc.) may need to be added
# Only asyncronous apis are supported
# Non-asnyc requests can be handled, but chat() should still be async, so include something like await asyncio.sleep(0)

from abc import ABC, abstractmethod
from typing import List, Dict
import os

import backoff
import openai
import anthropic


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


class OpenAIAPI(ChatAPI):
    """API implementation for OpenAI models."""

    def __init__(self, model: str):
        """Initialize the OpenAI API client.

        Parameters
        ----------
        model : str
            The name of the OpenAI model.
        """
        self.model = model
        self.client = openai.AsyncClient(api_key=os.environ.get("OPENAI_API_KEY"))

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request to OpenAI.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The list of messages for the conversation.
        **kwargs
            Additional parameters for the OpenAI API call.

        Returns
        -------
        str
            The generated response content.
        """
        if self.model.startswith("o1"):
            if messages[0]["role"] == "system":
                system_message = messages.pop(0)["content"]
                user_message = messages[0]["content"]
                messages[0] = {
                    "role": "user",
                    "content": f"<|BEGIN_SYSTEM_MESSAGE|>\n{system_message.strip()}\n<|END_SYSTEM_MESSAGE|>\n\n{user_message}",
                }

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        return response.choices[0].message.content


class AnthropicAPI(ChatAPI):
    """API implementation for Anthropic models."""

    def __init__(self, model: str):
        """Initialize the Anthropic API client.

        Parameters
        ----------
        model : str
            The name of the Anthropic model.
        """
        self.model = model
        self.client = anthropic.AsyncClient(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @backoff.on_exception(
        backoff.fibo, anthropic.AnthropicError, max_tries=5, max_value=30
    )
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a message request to Anthropic.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The messages to send to the model.
        **kwargs
            Additional parameters for the Anthropic API call.

        Returns
        -------
        str
            The model's response text.
        """
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = ""

        response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            system=system_message,
            # max_tokens=8192, # 4096 for claude-3-*
            **kwargs,
        )

        return response.content[0].text


class GeminiAPI(OpenAIAPI):
    """API implementation for Google Gemini models via Vertex AI OpenAI interface."""

    def __init__(self, model: str):
        """Initialize the Gemini API client using Google Cloud credentials.

        Parameters
        ----------
        model : str
            The Gemini model identifier.
        """
        import google.auth
        import google.auth.transport.requests

        # Programmatically get an access token, need to setup your google cloud account properly,
        # and get `gcloud auth application-default login` to be run first
        creds, _project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        # Note: the credential lives for 1 hour by default (https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.

        project_id = creds.quota_project_id
        # Pass the Vertex endpoint and authentication to the OpenAI SDK
        self.model = f"google/{model}"
        self.client = openai.AsyncClient(
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/us-central1/endpoints/openapi",
            api_key=creds.token,
        )

    @backoff.on_exception(
        backoff.fibo, (openai.OpenAIError), max_tries=10, max_value=30
    )
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat request to Google Gemini.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The messages for the conversation.
        **kwargs
            Additional parameters for the Gemini API call.

        Returns
        -------
        str
            The model's response content.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        message = response.choices[0].message
        if message is None:
            # This happens for Google Gemini under high concurrency
            raise openai.OpenAIError("No response from Google Gemini")
        return message.content


class TogetherAPI(ChatAPI):
    """API implementation for Together AI models."""

    def __init__(self, model: str):
        """Initialize the Together AI API client.

        Parameters
        ----------
        model : str
            The Together AI model name.
        """
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat completion request to Together AI.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The list of conversation messages.
        **kwargs
            Additional parameters for the API call.

        Returns
        -------
        str
            The generated response text.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content


class LocalAPI(ChatAPI):
    """API implementation for local model serving via vLLM."""

    def __init__(self, model: str):
        """Initialize the local API client pointing to localhost:8000.

        Parameters
        ----------
        model : str
            The name of the local model.
        """
        self.model = model
        self.client = openai.AsyncClient(
            base_url="http://localhost:8000/v1", api_key="EMPTY"
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat request to the local vLLM instance.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            The conversation messages.
        **kwargs
            Additional parameters for the completion call.

        Returns
        -------
        str
            The model response text.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a raw completion request (non-chat) to the local instance.

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
        response = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )
        return response.choices[0].text


def get_chat_api_from_model(model: str) -> ChatAPI:
    """Factory function to get the appropriate ChatAPI for a given model.

    Parameters
    ----------
    model : str
        The model identifier (e.g., 'gpt-4o', 'claude-3-opus', etc.).

    Returns
    -------
    ChatAPI
        An instance of a ChatAPI implementation for the model.
    """
    if model.startswith("gpt") or model.startswith("o1"):
        return OpenAIAPI(model)
    if model.startswith("claude"):
        return AnthropicAPI(model)
    if model.startswith("gemini"):
        return GeminiAPI(model)
    if model == "meta-llama/Meta-Llama-3.1-405B-Instruct":
        return TogetherAPI(model + "-turbo")
    return LocalAPI(model)
