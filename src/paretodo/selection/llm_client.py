"""Client for asking an LLM server for a structured answer."""

from typing import Any, Dict, Callable, Optional, List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

from paretodo.selection.llm_selector import LLMResponse


def ask_server(
    model: str, base_url: str, options: Optional[Dict[str, Any]] = None
) -> Callable[[str], str]:
    """
    Create a function that sends a prompt to an OpenAI-compatible server.

    Parameters:
        model (str): The name of the model to use.
        base_url (str): URL of the server.
        options (Optional[Dict[str, Any]]): extra request options, e.g. temperature.

    Returns:
        Callable[[str], str]: The function that returns the model's text answer.
    """
    client = OpenAI(base_url=base_url, api_key="not-needed")
    schema: ResponseFormatJSONSchema = {
        "type": "json_schema",
        "json_schema": {
            "name": "LLMResponse",
            "schema": LLMResponse.model_json_schema(),
        },
    }

    def ask(prompt: str) -> str:
        """Send a prompt to the server and return the model's answer.

        Parameters:
            prompt (str): The complete prompt to send to the model.

        Returns:
            str: The model's text answer, or an empty string if it gave none.
        """
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=schema,
            **(options or {}),
        )
        return response.choices[0].message.content or ""

    return ask
