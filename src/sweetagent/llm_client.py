from typing import Union, List, Optional
import json

from litellm.types.utils import ModelResponse

from litellm import completion, RateLimitError
from pydantic import BaseModel
from traceback_with_variables import format_exc

from sweetagent.core import RotatingList, LLMChatMessage
from sweetagent.io.base import BaseStaIO


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_keys_rotator: Union[list, RotatingList],
        stdio: BaseStaIO,
        base_url: str = None,
        completion_kwargs: Optional[dict] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_keys_rotator: RotatingList = (
            api_keys_rotator
            if isinstance(api_keys_rotator, RotatingList)
            else RotatingList(api_keys_rotator)
        )
        self.sta_stdio: BaseStaIO = stdio
        self.base_url: str = base_url
        self.completion_kwargs = completion_kwargs

    def complete(
        self, messages: List[dict], tools: List[dict], **completion_kwargs
    ) -> LLMChatMessage:
        if completion_kwargs:
            completion_kwargs_to_use = completion_kwargs.copy()
        elif self.completion_kwargs:
            completion_kwargs_to_use = self.completion_kwargs.copy()
        else:
            completion_kwargs_to_use = {}

        self.sta_stdio.log_debug(
            f"Using {self.base_url = } and {completion_kwargs_to_use = } Sending {json.dumps(messages, indent=4)}"
        )
        temperature = completion_kwargs_to_use.get("temperature")
        if temperature is None:
            raise ValueError("Temperature is None. Please provide a temperature.")

        last_error = None
        try:
            for i in range(self.api_keys_rotator.max_iter):
                try:
                    resp: ModelResponse = completion(
                        model=f"{self.provider}/{self.model}",
                        api_key=self.api_keys_rotator.current,
                        base_url=self.base_url,
                        temperature=completion_kwargs_to_use.pop("temperature", 0),
                        messages=messages,
                        tools=tools,
                        response_format=completion_kwargs_to_use.pop(
                            "response_format",
                            self.find_user_last_message_format(messages),
                        ),
                        **completion_kwargs_to_use,
                    )
                    break
                except RateLimitError as e:
                    last_error = e
                    self.api_keys_rotator.next()
            else:
                raise last_error

            llm_message = LLMChatMessage.from_model_response(resp)
            self.sta_stdio.log_debug(str(llm_message))

            if llm_message.content:
                parts = llm_message.content.split("</think>", maxsplit=1)
                if len(parts) == 1:
                    content = parts[0]
                else:
                    content = parts[1]

                llm_message.content = content

            return llm_message
        except Exception as e:
            self.sta_stdio.log_error(format_exc(e))

    def find_user_last_message_format(
        self, messages: List[dict]
    ) -> Union[dict, BaseModel, None]:
        for message in reversed(messages):
            if message["role"] == "user":
                return message.get("response_format")
