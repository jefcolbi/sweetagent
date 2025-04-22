from typing import Union, List, Optional
import json

from litellm.types.utils import ModelResponse

from litellm import completion, RateLimitError
from traceback_with_variables import format_exc

from sweetagent.core import RotatingList, LLMChatMessage
from sweetagent.io import BaseStaIO


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_keys_rotator: Union[list, RotatingList],
        stdio: BaseStaIO,
        base_url: str = None,
        litellm_complete_kwargs: Optional[dict] = None,
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
        self.litellm_completion_kwargs = litellm_complete_kwargs or {}

    def complete(
        self,
        messages: List[dict],
        tools: List[dict],
        force_tool_call: Union[bool, str] = False,
        temperature: int = 0,
    ) -> LLMChatMessage:
        self.sta_stdio.log_debug(
            f"Using {self.base_url = } Sending {json.dumps(messages, indent=4)}"
        )

        last_error = None
        try:
            for i in range(self.api_keys_rotator.max_iter):
                try:
                    resp: ModelResponse = completion(
                        model=f"{self.provider}/{self.model}",
                        api_key=self.api_keys_rotator.current,
                        base_url=self.base_url,
                        temperature=temperature,
                        messages=messages,
                        tools=tools,
                        **self.litellm_completion_kwargs,
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
