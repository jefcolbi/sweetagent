from itertools import cycle
from typing import Optional, List, Union

from litellm.types.utils import ModelResponse, ChatCompletionMessageToolCall
from dataclasses import dataclass
import json
from enum import Enum
from pydantic import BaseModel


@dataclass()
class ToolCall:
    name: str
    type: str
    tool_call_id: Optional[str] = None
    arguments: Optional[dict] = None

    def to_dict(self, provider: Optional[str] = None):
        return {
            "id": self.tool_call_id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments or {}),
            },
        }

    @classmethod
    def from_chat_message_tool_call(cls, msg_tool_call: ChatCompletionMessageToolCall):
        return ToolCall(
            name=msg_tool_call.function.name,
            arguments=json.loads(msg_tool_call.function.arguments),
            tool_call_id=msg_tool_call.id,
            type=msg_tool_call.type,
        )

    @classmethod
    def from_formatted_response_model(cls, response: "FormattedResponseModel"):  # noqa: F821
        return ToolCall(
            name=response.tool_name,
            arguments=response.tool_arguments or {},
            type="function",
        )


@dataclass()
class LLMChatMessage:
    role: Optional[str]
    content: Optional[str]
    name: Optional[str] = None  # name of the tool called
    tool_call_id: Optional[str] = None  # id of the tool called
    tool_calls: Optional[List[ToolCall]] = None
    type: Optional[str] = None  # the type of the tool called
    output: Optional[str] = None  # output of the tool called
    kind: Optional[str] = None  # used to classify assistant message
    response_format: Union[dict, BaseModel, None] = (
        None  # response format for the next llm message
    )
    data: Optional[dict] = None
    middlewares_data: Optional[dict] = None

    def to_dict(self, provider: Optional[str] = None) -> dict:
        res = {}
        if self.role:
            res["role"] = self.role
        if self.content:
            res["content"] = self.content
        if self.name:
            res["name"] = self.name
        if self.tool_call_id:
            res["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            res["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        if self.response_format:
            res["response_format"] = self.response_format
        return res

    @classmethod
    def from_model_response(cls, response: ModelResponse):
        if response.choices[0].message.tool_calls:
            tool_calls = [
                ToolCall.from_chat_message_tool_call(msg_tool_call)
                for msg_tool_call in response.choices[0].message.tool_calls
            ]
        else:
            tool_calls = None
        return LLMChatMessage(
            role=response.choices[0].message.role,
            content=response.choices[0].message.content,
            tool_calls=tool_calls,
        )


class RotatingList:
    def __init__(self, initial: list):
        self._initial: list = initial
        self._current = None
        self._cycle = cycle(self._initial)

    @property
    def current(self):
        if self._current:
            return self._current

        self._current = next(self._cycle)
        return self._current

    def next(self):
        self._current = next(self._cycle)

    @property
    def max_iter(self) -> int:
        return len(self._initial)


class WorkMode(Enum):
    TASK = "task"
    CHAT = "chat"


class RetryToFix(Exception):
    pass


class MiddlewareMessage:
    def __init__(self, content: str, **kwargs):
        self.content: str = content
        self.metadata = {}
