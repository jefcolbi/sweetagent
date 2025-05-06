from decouple import config
from unittest import TestCase
from pathlib import Path
import os

cur_dir = Path(os.path.abspath(__file__)).parent
src_path = cur_dir.parent / "src"

from sweetagent.llm_agent import LLMAgent
from sweetagent.llm_client import LLMClient
from sweetagent.io.base import ConsoleStaIO
from sweetagent.short_term_memory.session import SessionMemory
from sweetagent.core import WorkMode, ToolCall, LLMChatMessage
from sweetagent.prompt import SimplePromptEngine
from sweetagent.middlewares.base import BaseMiddleware
from traceback_with_variables import (
    format_exc,
)

LLM_PROVIDER = config("LLM_PROVIDER", default="azure")
LLM_MODEL = config("LLM_MODEL", default="gpt-4o")
LLM_API_KEYS = config("AZURE_API_KEYS").split(",")
AZURE_DEPLOYMENT_NAME = config("AZURE_DEPLOYMENT_NAME")
LLM_BASE_URL = config("AZURE_BASE_URL", default=None)


class WeatherAgent(LLMAgent):
    def get_weather(self, city: str = None):
        return "cloudy"

    def configure_tools(self):
        self.register_function_as_tool(self.get_weather)


class Assistant(LLMAgent):
    pass


class LogToolOutputMiddleware(BaseMiddleware):
    def after_tool_output(
        self,
        llm_agent: "LLMAgent",
        llm_client: "LLMClient",
        tool_call: ToolCall,
        chat_message: LLMChatMessage,
    ) -> LLMChatMessage:
        llm_agent.sta_stdio.log_debug(
            f"Response of {tool_call} => {chat_message.content}"
        )
        return chat_message

    def after_user_message(
        self,
        llm_agent: "LLMAgent",
        llm_client: "LLMClient",
        chat_message: LLMChatMessage,
    ) -> LLMChatMessage:
        llm_agent.sta_stdio.log_debug(f"User message => {chat_message.content}")
        return chat_message

    def after_agent_message(
        self,
        llm_agent: "LLMAgent",
        llm_client: "LLMClient",
        chat_message: LLMChatMessage,
    ) -> LLMChatMessage:
        llm_agent.sta_stdio.log_debug(f"LLM message => {chat_message.content}")
        return chat_message


class LLMAgentTestCase(TestCase):
    def test_01_tool_output_middleware(self):
        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER, LLM_MODEL, LLM_API_KEYS, stdio=stdio, base_url=LLM_BASE_URL
        )
        agent = WeatherAgent(
            "Weather Agent",
            "return the weather of cities",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
            prompt_engine=SimplePromptEngine(),
            after_tool_output_middlewares=[LogToolOutputMiddleware()],
        )
        agent.run("What is the current weather in Douala?")

    def test_02_user_message_middleware(self):
        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER, LLM_MODEL, LLM_API_KEYS, stdio=stdio, base_url=LLM_BASE_URL
        )
        agent = WeatherAgent(
            "Weather Agent",
            "return the weather of cities",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
            prompt_engine=SimplePromptEngine(),
            after_user_message_middlewares=[LogToolOutputMiddleware()],
            work_mode=WorkMode.CHAT,
        )
        agent.run("Hi")

    def test_03_user_agent_middleware(self):
        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER, LLM_MODEL, LLM_API_KEYS, stdio=stdio, base_url=LLM_BASE_URL
        )
        agent = WeatherAgent(
            "Weather Agent",
            "return the weather of cities",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
            prompt_engine=SimplePromptEngine(),
            after_agent_message_middlewares=[LogToolOutputMiddleware()],
            work_mode=WorkMode.CHAT,
        )
        agent.run("Hi")
