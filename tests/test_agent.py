from decouple import config
from unittest import TestCase
from pathlib import Path
import os

cur_dir = Path(os.path.abspath(__file__)).parent
src_path = cur_dir.parent / "src"

from sweetagent.llm_agent import LLMAgent
from sweetagent.llm_client import LLMClient
from sweetagent.io import ConsoleStaIO
from sweetagent.short_term_memory.session import SessionMemory
from sweetagent.core import WorkMode

LLM_PROVIDER = config("LLM_PROVIDER", default="openai")
LLM_MODEL = config("LLM_MODEL", default="gpt-4o")


class WeatherAgent(LLMAgent):
    def get_weather(self, city: str = None):
        return "cloudy"

    def configure_tools(self):
        self.register_function_as_tool(self.get_weather)


class Assistant(LLMAgent):
    pass


class LLMAgentTestCase(TestCase):
    def test_01_weather_agent(self):
        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER, LLM_MODEL, config("OPENAI_API_KEYS").split(","), stdio=stdio
        )
        agent = WeatherAgent(
            "Weather Agent",
            "return the weather of cities",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
        )
        agent.run("What is the current weather in Douala?")

    def test_02_assistant_agent(self):
        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER, LLM_MODEL, config("OPENAI_API_KEYS").split(","), stdio=stdio
        )
        weather_agent = WeatherAgent(
            "Weather Agent",
            "return the weather of cities",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
        )
        assistant = Assistant(
            "Assistant",
            "chat with user and help him as much as you can",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
            work_mode=WorkMode.CHAT,
        )
        assistant.register_agent_as_tool(weather_agent)

        # assistant.run("Hi")
