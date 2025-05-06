from decouple import config
from unittest import TestCase
from pathlib import Path
import os

# from mem0.configs.base import MemoryConfig
# from mem0.embeddings.configs import EmbedderConfig
# from mem0.llms.configs import LlmConfig

cur_dir = Path(os.path.abspath(__file__)).parent
src_path = cur_dir.parent / "src"

from sweetagent.llm_agent import LLMAgent
from sweetagent.llm_client import LLMClient
from sweetagent.io.base import ConsoleStaIO
from sweetagent.short_term_memory.session import SessionMemory
from sweetagent.core import WorkMode
from sweetagent.prompt import SimplePromptEngine

# from sweetagent.long_term_memory.memzero import Mem0LongTermMemory
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


class LLMAgentTestCase(TestCase):
    def test_01_weather_agent(self):
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
        )
        agent.run("What is the current weather in Douala?")

    def test_02_assistant_agent(self):
        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER, LLM_MODEL, LLM_API_KEYS, stdio=stdio, base_url=LLM_BASE_URL
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

    def test_03_weather_agent(self):
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

    def test_04_weather_agent_with_memory(self):
        "Only for python>=3.9"
        stdio = ConsoleStaIO("")
        client = LLMClient(
            LLM_PROVIDER, LLM_MODEL, LLM_API_KEYS, stdio=stdio, base_url=LLM_BASE_URL
        )
        try:
            memory_config = MemoryConfig(
                embedder=EmbedderConfig(
                    provider="azure_openai",
                    config={
                        "azure_kwargs": {
                            "api_key": LLM_API_KEYS[0],
                            "api_version": "2023-05-15",
                            "azure_endpoint": LLM_BASE_URL,
                            "azure_deployment": "text-embedding-3-small",
                        },
                        # 'embedding_dims': 3072,
                        "model": "text-embedding-3-small",
                    },
                ),
                llm=LlmConfig(
                    provider="azure_openai",
                    config={
                        "azure_kwargs": {
                            "api_key": LLM_API_KEYS[0],
                            "api_version": "2025-01-01-preview",
                            "azure_endpoint": LLM_BASE_URL,
                        }
                    },
                ),
            )
            agent = WeatherAgent(
                "Weather Agent",
                "return the weather of cities",
                client,
                short_term_memory=SessionMemory(),
                stdio=stdio,
                prompt_engine=SimplePromptEngine(),
                long_term_memory=Mem0LongTermMemory(memory_config),
            )
            agent.run(
                "What is the current weather in Douala?",
                use_memories=True,
                save_in_memories=True,
            )

            # agent.run("What is the current weather in Douala?", use_memories=True,
            #           save_in_memories=True)
        except Exception as e:
            print(format_exc(e))
