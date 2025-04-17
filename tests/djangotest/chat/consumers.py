import json

from channels.generic.websocket import WebsocketConsumer
from sweetagent.channels import ChannelWebsocketStaIO
from sweetagent.short_term_memory.session import SessionMemory
from sweetagent.core import WorkMode
from sweetagent.llm_agent import LLMAgent
from sweetagent.llm_client import LLMClient
from django.conf import settings
from queue import Queue
from .agents import WeatherAgent, Assistant
from threading import Thread


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

        self.agent: LLMAgent = None
        self.queue: Queue = Queue(maxsize=1)
        self.stdio = ChannelWebsocketStaIO("", self, self.queue)
        self.thread: Thread = None

    def configure_agent(self, initial_message: str) -> LLMAgent:
        """Configure the main agent and run it with the initial message"""
        client = LLMClient(
            settings.LLM_PROVIDER,
            settings.LLM_MODEL,
            settings.OPENAI_API_KEYS,
            stdio=self.stdio,
        )
        weather_agent = WeatherAgent(
            "Weather Agent",
            "return the weather of cities",
            client,
            short_term_memory=SessionMemory(),
            stdio=self.stdio,
        )
        assistant = Assistant(
            "Assistant",
            "chat with user and help him as much as you can",
            client,
            short_term_memory=SessionMemory(),
            stdio=self.stdio,
            work_mode=WorkMode.CHAT,
        )
        assistant.register_agent_as_tool(weather_agent)

        self.agent = assistant
        self.agent.run(initial_message)

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        self.stdio.log_debug(f"{text_data = }")
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]

        if self.agent:
            self.stdio.log_debug("Putting message in the queue")
            self.queue.put(message)
        else:
            self.thread = Thread(target=self.configure_agent, args=[message])
            self.thread.setDaemon(True)
            self.thread.setName("Assistant Agent Thread")
            self.thread.start()
