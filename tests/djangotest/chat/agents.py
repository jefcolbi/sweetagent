from sweetagent.llm_agent import LLMAgent


class WeatherAgent(LLMAgent):
    def get_weather(self, city: str = None):
        return "cloudy"

    def configure_tools(self):
        self.register_function_as_tool(self.get_weather)


class Assistant(LLMAgent):
    pass
