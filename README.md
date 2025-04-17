# Sweet Agent

This is not another Agent Framework. It is a library that gives you
only the basic you need to build or transform your current codebase in agents.

Why we build it?
Most agent frameworks around there don't integrate all the actors of
a real world case usage. These actors are the developer/admin, the LLMs,
the system/product and the user.

These frameworks let you start an agent which will do a task and return you
a result. But what most people really wants is the ability to chat with
an llm which can understand their need at some point and call the appropriate tool.

MCP came in and fixed this. But the LLM has to support it.

Sweet Agent gives you the following features:

- Support for LLM trained without tools use.
- Transparent tools use for different LLM
- Native support of human in the loop
- Easy class to agent transformation
- Tools as method
- Agent collaboration
- Multiple api keys support
- Short term memory
- Long term memory
- Easy prompt customization
- Integrations: django-channels

Sweet Agent gives you those primitives—nothing more, nothing less—so you can integrate them into **your** architecture.

---

## Why choose Sweet Agent?

| Feature | Sweet Agent | Typical frameworks |
|---------|-------------|---------------------|
| Any LLM (native tools or none) | ✅ | Often limited |
| Human‑in‑the‑loop prompts at every step | ✅ | Optional / hard |
| Class‑to‑agent in one line | ✅ | ❌ |
| Tools = plain Python functions **or** other agents | ✅ | Partial |
| API‑key rotation built in | ✅ | ❌ |
| Plug‑and‑play I/O layers (console, Django Channels, …) | ✅ | Rare |
| Memory backends swappable | ✅ | Often single choice |

---

## Installation

```bash
# From PyPI (soon)
pip install sweetagent

# Or from source
pip install git+https://github.com/your‑org/sweetagent.git
```

Sweet Agent requires **Python ⩾ 3.8**.

---

## Quick start

Below is the smallest useful agent: it exposes one tool (`get_weather`) and answers a user in the console.

```python
from sweetagent.io import ConsoleStaIO
from sweetagent.llm_client import LLMClient
from sweetagent.short_term_memory.session import SessionMemory
from sweetagent.llm_agent import LLMAgent

class WeatherAgent(LLMAgent):
    """Return current weather for a city (demo)."""

    def get_weather(self, city: str = None):
        # <-- Real code would query a weather API
        return "cloudy"

    def configure_tools(self):
        # Turn the method into an LLM‑callable tool
        self.register_function_as_tool(self.get_weather)

if __name__ == "__main__":
    stdio = ConsoleStaIO("weather")
    client = LLMClient(
        provider="openai",
        model="gpt-4o",
        api_keys_rotator=["sk‑...", "sk‑backup‑..."],
        stdio=stdio,
    )

    agent = WeatherAgent(
        name="Weather Agent",
        role="Return the weather of cities",
        llm_client=client,
        short_term_memory=SessionMemory(),
        stdio=stdio,
    )

    answer = agent.run("What is the weather in Douala?")
    print(answer)
```

Run it:

```bash
python examples/weather_agent.py
```

---

## Agent collaboration

Agents can register *other* agents as tools. The `Assistant` below delegates weather questions to the `WeatherAgent` above while chatting freely with the user.

```python
from sweetagent.core import WorkMode
from examples.weather_agent import WeatherAgent  # see previous snippet

class Assistant(LLMAgent):
    """General‑purpose helper that can invoke WeatherAgent."""
    pass

# wire‑up
assistant = Assistant(
    "Assistant",
    "chat with user and help as much as you can",
    client,
    short_term_memory=SessionMemory(),
    stdio=stdio,
    work_mode=WorkMode.CHAT,
)
assistant.register_agent_as_tool(weather_agent)

assistant.run("Hi!")
```

See [`test_agent.py`](./tests/test_agent.py) for a full example. citeturn1file1

---

## Prompt Engine

Sweet Agent ships a strict **response‑format contract** so your UI and downstream code never guess what the LLM meant.

* Sections are marked with `+++ section +++` fences.
* YAML blocks can carry structured data (choices, tables, …).
* A built‑in parser turns the raw reply into a `FormatResponseModel`.

```text
+++ kind +++        # message | tool_call | final_answer
+++ message +++     # free text (optional)
+++ data +++        # YAML (optional)
+++ tool_name +++   # when kind == tool_call
+++ tool_arguments +++
~~~ field ~~~       # arguments in their own fences
value

+++ end +++         # terminator (mandatory)
```

`PromptEngine` handles:

* building the initial **system prompt** with examples and tool JSON
* decoding every LLM response (see `test_prompting.py`). citeturn1file0

You can subclass it or plug your own templating engine—just return a markdown string.

---

## Memory

### Short‑term

Implement `BaseShortTermMemory` to implement a short term memory strategy.

`SessionMemory` keeps the last messages in RAM for the current execution.

### Long‑term (pluggable)



---

## Key rotation & retries

Provide **one or many** API keys. `LLMClient` cycles through them on `RateLimitError`:

```python
client = LLMClient("openai", "gpt-4o", ["k1", "k2", "k3"], stdio=stdio)
```

If *all* keys fail you get the last captured exception.

---

## Human‑in‑the‑loop

Because every model prompt can return `kind = message`, the agent pauses and calls `StaIO.user_input_text()`—letting a human decide before the loop continues.

Implement custom `BaseStaIO` subclasses to:

* stream to a React frontend
* post to Slack
* collect feedback for RLHF

---

## Django Channels integration

For real‑time websockets drop this into your `consumers.py` (excerpt from the demo project):

```python
class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

        self.agent: LLMAgent = None
        self.queue: Queue = Queue(maxsize=1)
        self.stdio = ChannelWebsocketStaIO('', self, self.queue)
        self.thread: Thread = None

    def configure_agent(self, initial_message: str) -> LLMAgent:
        """Configure the main agent and run it with the initial message"""
        client = LLMClient(settings.LLM_PROVIDER, settings.LLM_MODEL, settings.OPENAI_API_KEYS,
                           stdio=self.stdio)
        weather_agent = WeatherAgent("Weather Agent", "return the weather of cities", client,
                                     short_term_memory=SessionMemory(), stdio=self.stdio)
        assistant = Assistant("Assistant", "chat with user and help him as much as you can", client,
                              short_term_memory=SessionMemory(), stdio=self.stdio, work_mode=WorkMode.CHAT)
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
```

The full example lives in [`consumers.py`](./tests/djangotest/chat/consumers.py). citeturn1file2

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Provider identifier used by `litellm` |
| `LLM_MODEL` | `gpt-4o` | Model name (or `anthropic/claude‑...`) |
| `OPENAI_API_KEYS` | – | Comma‑separated list of keys for rotation |

Set them in `.env` and load with [python‑decouple](https://github.com/henriquebastos/python-decouple):

```ini
OPENAI_API_KEYS=sk‑primary,sk‑backup
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

---

## Running the test suite

```bash
python -m unittest tests
```

The tests exercise:

* prompt encoding/decoding (`test_prompting.py`)
* agent collaboration and tool calls (`test_agent.py`)

---

## Roadmap

* [ ] Async I/O & streaming partial tokens
* [ ] Built‑in Redis + SQLite memory backends

---

## License

Sweet Agent is released under the MIT License. See [LICENSE](./LICENSE) for the full text.

