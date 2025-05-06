import logging
from typing import Union
import json
import time

from sweetagent.io.base import (
    BaseStaIO,
    PredefinedLoggerMixinStaIO,
    ConsoleLoggerMixinStaIO,
)

try:
    import redis
except Exception:
    raise ImportError(
        "Error while import redis. Did you install redis? pip install redis"
    )


class RedisInputMixinStaIO:
    def __init__(self, run_id: str, config_url: str, **kwargs):
        self.redis_cli = redis.from_url(config_url)
        self.to_user_key = f"{run_id}-user"
        self.to_agent_key = f"{run_id}-agent"

    def user_info_text(self, message: str, **kwargs):
        """Send an information to the user"""
        self.log_info(f"User Info: {message}")
        return self.redis_cli.set(
            self.to_user_key, json.dumps({"type": "info", "message": message})
        )

    def user_info_text_with_data(self, message: str, data: Union[dict, list]):
        """Send an information to user and attach some data in json format"""
        self.log_info(f"User Info: {message}\n{json.dumps(data)}")
        return self.redis_cli.set(
            self.to_user_key,
            json.dumps({"type": "info", "message": message, "data": data}),
        )

    def user_input_text(self, message: str, **kwargs):
        """Request an input from the user"""
        self.log_info(f"User Input: {message}")
        self.redis_cli.set(self.to_user_key, json.dumps({"type": "input"}))
        while True:
            res = self.redis_cli.getdel(self.to_agent_key)
            if res:
                res = json.loads(res)
                break
            time.sleep(1)

        return res["input"]

    def user_input_text_with_data(self, message: str, data: Union[dict, list]):
        """Request an input user and attach some data in json format. Useful to show dropdown or buttons"""
        self.log_info(f"User Input: {message}\n{json.dumps(data)}")
        self.redis_cli.set(
            self.to_user_key, json.dumps({"type": "input", "message": message})
        )
        while True:
            res = self.redis_cli.getdel(self.to_agent_key)
            if res:
                res = json.loads(res)
                break
            time.sleep(1)
        return res["input"]


class RedisWithLoggerStaIO(PredefinedLoggerMixinStaIO, RedisInputMixinStaIO, BaseStaIO):
    def __init__(
        self,
        logger_name: str,
        run_id: str,
        config_url: str,
        level=logging.DEBUG,
        **kwargs,
    ):
        PredefinedLoggerMixinStaIO.__init__(self, logger_name, level=level)
        RedisInputMixinStaIO.__init__(self, run_id, config_url)


class RedisWithConsoleStaIO(ConsoleLoggerMixinStaIO, RedisInputMixinStaIO, BaseStaIO):
    def __init__(
        self,
        logger_name: str,
        run_id: str,
        config_url: str,
        level=logging.DEBUG,
        **kwargs,
    ):
        ConsoleLoggerMixinStaIO.__init__(self, logger_name, **kwargs)
        RedisInputMixinStaIO.__init__(self, run_id, config_url)
