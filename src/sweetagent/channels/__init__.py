from typing import Union

from sweetagent.io import BaseStaIO
from channels.generic.websocket import WebsocketConsumer
import logging
from traceback_with_variables import format_exc
import json
from queue import Queue


class ChannelWebsocketStaIO(BaseStaIO):
    def __init__(
        self,
        logger_name: str,
        consumer: WebsocketConsumer,
        queue: Queue,
        level=logging.DEBUG,
        **kwargs,
    ):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.hasHandlers():
            self.logger.setLevel(level)
            self.hdlr = logging.StreamHandler()
            self.hdlr.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s - %(pathname)s",
                    datefmt="%d/%b/%Y %H:%M:%S",
                )
            )
            self.hdlr.setLevel(level)
            self.logger.addHandler(self.hdlr)

        self.consumer: WebsocketConsumer = consumer
        self.queue: Queue = queue

    def log_info(self, message: str, **kwargs):
        return self.logger.info(message, stacklevel=2, **kwargs)

    def log_debug(self, message: str, **kwargs):
        return self.logger.debug(message, stacklevel=2, **kwargs)

    def log_warning(self, message: str, **kwargs):
        return self.logger.warning(message, stacklevel=2, **kwargs)

    def log_error(self, message: str, **kwargs):
        return self.logger.error(message, stacklevel=2, **kwargs)

    def log_traceback(self, exception: Exception, **kwargs):
        return self.logger.error(format_exc(exception), stacklevel=2, **kwargs)

    def user_info_text(self, message: str, **kwargs):
        """Send an information to the user"""
        return self.consumer.send(
            text_data=json.dumps({"type": "message", "message": message})
        )

    def user_info_text_with_data(self, message: str, data: Union[dict, list]):
        """Send an information to user and attach some data in json format"""
        return self.consumer.send(
            text_data=json.dumps(
                {"type": "message_data", "message": message, "data": data}
            )
        )

    def user_input_text(self, message: str, **kwargs):
        """Request an input from the user"""
        self.consumer.send(
            text_data=json.dumps({"type": "message", "message": message})
        )
        rcv_msg = self.queue.get(block=True)
        self.queue.task_done()
        return rcv_msg

    def user_input_text_with_data(self, message: str, data: Union[dict, list]):
        """Request an input user and attach some data in json format. Useful to show dropdown or buttons"""
        self.consumer.send(
            text_data=json.dumps(
                {"type": "message_data", "message": message, "data": data}
            )
        )
        rcv_msg = self.queue.get(block=True)
        self.queue.task_done()
        return rcv_msg
