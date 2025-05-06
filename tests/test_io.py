from unittest import TestCase
import os
from pathlib import Path
import logging
import logging.config

cur_dir = Path(os.path.abspath(__file__)).parent
src_path = cur_dir.parent / "src"

from sweetagent.io.redis import RedisWithConsoleStaIO
from sweetagent.io.base import (
    ConsoleStaIO,
    RotatingFileLoggerStaIO,
    PredefinedLoggerStaIO,
    ExistingLoggerStaIO,
)
from uuid import uuid4


class StaIOTestCase(TestCase):
    def test_01_console(self):
        sta_io = ConsoleStaIO("")
        sta_io.log_info("Logging info")
        sta_io.log_warning("Logging warning")
        sta_io.log_debug("Logging debug")
        sta_io.log_info(sta_io.user_input_text("what is your age?"))

    def test_02_rotating(self):
        sta_io = RotatingFileLoggerStaIO("", "test-rotating.log")
        sta_io.log_info("Logging info")
        sta_io.log_warning("Logging warning")
        sta_io.log_debug("Logging debug")
        sta_io.log_info(sta_io.user_input_text("what is your age?"))

    def test_03_existing(self):
        logger = logging.getLogger("")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        sta_io = ExistingLoggerStaIO(logger)
        sta_io.log_info("Logging info")
        sta_io.log_warning("Logging warning")
        sta_io.log_debug("Logging debug")
        sta_io.log_info(sta_io.user_input_text("what is your age?"))

    def test_04_predefined(self):
        LOGGING = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s - %(pathname)s",
                    "datefmt": "%d/%b/%Y %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
            },
            "loggers": {
                "yes": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": True,
                },
            },
        }
        logging.config.dictConfig(LOGGING)
        sta_io = PredefinedLoggerStaIO("yes")
        sta_io.log_info("Logging info")
        sta_io.log_warning("Logging warning")
        sta_io.log_debug("Logging debug")
        sta_io.log_info(sta_io.user_input_text("what is your age?"))

    def test_05_redis(self):
        sta_io = RedisWithConsoleStaIO("default", str(uuid4()), "")
