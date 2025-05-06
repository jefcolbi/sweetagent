"""IO (Input Output) plays an essential role in understanding what is going on inside the pipeline.
We define Stdout as Standard Output. It is a class whih is responsible of outputting the message
from the pipeline and nodes.
"""

import logging
from typing import Union
import json
from logging.handlers import RotatingFileHandler

from traceback_with_variables import format_exc


class BaseStaIO:
    """The base class for sweet agent input output classes"""

    def log_info(self, message: str, **kwargs):
        """Output an informational message"""
        raise NotImplementedError()

    def log_debug(self, message: str, **kwargs):
        """Output a debugging message"""
        raise NotImplementedError()

    def log_warning(self, message: str, **kwargs):
        """Output a warning message"""
        raise NotImplementedError()

    def log_error(self, message: str, **kwargs):
        """Output an error message"""
        raise NotImplementedError()

    def log_traceback(self, exception: Exception, **kwargs):
        """Output a traceback. If a Stdout return the message to the customer, it must avoid sending
        the traceback. So this method must simply 'pass'"""
        raise NotImplementedError()

    def user_info_text(self, message: str, **kwargs):
        """Send an information to the user"""
        raise NotImplementedError()

    def user_info_text_with_data(self, message: str, data: Union[dict, list]):
        """Send an information to user and attach json-compatible data"""
        raise NotImplementedError()

    def user_input_text(self, message: str, **kwargs) -> str:
        """Request an input from the user"""
        raise NotImplementedError()

    def user_input_text_with_data(self, message: str, data: Union[dict, list]) -> str:
        """Request an input user and attach some data in json format. Useful to show dropdown or buttons"""
        raise NotImplementedError()

    def admin_info(self, message: str, **kwargs):
        """Send an informational message to the admin"""
        raise NotImplementedError()

    def admin_error(self, message: str, **kwargs):
        """Send an error message to the admin"""
        raise NotImplementedError()


class ConsoleInputMixinStaIO:
    def user_info_text(self, message: str, **kwargs):
        """Send an information to the user"""
        self.log_info(f"User Info: {message}")
        return print(message)

    def user_info_text_with_data(self, message: str, data: Union[dict, list]):
        """Send an information to user and attach some data in json format"""
        self.log_info(f"User Info: {message}\n{json.dumps(data)}")
        return print(f"{message}\n{json.dumps(data)}")

    def user_input_text(self, message: str, **kwargs):
        """Request an input from the user"""
        self.log_info(f"User Input: {message}")
        print(message)
        return input(">> ")

    def user_input_text_with_data(self, message: str, data: Union[dict, list]):
        """Request an input user and attach some data in json format. Useful to show dropdown or buttons"""
        self.log_info(f"User Input: {message}\n{json.dumps(data)}")
        print(f"{message}\n{json.dumps(data)}")
        return input(">> ")


class ExistingLoggerMixinStaIO:
    """A mixin for BaseStaIO to use an existing logger for logging operations"""

    def __init__(self, logger: logging.Logger, *args, **kwargs):
        self.logger = logger

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


class PredefinedLoggerMixinStaIO(ExistingLoggerMixinStaIO):
    """A Stdout which use a predefined logger."""

    def __init__(self, name: str, *args, **kwargs):
        logger = logging.getLogger(name)
        ExistingLoggerMixinStaIO.__init__(self, logger)


class ConsoleLoggerMixinStaIO(ExistingLoggerMixinStaIO):
    def __init__(
        self,
        name: str,
        level=logging.DEBUG,
        log_format: str = "[%(asctime)s] [%(name)s] [%(levelname)s]  %(message)s - %(pathname)s#L%(lineno)s",
        log_date_format: str = "%d/%b/%Y %H:%M:%S",
        *args,
        **kwargs,
    ):
        self.console = logging.getLogger(name)
        self.console.setLevel(level)
        self.hdlr = logging.StreamHandler()
        self.hdlr.setFormatter(
            logging.Formatter(
                fmt=log_format,
                datefmt=log_date_format,
            )
        )
        self.hdlr.setLevel(level)
        self.console.addHandler(self.hdlr)
        ExistingLoggerMixinStaIO.__init__(self, self.console)


class RotatingFileLoggerMixinStaIO(ExistingLoggerMixinStaIO):
    def __init__(
        self,
        name: str,
        filename: str,
        backup_count=2,
        level=logging.DEBUG,
        log_format: str = "[%(asctime)s] [%(name)s] [%(levelname)s]  %(message)s - %(pathname)s#L%(lineno)s",
        log_date_format: str = "%d/%b/%Y %H:%M:%S",
        *args,
        **kwargs,
    ):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        hdlr = RotatingFileHandler(filename=filename, backupCount=backup_count)
        hdlr.setFormatter(
            logging.Formatter(
                fmt=log_format,
                datefmt=log_date_format,
            )
        )
        hdlr.setLevel(level)
        logger.addHandler(hdlr)
        ExistingLoggerMixinStaIO.__init__(self, logger)


class ConsoleStaIO(ConsoleLoggerMixinStaIO, ConsoleInputMixinStaIO, BaseStaIO):
    """A Stdout which write to the console."""

    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance:
            return cls.instance
        else:
            cls.instance = super().__new__(cls)
            return cls.instance

    def __init__(self, name: str, **kwargs):
        ConsoleLoggerMixinStaIO.__init__(self, name, **kwargs)


class ExistingLoggerStaIO(ExistingLoggerMixinStaIO, ConsoleInputMixinStaIO, BaseStaIO):
    def __init__(self, logger: logging.Logger, **kwargs):
        ExistingLoggerMixinStaIO.__init__(self, logger, **kwargs)


class PredefinedLoggerStaIO(
    PredefinedLoggerMixinStaIO, ConsoleInputMixinStaIO, BaseStaIO
):
    def __init__(self, name: str, **kwargs):
        PredefinedLoggerMixinStaIO.__init__(self, name, **kwargs)


class RotatingFileLoggerStaIO(
    RotatingFileLoggerMixinStaIO, ConsoleInputMixinStaIO, BaseStaIO
):
    def __init__(self, name: str, filename: str, **kwargs):
        RotatingFileLoggerMixinStaIO.__init__(self, name, filename, **kwargs)


if __name__ == "__main__":
    stdout = ConsoleStaIO("dummy")
    stdout.user_input_text_with_data("Are you major?", ["Yes", "No"])
