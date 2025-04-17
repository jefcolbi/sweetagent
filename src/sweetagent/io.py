"""IO (Input Output) plays an essential role in understanding what is going on inside the pipeline.
We define Stdout as Standard Output. It is a class whih is responsible of outputting the message
from the pipeline and nodes.
"""

import logging
from typing import Union
import json

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


class ConsoleStaIO(BaseStaIO):
    """A Stdout which write to the console."""

    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance:
            return cls.instance
        else:
            cls.instance = super().__new__(cls)
            return cls.instance

    def __init__(self, name: str, level=logging.DEBUG, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(name)
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

    def _console_info(self, message: str, **kwargs):
        self.logger.info(message, stacklevel=3, **kwargs)

    def _console_debug(self, message: str, **kwargs):
        self.logger.debug(message, stacklevel=3, **kwargs)

    def _console_warning(self, message: str, **kwargs):
        self.logger.warning(message, stacklevel=3, **kwargs)

    def _console_error(self, message: str, **kwargs):
        self.logger.error(message, stacklevel=3, **kwargs)

    def _console_traceback(self, exception: Exception, **kwargs):
        self.logger.error(format_exc(exception))

    def log_info(self, message: str, **kwargs):
        return self._console_info(message, **kwargs)

    def log_debug(self, message: str, **kwargs):
        return self._console_debug(message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        return self._console_warning(message, **kwargs)

    def log_error(self, message: str, **kwargs):
        return self._console_error(message, **kwargs)

    def log_traceback(self, exception: Exception, **kwargs):
        return self._console_traceback(exception, **kwargs)

    def user_info_text(self, message: str, **kwargs):
        """Send an information to the user"""
        return self._console_info(message)

    def user_info_text_with_data(self, message: str, data: Union[dict, list]):
        """Send an information to user and attach some data in json format"""
        return self._console_info(f"{message}\n{json.dumps(data)}")

    def user_input_text(self, message: str, **kwargs):
        """Request an input from the user"""
        self._console_info(message)
        return input(">> ")

    def user_input_text_with_data(self, message: str, data: Union[dict, list]):
        """Request an input user and attach some data in json format. Useful to show dropdown or buttons"""
        self._console_info(f"{message}\n{json.dumps(data)}")
        return input(">> ")


class PredefinedLoggerStaIO(BaseStaIO):
    """A Stdout which use a predefined logger."""

    def __init__(self, name: str, level=logging.DEBUG, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(name)

    def _console_info(self, message: str, **kwargs):
        self.logger.info(message, stacklevel=3, **kwargs)

    def _console_debug(self, message: str, **kwargs):
        self.logger.debug(message, stacklevel=3, **kwargs)

    def _console_warning(self, message: str, **kwargs):
        self.logger.warning(message, stacklevel=3, **kwargs)

    def _console_error(self, message: str, **kwargs):
        self.logger.error(message, stacklevel=3, **kwargs)

    def _console_traceback(self, message: str, **kwargs):
        self.logger.error(message, stacklevel=3, **kwargs)

    def log_info(self, message: str, **kwargs):
        return self._console_info(message, **kwargs)

    def log_debug(self, message: str, **kwargs):
        return self._console_debug(message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        return self._console_warning(message, **kwargs)

    def log_error(self, message: str, **kwargs):
        return self._console_error(message, **kwargs)

    def log_traceback(self, message: str, **kwargs):
        return self._console_traceback(message, **kwargs)


if __name__ == "__main__":
    stdout = ConsoleStaIO("dummy")
    stdout.user_input_text_with_data("Are you major?", ["Yes", "No"])
