import logging
from datetime import datetime
import utils
import os


log_lv = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


class Logger(logging.Logger):
    NAME = 'ConvS2S'

    def nofmt(self, msg, lvl=logging.INFO, *args, **kwargs):
        formatters = self.remove_formats()
        r = super().log(lvl, msg, *args, **kwargs)
        self.set_formats(formatters)
        return r

    def remove_formats(self):
        """ Remove all formats from logger """
        formatters = []
        for handler in self.handlers:
            formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        return formatters

    def set_formats(self, formatters):
        """ Set formats to every handler of logger """
        for handler, formatter in zip(self.handlers, formatters):
            handler.setFormatter(formatter)

    def set_file_handler(self, file_path):
        file_handler = logging.FileHandler(file_path)
        formatter = self.handlers[0].formatter
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    @classmethod
    def get(cls, file_path=None, level='info'):
        logging.setLoggerClass(cls)
        logger = logging.getLogger(cls.NAME)
        logging.setLoggerClass(logging.Logger) # restore

        if logger.hasHandlers():
            # If logger already got all handlers (# handlers == 2), use the logger.
            # else, re-set handlers.
            if len(logger.handlers) == 2:
                return logger

            logger.handlers.clear()

        log_format = '%(levelname)s::%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %H:%M:%S')

        # standard output handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if file_path:
            # file output handler
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(log_lv[level])
        logger.propagate = False

        return logger
