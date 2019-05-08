import logging
from datetime import datetime
import utils
import os


class Logger(logging.Logger):
    NAME = 'ConvS2S'

    def nofmt(self, msg, lvl=20, *args, **kwargs):
        # need lock?
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
    def get(cls, file_path=None):
        logging.setLoggerClass(cls)
        logger = logging.getLogger(cls.NAME)
        if logger.hasHandlers():
            assert file_path is None
            return logger

        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %H:%M:%S %p')

        # standard output handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if file_path is None:
            # set default
            timestamp = datetime.now().strftime('%y%m%d_%H-%M-%S')
            file_name = "{}.log".format(timestamp)
            file_path = os.path.join("logs", file_name)
            utils.makedirs("logs")

        # file output handler
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)

        return logger
