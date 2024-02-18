import logging

LogName = 'PerAct'

if LogName not in logging.Logger.manager.loggerDict.keys():
    logger = logging.getLogger(LogName)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler())


def get_logger() -> logging.Logger:
    return logging.getLogger(LogName)


def logger_add_file_handler(filename: str) -> None:
    if len(logger.handlers) > 1:
        return
    formatter = logging.Formatter(fmt='%(asctime)s;%(levelname)s  %(message)s',
                                  datefmt='%H:%M:%S')
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
