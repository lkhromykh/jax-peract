import logging

LogName = 'PerAct'

if LogName not in logging.Logger.manager.loggerDict.keys():
    logger = logging.getLogger(LogName)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def get_logger() -> logging.Logger:
    return logging.getLogger(LogName)


def logger_add_default_handlers(log_path: str) -> None:
    if len(logger.handlers):
        return
    formatter = logging.Formatter(fmt='%(asctime)s;%(levelname)s  %(message)s',
                                  datefmt='%H:%M:%S')
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
