import os
import logging
from logging.handlers import TimedRotatingFileHandler


def init_logger(logfile_name, debug_level=0):
    dirname = os.path.dirname(logfile_name)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    logging.basicConfig(
        # filename=logfile_name,
        level=logging.INFO if debug_level == 0 else logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        handlers=[
            TimedRotatingFileHandler(logfile_name, when="midnight", interval=1),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger
