import sys
from datetime import datetime

from loguru import logger as _logger


def define_log_level(print_level="DEBUG", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")
    log_name = f"{name}_{formatted_date}" if name else formatted_date  # name a log with prefix name

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(f"logs/{log_name}.txt", level=logfile_level)
    return _logger

logger = define_log_level()