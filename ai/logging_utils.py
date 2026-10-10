"""Console logging shared by the web app and training CLI."""
import logging
import os
import re
from urllib.parse import urlsplit


class QuietAccessFilter(logging.Filter):
    """Keep failed requests; hide successful polling and static-file traffic."""

    def filter(self, record):
        if record.levelno >= logging.WARNING or logging.getLogger('ai').isEnabledFor(logging.DEBUG):
            return True
        message = re.sub(r'\x1b\[[0-9;]*m', '', record.getMessage())
        match = re.search(r'"GET (\S+) HTTP/[\d.]+" (\d{3})\b', message)
        if not match:
            return True
        status = int(match[2])
        try:
            path = urlsplit(match[1]).path
        except ValueError:
            return True
        quiet = path.startswith(('/jobs/', '/train/progress/', '/static/'))
        return not (quiet and (200 <= status < 300 or status == 304))


def configure_logging():
    """Configure once without replacing handlers supplied by a host server."""
    level = os.environ.get('HANDWRITER_LOG_LEVEL', 'INFO').upper()
    if level not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
        level = 'INFO'
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('ai').setLevel(level)
    access_logger = logging.getLogger('werkzeug')
    access_logger.setLevel(logging.INFO)
    if not any(isinstance(item, QuietAccessFilter) for item in access_logger.filters):
        access_logger.addFilter(QuietAccessFilter())
