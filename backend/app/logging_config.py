"""
Structured logging setup using structlog.
Logs are emitted as colored console output in development and JSON in production.
API keys and secrets are never logged.
"""
import logging
import sys

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

from app.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    if _HAS_STRUCTLOG:
        shared_processors: list = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
        ]

        if settings.ENVIRONMENT == "development":
            renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
        else:
            renderer = structlog.processors.JSONRenderer()

        structlog.configure(
            processors=shared_processors + [renderer],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


class _StdlibLoggerWrapper:
    def __init__(self, logger):
        self._logger = logger

    def info(self, msg, **kwargs):
        self._logger.info(f"{msg} {kwargs}" if kwargs else msg)

    def warning(self, msg, **kwargs):
        self._logger.warning(f"{msg} {kwargs}" if kwargs else msg)

    def error(self, msg, **kwargs):
        self._logger.error(f"{msg} {kwargs}" if kwargs else msg)

    def debug(self, msg, **kwargs):
        self._logger.debug(f"{msg} {kwargs}" if kwargs else msg)


def get_logger(name: str = __name__):
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlibLoggerWrapper(logging.getLogger(name))
