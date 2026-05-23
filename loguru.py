"""Minimal runtime fallback for environments without the external `loguru` package.

This shim only implements the subset of APIs used by this repository so tests
and backend flows can run in constrained environments.
"""

from __future__ import annotations

import logging
from typing import Any


class _CompatLogger:
    def __init__(self) -> None:
        self._logger = logging.getLogger("trading_system")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def success(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(msg, *args, **kwargs)

    # Loguru-like fluent/no-op helpers used in the codebase.
    def add(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def remove(self, *args: Any, **kwargs: Any) -> None:
        return None

    def enable(self, *args: Any, **kwargs: Any) -> None:
        return None

    def disable(self, *args: Any, **kwargs: Any) -> None:
        return None

    def bind(self, *args: Any, **kwargs: Any) -> "_CompatLogger":
        return self

    def opt(self, *args: Any, **kwargs: Any) -> "_CompatLogger":
        return self

    def patch(self, *args: Any, **kwargs: Any) -> "_CompatLogger":
        return self


logger = _CompatLogger()

