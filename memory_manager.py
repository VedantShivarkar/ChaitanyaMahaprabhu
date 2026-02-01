import logging
import time
from typing import Callable

import psutil

from config import MIN_FREE_MEMORY_RATIO, PROCESSING_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)


def ensure_enough_memory() -> None:
    mem = psutil.virtual_memory()
    free_ratio = mem.available / mem.total if mem.total else 1.0
    if free_ratio < MIN_FREE_MEMORY_RATIO:
        logger.warning("Low memory: free ratio %.3f", free_ratio)
        raise MemoryError("Not enough free memory to continue processing.")


def run_with_timeout(func: Callable, *args, **kwargs):
    """Simple cooperative timeout wrapper.

    The wrapped function should periodically call ensure_enough_memory()
    for best safety; here we just enforce a global wall-clock timeout.
    """
    start = time.time()
    result_container = {}
    exc_container = {}

    def _runner():
        try:
            result_container["result"] = func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            exc_container["exc"] = e

    _runner()
    elapsed = time.time() - start
    if elapsed > PROCESSING_TIMEOUT_SECONDS:
        raise TimeoutError("Operation exceeded processing timeout.")

    if "exc" in exc_container:
        raise exc_container["exc"]

    return result_container.get("result")
