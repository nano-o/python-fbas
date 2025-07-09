from collections.abc import Sequence
import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Iterator, Generator, Set, TypeVar

T = TypeVar('T')


def powerset(s: Sequence[T]) -> Generator[Set[T], None, None]:
    """A generator for the powerset of s. Assume elements in s are unique."""
    x = len(s)
    # each x-bit number represents a subset of s:
    for i in range(1 << x):
        yield {s[j] for j in range(x) if (i & (1 << j))}


_TIMINGS: Dict[str, List[float]] = {}


@contextmanager
def timed(label: str) -> Iterator[None]:
    """
    Context manager that measures the execution time of the enclosed block,
    logs the duration, and stores the sample in _TIMINGS.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        logging.getLogger(__name__).info("%s took %.3f s", label, dur)
        _TIMINGS.setdefault(label, []).append(dur)


def timings() -> Dict[str, List[float]]:
    """Return all collected timing samples."""
    return _TIMINGS

