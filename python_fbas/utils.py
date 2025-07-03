from collections.abc import Sequence


def powerset(s: Sequence):
    """A generator for the powerset of s. Assume elements in s are unique."""
    x = len(s)
    # each x-bit number represents a subset of s:
    for i in range(1 << x):
        yield {s[j] for j in range(x) if (i & (1 << j))}

