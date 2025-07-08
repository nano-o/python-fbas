from dataclasses import dataclass, replace, fields
from typing import Literal, Optional
from contextlib import contextmanager

# ---------------------------------------------------------------------------#
# Immutable configuration object
# ---------------------------------------------------------------------------#
@dataclass(frozen=True, slots=True)
class Config:
    stellar_data_url: str = "https://radar.withobsrvr.com/api/v1/node"
    sat_solver: str = "cryptominisat5"
    card_encoding: Literal["naive", "totalizer"] = "totalizer"
    max_sat_algo: Literal["LSU", "RC2"] = "LSU"
    output: Optional[str] = None
    group_by: Optional[str] = None
    validator_display: Literal["id", "name", "both"] = "both"


_cfg: Config = Config()           # single authoritative instance


def get() -> Config:
    """Return current configuration (read-only)."""
    return _cfg


def update(**kwargs) -> None:
    """Atomically replace the configuration with a modified copy."""
    global _cfg
    _cfg = replace(_cfg, **kwargs)


@contextmanager
def temporary_config(**kwargs):
    """Temporarily override configuration values inside a *with* block."""
    old = _cfg
    update(**kwargs)
    try:
        yield
    finally:
        update(**{f.name: getattr(old, f.name) for f in fields(old)})
