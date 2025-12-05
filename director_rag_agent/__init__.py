"""
Director Agent Package - V2 (Fixed)

Fixes the issue where agent stops after search tools without
generating JSON or calling save_to_supabase.
"""

from .agent import (
    root_agent,
    DIRECTOR_DIRECT_PROMPT,
    DEFAULT_MODEL,
)

__all__ = [
    "root_agent",
    "DIRECTOR_DIRECT_PROMPT",
    "DEFAULT_MODEL",
]

__version__ = "2.2.0"