"""
NUCML-Next Examples Helper
==========================

Minimal convenience wrappers for notebooks and documentation.

This module provides simple, reusable snippets to reduce boilerplate
in notebooks and examples. It is intentionally kept small and stable.
"""

from nucml_next.examples.helpers import (
    load_sample_db_path,
    quick_ingest,
    load_dataset,
    print_dataset_summary,
)

__all__ = [
    'load_sample_db_path',
    'quick_ingest',
    'load_dataset',
    'print_dataset_summary',
]
