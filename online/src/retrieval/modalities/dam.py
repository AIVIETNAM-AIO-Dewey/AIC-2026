"""Canonical DAM modality boundary.

Branch-2 owns the production six-English-query DAM scorer.  The workbench adapter is
re-exported here for the existing non-branch endpoint contract.
"""

from .local import CpuQdrantSearch

__all__ = ["CpuQdrantSearch"]
