"""Science tool package. Only names in __all__ become MCP tools.

Layout:
  spec.py            AnalysisSpec (what an analysis provides) + ArtifactResult
  mu2e_job.py        running `mu2e -c <fcl> -s|-S <input>`; analysis-agnostic
  registry.py        the catalogue: name -> AnalysisSpec
  analyses/<name>.py one module per analysis (fcl + stdout parser)
  analysis_tools.py  the MCP tools: list_analyses, run_analysis

To add an analysis, write `analyses/<name>.py` with a `SPEC` and list it in
registry.py — the tools below pick it up with no changes here.
"""

from .analysis_tools import list_analyses, run_analysis
from .registry import ANALYSES, ANALYSIS_NAMES
from .spec import AnalysisSpec, ArtifactResult

__all__ = ["list_analyses", "run_analysis"]
