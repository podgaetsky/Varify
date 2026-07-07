"""Automated data extraction, analysis-hook dispatch and plotting suite."""

from src.analysis.parser import ResultParser
from src.analysis.analysis_dispatcher import AnalysisDispatcher
from src.analysis.plotting import PlotSuite

__all__ = ["ResultParser", "AnalysisDispatcher", "PlotSuite"]
