"""Automated data extraction, analysis-hook dispatch and plotting suite."""

from varify.src.analysis.parser import ResultParser
from varify.src.analysis.analysis_dispatcher import AnalysisDispatcher
from varify.src.analysis.plotting import PlotSuite
from varify.src.analysis.postprocess import load_xy, fit_spline, spline_mse

__all__ = [
    "ResultParser", "AnalysisDispatcher", "PlotSuite",
    "load_xy", "fit_spline", "spline_mse",
]
