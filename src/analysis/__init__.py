"""Automated data extraction, analysis-hook dispatch and plotting suite."""

from varify.src.analysis.parser import ResultParser
from varify.src.analysis.analysis_dispatcher import AnalysisDispatcher, PostJobDispatcher
from varify.src.analysis.plotting import PlotSuite
from varify.src.analysis.postprocess import (
    load_xy, write_xy, fit_spline, spline_mse, curve_loss, compute_loss,
    mse, rmse, mae, huber, chi2, LOSSES,
)

__all__ = [
    "ResultParser", "AnalysisDispatcher", "PostJobDispatcher", "PlotSuite",
    "load_xy", "write_xy", "fit_spline", "spline_mse", "curve_loss",
    "compute_loss", "mse", "rmse", "mae", "huber", "chi2", "LOSSES",
]
