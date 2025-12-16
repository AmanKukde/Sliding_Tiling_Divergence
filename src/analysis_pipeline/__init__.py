"""Analysis Pipeline for Gradient-Based Experiment Evaluation."""

from .core.plotting import plot_prediction_comparison
from .core.analysis import run_gradient_analysis_multi

__version__ = "0.2.0"

__all__ = [
    "plot_prediction_comparison",
    "run_gradient_analysis_multi",
    "__version__",
]
