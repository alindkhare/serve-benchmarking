from benchmarking import serve_benchmark
from benchmarking.experiment import Experiment, Plotter

SERVE_IMPLEMENTATIONS = ["vanilla"]


def get_name(module):
    if module.__name__ == serve_benchmark.__name__:
        return "vanilla"


__all__ = ["serve_benchmark", "get_serve", "Experiment", "Plotter"]
