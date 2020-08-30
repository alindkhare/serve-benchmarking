from abc import abstractmethod
import json


class Experiment:
    def __init__(self, name, config_path):
        self.name = name
        self.config = None
        with open(config_path, "r") as fp:
            self.config = json.load(fp)

    @abstractmethod
    def run(self):
        NotImplementedError()

    @abstractmethod
    def save(self, filepath):
        NotImplementedError()


class Plotter:
    def __init__(self, plot_config_path):
        self.config = None
        with open(plot_config_path, "r") as fp:
            self.config = json.load(fp)

    @abstractmethod
    def plot(self, experiment_file):
        NotImplementedError()

