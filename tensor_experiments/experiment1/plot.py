from benchmarking import Plotter
import pandas as pd
from benchmarking.utils import ROOT_DIR
import os
import matplotlib.pyplot as plt
import matplotlib
import json

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import seaborn as sns
import click

sns.set_style("white")
sns.set_context(
    "paper", font_scale=1.0,
)
SMALL_SIZE = 15

plt.rcParams.update({"font.size": 40})
plt.rc("axes", labelsize=20)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rcParams["legend.title_fontsize"] = SMALL_SIZE
plt.rc("axes", titlesize=SMALL_SIZE + 3)


class TensorComparison(Plotter):
    def _extract_dfs(self):
        self._comparison_dfs = dict()
        self._batch_sizes = None
        for key in self.config["profiles"].keys():
            df_path = os.path.join(ROOT_DIR, self.config["profiles"][key])
            df = pd.read_csv(df_path)
            if self._batch_sizes is None:
                self._batch_sizes = df["batch_size"].unique().tolist()
            else:
                assert (
                    self._batch_sizes == df["batch_size"].unique().tolist()
                ), "Incompatible profiles"
            df["latency_s"] = df["latency_s"].apply(json.loads)
            df = df.explode("latency_s")
            self._comparison_dfs[key] = df

    def __init__(self, plot_config_path):
        super().__init__(plot_config_path)
        self._extract_dfs()

    def plot(self, foldername):
        for batch_size in self._batch_sizes:
            bs_path = os.path.join(foldername, f"batch_size_{batch_size}")
            if not os.path.exists(bs_path):
                os.makedirs(bs_path)
            df_list = list()
            for key in self._comparison_dfs.keys():
                df = self._comparison_dfs[key].query(
                    f"batch_size=={batch_size}"
                )
                df.loc[:, "mechanisms"] = key
                df_list.append(df)

            plot_df = pd.concat(df_list)
            plt.figure(figsize=(16, 12))
            ax = sns.boxplot(
                y="latency_s",
                x="pipeline_length",
                data=plot_df,
                palette="Set2",
                showfliers=False,
                hue="mechanisms",
            )
            ax.set_ylabel("Latency (in seconds)")
            ax.set_xlabel("Pipeline Length (Chain)")
            ax.set_title("Torch Tensor Noop Chain Closed Loop Latency")
            ax.yaxis.grid(True)
            plt.savefig(os.path.join(bs_path, "latency_comparison.pdf"))


@click.command()
@click.option("--config-path", type=str, default="./plot_config.json")
@click.option("--save-path", type=str, default="plots/")
def main(config_path, save_path):
    plotter = TensorComparison(plot_config_path=config_path)
    plotter.plot(save_path)


if __name__ == "__main__":
    main()
