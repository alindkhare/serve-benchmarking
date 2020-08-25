import matplotlib.pyplot as plt
import utils
import os
import argparse
import json
import pandas as pd
import seaborn as sns

sns.set_style("white")
sns.set_context(
    "paper", font_scale=1.0,
)

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def read_file(filename):
    profile_information = None
    with open(filename, "r") as fp:
        profile_information = json.load(fp)
    return (
        profile_information["pipeline_lengths"],
        profile_information["latency"]["mean"],
        profile_information["latency"]["std"],
        profile_information["throughput"],
    )


def plot_df(df, current_platform):
    for tensor_type in utils.tensor_types:
        plot_df = df.loc[df["tensor_type"] == tensor_type]
        if len(plot_df) == 0:
            continue

        mechanisms = plot_df["mechanism"].unique()
        colors = sns.color_palette("deep", n_colors=len(mechanisms))
        pipeline_lengths = plot_df["pipeline_length"].unique()
        print("Pipeline Lengths are : {}".format(pipeline_lengths))
        print("Mechnaims are: {}".format(mechanisms))

        fig, axis_plots = plt.subplots(
            nrows=len(pipeline_lengths), figsize=(3.0, 4.0), sharex=True,
        )
        # for mechanism, axes in zip(mechanisms, axis_plots):
        for pipeline_length, ax in zip(pipeline_lengths, axis_plots):
            for mechanism, color in zip(mechanisms, colors):
                pipeline_df = plot_df.loc[
                    plot_df["pipeline_length"] == pipeline_length
                ]
                pipeline_mech_df = pipeline_df.loc[
                    pipeline_df["mechanism"] == mechanism
                ]
                # ax.set_ylim(0, 0.12)
                # ax.set_yscale("symlog")
                ax.plot(
                    pipeline_mech_df["max_batch_size"],
                    pipeline_mech_df["latency_s(mean)"],
                    label=mechanism,
                    color=color,
                )

            ax.set_title(f"length: {pipeline_length}")
        axis_plots[0].legend()
        mid = len(pipeline_lengths) // 2
        axis_plots[mid].set_ylabel(
            "Latency (mean) in seconds",
            # rotation=1, ha="right", va="center"
        )
        axis_plots[-1].set_xlabel("Maximum batch size")
        fig.subplots_adjust(hspace=0.4)
        fname = f"performance_plots_{tensor_type}.pdf"
        fpath = os.path.join(utils.RESULT_DIR, platform, tensor_type, fname)
        plt.savefig(fpath, bbox_inches="tight")

        # plt.plot(pipeline_df)


parser = argparse.ArgumentParser("Plot Configs")

parser.add_argument("-b", "--batches", nargs="+", type=int, default=[1, 4, 8])
parser.add_argument(
    "-p", "--pipelines", nargs="+", type=int, default=[1, 2, 4, 8]
)
parser.add_argument(
    "-s",
    "--shapes",
    nargs="+",
    type=tuple,
    default=[(32, 32, 3), (64, 64, 3), (224, 224, 3)],
    help="(H,W,C)",
)

args = parser.parse_args()

for platform in os.listdir(utils.RESULT_DIR):
    results = pd.DataFrame(
        columns=(
            "max_batch_size",
            "latency_s(mean)",
            "latency_s(std)",
            "throughput_qps",
            "pipeline_length",
            "mechanism",
            "tensor_type",
        )
    )
    for batch, shape in zip(args.batches, args.shapes):
        for tensor_type in utils.tensor_types:
            folder_names = list(utils.label_map.keys())
            folder_paths = [
                os.path.join(
                    utils.RESULT_DIR, platform, tensor_type, folder_name
                )
                for folder_name in folder_names
            ]
            print(folder_paths)
            if not all([os.path.exists(fpath) for fpath in folder_paths]):
                continue

            filename = (
                f"{tensor_type}: BS:{batch} "
                f"[{shape[0]}x{shape[1]}x{shape[2]}].json"
            )
            filepaths = [
                os.path.join(folder, filename) for folder in folder_paths
            ]
            print(filepaths)
            if not all([os.path.exists(fpath) for fpath in filepaths]):
                continue

            for filepath, mechanism in zip(filepaths, utils.label_map.values()):

                for pipeline, lat_mean, lat_std, throughput in zip(
                    *read_file(filepath)
                ):
                    results = results.append(
                        {
                            "max_batch_size": batch,
                            "latency_s(mean)": lat_mean,
                            "latency_s(std)": lat_std,
                            "throughput_qps": throughput,
                            "pipeline_length": pipeline,
                            "mechanism": mechanism,
                            "tensor_type": tensor_type,
                        },
                        ignore_index=True,
                    )
    print(results)
    csv_path = os.path.join(
        utils.RESULT_DIR, platform, "profile_information.csv"
    )
    results.to_csv(csv_path)

    plot_df(results, platform)

