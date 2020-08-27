from benchmarking import Experiment
from benchmarking import serve_benchmark
import torch
import ray
import pandas as pd
import time
import numpy as np
import click
import pickle
from pprint import pprint


@serve_benchmark.accept_batch
def noop(_, data):
    ret = [pickle.loads(v) for v in data]
    # some computation
    return [pickle.dumps(v) for v in data]


class Chain:
    def __init__(self, max_batch_size, pipeline_length):
        self.plength = pipeline_length
        self.handles = list()

        for index in range(self.plength):
            node_id = f"service-{index}"
            with serve_benchmark.using_router(node_id):
                serve_benchmark.create_endpoint(node_id)
                config = serve_benchmark.BackendConfig(
                    max_batch_size=max_batch_size, num_replicas=1
                )
                serve_benchmark.create_backend(
                    noop, node_id, backend_config=config
                )
                serve_benchmark.link(node_id, node_id)
                self.handles.append(serve_benchmark.get_handle(node_id))

    def remote(self, data):
        for index in range(self.plength):
            data = self.handles[index].remote(data=data)
        return data


def construct_tensor(config):
    tensor_shape = [1] + config["tensor_shape"]
    assert config["tensor_type"] == "torch", "Wrong configuration"
    return torch.zeros(tensor_shape)


class PickledTensorExperiment(Experiment):
    def __init__(self, name, config_path):
        super().__init__(name, config_path)
        assert self.config["serving_type"] == "vanilla", "Wrong configuration"
        assert (
            self.config["arrival_process"] == "closed_loop"
        ), "wrong configuration"
        columns = [
            "batch_size",
            "throughput_qps",
            "pipeline_length",
            "tensor_type",
            "tensor_shape",
            "serving_type",
        ]
        for perc in self.config["latency_percentile"]:
            columns.append(f"lat_s_{perc}")
        self._df = pd.DataFrame(columns=columns)

    def run(self):

        tensor_data = construct_tensor(self.config)
        for batch_size, pipeline_length in zip(
            self.config["max_batch_sizes"], self.config["pipeline_lengths"]
        ):
            df_row = dict(
                batch_size=batch_size,
                pipeline_length=pipeline_length,
                tensor_type=self.config["tensor_type"],
                tensor_shape=self.config["tensor_shape"],
                serving_type=self.config["serving_type"],
            )

            # initialize serve
            serve_benchmark.init(start_server=False)

            chain_pipeline = Chain(
                max_batch_size=batch_size, pipeline_length=pipeline_length
            )

            # warmup
            ray.wait(
                [
                    chain_pipeline.remote(pickle.dumps(tensor_data))
                    for _ in range(200)
                ],
                200,
            )

            # throughput calculation
            start_time = time.perf_counter()
            ray.wait(
                [
                    chain_pipeline.remote(pickle.dumps(tensor_data))
                    for _ in range(self.config["num_requests"])
                ],
                self.config["num_requests"],
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            qps = self.config["num_requests"] / duration
            df_row.update(throughput_qps=qps)

            serve_benchmark.clear_trace()

            # closed loop latency calculation
            closed_loop_latencies = list()
            for _ in range(self.config["num_requests"]):
                start_time = time.perf_counter()
                ray.wait([chain_pipeline.remote(pickle.dumps(tensor_data))], 1)
                end_time = time.perf_counter()
                latency = end_time - start_time
                closed_loop_latencies.append(latency)

            # percentile_values =
            df_row.update(
                {
                    f"lat_s_{percentile}": percentile_value
                    for percentile, percentile_value in zip(
                        self.config["latency_percentile"],
                        np.percentile(
                            closed_loop_latencies,
                            self.config["latency_percentile"],
                        ),
                    )
                }
            )

            pprint(df_row)
            self._df.append(df_row, ignore_index=True)

            # cleanup
            del closed_loop_latencies, chain_pipeline
            serve_benchmark.shutdown()

    def save(self, filepath):
        self._df.to_csv(filepath)


@click.command()
@click.option("--config-path", type=str, default="../config.json")
@click.option("--save-path", type=str, default="pickled_tensor_stats.csv")
def main(config_path, save_path):
    experiment = PickledTensorExperiment(name="pickle", config_path=config_path)
    experiment.run()
    experiment.save(save_path)


if __name__ == "__main__":
    main()

