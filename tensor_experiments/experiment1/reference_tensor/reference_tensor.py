from benchmarking import Experiment
from benchmarking import serve_reference
import torch
import ray
import pandas as pd
import time
import numpy as np
import click
from pprint import pprint
from itertools import product


@serve_reference.accept_batch
def noop(_, data):
    return data


class Chain:
    def __init__(self, max_batch_size, pipeline_length):
        self.plength = pipeline_length
        self.handles = list()

        for index in range(self.plength):
            node_id = f"service-{index}"
            with serve_reference.using_router(node_id):
                serve_reference.create_endpoint(node_id)
                config = serve_reference.BackendConfig(
                    max_batch_size=max_batch_size, num_replicas=1
                )
                serve_reference.create_backend(
                    noop, node_id, backend_config=config
                )
                serve_reference.link(node_id, node_id)
                self.handles.append(serve_reference.get_handle(node_id))

    def remote(self, data):
        for index in range(self.plength):
            data = self.handles[index].remote(data=data)
        return data


def construct_tensor(config):
    tensor_shape = [1] + config["tensor_shape"]
    assert config["tensor_type"] == "torch", "Wrong configuration"
    return torch.zeros(tensor_shape)


class ReferencedTensorExperiment(Experiment):
    def __init__(self, name, config_path):
        super().__init__(name, config_path)
        assert self.config["serving_type"] == "vanilla", "Wrong configuration"
        assert (
            self.config["arrival_process"] == "closed_loop"
        ), "wrong configuration"
        columns = [
            "batch_size",
            "pipeline_length",
            "tensor_type",
            "tensor_shape",
            "serving_type",
            "arrival_process",
            "throughput_qps",
            "latency_s",
        ]

        self._df = pd.DataFrame(columns=columns)

    def _throughput_calculation(self, chain_pipeline, tensor_data):
        start_time = time.perf_counter()
        fut = [
            chain_pipeline.remote(data=tensor_data)
            for _ in range(self.config["num_requests"])
        ]
        current = fut
        all_ready = False
        while True:
            if not all_ready:
                ready, unready = ray.wait(
                    current, num_returns=len(current), timeout=0
                )
            else:
                ready = current
            if len(ready) > 0:
                s_ready, s_unready = ray.wait(
                    ready, num_returns=len(ready), timeout=0
                )
                if len(s_unready) == 0:
                    break
            if len(unready) > 0:
                current = unready
            else:
                all_ready = True
                current = s_unready

        end_time = time.perf_counter()
        duration = end_time - start_time
        qps = self.config["num_requests"] / duration
        return qps

    def run(self):

        tensor_data = construct_tensor(self.config)
        for batch_size, pipeline_length in product(
            self.config["max_batch_sizes"], self.config["pipeline_lengths"]
        ):
            df_row = dict(
                batch_size=batch_size,
                pipeline_length=pipeline_length,
                tensor_type=self.config["tensor_type"],
                tensor_shape="x".join(
                    [str(shape) for shape in self.config["tensor_shape"]]
                ),
                serving_type=self.config["serving_type"],
                arrival_process=self.config["arrival_process"],
            )

            # initialize serve
            serve_reference.init(start_server=False)

            chain_pipeline = Chain(
                max_batch_size=batch_size, pipeline_length=pipeline_length
            )

            # warmup
            ready_refs, _ = ray.wait(
                [chain_pipeline.remote(tensor_data) for _ in range(200)], 200
            )
            ray.wait(ready_refs, num_returns=200)
            del ready_refs

            qps = self._throughput_calculation(chain_pipeline, tensor_data)
            df_row.update(throughput_qps=qps)

            serve_reference.clear_trace()

            # closed loop latency calculation
            closed_loop_latencies = list()
            for _ in range(self.config["num_requests"]):
                start_time = time.perf_counter()
                ready, _ = ray.wait([chain_pipeline.remote(tensor_data)], 1)
                ray.wait(ready, 1)
                end_time = time.perf_counter()
                latency = end_time - start_time
                closed_loop_latencies.append(latency)

            pprint(df_row)
            # percentile_values =
            df_row.update(latency_s=closed_loop_latencies)

            self._df = self._df.append(df_row, ignore_index=True)

            # cleanup
            del closed_loop_latencies, chain_pipeline
            serve_reference.shutdown()

    def save(self, filepath):
        self._df.to_csv(filepath)


@click.command()
@click.option("--config-path", type=str, default="../config.json")
@click.option("--save-path", type=str, default="referenced_tensor_stats.csv")
def main(config_path, save_path):
    experiment = ReferencedTensorExperiment(
        name="unpickle", config_path=config_path
    )
    experiment.run()
    experiment.save(save_path)


if __name__ == "__main__":
    main()

