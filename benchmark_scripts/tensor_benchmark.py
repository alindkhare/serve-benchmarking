"""
Implements echo tensor benchmarking
"""

import serve_benchmark
import argparse
import torch
import tensorflow as tf
import numpy as np
import time
import json
import ray


@serve_benchmark.accept_batch
def noop(_, data):
    return data


def construct_tensor(args):
    tensor_shape = (1, args.height, args.width, args.channel)
    if args.tensor_type == "torch":
        return torch.zeros(tensor_shape)
    elif args.tensor_type == "tf":
        return tf.zeros(tensor_shape)

    return np.zeros(tensor_shape)


class Chain:
    def __init__(self, args, pipeline_length):
        self.plength = pipeline_length
        self.handles = list()

        for index in range(plength):
            node_id = f"service-{index}"
            with serve_benchmark.using_router(node_id):
                serve_benchmark.create_endpoint(node_id)
                config = serve_benchmark.BackendConfig(
                    max_batch_size=args.max_batch_size,
                    num_replicas=1
                )
                serve_benchmark.create_backend(
                    noop, node_id, backend_config=config)
                serve_benchmark.link(node_id, node_id)
                self.handles.append(serve_benchmark.get_handle(node_id))

    def remote(self, data):
        for index in range(self.plength):
            data = self.handles[index].remote(data=data)
        return data


parser = argparse.ArgumentParser("Bechnmark Configs")
parser.add_argument("-H", "--height", default=224, type=int,
                    help="Height of Tensor: [hxwxc]")
parser.add_argument("-W", "--width", default=224, type=int,
                    help="Height of Tensor: [hxwxc]")
parser.add_argument("-C", "--channel", default=3, type=int,
                    help="channel of Tensor: [hxwxc]")
parser.add_argument("-b", "--max-batch-size", default=1, type=int)
parser.add_argument("-t", "--tensor-type", default="torch",
                    type=str, choices=["torch", "tf", "np"])
parser.add_argument("-r", "--num-requests", type=int, default=2000)

args = parser.parse_args()

pipeline_lengths = [1, 2, 4, 8]
throughput_values = list()
latency_means = list()
latency_stds = list()

print(f"Got {args.tensor_type} "
      f"tensor size: {(args.height, args.width, args.channel)}")
for plength in pipeline_lengths:

    serve_benchmark.init(start_server=False)

    tensor_data = construct_tensor(args)
    chain_pipeline = Chain(args, plength)

    start_time = time.perf_counter()
    ray.wait(
        [chain_pipeline.remote(tensor_data) for _ in range(args.num_requests)],
        args.num_requests
    )
    end_time = time.perf_counter()

    duration = end_time - start_time

    qps = num_queries / duration
    print(f"Throughput calculated: {qps} QPS pipeline: {plength} ")
    throughput_values.append(qps)

    closed_loop_latencies = list()
    for _ in range(args.num_requests):
        start_time = time.perf_counter()
        ray.wait([chain_pipeline.remote(tensor_data)], 1)
        end_time = time.perf_counter()
        latency = end_time - start_time
        closed_loop_latencies.append(latency)

    mean_latency = np.mean(closed_loop_latencies)
    std_latency = np.std(closed_loop_latencies)

    latency_means.append(mean_latency)
    latency_stds.append(std_latency)
    print(f"Latency calculated: {mean_latency} +- {std_latency} s "
          f"pipeline: {plength} ")

    serve_benchmark.shutdown()
    del closed_loop_latencies, tensor_data, chain_pipeline
    # time for system to get back to normal state
    time.sleep(0.2)

profile_data = {
    "pipeline_lengths": pipeline_lengths,
    "latency": {
        "mean": latency_means,
        "std": latency_stds,
    },
    "throughput": throughput_values,
    "tensor_type": args.tensor_type,
}

file_name = (
    f"{args.tensor_type}: BS:{args.max_batch_size} "
    f"[{args.height}x{args.width}x{args.channel}].json"
)

with open(file_name, "w") as fp:
    json.dumps(profile_data)
