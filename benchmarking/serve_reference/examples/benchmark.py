import time
import tempfile
import json

from tqdm import tqdm
import pandas as pd
import ray
import click
import torch

from benchmarking import serve_reference
from benchmarking.utils import throughput_calculation, throughput_calculation_1, get_latency

from benchmarking import HTTPProxyActor
import subprocess
import os

class ChainHandle:
    def __init__(self, handle_list):
        self.handle_list = handle_list

    def remote(self, data):
        for index in range(len(self.handle_list)):
            data = self.handle_list[index].remote(data=data)
        return data


@click.command()
@click.option("--batch-size", type=int, default=None)
@click.option("--num-warmups", type=int, default=200)
@click.option("--num-queries", type=int, default=500)
@click.option(
    "--return-type", type=click.Choice(["string", "torch"]), default="string"
)
def main(batch_size, num_warmups, num_queries, return_type):
    serve_reference.init(start_server=False)
    # serve_reference.init()

    # def noop(_, *args, **kwargs):
    def noop(_, val):
        bs = serve_reference.context.batch_size
        assert (
            bs == batch_size
        ), f"worker received {bs} which is not what expected"
        result = ""
        #time.sleep(.01)

        return val

        if return_type == "torch":
            result = torch.zeros((3, 224, 224))

        if bs is None:  # No batching
            return result
        else:
            return [result] * bs

    def noop2(_, data):
        # time.sleep(.01)
        # for _ in range(10000):
        #     data += 1
        return data

    if batch_size:
        noop = serve_reference.accept_batch(noop)

    with serve_reference.using_router("noop"):
        serve_reference.create_endpoint("noop", "/noop")
        config = serve_reference.BackendConfig(max_batch_size=batch_size)
        serve_reference.create_backend(noop2, "noop", backend_config=config)
        serve_reference.link("noop", "noop")
        handle = serve_reference.get_handle("noop").options()

    # Uncomment next line to use serve_reference
    with serve_reference.using_router("noop2"):
        serve_reference.create_endpoint("noop2", "/noop2")
        config = serve_reference.BackendConfig(max_batch_size=batch_size)
        serve_reference.create_backend(noop2, "noop2", backend_config=config)
        serve_reference.link("noop2", "noop2")
        handle2 = serve_reference.get_handle("noop2").options()

    # Switch commenting of next two lines to use serve_reference
    handle = ChainHandle([handle, handle2])
    # handle.next_handle = handle2

    closed = True
    qps = True
    dump = False
    open = True
    if closed:
        latency = []
        for i in tqdm(range(num_warmups + num_queries)):
            if i == num_warmups:
                serve_reference.clear_trace()

            start = time.perf_counter()

            if not batch_size:
                future = handle.remote(data=1)
                future = ray.get(future)
                ray.get(future)
            else:
                ray.get(handle.enqueue_batch(data=[1] * batch_size))
                # ray.get([handle.remote() for _ in range(batch_size)])

            end = time.perf_counter()
            latency.append(end - start)

        # Remove initial samples
        latency = latency[num_warmups:]

        series = pd.Series(latency) * 1000
        print("Closed Latency for two noop backend (ms)")
        print(series.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    if qps:
        qps = throughput_calculation_1(handle, {'data': 1}, num_queries)
        print(f"Throughput: {qps}")

    if dump:
        _, trace_file = tempfile.mkstemp(suffix=".json")
        with open(trace_file, "w") as f:
            json.dump(serve_reference.get_trace(), f)
        print(f"trace file written to {trace_file}")


    if open:
        filename_query = "arrival_trace.jsonl"
        route = "/noop"
        # handle
        image_path = "elephant.jpg"
        http_actor = HTTPProxyActor.remote(
            host="127.0.0.1",
            port=8000,
            serving_backend="reference",
            filename=filename_query,
        )
        ray.get(
            http_actor.register_route.remote(
                route, handle
            )
        )
        go_client_path = "client.go"

        arrival_curve = [10]*num_queries
        arrival_curve_str = [str(x) for x in arrival_curve]
        ls_output = subprocess.Popen(
            [
                "go",
                "run",
                go_client_path,
                image_path,
                route,
                *arrival_curve_str,
            ]
        )
        ls_output.communicate()

        latency_s = get_latency(filename_query)
        series = pd.Series(latency_s) * 1000
        print("Open Loop Latency for two noop backend (ms)")
        print(series.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
        os.remove(filename_query)

        # cleanup
        # del latency_s, handle, arrival_curve, arrival_curve_str
        # serve_reference.shutdown()


if __name__ == "__main__":
    main()
