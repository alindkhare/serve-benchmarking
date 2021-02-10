import time
import tempfile
import json

from tqdm import tqdm
import pandas as pd
import ray
import click
import torch

from benchmarking import serve_reference
from benchmarking.utils import throughput_calculation, throughput_calculation_1


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
    serve_reference.init()

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
        for _ in range(10000):
            data += 1
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

    latency = []
    # Switch commenting of next two lines to use serve_reference
    handle = ChainHandle([handle, handle2])
    # handle.next_handle = handle2
    for i in tqdm(range(num_warmups + num_queries)):
        if i == num_warmups:
            serve_reference.clear_trace()

        start = time.perf_counter()

        if not batch_size:
            future = handle.remote(data=1)
            future = ray.get(future)
            ray.get(future)
            # ray.get(ray.get(
                # This is how to pass a higher level metadata to the tracing
                # context
                # handle.remote(data=1)
                # handle.remote(val=handle2.remote(val=1))
                # handle.options(
                #     tracing_metadata={"demo": "pipeline-id"}
                # ).remote()
            # ))
        else:
            ray.get(handle.enqueue_batch(data=[1] * batch_size))
            # ray.get([handle.remote() for _ in range(batch_size)])

        end = time.perf_counter()
        latency.append(end - start)

    # Remove initial samples
    latency = latency[num_warmups:]

    series = pd.Series(latency) * 1000
    print("Latency for single noop backend (ms)")
    print(series.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    qps = throughput_calculation_1(handle, {'data': 1}, num_queries)
    print(f"Throughput: {qps}")
    _, trace_file = tempfile.mkstemp(suffix=".json")
    with open(trace_file, "w") as f:
        json.dump(serve_reference.get_trace(), f)
    print(f"trace file written to {trace_file}")


if __name__ == "__main__":
    main()
