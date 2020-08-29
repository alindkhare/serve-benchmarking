import time
import tempfile
import json

from tqdm import tqdm
import pandas as pd
import ray
import click
import torch

from benchmarking import serve_reference


@click.command()
@click.option("--batch-size", type=int, default=None)
@click.option("--num-warmups", type=int, default=200)
@click.option("--num-queries", type=int, default=5000)
@click.option(
    "--return-type", type=click.Choice(["string", "torch"]), default="string"
)
def main(batch_size, num_warmups, num_queries, return_type):
    serve_reference.init()

    def noop(_, *args, **kwargs):
        bs = serve_reference.context.batch_size
        assert (
            bs == batch_size
        ), f"worker received {bs} which is not what expected"
        result = ""

        if return_type == "torch":
            result = torch.zeros((3, 224, 224))

        if bs is None:  # No batching
            return result
        else:
            return [result] * bs

    if batch_size:
        noop = serve_reference.accept_batch(noop)

    with serve_reference.using_router("noop"):
        serve_reference.create_endpoint("noop", "/noop")
        config = serve_reference.BackendConfig(max_batch_size=batch_size)
        serve_reference.create_backend(noop, "noop", backend_config=config)
        serve_reference.link("noop", "noop")
        handle = serve_reference.get_handle("noop")

    latency = []
    for i in tqdm(range(num_warmups + num_queries)):
        if i == num_warmups:
            serve_reference.clear_trace()

        start = time.perf_counter()

        if not batch_size:
            ray.get(
                # This is how to pass a higher level metadata to the tracing
                # context
                handle.options(
                    tracing_metadata={"demo": "pipeline-id"}
                ).remote()
            )
        else:
            ray.get(handle.enqueue_batch(val=[1] * batch_size))
            # ray.get([handle.remote() for _ in range(batch_size)])

        end = time.perf_counter()
        latency.append(end - start)

    # Remove initial samples
    latency = latency[num_warmups:]

    series = pd.Series(latency) * 1000
    print("Latency for single noop backend (ms)")
    print(series.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    _, trace_file = tempfile.mkstemp(suffix=".json")
    with open(trace_file, "w") as f:
        json.dump(serve_reference.get_trace(), f)
    print(f"trace file written to {trace_file}")


if __name__ == "__main__":
    main()
