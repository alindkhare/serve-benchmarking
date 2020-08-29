import time
import tempfile
import json

import ray
import click
import torch
import base64

from benchmarking import serve_reference

serve_reference.init(start_server=False)

batch_size = 1
num_queries = 2000

raw_image_data = base64.b64encode(open("./elephant.jpg", "rb").read())
image_data = ray.put(raw_image_data)


@serve_reference.accept_batch
def noop(_, sleep_time=[], data=[]):
    time.sleep(sleep_time[0])
    return [torch.ones((1, 224, 224, 3))] * serve_reference.context.batch_size


@click.command()
@click.option("--num-replicas", type=int, default=1)
@click.option(
    "--method", type=click.Choice(["chain", "group"]), default="chain"
)
def main(num_replicas, method):
    for node_id in ["up", "down"]:
        with serve_reference.using_router(node_id):
            serve_reference.create_endpoint(node_id)
            config = serve_reference.BackendConfig(
                max_batch_size=1, num_replicas=num_replicas
            )
            serve_reference.create_backend(noop, node_id, backend_config=config)
            serve_reference.link(node_id, node_id)

    with serve_reference.using_router("up"):
        up_handle = serve_reference.get_handle("up")
    with serve_reference.using_router("down"):
        down_handle = serve_reference.get_handle("down")

    start = time.perf_counter()
    oids = []

    if method == "chain":
        for i in range(num_queries):
            r = up_handle.options(tracing_metadata={"pipeline-id": i}).remote(
                sleep_time=0.01, data=image_data
            )
            r = down_handle.options(tracing_metadata={"pipeline-id": i}).remote(
                sleep_time=0.02, data=r  # torch tensor
            )
            oids.append(r)
    elif method == "group":
        res = [
            up_handle.options(tracing_metadata={"pipeline-id": i}).remote(
                sleep_time=0.01, data=image_data
            )
            for i in range(num_queries)
        ]
        oids = [
            down_handle.options(tracing_metadata={"pipeline-id": i}).remote(
                sleep_time=0.02, data=d  # torch tensor
            )
            for i, d in enumerate(res)
        ]
    else:
        raise RuntimeError("Unreachable")
    print(f"Submission time {time.perf_counter() - start}")

    ray.wait(oids, len(oids))
    end = time.perf_counter()

    duration = end - start
    qps = num_queries / duration

    print(f"Throughput {qps}")

    _, trace_file = tempfile.mkstemp(suffix=".json")
    with open(trace_file, "w") as f:
        json.dump(serve_reference.get_trace(), f)
    print(f"trace file written to {trace_file}")


if __name__ == "__main__":
    main()
