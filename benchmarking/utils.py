import git
import os
import json
import jsonlines
import time
import ray

ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir

import numpy as np


def gamma(mean, cv, size):
    if cv == 0.0:
        return np.ones(size) * mean
    else:
        return np.random.gamma(1.0 / cv, cv * mean, size=size)


def generate_fixed_arrival_process(mean_qps, cv, num_requests):
    """
    mean_qps : float
        Mean qps
    cv : float
    duration: float
        Duration of the trace in seconds
    """
    # deltas_path = os.path.join(arrival_process_dir,
    #                            "fixed_{mean_qps}_{cv}_{dur}_{ts:%y%m%d_%H%M%S}.deltas".format(
    #                                mean_qps=mean_qps, cv=cv, dur=duration, ts=datetime.now()))
    inter_request_delay_ms = 1.0 / float(mean_qps) * 1000.0
    num_deltas = num_requests - 1
    if cv == 0:
        deltas = np.ones(num_deltas) * inter_request_delay_ms
    else:
        deltas = gamma(inter_request_delay_ms, cv, size=num_deltas)
    deltas = np.clip(deltas, a_min=2.5, a_max=None)
    return deltas


class BytesEncoder(json.JSONEncoder):
    """Allow bytes to be part of the JSON document.
    BytesEncoder will walk the JSON tree and decode bytes with utf-8 codec.
    (Adopted from serve 0.8.2)
    Example:
    >>> json.dumps({b'a': b'c'}, cls=BytesEncoder)
    '{"a":"c"}'
    """

    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, bytes):
            return o.decode("utf-8")
        return super().default(o)


def get_latency(filename):
    latency = list()
    with jsonlines.open(filename) as reader:
        for obj in reader:
            latency.append((obj["end"] - obj["start"]))
    return latency


def throughput_calculation(serve_handle, data_kwarg, num_requests):
    start_time = time.perf_counter()
    fut = [serve_handle.remote(**data_kwarg) for _ in range(num_requests)]
    all_ready = False
    current_router = fut
    current_result = list()
    all_ready = False
    cnt = 0
    cnt_all_ready = 0
    while True:
        if not all_ready:
            ready, unready = ray.wait(
                current_router, num_returns=len(current_router), timeout=0
            )
            cnt_all_ready += len(ready)
        else:
            ready, unready = [], []

        if all_ready or len(ready) > 0:
            result_wait = ray.get(ready) + current_result
            s_ready, s_unready = ray.wait(
                result_wait, num_returns=len(result_wait), timeout=0
            )
            cnt += len(s_ready)
            if cnt == num_requests:
                assert len(s_unready) == 0, "Wrong throughput calculation"
                break
            else:
                current_result = s_unready
        if len(unready) > 0:
            current_router = unready
        else:
            all_ready = True
            assert cnt_all_ready == num_requests, "Wrong throughput calculation"
            # print(f"All fired queries ready: {cnt_all_ready}")
            # current_router = s_unready

    end_time = time.perf_counter()
    duration = end_time - start_time
    qps = num_requests / duration
    return qps


def throughput_calculation_1(serve_handle, data_kwarg, num_requests):
    start_time = time.perf_counter()
    fut = [serve_handle.remote(**data_kwarg) for _ in range(num_requests)]
    ready_refs, _ = ray.wait(fut, num_returns=num_requests)
    ray.wait(ready_refs, num_returns=num_requests)
    end_time = time.perf_counter()
    duration = end_time - start_time
    qps = num_requests / duration
    return qps
