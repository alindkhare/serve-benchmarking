from collections import deque
import asyncio
import uvloop
import pickle
import copy
from typing import DefaultDict, List
from benchmarking.serve_benchmark import context as serve_context


def _make_future_unwrapper(
    client_futures: List[asyncio.Future], host_future: asyncio.Future
):
    """Distribute the result of host_future to each of client_future"""
    for client_future in client_futures:
        # Keep a reference to host future so the host future won't get
        # garbage collected.
        client_future.host_ref = host_future

    def unwrap_future(_):
        result = host_future.result()

        if isinstance(result, list):
            for client_future, result_item in zip(client_futures, result):
                client_future.set_result(result_item)
        else:  # Result is an exception.
            for client_future in client_futures:
                client_future.set_exception(result)

    return unwrap_future


class NewQuery:
    def __init__(self, request_args, request_kwargs, async_future=None):
        self.request_args = request_args
        self.request_kwargs = request_kwargs
        self.async_future = async_future
        self.backend_worker = None
        self.call_method = "__call__"
        self.request_context = serve_context.TaskContext.Python
        self.first = None

    def on_assigned(self, worker, first=True):
        self.backend_worker = worker
        self.first = first

    def done(self, router):
        assert self.backend_worker is not None, "error in router"
        if self.first:
            router.dequeue_request(self.backend_worker)

    def ray_serialize(self):
        # NOTE: this method is needed because Query need to be serialized and
        # sent to the replica worker. However, after we send the query to
        # replica worker the async_future is still needed to retrieve the final
        # result. Therefore we need a way to pass the information to replica
        # worker without removing async_future.
        clone = copy.copy(self).__dict__
        clone.pop("async_future")
        clone.pop("backend_worker")
        clone.pop("call_method")
        clone.pop("request_context")
        clone.pop("first")
        return pickle.dumps(clone, protocol=4)

    @staticmethod
    def ray_deserialize(value):
        kwargs = pickle.loads(value)
        return NewQuery(**kwargs)


class NewQueues:
    def __init__(self):
        self._service_queues = deque()
        self._worker_queues = deque()
        self.config_dict = None
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    async def enqueue_request(
        self, request_meta, request_args=list(), request_kwargs=dict()
    ):
        query = NewQuery(
            request_args,
            request_kwargs,
            asyncio.get_event_loop().create_future(),
        )
        self._service_queues.append(query)
        await self.flush()
        result = await query.async_future
        query.done(self)
        return result

    async def dequeue_request(self, replica_handle):
        self._worker_queues.append(replica_handle)
        await self.flush()

    async def set_backend_config(self, backend, config_dict):
        self.config_dict = config_dict

    async def link(self, service, backend):
        pass

    async def set_traffic(self, service, traffic_dict):
        await self.flush()

    async def flush(self):
        """In the default case, flush calls ._flush.

        When this class is a Ray actor, .flush can be scheduled as a remote
        method invocation.
        """
        while len(self._service_queues) and len(self._worker_queues):
            worker = self._worker_queues.pop()
            max_batch_size = self.config_dict["max_batch_size"]
            if max_batch_size is None:  # No batching
                request = self._service_queues.pop()
                future = worker._ray_serve_call.remote(request).as_future()
                # chaining satisfies request.async_future with future result.
                asyncio.futures._chain_future(future, request.async_future)
                request.on_assigned(worker)
            else:
                real_batch_size = min(len(self._service_queues), max_batch_size)
                requests = [
                    self._service_queues.pop() for _ in range(real_batch_size)
                ]

                future = worker._ray_serve_call.remote(requests).as_future()

                complete_all_future = _make_future_unwrapper(
                    client_futures=[req.async_future for req in requests],
                    host_future=future,
                )

                [
                    q.on_assigned(worker, first=(i == 0))
                    for i, q in enumerate(requests)
                ]
                future.add_done_callback(complete_all_future)
