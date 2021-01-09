import asyncio
import uvloop

import copy
from collections import defaultdict
from typing import DefaultDict, List

import ray
import ray.cloudpickle as pickle
import blist

from benchmarking.serve_benchmark.utils import logger, tracer


class Query:
    def __init__(
        self,
        request_id,
        request_args,
        request_kwargs,
        request_context,
        request_slo_ms,
        call_method="__call__",
        async_future=None,
        router_name=None,
    ):
        self.request_id = request_id
        self.request_args = request_args
        self.request_kwargs = request_kwargs
        self.request_context = request_context
        self.router_name = router_name

        self.async_future = async_future

        # Service level objective in milliseconds. This is expected to be the
        # absolute time since unix epoch.
        self.request_slo_ms = request_slo_ms

        self.call_method = call_method

    def on_enqueue(self, endpoint, metadata=None):
        tracer.add(
            self.request_id,
            "router_enqueue",
            router_name=metadata.get("router_name", None)
            if metadata is not None
            else None,
        )
        # tracer.add_metadata(self.request_id, endpoint=endpoint)
        if metadata:
            assert isinstance(metadata, dict)
            tracer.add_metadata(self.request_id, **metadata)

    def on_assigned(self, backend_name, worker, batch_id, idx_in_batch):
        self.backend_name = backend_name
        self.backend_worker = worker
        self.batch_id = batch_id
        self.idx_in_batch = idx_in_batch

        tracer.add(
            self.request_id, "router_dequeue", router_name=self.router_name
        )
        tracer.add_metadata(
            self.request_id,
            backend=backend_name,
            batch_id=batch_id,
            router_name=self.router_name,
        )

    def on_worker_start(self):
        tracer.add(
            self.request_id, "worker_start", router_name=self.router_name
        )

    def on_worker_done(self):
        tracer.add(self.request_id, "worker_done", router_name=self.router_name)

    async def on_complete(self, router):
        tracer.add(
            self.request_id, "router_recv_result", router_name=self.router_name
        )
        if self.batch_id is None:
            await router.dequeue_request(self.backend_name, self.backend_worker)
        elif self.idx_in_batch == 0:
            await router.dequeue_request(self.backend_name, self.backend_worker)

    def ray_serialize(self):
        # NOTE: this method is needed because Query need to be serialized and
        # sent to the replica worker. However, after we send the query to
        # replica worker the async_future is still needed to retrieve the final
        # result. Therefore we need a way to pass the information to replica
        # worker without removing async_future.
        clone = copy.copy(self).__dict__
        clone.pop("async_future")
        return pickle.dumps(clone, protocol=5)

    @staticmethod
    def ray_deserialize(value):
        kwargs = pickle.loads(value)
        return Query(**kwargs)

    # adding comparator fn for maintaining an
    # ascending order sorted list w.r.t request_slo_ms
    def __lt__(self, other):
        return self.request_slo_ms < other.request_slo_ms


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


class CentralizedQueues:
    """A router that routes request to available workers.

    Router aceepts each request from the `enqueue_request` method and enqueues
    it. It also accepts worker request to work (called work_intention in code)
    from workers via the `dequeue_request` method. The traffic policy is used
    to match requests with their corresponding workers.

    Behavior:
        >>> # psuedo-code
        >>> queue = CentralizedQueues()
        >>> queue.enqueue_request(
            "service-name", request_args, request_kwargs, request_context)
        # nothing happens, request is queued.
        # returns result ObjectID, which will contains the final result
        >>> queue.dequeue_request('backend-1', replica_handle)
        # nothing happens, work intention is queued.
        # return work ObjectID, which will contains the future request payload
        >>> queue.link('service-name', 'backend-1')
        # here the enqueue_requester is matched with replica, request
        # data is put into work ObjectID, and the replica processes the request
        # and store the result into result ObjectID

    Traffic policy splits the traffic among different replicas
    probabilistically:

    1. When all backends are ready to receive traffic, we will randomly
       choose a backend based on the weights assigned by the traffic policy
       dictionary.

    2. When more than 1 but not all backends are ready, we will normalize the
       weights of the ready backends to 1 and choose a backend via sampling.

    3. When there is only 1 backend ready, we will only use that backend.
    """

    def __init__(self):
        # Note: Several queues are used in the router
        # - When a request come in, it's placed inside its corresponding
        #   service_queue.
        # - The service_queue is dequed during flush operation, which moves
        #   the queries to backend buffer_queue. Here we match a request
        #   for a service to a backend given some policy.
        # - The worker_queue is used to collect idle actor handle. These
        #   handles are dequed during the second stage of flush operation,
        #   which assign queries in buffer_queue to actor handle.

        # -- Queues -- #

        # service_name -> request queue
        self.service_queues: DefaultDict[asyncio.Queue[Query]] = defaultdict(
            asyncio.Queue
        )
        # backend_name -> worker request queue
        self.worker_queues: DefaultDict[
            asyncio.Queue[ray.actor.ActorHandle]
        ] = defaultdict(asyncio.Queue)
        # backend_name -> worker payload queue
        self.buffer_queues = defaultdict(blist.sortedlist)

        # -- Metadata -- #

        # service_name -> traffic_policy
        self.traffic = defaultdict(dict)
        # backend_name -> backend_config
        self.backend_info = dict()

        # -- Synchronization -- #

        # This lock guarantee that only one flush operation can happen at a
        # time. Without the lock, multiple flush operation can pop from the
        # same buffer_queue and worker_queue and create deadlock. For example,
        # an operation holding the only query and the other flush operation
        # holding the only idle replica. Additionally, allowing only one flush
        # operation at a time simplifies design overhead for custom queuing and
        # batching polcies.
        self.flush_lock = asyncio.Lock()

        self.request_id_counter = 0
        self.batch_id_counter = 0
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        # ray_pin_to_core()

    def is_ready(self):
        return True

    def _make_query(self, request_meta, args, kwargs):
        self.request_id_counter += 1

        # check if the slo specified is directly the
        # wall clock time
        if request_meta.absolute_slo_ms is not None:
            request_slo_ms = request_meta.absolute_slo_ms
        else:
            request_slo_ms = request_meta.adjust_relative_slo_ms()

        request_context = request_meta.request_context
        query = Query(
            self.request_id_counter,
            args,
            kwargs,
            request_context,
            request_slo_ms,
            call_method=request_meta.call_method,
            async_future=asyncio.get_event_loop().create_future(),
            router_name=request_meta.tracing_metadata.get("router_name", None)
            if request_meta.tracing_metadata is not None
            else None,
        )
        return query

    async def enqueue_request(
        self, request_meta, request_args=list(), request_kwargs=dict()
    ):
        service = request_meta.service
        logger.debug("Received a request for service {}".format(service))

        query = self._make_query(request_meta, request_args, request_kwargs)
        query.on_enqueue(service, metadata=request_meta.tracing_metadata)

        await self.service_queues[service].put(query)
        await self.flush()

        # Note: a future change can be to directly return the ObjectID from
        # replica task submission
        result = await query.async_future

        asyncio.get_event_loop().create_task(query.on_complete(self))

        return result

    async def enqueue_batch(self, request_meta, **input_kwargs):
        # Check and find endpoint
        endpoint_name = request_meta.service
        backends = list(self.traffic[endpoint_name].keys())
        assert len(backends) == 1, "Multiple backend detected"
        backend_name = backends[0]

        # Check batch size for the backend
        backend_batch_size = self.backend_info[backend_name]["max_batch_size"]
        input_batch_sizes = set(map(len, input_kwargs.values()))
        assert (
            len(input_batch_sizes) == 1
        ), "input args have different batch size or no batch input args"
        input_batch_size = input_batch_sizes.pop()
        assert input_batch_size == backend_batch_size, (
            "Enqueue batch requires the same batch size as max_batch_size, "
            "which is {}".foramt(backend_batch_size)
        )

        # Make and enqueue all queries
        queries = []
        for i in range(input_batch_size):
            kwargs = {k: v[i] for k, v in input_kwargs.items()}
            query = self._make_query(request_meta, args=[], kwargs=kwargs)
            queries.append(query)
            await self.service_queues[endpoint_name].put(query)

        # Flush at once
        await self.flush()

        # Make sure they end up on the same batch
        assert (
            len(set(q.batch_id for q in queries)) == 1
        ), "Queries end up with different batch size! This is unexpected."

        # Wait on the output
        results = await asyncio.gather(*[q.async_future for q in queries])
        for q in queries:
            asyncio.get_event_loop().create_task(q.on_complete(self))
        return results

    async def dequeue_request(self, backend, replica_handle):
        logger.debug(
            "Received a dequeue request for backend {}".format(backend)
        )
        await self.worker_queues[backend].put(replica_handle)
        await self.flush()

    async def remove_and_destory_replica(self, backend, replica_handle):
        # We need this lock because we modify worker_queue here.
        async with self.flush_lock:
            old_queue = self.worker_queues[backend]
            new_queue = asyncio.Queue()
            target_id = replica_handle._actor_id

            while not old_queue.empty():
                replica_handle = await old_queue.get()
                if replica_handle._actor_id != target_id:
                    await new_queue.put(replica_handle)

            self.worker_queues[backend] = new_queue
            # TODO: consider await this with timeout, or use ray_kill
            replica_handle.__ray_terminate__.remote()

    async def link(self, service, backend):
        logger.debug("Link %s with %s", service, backend)
        await self.set_traffic(service, {backend: 1.0})

    async def set_traffic(self, service, traffic_dict):
        logger.debug(
            "Setting traffic for service %s to %s", service, traffic_dict
        )
        self.traffic[service] = traffic_dict
        await self.flush()

    async def set_backend_config(self, backend, config_dict):
        logger.debug(
            "Setting backend config for "
            "backend {} to {}".format(backend, config_dict)
        )
        self.backend_info[backend] = config_dict

    async def flush(self):
        """In the default case, flush calls ._flush.

        When this class is a Ray actor, .flush can be scheduled as a remote
        method invocation.
        """
        async with self.flush_lock:
            await self._flush_service_queues()
            await self._flush_buffer_queues()

    def _get_available_backends(self, service):
        backends_in_policy = set(self.traffic[service].keys())
        available_workers = {
            backend
            for backend, queues in self.worker_queues.items()
            if queues.qsize() > 0
        }
        return list(backends_in_policy.intersection(available_workers))

    async def _flush_service_queues(self):
        # perform traffic splitting for requests
        for service, queue in self.service_queues.items():
            # while there are incoming requests and there are backends
            while queue.qsize() and len(self.traffic[service]):
                backend_names = list(self.traffic[service].keys())
                assert len(backend_names) == 1, "Expect only one backend"
                chosen_backend = backend_names[0]
                request = await queue.get()
                self.buffer_queues[chosen_backend].add(request)

    # flushes the buffer queue and assigns work to workers
    async def _flush_buffer_queues(self):
        for service in self.traffic.keys():
            ready_backends = self._get_available_backends(service)
            for backend in ready_backends:
                # no work available
                if len(self.buffer_queues[backend]) == 0:
                    continue

                buffer_queue = self.buffer_queues[backend]
                worker_queue = self.worker_queues[backend]

                logger.debug(
                    "Assigning queries for backend {} with buffer "
                    "queue size {} and worker queue size {}".format(
                        backend, len(buffer_queue), worker_queue.qsize()
                    )
                )

                max_batch_size = None
                if backend in self.backend_info:
                    max_batch_size = self.backend_info[backend][
                        "max_batch_size"
                    ]

                await self._assign_query_to_worker(
                    backend, buffer_queue, worker_queue, max_batch_size
                )

    async def _assign_query_to_worker(
        self, backend_name, buffer_queue, worker_queue, max_batch_size=None
    ):

        while len(buffer_queue) and worker_queue.qsize():
            worker = await worker_queue.get()
            if max_batch_size is None:  # No batching
                request = buffer_queue.pop(0)
                future = worker._ray_serve_call.remote(request).as_future()
                # chaining satisfies request.async_future with future result.
                asyncio.futures._chain_future(future, request.async_future)
                request.on_assigned(
                    backend_name, worker, batch_id=None, idx_in_batch=None
                )
            else:
                real_batch_size = min(len(buffer_queue), max_batch_size)
                requests = [buffer_queue.pop(0) for _ in range(real_batch_size)]

                # split requests by method type
                requests_group = defaultdict(list)
                for request in requests:
                    requests_group[request.call_method].append(request)

                for group in requests_group.values():
                    future = worker._ray_serve_call.remote(group).as_future()

                    complete_all_future = _make_future_unwrapper(
                        client_futures=[req.async_future for req in group],
                        host_future=future,
                    )

                    [
                        q.on_assigned(
                            backend_name,
                            worker,
                            batch_id=self.batch_id_counter,
                            idx_in_batch=i,
                        )
                        for i, q in enumerate(group)
                    ]
                    self.batch_id_counter += 1

                    future.add_done_callback(complete_all_future)

    def get_trace(self):
        return tracer.sink, tracer.metadata

    def clear_trace(self):
        tracer.clear()
