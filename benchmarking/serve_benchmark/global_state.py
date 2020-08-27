from collections import defaultdict
from contextlib import contextmanager
import ray
from serve_benchmark.constants import (
    BOOTSTRAP_KV_STORE_CONN_KEY,
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    SERVE_MASTER_NAME,
    ASYNC_CONCURRENCY,
)
from serve_benchmark.kv_store_service import (
    BackendTable,
    RoutingTable,
    TrafficPolicyTable,
)

from serve_benchmark.policy import RoutePolicy
from serve_benchmark.http_proxy import HTTPProxyActor
from serve_benchmark.utils import ray_pin_to_core


def start_initial_state(kv_store_connector):
    master_handle = ServeMaster.remote()
    ray.util.register_actor(SERVE_MASTER_NAME, master_handle)

    ray.get(
        master_handle.store_bootstrap_state.remote(
            BOOTSTRAP_KV_STORE_CONN_KEY, kv_store_connector
        )
    )
    return master_handle


@ray.remote(num_cpus=1)
class ServeMaster:
    """Initialize and store all actor handles.

    Note:
        This actor is necessary because ray will destory actors when the
        original actor handle goes out of scope (when driver exit). Therefore
        we need to initialize and store actor handles in a seperate actor.
    """

    def __init__(self):
        self.tag_to_actor_handles = dict()

        self.bootstrap_state = dict()

        ray_pin_to_core()

    def start_actor(
        self, actor_cls, tag, init_args=(), init_kwargs={}, is_asyncio=False
    ):
        """Start an actor and add it to the nursery"""
        # Avoid double initialization
        if tag in self.tag_to_actor_handles.keys():
            return [self.tag_to_actor_handles[tag]]

        max_concurrency = ASYNC_CONCURRENCY if is_asyncio else None
        handle = actor_cls.options(max_concurrency=max_concurrency).remote(
            *init_args, **init_kwargs
        )
        self.tag_to_actor_handles[tag] = handle
        return [handle]

    def start_actor_with_creator(self, creator, kwargs, tag):
        """
        Args:
            creator (Callable[Dict]): a closure that should return
                a newly created actor handle when called with kwargs.
                The kwargs input is passed to `ActorCls_remote` method.
        """
        handle = creator(kwargs)
        self.tag_to_actor_handles[tag] = handle
        return [handle]

    def get_all_handles(self):
        return self.tag_to_actor_handles

    def get_handle(self, actor_tag):
        return [self.tag_to_actor_handles[actor_tag]]

    def remove_handle(self, actor_tag):
        if actor_tag in self.tag_to_actor_handles.keys():
            self.tag_to_actor_handles.pop(actor_tag)

    def store_bootstrap_state(self, key, value):
        self.bootstrap_state[key] = value

    def get_bootstrap_state_dict(self):
        return self.bootstrap_state

    def get_all_workers(self):
        return [
            handle
            for tag, handle in self.tag_to_actor_handles.items()
            if tag.startswith("worker:")
        ]


class GlobalState:
    """Encapsulate all global state in the serving system.

    The information is fetch lazily from
        1. A collection of namespaced key value stores
        2. A actor supervisor service
    """

    def __init__(self, master_actor_handle=None, start_server=True):
        # Get actor nursery handle
        if master_actor_handle is None:
            master_actor_handle = ray.util.get_actor(SERVE_MASTER_NAME)
        self.master_actor_handle = master_actor_handle
        self.start_server = start_server

        # Connect to all the table
        bootstrap_config = ray.get(
            self.master_actor_handle.get_bootstrap_state_dict.remote()
        )
        kv_store_connector = bootstrap_config[BOOTSTRAP_KV_STORE_CONN_KEY]
        self.route_table = RoutingTable(kv_store_connector)
        self.backend_table = BackendTable(kv_store_connector)
        self.policy_table = TrafficPolicyTable(kv_store_connector)

        self.refresh_actor_handle_cache()

        self.router_name = "default"

    @contextmanager
    def using_router(self, name="default"):
        old_name, self.router_name = self.router_name, name
        yield
        self.router_name = old_name

    def refresh_actor_handle_cache(self):
        self.actor_handle_cache = ray.get(
            self.master_actor_handle.get_all_handles.remote()
        )

    def init_or_get_http_proxy(
        self, host=DEFAULT_HTTP_HOST, port=DEFAULT_HTTP_PORT
    ):
        if "http_proxy" not in self.actor_handle_cache:
            [handle] = ray.get(
                self.master_actor_handle.start_actor.remote(
                    HTTPProxyActor, tag="http_proxy"
                )
            )

            handle.run.remote(host=host, port=port)
            self.refresh_actor_handle_cache()
        return self.actor_handle_cache["http_proxy"]

    def init_or_get_router(
        self, queueing_policy=RoutePolicy.Random, policy_kwargs={}
    ):
        assert self.router_name != "default", (
            "Default router detected, did you forgot to use "
            "`with serve_benchmark.using_router(name):` to assign router "
            "per endpiont?"
        )

        queue_actor_tag = f"queue::{self.router_name}::" + queueing_policy.name
        if queue_actor_tag not in self.actor_handle_cache:
            [handle] = ray.get(
                self.master_actor_handle.start_actor.remote(
                    queueing_policy.value,
                    init_kwargs=policy_kwargs,
                    tag=queue_actor_tag,
                    is_asyncio=True,
                )
            )
            # handle.register_self_handle.remote(handle)
            self.refresh_actor_handle_cache()
        return self.actor_handle_cache[queue_actor_tag]

    def _iter_all_routers(self):
        for k, handle in self.actor_handle_cache.items():
            if k.startswith("queue::"):
                yield handle

    def get_trace(self):
        all_traces = []
        all_metadata = defaultdict(dict)

        for router in self._iter_all_routers():
            sink, metadata = ray.get(router.get_trace.remote())
            all_traces.extend(sink)
            all_metadata.update(metadata)

        workers = ray.get(self.master_actor_handle.get_all_workers.remote())
        for worker in workers:
            sink, metadata = ray.get(worker._ray_serve_get_trace.remote())
            all_traces.extend(sink)
            all_metadata.update(metadata)

        return {"traces": all_traces, "metadata": all_metadata}

    def clear_trace(self):
        for router in self._iter_all_routers():
            ray.get(router.clear_trace.remote())

        workers = ray.get(self.master_actor_handle.get_all_workers.remote())
        for worker in workers:
            ray.get(worker._ray_serve_clear_trace.remote())
