from benchmarking import serve_benchmark
from benchmarking.serve_benchmark.context import TaskContext
from benchmarking.serve_benchmark.exceptions import RayServeException
from benchmarking.serve_benchmark.constants import DEFAULT_HTTP_ADDRESS
from benchmarking.serve_benchmark.request_params import RequestMetadata


class RayServeHandle:
    """A handle to a service endpoint.

    Invoking this endpoint with .remote is equivalent to pinging
    an HTTP endpoint.

    Example:
       >>> handle = serve_benchmark.get_handle("my_endpoint")
       >>> handle
       RayServeHandle(
            Endpoint="my_endpoint",
            URL="...",
            Traffic=...
       )
       >>> handle.remote(my_request_content)
       ObjectID(...)
       >>> ray.get(handle.remote(...))
       # result
       >>> ray.get(handle.remote(let_it_crash_request))
       # raises RayTaskError Exception
    """

    def __init__(
        self,
        router_handle,
        router_name,
        endpoint_name,
        relative_slo_ms=None,
        absolute_slo_ms=None,
        method_name=None,
        tracing_metadata=None,
    ):
        self.router_handle = router_handle
        self.router_name = router_name
        self.endpoint_name = endpoint_name
        assert relative_slo_ms is None or absolute_slo_ms is None, (
            "Can't specify both " "relative and absolute " "slo's together!"
        )
        self.relative_slo_ms = self._check_slo_ms(relative_slo_ms)
        self.absolute_slo_ms = self._check_slo_ms(absolute_slo_ms)
        self.method_name = method_name
        self.tracing_metadata = tracing_metadata or {
            "router_name": self.router_name
        }

    def _check_slo_ms(self, slo_value):
        if slo_value is not None:
            try:
                slo_value = float(slo_value)
                if slo_value < 0:
                    raise ValueError(
                        "Request SLO must be positive, it is {}".format(
                            slo_value
                        )
                    )
                return slo_value
            except ValueError as e:
                raise RayServeException(str(e))
        return None

    def remote(self, *args, **kwargs):
        if len(args) != 0:
            raise RayServeException(
                "handle.remote must be invoked with keyword arguments."
            )

        return self.router_handle.enqueue_request.remote(
            self._make_metadata(), **kwargs
        )

    def _make_metadata(self):
        method_name = self.method_name
        if method_name is None:
            method_name = "__call__"

        # create RequestMetadata instance
        request_in_object = RequestMetadata(
            self.endpoint_name,
            TaskContext.Python,
            self.relative_slo_ms,
            self.absolute_slo_ms,
            call_method=method_name,
            tracing_metadata=self.tracing_metadata,
        )
        return request_in_object

    def enqueue_batch(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, list), f"Input to argument {k} is not a list"

        return self.router_handle.enqueue_batch.remote(
            self._make_metadata(), **kwargs
        )

    def options(
        self,
        method_name=None,
        relative_slo_ms=None,
        absolute_slo_ms=None,
        tracing_metadata=None,
    ):
        # If both the slo's are None then then we use a high default
        # value so other queries can be prioritize and put in front of these
        # queries.
        assert not all([absolute_slo_ms, relative_slo_ms]), (
            "Can't specify both " "relative and absolute " "slo's together!"
        )

        # Don't override existing method
        if method_name is None and self.method_name is not None:
            method_name = self.method_name

        return RayServeHandle(
            self.router_handle,
            self.router_name,
            self.endpoint_name,
            relative_slo_ms,
            absolute_slo_ms,
            method_name=method_name,
            tracing_metadata={
                **tracing_metadata,
                "router_name": self.router_name,
            }
            if tracing_metadata is not None
            else self.tracing_metadata,
        )

    def get_traffic_policy(self):
        policy_table = serve_benchmark.api._get_global_state().policy_table
        all_services = policy_table.list_traffic_policy()
        return all_services[self.endpoint_name]

    def get_http_endpoint(self):
        return DEFAULT_HTTP_ADDRESS

    def _ensure_backend_unique(self, backend_tag=None):
        traffic_policy = self.get_traffic_policy()
        if backend_tag is None:
            assert len(traffic_policy) == 1, (
                "Multiple backends detected. "
                "Please pass in backend_tag=... argument to specify backend."
            )
            backends = set(traffic_policy.keys())
            return backends.pop()
        else:
            assert (
                backend_tag in traffic_policy
            ), "Backend {} not found in avaiable backends: {}.".format(
                backend_tag, list(traffic_policy.keys())
            )
            return backend_tag

    def scale(self, new_num_replicas, backend_tag=None):
        with serve_benchmark.using_router(self.endpoint_name):
            backend_tag = self._ensure_backend_unique(backend_tag)
            config = serve_benchmark.get_backend_config(backend_tag)
            config.num_replicas = new_num_replicas
            serve_benchmark.set_backend_config(backend_tag, config)

    def set_max_batch_size(self, new_max_batch_size, backend_tag=None):
        with serve_benchmark.using_router(self.endpoint_name):
            backend_tag = self._ensure_backend_unique(backend_tag)
            config = serve_benchmark.get_backend_config(backend_tag)
            config.max_batch_size = new_max_batch_size
            serve_benchmark.set_backend_config(backend_tag, config)

    def set_backend_config(self, backend_config, backend_tag=None):
        with serve_benchmark.using_router(self.endpoint_name):
            backend_tag = self._ensure_backend_unique(backend_tag)
            serve_benchmark.set_backend_config(backend_tag, backend_config)

    def get_backend_config(self, backend_tag=None):
        with serve_benchmark.using_router(self.endpoint_name):
            backend_tag = self._ensure_backend_unique(backend_tag)
            return serve_benchmark.get_backend_config(backend_tag)

    def __repr__(self):
        return """
RayServeHandle(
    Endpoint="{endpoint_name}",
    URL="{http_endpoint}/{endpoint_name}",
    Traffic={traffic_policy}
)
""".format(
            endpoint_name=self.endpoint_name,
            http_endpoint=self.get_http_endpoint(),
            traffic_policy=self.get_traffic_policy(),
        )

    # TODO(simon): a convenience function that dumps equivalent requests
    # code for a given call.
