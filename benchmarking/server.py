import asyncio
import socket

import uvicorn
from benchmarking.utils import BytesEncoder
import ray
import json
import time

# # The maximum number of times to retry a request due to actor failure.
# # TODO(edoakes): this should probably be configurable.
# MAX_ACTOR_DEAD_RETRIES = 10


class Response:
    """ASGI compliant response class.
    It is expected to be called in async context and pass along
    `scope, receive, send` as in ASGI spec.
    (Adopted from serve 0.8.2)
    >>> await JSONResponse({"k": "v"})(scope, receive, send)
    """

    def __init__(self, content=None, status_code=200):
        """Construct a JSON HTTP Response.
        Args:
            content (optional): Any JSON serializable object.
            status_code (int, optional): Default status code is 200.
        """
        self.body = self.render(content)
        self.status_code = status_code
        self.raw_headers = [[b"content-type", b"application/json"]]

    def render(self, content):
        if content is None:
            return b""
        if isinstance(content, bytes):
            return content
        return json.dumps(content, cls=BytesEncoder, indent=2).encode()

    async def __call__(self, scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        await send({"type": "http.response.body", "body": self.body})


class HTTPProxy:
    """
    This class should be instantiated and ran by ASGI server.
    (Adopted and modified from serve 0.8.2)
    >>> import uvicorn
    >>> uvicorn.run(HTTPProxy(kv_store_actor_handle, router_handle))
    # blocks forever
    """

    # async def fetch_config_from_master(self):
    #     assert ray.is_initialized()
    #     master = ray.util.get_actor(SERVE_MASTER_NAME)
    #     self.route_table, [self.router_handle
    #                        ] = await master.get_http_proxy_config.remote()
    methods_allowed = ["POST", "GET"]

    def initalize_route_table(self):
        self.route_table = dict()

    def initialize_file(self, filename):
        self.profile_file = open(filename, "w")

    def set_serving_backend(self, serving_backend):
        self.serving_backend = serving_backend

    def register_route(self, route, handle):
        if route not in self.route_table:
            self.route_table[route] = handle
            return True
        return False

    async def handle_lifespan_message(self, scope, receive, send):
        assert scope["type"] == "lifespan"

        message = await receive()
        if message["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})

    async def receive_http_body(self, scope, receive, send):
        body_buffer = []
        more_body = True
        while more_body:
            message = await receive()
            assert message["type"] == "http.request"

            more_body = message["more_body"]
            body_buffer.append(message["body"])

        return b"".join(body_buffer)

    def _make_error_sender(self, scope, receive, send):
        async def sender(error_message, status_code):
            response = Response(error_message, status_code=status_code)
            await response(scope, receive, send)

        return sender

    async def __call__(self, scope, receive, send):
        # NOTE: This implements ASGI protocol specified in
        #       https://asgi.readthedocs.io/en/latest/specs/index.html

        if scope["type"] == "lifespan":
            await self.handle_lifespan_message(scope, receive, send)
            return

        error_sender = self._make_error_sender(scope, receive, send)

        assert (
            self.route_table is not None
        ), "Route table must be set via set_route_table."
        assert scope["type"] == "http"
        current_path = scope["path"]
        if current_path == "/-/routes":
            await Response(list(self.route_table.keys()))(scope, receive, send)
            return

        try:
            handle = self.route_table[current_path]
        except KeyError:
            error_message = (
                "Path {} not found. "
                "Please ping http://.../-/routes for routing table"
            ).format(current_path)
            await error_sender(error_message, 404)
            return

        if scope["method"] not in self.methods_allowed:
            error_message = (
                "Methods {} not allowed. " "Avaiable HTTP methods are {}."
            ).format(scope["method"], self.methods_allowed)
            await error_sender(error_message, 405)
            return

        http_body_bytes = await self.receive_http_body(scope, receive, send)
        if scope["method"] == "POST":
            try:
                kwargs = json.loads(http_body_bytes)
            except Exception:
                error_message = "Got body which is not json decodable"
                await error_sender(error_message, 405)
                return
            try:
                if self.serving_backend == "vanilla":

                    request_sent_time = time.perf_counter()

                    result = await handle.remote(**kwargs)

                    result_received_time = time.perf_counter()
                    self.profile_file.write(
                        json.dumps(
                            {
                                "start": request_sent_time,
                                "end": result_received_time,
                            }
                        )
                    )
                    self.profile_file.write("\n")
                    self.profile_file.flush()

                elif self.serving_backend == "reference":

                    request_sent_time = time.perf_counter()

                    oid = await handle.remote(**kwargs)
                    result = await oid

                    result_received_time = time.perf_counter()
                    self.profile_file.write(
                        json.dumps(
                            {
                                "start": request_sent_time,
                                "end": result_received_time,
                            }
                        )
                    )
                    self.profile_file.write("\n")
                    self.profile_file.flush()
            except Exception:
                error_message = (
                    "Wrong keyword arguments specified in POST "
                    "request for the pipeline"
                )
                await error_sender(error_message, 405)
                return
            await Response(result)(scope, receive, send)
            return
        elif scope["method"] == "GET":
            await Response(http_body_bytes)(scope, receive, send)
            return


@ray.remote(num_cpus=0)
class HTTPProxyActor:
    async def __init__(self, host, port, serving_backend, filename):
        self.app = HTTPProxy()
        self.app.initalize_route_table()
        self.app.set_serving_backend(serving_backend)
        self.app.initialize_file(filename)
        self.host = host
        self.port = port

        # Start running the HTTP server on the event loop.
        asyncio.get_event_loop().create_task(self.run())

    async def run(self):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.set_inheritable(True)

        config = uvicorn.Config(self.app, lifespan="on", access_log=False)
        server = uvicorn.Server(config=config)
        # TODO(edoakes): we need to override install_signal_handlers here
        # because the existing implementation fails if it isn't running in
        # the main thread and uvicorn doesn't expose a way to configure it.
        server.install_signal_handlers = lambda: None
        await server.serve(sockets=[sock])

    async def register_route(self, route, handle):
        return self.app.register_route(route, handle)
