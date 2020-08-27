import traceback
import inspect

import ray
from benchmarking.serve_benchmark import context as serve_context
from benchmarking.serve_benchmark.context import FakeFlaskRequest
from collections import defaultdict
from benchmarking.serve_benchmark.utils import (
    parse_request_item,
    ray_pin_to_core,
    tracer,
)
from benchmarking.serve_benchmark.exceptions import RayServeException


class TaskRunner:
    """A simple class that runs a function.

    The purpose of this class is to model what the most basic actor could be.
    That is, a ray serve_benchmark actor should implement the TaskRunner interface.
    """

    def __init__(self, func_to_run):
        self.func = func_to_run

        # This parameter let argument inspection work with inner function.
        self.__wrapped__ = func_to_run

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def wrap_to_ray_error(exception):
    """Utility method that catch and seal exceptions in execution"""
    try:
        # Raise and catch so we can access traceback.format_exc()
        raise exception
    except Exception as e:
        traceback_str = ray.utils.format_error_message(traceback.format_exc())
        return ray.exceptions.RayTaskError(str(e), traceback_str, e.__class__)


class RayServeMixin:
    """This mixin class adds the functionality to fetch from router queues.

    Warning:
        It assumes the main execution method is `__call__` of the user defined
        class. This means that serve_benchmark will call `your_instance.__call__` when
        each request comes in. This behavior will be fixed in the future to
        allow assigning artibrary methods.

    Example:
        >>> # Use ray.remote(num_cpus=1) decorator and RayServeMixin
        >>> # to make MyClass servable.
        >>> @ray.remote(num_cpus=1)
            class RayServeActor(RayServeMixin, MyClass):
                pass
    """

    _ray_serve_self_handle = None
    _ray_serve_router_handle = None
    _ray_serve_setup_completed = False
    _ray_serve_dequeue_requester_name = None

    # Work token can be unfullfilled from last iteration.
    # This cache will be used to determine whether or not we should
    # work on the same task as previous iteration or we are ready to
    # move on.
    _ray_serve_cached_work_token = None

    def _ray_serve_setup(self, my_name, router_handle, my_handle):
        self._ray_serve_dequeue_requester_name = my_name
        self._ray_serve_router_handle = router_handle
        self._ray_serve_self_handle = my_handle
        self._ray_serve_setup_completed = True

        ray_pin_to_core()

    def _ray_serve_fetch(self):
        assert self._ray_serve_setup_completed

        self._ray_serve_router_handle.dequeue_request.remote(
            self._ray_serve_dequeue_requester_name, self._ray_serve_self_handle
        )

    def _ray_serve_get_runner_method(self, request_item):
        method_name = request_item.call_method
        if not hasattr(self, method_name):
            raise RayServeException(
                "Backend doesn't have method {} "
                "which is specified in the request. "
                "The avaiable methods are {}".format(method_name, dir(self))
            )
        return getattr(self, method_name)

    def _ray_serve_count_num_positional(self, f):
        # NOTE:
        # In the case of simple functions, not actors, the f will be
        # a TaskRunner.__call__. What we really want here is the wrapped
        # functionso inspect.signature will figure out the underlying f.
        if hasattr(self, "__wrapped__"):
            f = self.__wrapped__

        signature = inspect.signature(f)
        counter = 0
        for param in signature.parameters.values():
            if (
                param.kind == param.POSITIONAL_OR_KEYWORD
                and param.default is param.empty
            ):
                counter += 1
        return counter

    def invoke_single(self, request_item):
        args, kwargs, is_web_context = parse_request_item(request_item)
        serve_context.web = is_web_context

        method_to_call = self._ray_serve_get_runner_method(request_item)
        args = (
            args if self._ray_serve_count_num_positional(method_to_call) else []
        )
        try:
            result = method_to_call(*args, **kwargs)
        except Exception as e:
            result = wrap_to_ray_error(e)

        return result

    def invoke_batch(self, request_item_list):
        # TODO(alind) : create no-http services. The enqueues
        # from such services will always be TaskContext.Python.

        # Assumption : all the requests in a bacth
        # have same serve_benchmark context.

        # For batching kwargs are modified as follows -
        # kwargs [Python Context] : key,val
        # kwargs_list             : key, [val1,val2, ... , valn]
        # or
        # args[Web Context]       : val
        # args_list               : [val1,val2, ...... , valn]
        # where n (current batch size) <= max_batch_size of a backend

        arg_list = []
        kwargs_list = defaultdict(list)
        context_flags = set()
        batch_size = len(request_item_list)
        call_methods = set()

        for item in request_item_list:
            args, kwargs, is_web_context = parse_request_item(item)
            context_flags.add(is_web_context)

            call_method = self._ray_serve_get_runner_method(item)
            call_methods.add(call_method)

            if is_web_context:
                # Python context only have kwargs
                flask_request = args[0]
                arg_list.append(flask_request)
            else:
                # Web context only have one positional argument
                for k, v in kwargs.items():
                    kwargs_list[k].append(v)

                # Set the flask request as a list to conform
                # with batching semantics: when in batching
                # mode, each argument it turned into list.
                if self._ray_serve_count_num_positional(call_method):
                    arg_list.append(FakeFlaskRequest())

        try:
            # check mixing of query context
            # unified context needed
            if len(context_flags) != 1:
                raise RayServeException(
                    "Batched queries contain mixed context. Please only send "
                    "the same type of requests in batching mode."
                )
            serve_context.web = context_flags.pop()

            if len(call_methods) != 1:
                raise RayServeException(
                    "Queries contain mixed calling methods. Please only send "
                    "the same type of requests in batching mode."
                )
            call_method = call_methods.pop()

            serve_context.batch_size = batch_size
            # Flask requests are passed to __call__ as a list
            arg_list = [arg_list]

            result_list = call_method(*arg_list, **kwargs_list)

            if (not isinstance(result_list, list)) or (
                len(result_list) != batch_size
            ):
                raise RayServeException(
                    "__call__ function "
                    "doesn't preserve batch-size. "
                    "Please return a list of result "
                    "with length equals to the batch "
                    "size."
                )
            return result_list
        except Exception as e:
            wrapped_exception = wrap_to_ray_error(e)
            return [wrapped_exception for _ in range(batch_size)]

    def _ray_serve_call(self, request):
        # check if work_item is a list or not
        # if it is list: then batching supported
        if not isinstance(request, list):
            request.on_worker_start()
            result = self.invoke_single(request)
            request.on_worker_done()
        else:
            [r.on_worker_start() for r in request]
            result = self.invoke_batch(request)
            [r.on_worker_done() for r in request]

        # re-assign to default values
        serve_context.web = False
        serve_context.batch_size = None

        return result

    def _ray_serve_get_trace(self):
        return tracer.sink, tracer.metadata

    def _ray_serve_clear_trace(self):
        tracer.clear()


class TaskRunnerBackend(TaskRunner, RayServeMixin):
    """A simple function serving backend

    Note that this is not yet an actor. To make it an actor:

    >>> @ray.remote(num_cpus=1)
        class TaskRunnerActor(TaskRunnerBackend):
            pass

    Note:
    This class is not used in the actual ray serve_benchmark system. It exists
    for documentation purpose.
    """


@ray.remote(num_cpus=1)
class TaskRunnerActor(TaskRunnerBackend):
    pass
