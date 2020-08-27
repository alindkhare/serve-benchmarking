import os
import subprocess
import tempfile

import ray
from benchmarking import serve_benchmark


def test_new_driver(serve_instance):
    script = """
import ray
ray.init(address="{}")

from benchmarking import serve_benchmark
serve_benchmark.init()

@serve_benchmark.route("/driver")
def driver(flask_request):
    return "OK!"
""".format(
        ray.worker._global_node._redis_address
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        path = f.name
        f.write(script)

    proc = subprocess.Popen(["python", path])
    return_code = proc.wait(timeout=10)
    assert return_code == 0

    handle = serve_benchmark.get_handle("driver")
    assert ray.get(handle.remote()) == "OK!"

    os.remove(path)
