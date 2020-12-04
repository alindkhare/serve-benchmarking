from benchmarking import Experiment
from benchmarking import serve_reference
from benchmarking.utils import throughput_calculation
import torch
import ray
import pandas as pd
import time
import numpy as np
import click
from pprint import pprint
from itertools import product
from benchmarking.utils import throughput_calculation
from torch.autograd import Variable
from PIL import Image
import base64
import torch
import torchvision.transforms as transforms
from torchvision import models
import io
from typing import Any


class PredictModelPytorch:
    """
    Standard pytorch prediction functionality
    - gets a preprocessed tensor
    - predicts it's class
    """

    def __init__(
        self, transform: Any, model_name: str, is_cuda: bool = False
    ) -> None:
        self.transform = transform
        self.model = models.__dict__[model_name](pretrained=True)
        self.is_cuda = is_cuda
        if is_cuda:
            self.model = self.model.cuda()

    @serve_reference.accept_batch
    def __call__(self, _, data: list) -> list:
        data_list = list()
        for img in data:
            data = Image.open(io.BytesIO(base64.b64decode(img)))
            if data.mode != "RGB":
                data = data.convert("RGB")
            data = self.transform(data)
            data_list.append(data)
        data = data_list
        data = torch.stack(data)
        data = Variable(data)
        if self.is_cuda:
            data = data.cuda()
        outputs = self.model(data)
        _, predicted = outputs.max(1)
        return predicted.cpu().numpy().tolist()


def main():
    TAG = "Resnet18"
    for num_replica in range(1, 9):
        # initialize serve
        serve_reference.init(start_server=False)

        serve_handle = None
        with serve_reference.using_router(TAG):
            serve_reference.create_endpoint(TAG)
            config = serve_reference.BackendConfig(
                max_batch_size=8, num_replicas=num_replica, num_gpus=1
            )
            serve_reference.create_backend(
                PredictModelPytorch,
                TAG,
                "resnet18",
                True,
                backend_config=config,
            )
            serve_reference.link(TAG, TAG)
            serve_handle = serve_reference.get_handle(TAG)

        img = base64.b64encode(open("elephant.jpg", "rb").read())

        # warmup
        ready_refs, _ = ray.wait(
            [serve_handle.remote(data=img) for _ in range(200)], 200
        )
        ray.wait(ready_refs, num_returns=200)
        del ready_refs

        qps = throughput_calculation(serve_handle, {"data": img}, 2000)
        print(
            f"[Resnet18] Batch Size: 8 Replica: {num_replica} "
            f"Throughput: {qps} QPS"
        )

        serve_reference.shutdown()


if __name__ == "__main__":
    main()