from pprint import pprint
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.resnet import resnet50
import io
from PIL import Image
import base64
import torch

from benchmarking import serve_benchmark
from benchmarking import Experiment, HTTPProxyActor
import pandas as pd
import json
import time
import ray
from benchmarking.utils import (
    generate_fixed_arrival_process,
    get_latency,
    ROOT_DIR,
)
import os
import subprocess
import click
from typing import Any
from torch.autograd import Variable
from tqdm import tqdm


class Transform:
    """
    Standard pytorch pre-processing functionality
    - gets a raw image
    - converts it to tensor
    """

    def __init__(self, transform: Any) -> None:

        self.transform = transform

    @serve_benchmark.accept_batch
    def __call__(self, _, data: list) -> list:
        data_list = list()
        for img in data:
            data = Image.open(io.BytesIO(base64.b64decode(img)))
            if data.mode != "RGB":
                data = data.convert("RGB")
            data = self.transform(data)
            data_list.append(data)
        return data_list


class PredictModelPytorch:
    """
    Standard pytorch prediction functionality
    - gets a preprocessed tensor
    - predicts it's class
    """

    def __init__(self, model_name, is_cuda: bool = False) -> None:
        self.model = models.__dict__[model_name](pretrained=True)
        self.is_cuda = is_cuda
        if is_cuda:
            self.model = self.model.cuda()

    @serve_benchmark.accept_batch
    def __call__(self, _, data: list) -> list:
        data = torch.stack(data)
        data = Variable(data)
        if self.is_cuda:
            data = data.cuda()
        outputs = self.model(data)
        _, predicted = outputs.max(1)
        return predicted.cpu().numpy().tolist()


class ChainHandle:
    def __init__(self, handle_list):
        self.handle_list = handle_list

    def remote(self, data):
        for index in range(len(self.handle_list)):
            data = self.handle_list[index].remote(data=data)
        return data


class ImagePrepocPipeline:
    def __init__(self, vertex_config, model_name):
        super().__init__()
        handle_list = list()
        for node_id in vertex_config.keys():
            backend_config = vertex_config[node_id]
            with serve_benchmark.using_router(node_id):
                serve_benchmark.create_endpoint(node_id)
                config = serve_benchmark.BackendConfig(**backend_config)
                if node_id == "prepoc":
                    min_img_size = 224
                    transform = transforms.Compose(
                        [
                            transforms.Resize(min_img_size),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ]
                    )
                    serve_benchmark.create_backend(
                        Transform, node_id, transform, backend_config=config
                    )
                elif node_id == "model":
                    serve_benchmark.create_backend(
                        PredictModelPytorch,
                        node_id,
                        model_name,
                        True,
                        backend_config=config,
                    )
                serve_benchmark.link(node_id, node_id)
                handle_list.append(serve_benchmark.get_handle(node_id))

        self.chain_handle = ChainHandle(handle_list)

    def remote(self, data):
        return self.chain_handle.remote(data)


class ReferencedTensorExperiment(Experiment):
    def __init__(self, name, config_path):
        super().__init__(name, config_path)
        self.config["serving_type"] = "vanilla"
        columns = [
            "vertex_config",
            "serving_type",
            "arrival_process",
            "throughput_qps",
            "latency_s",
        ]
        self._df = pd.DataFrame(columns=columns)
        self._model_dir = self.config["model_type"]
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

    def _throughput_calculation(
        self, chain_pipeline, tensor_data, num_requests
    ):

        print("warmming up")
        # warmup
        ready, _ = ray.wait(
            [chain_pipeline.remote(tensor_data) for _ in range(100)], 100,
        )
        print(f"Warmup {len(ready)}")
        print("Warmup done")
        print("Onto Throughput calculation")
        # del ready

        num_requests = 100

        start_time = time.perf_counter()
        queries_fired = [
            chain_pipeline.remote(data=tensor_data) for _ in range(num_requests)
        ]
        print(f" {len(queries_fired)} queries fired waiting for them")
        ready, _ = ray.wait(queries_fired, num_requests)

        print(f"Throughput {len(ready)}")

        end_time = time.perf_counter()
        duration = end_time - start_time
        qps = num_requests / duration
        del ready
        return qps

    def run(self):
        for vertex_config in self.config["vertex_configs"]:

            serve_benchmark.init(start_server=False)
            filename_query = "arrival_trace.jsonl"
            route = "/prepoc"

            pipeline = ImagePrepocPipeline(
                vertex_config, self.config["model_type"]
            )
            vertex_config_name = json.dumps(vertex_config)
            df_row = dict(
                vertex_config=vertex_config_name,
                serving_type=self.config["serving_type"],
                arrival_process=self.config["arrival_process"],
            )

            image_path = os.path.join(ROOT_DIR, self.config["image_file_path"])
            tensor_data = base64.b64encode(open(image_path, "rb").read())

            throughput_qps = self._throughput_calculation(
                pipeline, tensor_data, self.config["num_requests"]
            )
            df_row.update(throughput_qps=throughput_qps)

            pprint(df_row)

            # closed loop latency calculation
            closed_loop_latencies = list()
            for _ in tqdm(range(30)):
                start_time = time.perf_counter()
                ready, _ = ray.wait([pipeline.remote(tensor_data)], 1)
                end_time = time.perf_counter()
                latency = end_time - start_time
                closed_loop_latencies.append(latency)

            df_row.update(latency_s=closed_loop_latencies)

            self._df = self._df.append(df_row, ignore_index=True)

            # cleanup
            del closed_loop_latencies, pipeline
            serve_benchmark.shutdown()

    def save(self, filepath):
        self._df.to_csv(os.path.join(self._model_dir, filepath))


@click.command()
@click.option(
    "--config-path", type=str, default="../resnet50_config_closed_loop.json"
)
@click.option(
    "--save-path", type=str, default="image_prepoc_vanilla_closed_loop.csv"
)
def main(config_path, save_path):
    experiment = ReferencedTensorExperiment(
        name="ref_prepoc", config_path=config_path
    )
    experiment.run()
    experiment.save(save_path)


if __name__ == "__main__":
    main()
