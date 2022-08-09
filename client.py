#!/usr/bin/env python3

from __future__ import annotations

import os
import platform  # type: ignore
import sys  # type: ignore

import cpuinfo  # type: ignore

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}  # noqa: E402

import flwr as fl  # type: ignore
from tensorflow.keras import callbacks  # type: ignore

from cinic10_ds import batch_size, get_test_val_ds, get_train_ds
from model import create_model


class FLClient(fl.client.NumPyClient):
    def __init__(self, data_path: str, poisoning: str = None):
        self.poisoning = poisoning

        if poisoning == "label":
            self.train_ds = get_train_ds(data_path, "train/train_label_poison")
        elif poisoning == "data":
            self.train_ds = get_train_ds(data_path, "train/train_data_poison")
        else:
            self.train_ds = get_train_ds(data_path, "train/train")

        self.test_ds, self.val_ds = get_test_val_ds(data_path)

        self.train_count = len(self.train_ds) * batch_size
        self.test_count = len(self.test_ds) * batch_size
        self.val_count = len(self.val_ds) * batch_size

        self.model = create_model()

    def get_properties(self, ins: fl.common.PropertiesIns) -> dict:
        return {
            "poisoning": str(self.poisoning),
            "version": sys.version,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor_brand": cpuinfo.get_cpu_info()["brand_raw"],
            "processor_version": cpuinfo.get_cpu_info()["cpuinfo_version_string"],
            "processor_flags": ",".join(cpuinfo.get_cpu_info()["flags"]),
            "type": "client",
        }

    def get_parameters(self) -> fl.common.Weights:
        return self.model.get_weights()

    def fit(
        self, parameters: fl.common.Parameters, config: dict
    ) -> tuple[fl.common.Weights, int, dict]:
        self.model.set_weights(parameters)

        callback = callbacks.EarlyStopping(monitor="loss", patience=3)

        if self.poisoning == "model":
            self.model = create_model(initializer="random_normal")
        elif self.poisoning == "lazy":
            pass
        else:
            history = self.model.fit(
                self.train_ds,
                epochs=config["epochs"],
                batch_size=128,
                validation_data=self.val_ds,
                callbacks=[callback],
            )
            metrics = {metric: value[-1] for metric, value in history.history.items()}
            return self.model.get_weights(), self.train_count, metrics
        return self.model.get_weights(), self.train_count, {"poisoning": self.poisoning}

    def evaluate(
        self, parameters: fl.common.Parameters, config: dict
    ) -> tuple[float, int, dict]:
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_ds)
        return loss, self.test_count, {"accuracy": accuracy}

    def start(self, server_address: str) -> None:
        fl.client.start_numpy_client(server_address + ":8080", client=self)
