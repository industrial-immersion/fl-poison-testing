#!/usr/bin/env python3

from __future__ import annotations

import math
import platform
import sys
from typing import TYPE_CHECKING

import cpuinfo  # type: ignore  # missing stub
import flwr as fl  # type: ignore
from pip._internal.operations import freeze as pip_freeze

from cinic10_ds import get_test_val_ds
from logger import DictLogger
from model import create_model
from validation_server import ValidationServer

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Union

    import tensorflow as tf


l = DictLogger()  # noqa: E741

get_properties_ins = fl.common.PropertiesIns({})
all_metrics: dict[int, dict] = {}


class SelectClientsCritertion(fl.server.criterion.Criterion):
    def __init__(self, poisoned: bool):
        self.poisoned = poisoned

    def select(self, client: fl.server.client_proxy.ClientProxy) -> bool:
        if self.poisoned:
            return (
                client.get_properties(get_properties_ins, None).properties["poisoning"]
                != "None"
            )
        else:
            return (
                client.get_properties(get_properties_ins, None).properties["poisoning"]
                == "None"
            )


get_properties_ins = fl.common.PropertiesIns({})


class SelectNonPoisonedClientsCritertion(fl.server.criterion.Criterion):
    def select(self, client: fl.server.client_proxy.ClientProxy) -> bool:
        return (
            client.get_properties(get_properties_ins, None).properties["poisoning"]
            == "None"
        )


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *args: Any,
        data_path: str,
        num_fl_clients: int,
        poisoned_client_selection: Union[str, float],
        **kwargs: Any,
    ):
        self.total_fl_clients_in_inventory = num_fl_clients
        self.poisoned_client_selection = poisoned_client_selection
        self.blacklist: set[str] = set()

        super().__init__(*args, **kwargs)

        self.criterion = SelectNonPoisonedClientsCritertion()

        self.validator = ValidationServer(
            num_of_models=3,
            data_path=data_path,
        )

        self.blacklisted_clients: list[int] = []

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> None:
        """Wait for all configured clients to come up online and connect to server."""
        client_manager.wait_for(self.total_fl_clients_in_inventory)

    def aggregate_fit(
        self,
        rnd: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[BaseException],
    ) -> Optional[fl.common.Weights]:
        self.round = rnd

        self.update_blacklist(results)

        valid_results = [
            (client_proxy, fit_result)
            for client_proxy, fit_result in results
            if client_proxy.cid not in self.blacklist
        ]

        all_metrics[rnd] = {}
        for result in results:
            all_metrics[rnd]["client://" + str(result[0].cid)] = result[1].metrics
            all_metrics[rnd]["client://" + str(result[0].cid)]["num_examples"] = result[
                1
            ].num_examples

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd, valid_results, failures
        )

        self.validator.update(
            fl.common.parameters_to_weights(aggregated_parameters), rnd
        )

        return aggregated_parameters, aggregated_metrics

    def update_blacklist(
        self, results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]
    ) -> None:
        validation_results = self.validator.validate_clients(results)

        self.blacklist |= {
            cid for cid, value in validation_results.items() if value is True
        }

    def configure_fit(
        self,
        rnd: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = fl.common.FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        poisoned_clients_criterion = SelectClientsCritertion(poisoned=True)
        non_poisoned_clients_criterion = SelectClientsCritertion(poisoned=False)

        selected_clients = []

        if self.poisoned_client_selection == "random":
            selected_clients.extend(
                client_manager.sample(
                    num_clients=sample_size,
                    min_num_clients=min_num_clients,
                )
            )
        else:  # poisoned_client_selection is a float between 0 and 1
            selected_clients.extend(
                client_manager.sample(
                    num_clients=math.ceil(
                        sample_size * self.fraction_fit * self.poisoned_client_selection
                    ),
                    min_num_clients=min_num_clients,
                    criterion=poisoned_clients_criterion,
                )
            )

            selected_clients.extend(
                client_manager.sample(
                    num_clients=math.floor(sample_size * self.fraction_fit),
                    min_num_clients=min_num_clients,
                    criterion=non_poisoned_clients_criterion,
                )
            )

        # Return client/config pairs
        return [(client, fit_ins) for client in selected_clients]


class FLServer(fl.server.server.Server):
    def __init__(
        self,
        data_path: str,
        num_fl_clients: int,
        poisoned_client_selection: Union[str, float],
        fraction_fit: float,
        rounds: int,
        epochs: int,
    ):
        self.poisoned_client_selection = poisoned_client_selection
        self.rounds = rounds
        self.epochs = epochs
        self.data_path = data_path
        self.val_ds = get_test_val_ds(data_path)[1]
        model = create_model()
        model.summary()

        self.strategy = SaveModelStrategy(
            initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
            on_evaluate_config_fn=FLServer.evaluate_config,
            min_fit_clients=0,
            min_available_clients=num_fl_clients,
            fraction_fit=fraction_fit,
            min_eval_clients=0,
            fraction_eval=0.0,
            eval_fn=FLServer.get_eval_fn(model, self.val_ds),
            on_fit_config_fn=self.on_fit_config,
            # custom arguments
            data_path=self.data_path,
            num_fl_clients=num_fl_clients,
            poisoned_client_selection=poisoned_client_selection,
        )

        super().__init__(
            client_manager=fl.server.client_manager.SimpleClientManager(),
            strategy=self.strategy,
        )

        l.log(
            {
                "pip_info": {
                    dep.split(sep="==")[0]: dep.split(sep="==")[1]
                    for dep in pip_freeze.freeze()
                }
            }
        )

        l.log(
            {
                "server_info": {
                    "version": sys.version,
                    "platform": platform.system(),
                    "architecture": platform.machine(),
                    "processor": cpuinfo.get_cpu_info()["brand_raw"],
                    "type": "server",
                }
            }
        )

    def on_fit_config(self, rnd: int) -> dict[str, int]:
        return {
            "rnd": rnd,
            "epochs": self.epochs,
            "rounds": self.rounds,
        }

    def start(self) -> None:
        fl.server.start_server(
            server=self, config={"num_rounds": self.rounds}, strategy=self.strategy
        )

    @staticmethod
    def get_eval_fn(model: tf.keras.models.Model, val_ds: tf.data.Dataset) -> Callable:
        """Return an evaluation function for server-side evaluation.

        The returned function will be called on every round by the server.
        """

        def evaluate(weights: fl.common.Weights) -> tuple[float, dict[str, Any]]:
            model.set_weights(weights)  # Update model with the latest parameters
            metrics = model.evaluate(val_ds)

            accuracy = metrics[model.metrics_names.index("accuracy")]
            loss = metrics[model.metrics_names.index("loss")]

            round_metrics = {
                model.metrics_names[k]: metrics[k] for k in range(len(metrics))
            }

            if len(all_metrics) == 0:
                all_metrics[0] = {"server": round_metrics}
            else:
                all_metrics[len(all_metrics) - 1] |= {"server": round_metrics}

            return loss, {"accuracy": accuracy}

        return evaluate

    @staticmethod
    def evaluate_config(rnd: int) -> dict[str, Any]:
        """Return evaluation configuration dict for each round.

        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if rnd < 4 else 10
        return {"val_steps": val_steps}

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        l.log(
            {
                "client_system_info": {
                    "client_info": {
                        cid: client.get_properties(get_properties_ins, None).properties
                        for cid, client in self.client_manager().clients.items()
                    }
                }
            }
        )

        l.log({"all_metrics": all_metrics})

        super().disconnect_all_clients(timeout)
