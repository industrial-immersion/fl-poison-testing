#!/usr/bin/env python3

from __future__ import annotations

from typing import TYPE_CHECKING

import flwr as fl  # type: ignore

from cinic10_ds import get_test_val_ds
from model import create_model

if TYPE_CHECKING:
    from typing import Union


class ValidationServer:
    def __init__(self, num_of_models: int = 3, data_path: str = "data/"):
        self.num_of_models = num_of_models
        if self.num_of_models % 2 == 0:
            self.num_of_models += 1

        self.models = []
        for _ in range(self.num_of_models):
            self.models.append(create_model())

        _, self.val_ds = get_test_val_ds(data_path)

    def update(self, model_weights: fl.common.Weights, cur_round: int) -> None:
        self.models[cur_round % self.num_of_models].set_weights(model_weights)

    @staticmethod
    def compare_model_performance(
        metrics_a: dict[str, Union[float, int]], metrics_b: dict[str, Union[float, int]]
    ) -> bool:
        return metrics_b["accuracy"] > metrics_a["accuracy"]

    def validate_clients(
        self,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> dict[str, bool]:
        votes: dict[str, list[bool]] = {}

        model_to_validate = create_model()

        reference_model_metrics = []
        for reference_model in self.models:
            reference_model_metrics.append(
                {
                    reference_model.metrics_names[index]: metric
                    for index, metric in enumerate(
                        reference_model.evaluate(self.val_ds)
                    )
                }
            )

        for client_proxy, fit_res in results:
            votes[client_proxy.cid] = []

            client_model_weights = fl.common.parameters_to_weights(fit_res.parameters)

            for index, reference_model in enumerate(self.models):
                model_to_validate.set_weights(
                    fl.server.strategy.aggregate.aggregate(
                        [(client_model_weights, 1), (reference_model.get_weights(), 1)]
                    )
                )

                model_to_validate_metrics = {
                    model_to_validate.metrics_names[i]: metric
                    for i, metric in enumerate(model_to_validate.evaluate(self.val_ds))
                }

                votes[client_proxy.cid].append(
                    self.compare_model_performance(
                        reference_model_metrics[index], model_to_validate_metrics
                    )
                )

        return {
            cid: sum(comparison_results) > self.num_of_models / 2
            for cid, comparison_results in votes.items()
        }
