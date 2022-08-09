#!/usr/bin/env python3

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorflow.keras import layers, metrics, regularizers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

if TYPE_CHECKING:
    from tensorflow.keras.models import Model


def create_model(initializer: str = "glorot_uniform") -> Model:
    model = Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=(32, 32, 3),
            kernel_initializer=initializer,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_initializer=initializer,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_initializer=initializer,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_initializer=initializer,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_initializer=initializer,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_initializer=initializer,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu", kernel_initializer=initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(
            10,
            activation="softmax",
            kernel_regularizer=regularizers.l2(0.0005),
            kernel_initializer=initializer,
        )
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            metrics.AUC(),
            metrics.Recall(),
            metrics.Precision(),
            metrics.TruePositives(),
            metrics.FalsePositives(),
            metrics.TrueNegatives(),
            metrics.FalseNegatives(),
        ],
    )

    return model
