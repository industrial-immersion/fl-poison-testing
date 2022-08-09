#!/usr/bin/env python3

from __future__ import annotations

import os
import pathlib

import tensorflow as tf  # type: ignore

batch_size = 32


def get_train_ds(data_path: str, subdir: str = None) -> tf.data.Dataset:
    train_data_dir = pathlib.Path(os.path.join(data_path, subdir or "train"))

    img_height = 32
    img_width = 32

    train_folder = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels="inferred",
        label_mode="categorical",
    )

    return train_folder


def get_test_val_ds(data_path: str) -> tf.data.Dataset:

    val_data_dir = pathlib.Path(os.path.join(data_path, "valid"))
    test_data_dir = pathlib.Path(os.path.join(data_path, "test"))

    img_height = 32
    img_width = 32

    test_folder = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels="inferred",
        label_mode="categorical",
    )

    val_folder = tf.keras.utils.image_dataset_from_directory(
        val_data_dir,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels="inferred",
        label_mode="categorical",
    )

    return test_folder, val_folder
