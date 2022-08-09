#!/usr/bin/env python

from __future__ import annotations

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # noqa: E402
# silence tensorflow warnings, needs to run before importing flower

from typing import TYPE_CHECKING

import yaml  # type: ignore

if TYPE_CHECKING:
    from typing import Any

from client import FLClient
from logger import DictLogger
from server import FLServer


def get_server_and_client_info(config: dict[str, Any]) -> tuple:
    server = [
        machine
        for machine in config["machines"]
        if isinstance(machine, dict) and machine.get("role") == "server"
    ][0]

    clients = config["machines"]
    clients.remove(server)

    num_fl_clients = sum(
        [
            isinstance(client, str)
            and config["containers_per_machine"]
            or client.get("num_containers", config["containers_per_machine"])
            for client in clients
        ]
    )

    num_poisoned_clients = (
        config["poisoning"]
        and 0
        or sum(
            [
                len(client.get("poisoned_containers", []))
                for client in clients
                if isinstance(client, dict)
            ]
        )
    )

    return server, clients, num_fl_clients, num_poisoned_clients


def get_server_and_client_info(config):
    server = [
        machine
        for machine in config["machines"]
        if isinstance(machine, dict) and machine.get("role") == "server"
    ][0]

    clients = config["machines"]
    clients.remove(server)

    num_fl_clients = sum(
        [
            isinstance(client, str)
            and config["containers_per_machine"]
            or client.get("num_containers", config["containers_per_machine"])
            for client in clients
        ]
    )

    num_poisoned_clients = (
        config["poisoning"]
        and 0
        or sum(
            [
                len(client.get("poisoned_containers", []))
                for client in clients
                if isinstance(client, dict)
            ]
        )
    )

    return server, clients, num_fl_clients, num_poisoned_clients


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--poisoned", action="store_true")
    args = parser.parse_args()

    config_file = args.config
    config = {}

    if os.path.isfile(config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        raise FileNotFoundError(f"config file {config_file} not found")

    server, clients, num_fl_clients, num_poisoned_clients = get_server_and_client_info(
        config
    )

    if args.server:
        try:
            l = DictLogger()  # noqa: E741
            l.log({"config": config})

            FLServer(
                config["data_path"],
                num_fl_clients,
                config["poisoned_client_selection"],
                config["fraction_fit"],
                config["model"]["rounds"],
                config["model"]["epochs"],
            ).start()
        except Exception as e:
            l.delete_log_file()
            raise (e)

    else:
        FLClient(
            config["data_path"],
            poisoning=args.poisoned and config["poisoning"] or None,
        ).start(server["address"])
