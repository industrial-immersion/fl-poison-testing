#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time


class DictLogger(object):
    _json_file_name: str = ""

    def __new__(cls) -> "DictLogger":
        if not cls._json_file_name:
            cls._json_file_name = time.strftime("%y.%m.%d-%H:%M:%S") + ".json"
            cls.content: dict = {}

        return super().__new__(cls)

    def log(self, dictionary: dict) -> None:
        with open(self._json_file_name, "w") as f:
            self.content |= dictionary
            json.dump(self.content, f, indent=2)

    def delete_log_file(self) -> None:
        os.remove(self._json_file_name)
