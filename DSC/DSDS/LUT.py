import numpy as np
from loguru import logger
import math


class LUT_1D:
    def __init__(
        self,
        default_value=0.0,
        verbose=False,
        store_all_data=False,
    ):
        logger.info("Initializing new 1D LUT")
        self._lut = {}
        self._default_value = default_value
        self._verbose = verbose
        self._store_all_data = store_all_data

    def _init_lut_entry_if_needed(self, index, key):
        if index not in self._lut.keys():
            self._lut[index] = {}

        if key not in self._lut[index].keys():
            self._lut[index][key] = {
                "n_samples": 0,
                "data": [],
                "sum_x": 0.0,
                "sum_x2": 0.0,
                "mean": 0.0,
                "std": 0.0,
            }

    def train_sample(self, index, key, value):
        self._init_lut_entry_if_needed(index=index, key=key)

        self._lut[index][key]["n_samples"] += 1
        self._lut[index][key]["sum_x"] += value
        self._lut[index][key]["sum_x2"] += math.pow(value, 2)

        if self._store_all_data:
            self._lut[index][key]["data"].append(value)

        _d = self._lut[index][key]

        self._lut[index][key]["mean"] = _d["sum_x"] / _d["n_samples"]
        self._lut[index][key]["std"] = math.sqrt(
            _d["sum_x2"] / _d["n_samples"] - math.pow(_d["sum_x"] / _d["n_samples"], 2)
        )

    def get_mean_std(self, index, key):
        if index not in self._lut.keys():
            if self._verbose:
                logger.warning("LUT has no entry at {index}; returning default value")
            return self._default_value, 0.0

        if key not in self._lut[index].keys():
            if self._verbose:
                logger.warning(
                    "LUT has no entry for {key} at {index}; returning default value"
                )
            return self._default_value, 0.0

        return self._lut[index][key]["mean"], self._lut[index][key]["std"]

    def get_n_samples_per_entry(self):
        n_samples = {}

        for i in self._lut.keys():
            for k in self._lut[i].keys():

                if k not in n_samples.keys():
                    n_samples[k] = {}

                n_samples[k][i] = self._lut[i][k]["n_samples"]

        return n_samples
