from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class _DataWithProducer(np.ndarray):
    producer: BaseEstimator | None = None

    def __new__(cls, data: np.ndarray, producer: BaseEstimator):
        obj = np.asarray(data).view(cls)
        obj.producer = producer
        return obj

    def __array_finalize__(self, obj: np.ndarray | None):
        if obj is None:
            return
        self.producer = getattr(obj, "producer", None)


def _patched_predict(self: BaseEstimator, *args, **kwargs) -> _DataWithProducer:
    y_pred = super(self.__class__, self).predict(*args, **kwargs)
    y_pred = _DataWithProducer(y_pred, self)
    return y_pred


def patch_estimator_class(cls: type) -> type:
    return type(
        f"{cls.__name__}_Patched4InferenceTimer",
        (cls,),
        {"predict": _patched_predict},
    )


def patch_estimator(estimator: BaseEstimator) -> BaseEstimator:
    estimator.__class__ = patch_estimator_class(estimator.__class__)
    return estimator


class TargetLib(Enum):
    pycaret = "pycaret"
    flaml = "flaml"


supported_libs = [e.value for e in TargetLib]


class InferenceTimer:
    def __init__(
        self,
        inference_data: pd.DataFrame | np.ndarray,
        target_lib: TargetLib | str | None = None,
    ):
        self.X: np.ndarray = (
            inference_data
            if isinstance(inference_data, np.ndarray)
            else inference_data.values
        )
        self.__name__: str = "inference_time"

        if isinstance(target_lib, str):
            target_lib = TargetLib[target_lib]
        self.target_lib: TargetLib | None = target_lib

    def _infer(self, estimator: BaseEstimator):
        _ = estimator.predict(self.X)

    def _infer_pycaret(self, y_true: np.ndarray, y_pred: _DataWithProducer, **kwargs):
        _ = y_pred.producer.predict(self.X)

    def _infer_flaml(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        estimator: BaseEstimator,
        *args,
        **kwargs,
    ):
        _ = estimator.predict(self.X)

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        match self.target_lib:
            case TargetLib.pycaret:
                infer_func = self._infer_pycaret
            case TargetLib.flaml:
                infer_func = self._infer_flaml
            case None:
                infer_func = self._infer

        start = datetime.now()
        infer_func(*args, **kwargs)
        elapsed = datetime.now() - start

        match self.target_lib:
            case TargetLib.flaml:
                return (
                    elapsed.total_seconds(),
                    {"inference_time": elapsed.total_seconds()},
                )
            case _:
                return elapsed.total_seconds()
