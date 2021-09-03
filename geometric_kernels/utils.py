"""
Convenience utilities.
"""
from importlib import import_module
from typing import Callable

import eagerpy as ep
import numpy as np
from multipledispatch import Dispatcher

__all__ = [
    "cast_to_int",
    "take_along_axis",
]

tf = None
torch = None

cast_to_int = Dispatcher("cast_to_int")
take_along_axis = Dispatcher("take_along_axis")


def requires_tensorflow(func: Callable) -> Callable:
    """A decorator to import `tensorflow` upon calling the decorated function.
    The import is performed once.
    Helps avoid importing every backend simultaneously.

    It is suggested to disable mypy inside the decorated function with
    `# type: ignore[misc]` type comment.

    :param func: decorated function
    :return: decorated function
    """
    global tf
    if tf is None:
        tf = import_module("tensorflow")

    return func


def requires_torch(func: Callable) -> Callable:
    """A decorator to import `torch` upon calling the decorated function.
    The import is performed once.
    Helps avoid importing every backend simultaneously.

    It is suggested to disable mypy inside the decorated function with
    `# type: ignore[misc]` type comment.

    :param func: decorated function
    :return: decorated function
    """
    global torch
    if torch is None:
        torch = import_module("torch")

    return func


@requires_tensorflow
@cast_to_int.register(ep.TensorFlowTensor)
def _cast_to_int_tf(t: ep.TensorFlowTensor):
    return ep.astensor(tf.cast(t.raw, tf.int64))  # type: ignore[misc]


@requires_torch
@cast_to_int.register(ep.PyTorchTensor)
def _cast_to_int_torch(t: ep.PyTorchTensor):
    return ep.astensor(t.raw.to(torch.int64))  # type: ignore[misc]


@cast_to_int.register(ep.NumPyTensor)
def _cast_to_int_numpy(t: ep.NumPyTensor):
    return t.astype(np.int64)


@requires_torch
@take_along_axis.register(ep.PyTorchTensor, ep.PyTorchTensor)
def _take_along_axis_torch(t: ep.PyTorchTensor, index: ep.PyTorchTensor, axis=0):
    return type(t)(torch.index_select(t.raw, axis, index.raw.ravel())).astype(  # type: ignore[misc]
        t.dtype
    )


@requires_tensorflow
@take_along_axis.register(ep.TensorFlowTensor, ep.TensorFlowTensor)
def _take_along_axis_tf(t: ep.TensorFlowTensor, index: ep.TensorFlowTensor, axis=0):
    return type(t)(
        tf.gather(t.raw, tf.reshape(index.raw, (-1,)), axis=axis)  # type: ignore[misc]
    ).astype(t.dtype)


@requires_tensorflow
@take_along_axis.register(ep.TensorFlowTensor, ep.NumPyTensor)
def _take_along_axis_tf_np(t: ep.TensorFlowTensor, index: ep.NumPyTensor, axis=0):
    return type(t)(tf.gather(t.raw, index.raw.ravel(), axis=axis)).astype(  # type: ignore[misc]
        t.dtype
    )


@take_along_axis.register(ep.NumPyTensor, ep.NumPyTensor)
def _take_along_axis_numpy(t: ep.NumPyTensor, index: ep.NumPyTensor, axis=0):
    return type(t)(np.take_along_axis(t.raw, index.raw, axis=axis))  # type: ignore[misc]
