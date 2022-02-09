# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod
from functools import reduce
from operator import or_
from typing import Callable, Mapping

from jax import jit
from plum import dispatch

from pysages.backends import ContextWrapper
from pysages.collective_variables.core import build
from pysages.utils import identity


# ================ #
#   Base Classes   #
# ================ #

class SamplingMethod(ABC):
    """
    Abstract base class for all sampling methods.

    Defines the constructor that expects the collective variables, the build method to
    initialize the execution for the biasing and the run method that executes the
    simulation run. All these are intended be enhanced/overwritten by inheriting classes.
    """
    snapshot_flags = set()

    def __init__(self, cvs, *args, **kwargs):
        self.cvs = cvs
        self.cv = build(*cvs)
        self.requires_box_unwrapping = reduce(
            or_, (cv.requires_box_unwrapping for cv in cvs), False
        )
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Returns the snapshot, and two functions: `initialize` and `update`.
        `initialize` is intended to allocate any runtime information required
        by `update`, while `update` is intended to be called after each call to
        the wrapped context's `run` method.
        """
        pass


@dispatch
def run(
    method: SamplingMethod, context_generator: Callable, timesteps: int,
    callback: Callable = None, context_args: Mapping = dict(), **kwargs
):
    """
    Base implementation for running a single simulation/replica with a sampling method.

    Arguments
    ---------
    method: SamplingMethod

    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
        and `openmm.Simulation` for OpenMM. The function gets `context_args`
        unpacked for additional user arguments.

    timesteps: int
        Number of timesteps the simulation is running.

    callback: Optional[Callable]
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function.

    Note to developers: `method` must be pickle-able object, so make sure to implement
    appropriate `__getstate__` and `__setstate__` attributes.
    """
    # Make sure the sampling methods a pickle-able
    assert hasattr(method, "__getstate__") and hasattr(method, "__setstate__")

    context = context_generator(**context_args)
    wrapped_context = ContextWrapper(context, method, callback)
    with wrapped_context:
        wrapped_context.run(timesteps, **kwargs)


class GriddedSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, *args, **kwargs):
        check_dims(cvs, grid)
        super().__init__(cvs, *args, **kwargs)
        self.grid = grid

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


class NNSamplingMethod(GriddedSamplingMethod):
    def __init__(self, cvs, grid, topology, *args, **kwargs):
        super().__init__(cvs, grid, *args, **kwargs)
        self.topology = topology

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


# ========= #
#   Utils   #
# ========= #

def check_dims(cvs, grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generalize(concrete_update, helpers, jit_compile = True):
    if jit_compile:
        _jit = jit
    else:
        _jit = identity

    _update = _jit(concrete_update)

    def update(snapshot, state):
        return _update(state, helpers.query(snapshot))

    return _jit(update)
