# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Adaptive Biasing Force (ABF) sampling method.

ABF partitions the collective variable space into bins determined by a user
provided grid, and keeps a tabulation of the number of visits to each bin
as well as the sum of generalized forces experienced by the system at each
configuration bin. These provide an estimate for the mean generalized force,
which can be integrated to yield the free energy.

The implementation of the adaptive biasing force method here closely follows
https://doi.org/10.1063/1.2829861. One important difference is that the time
derivative of the product :math:`W\\cdot p` (equation 9 of reference) is approximated by a
second order backward finite difference in the simulation time step.
"""

from functools import partial
from typing import NamedTuple

from jax import jit
from jax import numpy as np
from jax import vmap
from jax.lax import cond
from jax.scipy import linalg

from pysages.approxfun.core import compute_mesh
from pysages.approxfun.core import scale as _scale
from pysages.grids import build_indexer
from pysages.methods.core import GriddedSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.ml.models import MLP
from pysages.ml.objectives import GradientsSSE, L2Regularization
from pysages.ml.optimizers import LevenbergMarquardt
from pysages.ml.training import NNData, build_fitting_function, convolve
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.utils import JaxArray, dispatch


class ABFState(NamedTuple):
    """
    ABF internal state.

    Parameters
    ----------

    xi: JaxArray (CV shape)
        Last collective variable recorded in the simulation.

    bias: JaxArray (Nparticles, d)
        Array with biasing forces for each particle.

    hist: JaxArray (grid.shape)
        Histogram of visits to the bins in the collective variable grid.

    Fsum: JaxArray (grid.shape, CV shape)
        Cumulative forces at each bin in the CV grid.

    force: JaxArray (grid.shape, CV shape)
        Average force at each bin of the CV grid.

    Wp: JaxArray (CV shape)
        Product of W matrix and momenta matrix for the current step.

    Wp_: JaxArray (CV shape)
        Product of W matrix and momenta matrix for the previous step.
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    force: JaxArray
    Wp: JaxArray
    Wp_: JaxArray

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ABF(GriddedSamplingMethod):
    """
    Adaptive Biasing Force Method.

    Attributes
    ----------

    snapshot_flags:
        Indicate the system properties required from a snapshot.

    Parameters
    ----------

    cvs: Union[List, Tuple]
        Set of user selected collective variable.

    grid: Grid
        Specifies the collective variables domain and number of bins for discretizing
        the CV space along each CV dimension.

    N: Optional[int] = 500
        Threshold parameter before accounting for the full average
        of the adaptive biasing force.

    restraints: Optional[CVRestraints] = None
        If provided, indicate that harmonic restraints will be applied when any
        collective variable lies outside the box from `restraints.lower` to
        `restraints.upper`.
    """

    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(self, cvs, grid, **kwargs):
        super().__init__(cvs, grid, **kwargs)
        self.N = np.asarray(self.kwargs.get("N", 500))

    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Build the functions for the execution of ABF

        Parameters
        ----------

        snapshot:
            PySAGES snapshot of the simulation (backend dependent).

        helpers:
            Helper function bundle as generated by
            `SamplingMethod.context[0].get_backend().build_helpers`.

        Returns
        -------

        Tuple `(snapshot, initialize, update)` to run ABF simulations.
        """
        return _abf(self, snapshot, helpers)


def _abf(method, snapshot, helpers):
    """
    Internal function that generates the init and update functions.

    Parameters
    ----------

    method: ABF
        Class that generates the functions.
    snapshot:
        PySAGES snapshot of the simulation (backend dependent).
    helpers
        Helper function bundle as generated by
        `SamplingMethod.context[0].get_backend().build_helpers`.

    Returns
    -------
    Tuple `(snapshot, initialize, update)` to run ABF simulations.
    """
    cv = method.cv
    grid = method.grid

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)
    estimate_force = build_force_estimator(method)

    def initialize():
        """
        Internal function that generates the first ABFstate with correctly shaped JaxArrays.

        Returns
        -------
        ABFState
            Initialized State
        """
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        return ABFState(xi, bias, hist, Fsum, force, Wp, Wp_)

    def update(state, data):
        """
        Advance the state of the ABF simulation.

        Parameters
        ----------

        state: ABFstate
            Old ABFstate from the previous simutlation step.
        data: JaxArray
            Snapshot to access simulation data.

        Returns
        -------
        ABFState
            Updated internal state.
        """
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)

        p = data.momenta
        # The following could equivalently be computed as `linalg.pinv(Jxi.T) @ p`
        # (both seem to have the same performance).
        # Another option to benchmark against is
        # Wp = linalg.tensorsolve(Jxi @ Jxi.T, Jxi @ p)
        Wp = linalg.solve(Jxi @ Jxi.T, Jxi @ p, sym_pos="sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt

        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        # Add previous force to remove bias
        Fsum = state.Fsum.at[I_xi].add(dWp_dt + state.force)

        force = estimate_force(xi, I_xi, Fsum, hist).reshape(dims)
        bias = np.reshape(-Jxi.T @ force, state.bias.shape)

        return ABFState(xi, bias, hist, Fsum, force, Wp, state.Wp)

    return snapshot, initialize, generalize(update, helpers)


@dispatch
def build_force_estimator(method: ABF):
    """
    Returns a function that computes the average forces
    (or the harmonic restraints forces if provided).
    """
    N = method.N
    grid = method.grid

    def average_force(data):
        _, I_xi, Fsum, hist = data
        return Fsum[I_xi] / np.maximum(N, hist[I_xi])

    if method.restraints is None:
        estimate_force = jit(lambda *args: average_force(args))
    else:
        lo, hi, kl, kh = method.restraints

        def restraints_force(data):
            xi, *_ = data
            xi = xi.reshape(grid.shape.size)
            return apply_restraints(lo, hi, kl, kh, xi)

        def estimate_force(xi, I_xi, Fsum, hist):
            ob = np.any(np.array(I_xi) == grid.shape)  # Out of bounds condition
            data = (xi, I_xi, Fsum, hist)
            return cond(ob, restraints_force, average_force, data)

    return estimate_force


@dispatch
def analyze(result: Result[ABF], **kwargs):
    """
    Computes the free energy from the result of an `ABF` run.
    Integrates the forces using Sobolev approximation of functions by gradient learning.

    Parameters
    ----------

    result: Result[ABF]:
        Result bundle containing method, final ABF state, and callback.

    topology: Optional[Tuple[int]] = (4, 4)
        Defines the architecture of the neural network (number of nodes of each hidden layer).

    Returns
    -------

    dict: A dictionary with the following keys:

        histogram: JaxArray
            Histogram for the states visited during the method.

        mean_force: JaxArray
            Average force at each bin of the CV grid.

        free_energy: JaxArray
            Free Energy at each bin of the CV grid.

        mesh: JaxArray
            Grid used in the method.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the CV domain defined
            by the grid.
    """
    topology = kwargs.get("topology", (4, 4))
    method = result.method
    state = result.states[0]
    grid = method.grid
    inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower

    model = MLP(grid.shape.size, 1, topology, transform=partial(_scale, grid=grid))
    optimizer = LevenbergMarquardt(loss=GradientsSSE(), max_iters=2500, reg=L2Regularization(1e-3))
    fit = build_fitting_function(model, optimizer)

    @vmap
    def smooth(data):
        boundary = "wrap" if grid.is_periodic else "edge"
        kernel = np.float32(blackman_kernel(grid.shape.size, 7))
        return convolve(data.T, kernel, boundary=boundary).T

    def train(nn, data):
        axes = tuple(range(data.ndim - 1))
        std = data.std(axis=axes).max()
        reference = np.float64(smooth(np.float32(data / std)))
        params = fit(nn.params, inputs, reference).params
        return NNData(params, nn.mean, std / reference.std(axis=axes).max())

    def build_fes_fn(state):
        hist = np.expand_dims(state.hist, state.hist.ndim)
        F = state.Fsum / np.maximum(hist, 1)

        ps, layout = unpack(model.parameters)
        nn = train(NNData(ps, 0.0, 1.0), F)

        def fes_fn(x):
            params = pack(nn.params, layout)
            A = nn.std * model.apply(params, x) + nn.mean
            return A.max() - A

        return jit(fes_fn)

    def average_forces(state):
        Fsum = state.Fsum
        shape = (*Fsum.shape[:-1], 1)
        return Fsum / np.maximum(state.hist.reshape(shape), 1)

    fes_fn = build_fes_fn(state)

    ana_result = dict(
        histogram=state.hist,
        mean_force=average_forces(state),
        free_energy=fes_fn(inputs).reshape(grid.shape),
        mesh=inputs,
        fes_fn=fes_fn,
    )
    return numpyfy_vals(ana_result)
