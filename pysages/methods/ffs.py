# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import Callable, Mapping, NamedTuple, Optional

from pysages.backends import ContextWrapper
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, copy

import jax.numpy as np


class FFSState(NamedTuple):
    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class FFS(SamplingMethod):
    snapshot_flags = {"positions", "indices"}

    def build(self, snapshot, helpers):
        self.helpers=helpers
        return _ffs(self,snapshot,helpers)

    # We override the default run method as FFS is algorithmically fairly different
    def run(
        self,
        context_generator: Callable,
        timesteps: int,
        dt: float,
        win_i: float,
        win_l: float,
        Nw: int,
        sampling_steps_basin: int,
        Nmax_replicas: int,
        verbose: bool = False,
        callback: Optional[Callable] = None,
        context_args: Mapping = dict(),
        **kwargs
    ):
        """
        Direct version of the Forward Flux Sampling algorithm.

        Arguments
        ---------
        context_generator: Callable
            User defined function that sets up a simulation context with the backend.
            Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
            and `simtk.openmm.Simulation` for OpenMM. The function gets `context_args`
            unpacked for additional user arguments.

        timesteps: int
            Number of timesteps the simulation is running.

        dt: float
            timestep of the simulation

        win_i: float
            initial window for the system

        win_l: float
            last window to be calculated in ffs

        Nw: int
            number of equally spaced windows

        sampling_steps_basin: int
            period for sampling configurations in the basin

        Nmax_replicas: int
            number of stored configuration for each window

        verbose: bool
            If True more information will be logged (useful for debbuging).

        callback: Optional[Callable]
            Allows for user defined actions into the simulation workflow of the method.
            `kwargs` gets passed to the backend `run` function.

        NOTE:
            The current implementation runs a single simulation/replica,
            but multiple concurrent simulations can be scripted on top of this.
        """

        context = context_generator(**context_args)
        wrapped_context = ContextWrapper(context, self, callback)

        with wrapped_context:
            sampler = wrapped_context.sampler
            xi = sampler.state.xi.block_until_ready()
            windows = np.linspace(win_i, win_l, num = Nw)

            is_configuration_good = check_input(windows, xi, verbose = verbose)
            if not is_configuration_good:
                raise ValueError("Bad initial configuration")

            run = wrapped_context.run
            helpers = self.helpers
            cv = self.cv

            reference_snapshot = copy(sampler.snapshot)

            # We Initially sample from basin A
            # TODO: bundle the arguments into data structures
            ini_snapshots = basin_sampling(
                Nmax_replicas,
                sampling_steps_basin,
                windows,
                run,
                sampler,
                reference_snapshot,
                helpers,
                cv
            )

            # Calculate initial flow
            phi_a, snaps_0 = initial_flow(
                Nmax_replicas,
                dt,
                windows,
                ini_snapshots,
                run,
                sampler,
                helpers,
                cv
            )

            write_to_file(phi_a)
            hist = np.zeros(len(windows))
            hist = hist.at[0].set(phi_a)

            # Calculate conditional probability for each window
            for k in range(1,len(windows)):
                if k == 1:
                    old_snaps = snaps_0
                prob, w1_snapshots = running_window(
                    windows, k, old_snaps, run, sampler, helpers, cv
                )
                write_to_file(prob)
                hist = hist.at[k].set(prob)
                old_snaps = increase_snaps(w1_snapshots, snaps_0)
                print(f"size of snapshots: {len(old_snaps)}\n")

            K_t = np.prod(hist)
            write_to_file("# Flux Constant")
            write_to_file(K_t)


def _ffs(method,snapshot,helpers):
    cv = method.cv
    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    #initialize method
    def initialize():
        bias = np.zeros((natoms, 3))
        xi, _ = cv(helpers.query(snapshot))
        return FFSState(bias, xi)

    def update(state, data):
        xi, _ = cv(data)
        bias = state.bias
        return FFSState(bias, xi)

    return snapshot, initialize, generalize(update,helpers)


def write_to_file(value):
    with open("ffs_results.dat", "a+") as f:
        f.write(str(value) + "\n")


# Since snapshots are depleted each window, this function restores the list to
# its initial values. This only works with stochastic integrators like BD or
# Langevin, for other, velocity resampling is needed
def increase_snaps(windows, initial_w):
    if len(windows) > 0:
        ratio = len(initial_w) // len(windows)
        windows = windows * ratio

    return windows


def check_input(grid, xi, verbose = False):
    """Verify if the initial configuration is a good one."""
    is_good = xi < grid[0]

    if is_good:
        print("Good initial configuration\n")
        print(xi)
    elif verbose:
        print(xi)

    return is_good


def basin_sampling(
    max_num_snapshots,
    sampling_time,
    grid,
    run,
    sampler,
    reference_snapshot,
    helpers,
    cv
):
    """
    Sampling of basing configurations for initial flux calculations.
    """
    basin_snapshots = []
    win_A = grid[0]
    xi = sampler.state.xi.block_until_ready()

    print("Starting basin sampling\n")
    while len(basin_snapshots) < int(max_num_snapshots):
        run(sampling_time)
        xi = sampler.state.xi.block_until_ready()

        if np.all(xi < win_A):
            snap = copy(sampler.snapshot)
            basin_snapshots.append(snap)
            print("Storing basing configuration with cv value:\n")
            print(xi)
        else:
            helpers.restore(sampler.snapshot, reference_snapshot)
            xi, _ = cv(helpers.query(sampler.snapshot))
            print("Restoring basing configuration since system leave basin with cv value:\n")
            print(xi)

    print(f"Finish sampling Basin with {max_num_snapshots} snapshots\n")

    return basin_snapshots


def initial_flow(
    Num_window0, timestep, grid, initial_snapshots, run, sampler, helpers, cv
):
    """Selects snapshots from list generated with `basin_sampling`."""

    success = 0
    time_count = 0.0
    window0_snaps = []
    win_A = grid[0]

    for i in range(0, Num_window0):
        print(f"Initial stored configuration: {i}\n")
        helpers.restore(sampler.snapshot, initial_snapshots[i])
        xi, _= cv(helpers.query(sampler.snapshot))
        print(xi)

        has_reached_A = False
        while not has_reached_A:
            # this can be used not every timestep
            run(1)
            time_count += timestep
            xi = sampler.state.xi.block_until_ready()

            if np.all(xi >= win_A):
                success += 1
                has_reached_A = True

                if len(window0_snaps) <= Num_window0:
                    snap = copy(sampler.snapshot)
                    window0_snaps.append(snap)

                break

    print(f"Finish Initial Flow with {success} succeses over {time_count} time\n")
    phi_a = float(success) / (time_count)

    return phi_a, window0_snaps    


def running_window(grid, step, old_snapshots, run, sampler, helpers, cv):
    success = 0
    new_snapshots = []
    win_A = grid[0]
    win_value = grid[int(step)]
    has_conf_stored = False

    for i in range(0,len(old_snapshots)):
        helpers.restore(sampler.snapshot, old_snapshots[i])
        xi, _ = cv(helpers.query(sampler.snapshot))
        print(f"Stored configuration: {i} of window: {step}\n")
        print(xi)
        
        # limit running time to avoid zombie trajectories
        # this can be probably be improved
        running = True
        while running:
            run(1)
            xi = sampler.state.xi.block_until_ready()
            
            if np.all(xi < win_A):
                running = False
            elif np.all(xi >= win_value):
                snap = copy(sampler.snapshot)
                new_snapshots.append(snap)
                success += 1
                running = False
                if not has_conf_stored:
                    has_conf_stored = True

    if success == 0:
        raise Exception(f"Error in window {step}\n")

    if len(new_snapshots) > 0:
        prob_local = float(success) / len(old_snapshots)
        print(f"Finish window {step} with {len(new_snapshots)} snapshots\n")
        return prob_local, new_snapshots
