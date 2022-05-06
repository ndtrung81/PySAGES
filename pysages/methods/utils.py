# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collection of helpful classes for methods.

This includes callback functors (callable classes).
"""

from jax import numpy as np


class HistogramLogger:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable to generate histograms.
    """

    def __init__(self, period: int, offset: int = 0):
        """
        HistogramLogger constructor.

        Arguments
        ---------
        period:
            Timesteps between logging of collective variables.

        offset:
            Timesteps at the beginning of a run used for equilibration.
        """
        self.period = period
        self.counter = 0
        self.offset = offset
        self.data = []

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        self.counter += 1
        if self.counter > self.offset and self.counter % self.period == 0:
            self.data.append(state.xi[0])

    def get_histograms(self, **kwargs):
        """
        Helper function to generate histrograms from the collected CV data.
        `kwargs` are passed on to `numpy.histogramdd` function.
        """
        data = np.asarray(self.data)
        if "density" not in kwargs:
            kwargs["density"] = True
        return np.histogramdd(data, **kwargs)

    def get_means(self):
        """
        Returns mean values of the histogram data.
        """
        data = np.asarray(self.data)
        return np.mean(data, axis=0)

    def get_cov(self):
        """
        Returns covariance matrix of the histgram data.
        """
        data = np.asarray(self.data)
        return np.cov(data.T)

    def reset(self):
        """
        Reset internal state.
        """
        self.counter = 0
        self.data = []


# NOTE: for OpenMM; issue #16 on openmm-dlext should be resolved for this to work properly.
class MetaDLogger:
    """
    Logs the state of the collective variable and other parameters in Metadynamics.
    """

    def __init__(self, hills_file, log_period):
        """
        MetaDLogger constructor.

        Arguments
        ---------
        hills_file:
            Name of the output hills log file.

        log_period:
            Timesteps between logging of collective variables and metadynamics parameters.
        """
        self.hills_file = hills_file
        self.log_period = log_period
        self.counter = 0

    def save_hills(self, xi, sigma, height):
        """
        Append the centers, standard deviations and heights to log file.
        """
        with open(self.hills_file, "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, xi.flatten())) + "\t")
            f.write("\t".join(map(str, sigma.flatten())) + "\t")
            f.write(str(height) + "\n")

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            idx = state.idx - 1 if state.idx > 0 else 0
            self.save_hills(state.centers[idx], state.sigmas, state.heights[idx])

        self.counter += 1
