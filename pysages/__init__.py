# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
PySAGES: Advanced sampling methods
"""

import os
# Check for user set memory environment for XLA/JAX
if not ("XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ
        or "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ
        or "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ):
    # If not set be user, disable preallocate to enable multiple/growing
    # simulation memory footprints
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# TODO: detect local number of GPUs per node (currently assumes 4)
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) % 4)
except KeyError:
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['MV2_COMM_WORLD_LOCAL_RANK']) % 4)
    except KeyError:
        # raise EnvironmentError("A valid MPI library was not found")
        pass

from ._version import version as __version__
from ._version import version_tuple

from .backends import (
    ContextWrapper,
    supported_backends,
)

from .grids import (
    Grid,
)

from . import (
    collective_variables,
    methods,
)
