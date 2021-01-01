# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Hummingbird.ml is a compiler for translating traditional ML operators (e.g., tree-based models) and featurizers
(e.g., one-hot encoding) into tensor operations.
Through Hummingbird, DNN frameworks can be used for both optimizing and enabling seamless hardware acceleration of traditional ML.
"""


# Register constants used for Hummingbird extra configs.
from . import supported as hummingbird_constants
from ._utils import _Constants

# Add constants in scope.
constants = _Constants(hummingbird_constants)

# Add the converters in the Hummingbird scope.
from .convert import convert, convert_batch  # noqa: F401, E402

# Add the supported backends in scope.
from .supported import backends  # noqa: F401, E402

# Add load capabilities.
from .containers import PyTorchSklearnContainer as TorchContainer  # noqa: F401, E402
from .containers import TVMSklearnContainer as TVMContainer  # noqa: F401, E402
from .containers import ONNXSklearnContainer as ONNXContainer  # noqa: F401, E402
from ._utils import load  # noqa: F401, E402
