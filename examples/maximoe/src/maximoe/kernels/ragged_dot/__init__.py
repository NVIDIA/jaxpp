# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-independent ragged-dot operations for MaxiMoE.

Operations accept ``implementation="xla"``, ``"sm90"``, or ``"sm100"``;
``None`` selects the local platform's default. The HiJAX primitives in
:mod:`maximoe.kernels.ragged_dot.ops` own differentiation and Ref
accumulation, while ``xla``, ``sm90``, and ``sm100`` implement one raw kernel
surface. The ``xla`` module is also an independently differentiable
specification.
"""

from maximoe.kernels.ragged_dot.common import (
    SWIGLU_EPILOGUE_BLOCK as SWIGLU_EPILOGUE_BLOCK,
)
from maximoe.kernels.ragged_dot.ops import IMPLEMENTATIONS as IMPLEMENTATIONS
from maximoe.kernels.ragged_dot.ops import RaggedDot as RaggedDot
from maximoe.kernels.ragged_dot.ops import RaggedDotCombine as RaggedDotCombine
from maximoe.kernels.ragged_dot.ops import RaggedDotSwiGLU as RaggedDotSwiGLU
from maximoe.kernels.ragged_dot.ops import get_implementation as get_implementation
from maximoe.kernels.ragged_dot.ops import ragged_dot as ragged_dot
from maximoe.kernels.ragged_dot.ops import ragged_dot_combine as ragged_dot_combine
from maximoe.kernels.ragged_dot.ops import ragged_dot_dw as ragged_dot_dw
from maximoe.kernels.ragged_dot.ops import ragged_dot_dx as ragged_dot_dx
from maximoe.kernels.ragged_dot.ops import ragged_dot_swiglu as ragged_dot_swiglu

# Groups need no padding; only the whole buffer keeps tile alignment.
RAGGED_BUFFER_ALIGNMENT = 256
