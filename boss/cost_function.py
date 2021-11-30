from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior, scalarize_posterior
from botorch.utils import apply_constraints
from torch import Tensor
from torch.nn import Module


class CostFunction(Module, ABC):
    r"""Abstract base class for cost functions."""

    ...


class GenericCostFunction(CostFunction):
    r"""Cost function generated from a generic callable."""

    def __init__(
        self, cost_function: Callable, reference_point: Optional[Tensor] = None
    ) -> None:
        r"""Cost function generated from a generic callable.
        Args:
            cost_function: .
            reference_point: A `ref_batch_shape x 1 x d`-dim tensor.
        """
        super().__init__()
        self.cost_function = cost_function
        self.reference_point = reference_point

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the cost function at `X`.
        Args:
            X: A `input_batch_shape x 1 x d`-dim tensor of inputs.
        Returns:
            A `input_batch_shape x ref_batch_shape`-dim Tensor of cost values.
        """
        return self.cost_function(self.reference_point, X)

    def update_reference_point(self, reference_point: Tensor) -> CostFunction:
        r"""."""
        return GenericCostFunction(self.cost_function, reference_point)
