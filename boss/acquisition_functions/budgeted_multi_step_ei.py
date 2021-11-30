#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_step_lookahead import (
    qMultiStepLookahead,
    TAcqfArgConstructor,
    _compute_stage_value,
    _construct_sample_weights,
)
from botorch.acquisition.objective import AcquisitionObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    t_batch_mode_transform,
)
from boss.acquisition_functions.budgeted_ei import (
    BudgetedExpectedImprovement,
)
from boss.samplers.posterior_mean_sampler import PosteriorMeanSampler
from torch import Tensor
from torch.nn import Module


class BudgetedMultiStepExpectedImprovement(qMultiStepLookahead):
    r"""Budgeted Multi-Step Look-Ahead Expected Improvement (one-shot optimization)."""

    def __init__(
        self,
        model: Model,
        cost_function: Callable,
        budget: Union[float, Tensor],
        num_fantasies: Optional[List[int]] = None,
        samplers: Optional[List[MCSampler]] = None,
        X_pending: Optional[Tensor] = None,
        collapse_fantasy_base_samples: bool = True,
    ) -> None:
        r"""Budgeted Multi-Step Expected Improvement.

        Args:
            model: .
            cost_function: .
            budget: A value determining the budget constraint.
            batch_size: Batch size of the current step.
            lookahead_batch_sizes: A list `[q_1, ..., q_k]` containing the batch sizes for the
            `k` look-ahead steps.
            num_fantasies: A list `[f_1, ..., f_k]` containing the number of fantasy
                for the `k` look-ahead steps.
            samplers: A list of MCSampler objects to be used for sampling fantasies in
                each stage.
            X_pending: A `m x d`-dim Tensor of `m` design points that have points that
                have been submitted for function evaluation but have not yet been
                evaluated. Concatenated into `X` upon forward call. Copied and set to
                have no gradient.
            collapse_fantasy_base_samples: If True, collapse_batch_dims of the Samplers
                will be applied on fantasy batch dimensions as well, meaning that base
                samples are the same in all subtrees starting from the same level.
        """
        # TODO: This objective is never really used.
        weights = torch.zeros(model.num_outputs, dtype=torch.double)
        weights[0] = 1.0
        objective = ScalarizedObjective(weights=weights)

        lookahead_batch_sizes = [1 for _ in num_fantasies]

        n_lookahead_steps = len(lookahead_batch_sizes) + 1

        valfunc_cls = [BudgetedExpectedImprovement for _ in range(n_lookahead_steps)]

        valfunc_argfacs = [budgeted_ei_argfac for _ in range(n_lookahead_steps)]

        # Set samplers
        if samplers is None:
            # The batch_range is not set here and left to sampler default of (0, -2),
            # meaning that collapse_batch_dims will be applied on fantasy batch dimensions.
            # If collapse_fantasy_base_samples is False, the batch_range is updated during
            # the forward call.
            samplers: List[MCSampler] = [
                PosteriorMeanSampler(collapse_batch_dims=True)
                if nf == 1
                else SobolQMCNormalSampler(
                    num_samples=nf, resample=False, collapse_batch_dims=True
                )
                for nf in num_fantasies
            ]

        super().__init__(
            model=model,
            batch_sizes=lookahead_batch_sizes,
            samplers=samplers,
            valfunc_cls=valfunc_cls,
            valfunc_argfacs=valfunc_argfacs,
            objective=objective,
            X_pending=X_pending,
            collapse_fantasy_base_samples=collapse_fantasy_base_samples,
        )
        self.cost_function = cost_function
        self.budget = budget

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qMultiStepLookahead on the candidate set X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        Xs = self.get_multi_step_tree_input_representation(X)

        # set batch_range on samplers if not collapsing on fantasy dims
        if not self._collapse_fantasy_base_samples:
            self._set_samplers_batch_range(batch_shape=X.shape[:-2])

        return _step(
            model=self.model,
            cost_function=self.cost_function,
            Xs=Xs,
            samplers=self.samplers,
            valfunc_cls=self._valfunc_cls,
            valfunc_argfacs=self._valfunc_argfacs,
            inner_samplers=self.inner_samplers,
            objective=self.objective,
            budget=self.budget,
            running_val=None,
        )


def _step(
    model: Model,
    cost_function: Callable,
    Xs: List[Tensor],
    samplers: List[Optional[MCSampler]],
    valfunc_cls: List[Optional[Type[AcquisitionFunction]]],
    valfunc_argfacs: List[Optional[TAcqfArgConstructor]],
    inner_samplers: List[Optional[MCSampler]],
    objective: AcquisitionObjective,
    budget: Tensor,
    running_val: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    step_index: int = 0,
) -> Tensor:
    r"""Recursive multi-step look-ahead computation.

    Helper function computing the "value-to-go" of a multi-step lookahead scheme.

    Args:
        model: A Model of appropriate batch size. Specifically, it must be possible to
            evaluate the model's posterior at `Xs[0]`.
        Xs: A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.
        samplers: A list of `k - j` samplers, such that the number of samples of sampler
            `i` is `f_i`. The last element of this list is considered the
            "inner sampler", which is used for evaluating the objective in case it is an
            MCAcquisitionObjective.
        valfunc_cls: A list of acquisition function class to be used as the (stage +
            terminal) value functions. Each element (except for the last one) can be
            `None`, in which case a zero stage value is assumed for the respective
            stage.
        valfunc_argfacs: A list of callables that map a `Model` and input tensor `X` to
            a dictionary of kwargs for the respective stage value function constructor.
            If `None`, only the standard `model`, `sampler` and `objective` kwargs will
            be used.
        inner_samplers: A list of `MCSampler` objects, each to be used in the stage
            value function at the corresponding index.
        objective: The AcquisitionObjective under which the model output is evaluated.
        running_val: As `batch_shape`-dim tensor containing the current running value.
        sample_weights: A tensor of shape `f_i x .... x f_1 x batch_shape` when called
            in the `i`-th step by which to weight the stage value samples. Used in
            conjunction with Gauss-Hermite integration or importance sampling. Assumed
            to be `None` in the initial step (when `step_index=0`).
        step_index: The index of the look-ahead step. `step_index=0` indicates the
            initial step.

    Returns:
        A `b`-dim tensor containing the multi-step value of the design `X`.
    """
    X = Xs[0]
    if sample_weights is None:  # only happens in the initial step
        sample_weights = torch.ones(*X.shape[:-2], device=X.device, dtype=X.dtype)

    # compute stage value
    stage_val = _compute_stage_value(
        model=model,
        valfunc_cls=valfunc_cls[0],
        X=X,
        objective=objective,
        inner_sampler=inner_samplers[0],
        arg_fac=valfunc_argfacs[0](budget),
    )
    if stage_val is not None:  # update running value
        # if not None, running_val has shape f_{i-1} x ... x f_1 x batch_shape
        # stage_val has shape f_i x ... x f_1 x batch_shape

        # this sum will add a dimension to running_val so that
        # updated running_val has shape f_i x ... x f_1 x batch_shape
        running_val = stage_val if running_val is None else running_val + stage_val

    # base case: no more fantasizing, return value
    if len(Xs) == 1:
        # compute weighted average over all leaf nodes of the tree
        batch_shape = running_val.shape[step_index:]
        # expand sample weights to make sure it is the same shape as running_val,
        # because we need to take a sum over sample weights for computing the
        # weighted average
        sample_weights = sample_weights.expand(running_val.shape)
        return (running_val * sample_weights).view(-1, *batch_shape).sum(dim=0)

    # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
    prop_grads = step_index > 0  # need to propagate gradients for steps > 0
    fantasy_model = model.fantasize(
        X=X, sampler=samplers[0], observation_noise=True, propagate_grads=prop_grads
    )

    # augment sample weights appropriately
    sample_weights = _construct_sample_weights(
        prev_weights=sample_weights, sampler=samplers[0]
    )

    # update budget
    new_budget = budget - cost_function(X)

    # update cost function
    new_cost_function = cost_function.update_reference_point(X)

    return _step(
        model=fantasy_model,
        cost_function=new_cost_function,
        Xs=Xs[1:],
        samplers=samplers[1:],
        valfunc_cls=valfunc_cls[1:],
        valfunc_argfacs=valfunc_argfacs[1:],
        inner_samplers=inner_samplers[1:],
        objective=objective,
        budget=new_budget,
        running_val=running_val,
        sample_weights=sample_weights,
        step_index=step_index + 1,
    )


class budgeted_ei_argfac(Module):
    r"""Extract the best observed value and reamaining budget from the model."""

    def __init__(self, budget: Union[float, Tensor]) -> None:
        super().__init__()
        self.budget = budget

    def forward(self, model: Model, X: Tensor) -> Dict[str, Any]:
        y = torch.transpose(model.train_targets, -2, -1)
        y_original_scale = model.outcome_transform.untransform(y)[0]
        obj_vals = y_original_scale[..., 0]
        params = {
            "best_f": obj_vals.max(dim=-1, keepdim=True).values,
            "budget": self.budget,
        }
        return params
