#!/usr/bin/env python3

from locale import Error
from typing import Callable, Dict, List, Optional

import logging
import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import ExpectedImprovement
from torch import Tensor

from boss.acquisition_functions.budgeted_multi_step_ei import (
    BudgetedMultiStepExpectedImprovement,
)
from boss.budget_scheduling_strategies import largest_one_step_cost
from boss.utils import (
    fit_model,
    generate_initial_design,
    optimize_acqf_and_get_suggested_point,
)


def boss_trial(
    problem: str,
    algo: str,
    algo_params: Optional[Dict],
    trial: int,
    restart: bool,
    objective_function: Callable,
    cost_function: Callable,
    input_dim: int,
    n_init_evals: int,
    budget: float,
    n_max_iter: int,
    ignore_failures: bool = False,
) -> None:
    # Modify algo's name to account for hyperparameters
    if algo == "B-MS-EI":
        algo_id = algo + "_"

        for n in algo_params.get("lookahead_n_fantasies"):
            algo_id += str(n)

        algo_id += "_" + str(int(budget))
    else:
        algo_id = algo

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + algo_id + "/"
    )

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            X = torch.tensor(np.loadtxt(results_folder + "X/X_" + str(trial) + ".txt"))
            Y = torch.tensor(np.loadtxt(results_folder + "Y/Y_" + str(trial) + ".txt"))

            # Historical best observed objective values and running times
            hist_best_obs_vals = list(
                np.loadtxt(results_folder + "best_obs_vals_" + str(trial) + ".txt")
            )
            costs = list(
                np.loadtxt(results_folder + "costs/costs_" + str(trial) + ".txt")
            )
            runtimes = list(
                np.loadtxt(results_folder + "runtimes/runtimes_" + str(trial) + ".txt")
            )

            # Current best observed objective value and cumulative cost
            best_obs_val = torch.tensor(hist_best_obs_vals[-1])
            cumulative_cost = sum(costs)

            iteration = len(hist_best_obs_vals) - 1
            print("Restarting experiment from available data.")

        except:

            # Initial evaluations
            X = generate_initial_design(
                num_samples=n_init_evals, input_dim=input_dim, seed=trial
            )
            Y = objective_function(X)

            # Current best observed objective value and cumulative cost
            best_obs_val = Y.max().item()
            cumulative_cost = 0.0

            # Historical best observed objective values and running times
            hist_best_obs_vals = [best_obs_val]
            costs = []
            runtimes = []

            iteration = 0
    else:
        # Initial evaluations
        X = generate_initial_design(
            num_samples=n_init_evals, input_dim=input_dim, seed=trial
        )
        Y = objective_function(X)

        # Current best observed objective value and cumulative cost
        best_obs_val = Y.max().item()
        cumulative_cost = 0.0

        # Historical best observed objective values and runtimes
        hist_best_obs_vals = [best_obs_val]
        costs = []
        runtimes = []

        iteration = 0

    algo_params["init_budget"] = budget
    cost_function = cost_function.update_reference_point(X[[-1]])

    while cumulative_cost <= budget and iteration <= n_max_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()
        new_x = get_new_suggested_point(
            algo=algo,
            X=X,
            Y=Y,
            costs=costs,
            cost_function=cost_function,
            budget_left=budget - cumulative_cost,
            input_dim=input_dim,
            algo_params=algo_params,
        )

        t1 = time.time()
        runtimes.append(t1 - t0)

        # Evaluate objective at new point
        objective_new_x = objective_function(new_x)
        cost_new_x = cost_function(new_x.unsqueeze(0)).item()
        # Update training data
        X = torch.cat([X, new_x], 0)
        Y = torch.cat([Y, objective_new_x], 0)
        costs.append(cost_new_x)
        cost_function = cost_function.update_reference_point(X[[-1]])

        # Update historical best observed objective values and cumulative cost
        cumulative_cost += cost_new_x
        best_obs_val = Y.max().item()
        hist_best_obs_vals.append(best_obs_val)
        print("Best value found so far: " + str(best_obs_val))
        print("Remaining budget: " + str(budget - cumulative_cost))

        # Save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")
        if not os.path.exists(results_folder + "X/"):
            os.makedirs(results_folder + "X/")
        if not os.path.exists(results_folder + "Y/"):
            os.makedirs(results_folder + "Y/")
        if not os.path.exists(results_folder + "costs/"):
            os.makedirs(results_folder + "costs/")
        np.savetxt(results_folder + "X/X_" + str(trial) + ".txt", X.numpy())
        np.savetxt(
            results_folder + "Y/Y_" + str(trial) + ".txt",
            Y.numpy(),
        )
        np.savetxt(
            results_folder + "costs/costs_" + str(trial) + ".txt", np.atleast_1d(costs)
        )
        np.savetxt(
            results_folder + "best_obs_vals_" + str(trial) + ".txt",
            np.atleast_1d(hist_best_obs_vals),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )


def get_new_suggested_point(
    algo: str,
    X: Tensor,
    Y: Tensor,
    costs: Tensor,
    cost_function: Callable,
    budget_left: float,
    input_dim: int,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    input_dim = X.shape[-1]
    algo_params["budget_left"] = budget_left

    if algo == "Random":
        return torch.rand([1, input_dim])
    elif algo == "B-MS-EI":
        # Model
        model = fit_model(
            X=X,
            Y=Y,
            noiseless_obs=True,
        )

        # Acquisition function
        budget = largest_one_step_cost(
            cost_function, budget_left, input_dim
        )  # torch.clone(torch.tensor(budget_left)).clamp_max(10.0)
        acquisition_function = BudgetedMultiStepExpectedImprovement(
            model=model,
            cost_function=cost_function,
            budget=budget,
            num_fantasies=algo_params.get("lookahead_n_fantasies"),
        )
    elif algo == "EI":
        # Model
        model = fit_model(
            X=X,
            Y=Y,
            noiseless_obs=True,
        )

        acquisition_function = ExpectedImprovement(model=model, best_f=Y.max().item())

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])

    new_x = optimize_acqf_and_get_suggested_point(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=1,
        algo_params=algo_params,
    )

    return new_x
