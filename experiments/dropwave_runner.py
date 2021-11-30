import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.synthetic import DropWave

from torch import Tensor

torch.set_default_dtype(torch.float64)
debug._set_state(True)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from boss.cost_function import GenericCostFunction
from boss.experiment_manager import experiment_manager


# Objective and cost functions
def objective_function(X: Tensor) -> Tensor:
    X_unnorm = (X * 10.24) - 5.12
    dropwave = DropWave()
    objective_X = -dropwave.evaluate_true(X_unnorm)
    return objective_X


def cost_function_callable(reference_point: Tensor, X: Tensor) -> Tensor:
    cost_X = torch.cdist(
        reference_point.view(
            torch.Size([1]) + reference_point.shape[:-2] + reference_point.shape[-1:]
        ),
        X.view(torch.Size([1]) + X.shape[:-2] + X.shape[-1:]),
    )
    cost_X *= 10.24
    cost_X += 0.1
    return cost_X


cost_function = GenericCostFunction(cost_function_callable)


# Algos
algo = "B-MS-EI"
if algo == "B-MS-EI":
    algo_params = {"lookahead_n_fantasies": [1, 1, 1]}
else:
    algo_params = {}

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="dropwave",
    algo=algo,
    algo_params=algo_params,
    restart=False,
    first_trial=first_trial,
    last_trial=last_trial,
    objective_function=objective_function,
    cost_function=cost_function,
    input_dim=2,
    n_init_evals=6,
    budget=10.0,
)
