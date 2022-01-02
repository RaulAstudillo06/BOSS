from itertools import product
import torch


def largest_one_step_cost(cost_function, remaining_budget, input_dim):
    vertices = torch.tensor(list(product((0.0, 1.0), repeat=input_dim))).unsqueeze(0)
    costs = cost_function(vertices)
    suggested_budget = costs.max().item() + 3.0
    suggested_budget = min(suggested_budget, remaining_budget)
    return suggested_budget


# def base_policy_cost(cost_function, remaining_budget, X, Y):
