# gp_interface.py
from __future__ import annotations

from typing import Callable, Tuple

import gpytorch
import numpy as np
import torch


def train_gp_model(
    gp_model, torch_seed=None, training_iterations=200, learning_rate=0.1
):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    likelihood = gp_model.likelihood
    train_x = gp_model.train_inputs[0]
    train_y = gp_model.train_targets

    # Find optimal model hyperparameters
    gp_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gp_model.parameters()), lr=learning_rate
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    prev_loss = np.inf
    num_loss_below_threshold = 0

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = likelihood(gp_model(train_x))
        loss = -mll(output, train_y.reshape((train_y.numel(),)))

        num_loss_below_threshold = (
            num_loss_below_threshold + 1 if abs(loss - prev_loss) < 5e-4 else 0
        )

        if num_loss_below_threshold > 5:
            print(f"stopping GP optimization early after {i} iterations.")
            break

        prev_loss = loss

        loss.backward()
        if (i + 1) % 20 == 0:
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))

        optimizer.step()

    gp_model.eval()
    likelihood.eval()
    return gp_model, likelihood
