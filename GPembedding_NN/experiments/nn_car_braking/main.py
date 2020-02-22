# File: main.py
# File Created: Friday, 21st February 2020 7:56:44 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Demonstrate embedding inputs with fully-connected neural nets.

Toy data is the "car braking" problem (inspired by Dai et al., 2017): given 
a car type (general input) and initial speed (real input), predict the stopping
distance (output).

You could extend this by...
* Having a "driver" as well as a car (or other general input dimensions)
* Minibatching the loss computation
* More efficient predictions when doing multiple samples from the latents' posterior.

Usage:
python main.py
"""

import sys
import os

from gptorch.param import Param
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.transforms import ExpTransform
from tqdm import tqdm

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if not src_path in sys.path:
    sys.path.append(src_path)

from src.embedders import Embedder, GaussianEmbedder

np.random.seed(42)
torch.manual_seed(42)


class Model(nn.Module):
    """
    Embedder into NN.

    p(y|xr, xg) = \int p(y|xr, z) q(z|xg) dz
    """

    def __init__(self, net: nn.Module, dz, n_train):
        """
        :param net: module to predict the mean of the predicted outputs 
        (will build a Gaussian around them.)
        """
        super().__init__()

        self._net = net
        self.embedder = GaussianEmbedder(dz)
        # Need this to properly scale the minibatch ELBO estimate
        self.n_train = n_train
        # Point estimate of noise
        self._sigma_y = Param(torch.ones(1), transform=ExpTransform())

    @property
    def sigma_y(self):
        return self._sigma_y.transform()

    def forward(
        self, xr: torch.Tensor, xg: np.ndarray = None, z=None, mode="posterior"
    ) -> torch.Tensor:
        """
        Sample z and compute the conditional output mean for y.
        """

        if z is None:
            assert xg is not None, "Need xg if not z"
            assert mode in ["prior", "posterior"]
            self.embedder.mode = mode
            # First, get a sample of the latents:
            z = self.embedder(xg[:, 0])  # Could do multiple general input spaces...
        else:
            assert xg is None, "can't give z and xg"

        # Then, map these and the real inputs through the net.
        return self._net(torch.cat([xr, z], dim=1))

    def loss(self, xr, xg, targets):
        """
        Minibatch estimate of the ELBO
        """

        self.embedder.mode = "posterior"
        z = self.embedder(xg[:, 0])  # Could do multiple general input spaces...
        mu_y = self(xr, z=z)

        mll = (
            self.n_train
            * Normal(mu_y, self.sigma_y).log_prob(targets).mean(dim=0).sum()
        )
        # Again, only a single general input dimension here.
        kl = self._kl_xg_gaussian(xg[:, 0])

        elbo = mll - kl

        return -elbo

    def _kl_xg_gaussian(self, xg):
        """
            KL divergence term.

            :param e: The embedder
            :type e: Embedder
            :param xg: general inputs (a single dimension)
            :type xg: np.ndarray, 1D

            :return: TensorType(0D?)
            """

        # Stack over the unique general inputs (don't double-count)
        e = self.embedder
        return torch.stack(
            [
                torch.distributions.kl.kl_divergence(
                    torch.distributions.Normal(
                        e.loc[xg_ij], e.scale[xg_ij].transform()
                    ),
                    torch.distributions.Normal(
                        torch.zeros(e.d_out), torch.ones(e.d_out)
                    ),
                )
                for xg_ij in Embedder.clean_inputs(np.unique(xg))
            ]
        ).sum()


def make_data():
    data = {}
    n = 32
    for name, coef in zip(["truck", "car", "sport"], [1.0, 0.5, 0.25]):
        v = torch.rand(n, 1)
        d = coef * v ** 2
        data[name] = {"xg": np.array([[name]] * n), "xr": v, "y": d}

    # plt.figure()
    # for vehicle_type in ["truck", "car", "sport"]:
    #     plt.plot(
    #         data[vehicle_type]["xr"].detach().cpu().numpy(),
    #         data[vehicle_type]["y"].detach().cpu().numpy(),
    #         linestyle="none",
    #         marker=".",
    #         label=vehicle_type
    #     )
    # plt.xlabel("Initial velocity")
    # plt.ylabel("Stopping distance")
    # plt.legend()
    # plt.show()

    return data


def get_data():
    data = make_data()

    # Watch out: xr and y are torch, but xg is np (torch doesn't do strings?)
    train, test = {}, {}
    train["xr"] = torch.cat(
        [data["truck"]["xr"], data["car"]["xr"], data["sport"]["xr"][:2]]
    )
    train["xg"] = np.concatenate(
        [data["truck"]["xg"], data["car"]["xg"], data["sport"]["xg"][:2]]
    )
    train["y"] = torch.cat(
        [data["truck"]["y"], data["car"]["y"], data["sport"]["y"][:2]]
    )

    test["xr"] = data["sport"]["xr"][2:]
    test["xg"] = data["sport"]["xg"][2:]
    test["y"] = data["sport"]["y"][2:]

    return train, test


def show_predictions(model, data, title=None, vehicle_types=None):
    n_test = 200
    vehicle_types = (
        ["truck", "car", "sport"] if vehicle_types is None else vehicle_types
    )
    with torch.no_grad():
        xr_test = torch.linspace(0, 1.3, n_test)[:, None]
        dy = 2.0 * np.sqrt(model.sigma_y.detach().cpu().numpy())

        plt.figure()
        for i, vehicle_type in enumerate(vehicle_types):
            xg_test = np.array([[vehicle_type]] * n_test)
            for _ in range(32):  # Samples of q(z|xg)
                y = model(xr_test, xg=xg_test).detach().cpu().numpy()[:, 0]

                # plt.fill_between(
                #     xr_test.detach().cpu().numpy().flatten(),
                #     y - dy,
                #     y + dy,
                #     color="C%i" % i,
                #     alpha=0.01
                # )
                plt.plot(
                    xr_test.detach().cpu().numpy().flatten(),
                    y,
                    color="C%i" % i,
                    alpha=0.3,
                )

            idx = np.where(data["xg"][:, 0] == vehicle_type)
            plt.plot(
                data["xr"][idx],
                data["y"][idx],
                ".",
                color="C%i" % i,
                label=vehicle_type,
            )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_train, data_test = get_data()
    dx = data_train["xr"].shape[1]
    dz = 4
    dy = data_train["y"].shape[1]

    model = Model(
        nn.Sequential(
            nn.Linear(dx + dz, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dy),
        ),
        dz,
        data_train["xr"].shape[0],
    )

    # Train
    # NOTE: I have to call the model's embedder (in posterior mode) so that the
    # (variational) parameters in the embedder exist. Otherwise, the optimizer
    # won't track them!
    model.embedder.mode = "posterior"
    model.embedder(data_train["xg"][:, 0])  # Single general input space, so [:,0]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    print("Training...")
    for _ in tqdm(range(1000)):
        optimizer.zero_grad()
        loss = model.loss(data_train["xr"], data_train["xg"], data_train["y"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    plt.figure()
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.show()

    show_predictions(model, data_train, title="Train data")
    show_predictions(model, data_test, title="Test data", vehicle_types=["sport"])
