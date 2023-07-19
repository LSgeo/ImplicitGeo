import random

import numpy as np
import torch
from torch import nn

torch.manual_seed(seed := 0)
random.seed(seed)
rng = np.random.default_rng()


class RLoss:
    """Regularisation loss for coordinate MLPs.
    From: Ramasinghe, S., MacDonald, L.E., Lucey, S., 2022.
    On the Frequency-bias of Coordinate-MLPs.
    Presented at the Advances in Neural Information Processing Systems.
    """

    def __init__(self, Sigma: float = 1e-3) -> None:
        """Regularisation loss for coordinate MLPs.
        Args:
            val: Represents search radius for the regularisation loss. "Small values".
                Possible hyperparameter (controlling... local smoothness?)
                These vals are relative to the Norm coord space - i.e. (-1,1)
        """
        self.Eps = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(3),
            covariance_matrix=torch.eye(3) * Sigma,
        )

    def __call__(
        self, model: torch.nn.Module, xbar: torch.Tensor, n_samples: int, device="cuda"
    ) -> torch.Tensor:
        """Equation 11
        Args:
            model: Coordinate MLP we are using
            xbar: Randomly selected from the coordinate space
            n_samples: Number of random coord samples
        """
        Eps = self.Eps.sample((1, n_samples)).to(device)  # batched 1

        return torch.linalg.vector_norm(
            (model.forward_until_g(xbar)) - model.forward_until_g(xbar + Eps)
        ) / torch.linalg.vector_norm(Eps)


### Below attr: SIREN. Sitzmann, Martel, Bergman, Lindell, Wetzstein, 2020.
# Implicit Neural Representations with Periodic Activation Functions,
# in: Advances in Neural Information Processing Systems.
# Originally written by the above authors at https://github.com/vsitzmann/siren
# LS modified for 3rd (z) dimension (upwards)


def get_mgrid(sidelen, dim=3):
    """Generates a flattened grid of ((x,y,z),...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    z: np.array(), raw sample height data. unnormalised at this stage.
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_until_g(self, coords):
        """Forward pass until penultimate layer for RLoss"""
        g = self.net[:-2]
        return g(coords)

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


####
####
####

## WIRE:
# Saragadam, LeJeune, Tan, Balakrishnan, Veeraraghavan, Baraniuk, 2023
# WIRE: Wavelet Implicit Neural Representations.
# https://doi.org/10.48550/arXiv.2301.05187
# Originally written by the above authors at https://github.com/vishwa91/


class RealGaborLayer(nn.Module):
    """
    Implicit representation with Gabor nonlinearity

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega: Frequency of Gabor sinusoid term
        scale: Scaling of Gabor Gaussian term
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=10.0,
        trainable=False,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale**2))


class ComplexGaborLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega0: Frequency of Gabor sinusoid term
        sigma0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=40.0,
        trainable=False,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30.0,
        scale=10.0,
        pos_encode=False,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
    ):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 2
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = "gabor"

        # Legacy parameter
        self.pos_encode = False

        self.net = []
        self.net.append(
            self.nonlin(
                in_features,
                hidden_features,
                omega0=first_omega_0,
                sigma0=scale,
                is_first=True,
                trainable=False,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    omega0=hidden_omega_0,
                    sigma0=scale,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input

        if self.wavelet == "gabor":
            return output.real, coords

        return output, coords

    def forward_until_g(self, coords):
        """Forward pass until penultimate layer for RLoss"""
        g = self.net[:-2]
        return g(coords)


class ComplexGaborLayer2D(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity with 2D activation function

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega0: Frequency of Gabor sinusoid term
        sigma0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=10.0,
        trainable=False,
        mode_3d=False,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.mode_3d = mode_3d

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        if self.mode_3d:
            # Third Guassian window
            self.scale_orth_z = nn.Linear(
                in_features, out_features, bias=bias, dtype=dtype
            )

    def forward(self, input):
        lin = self.linear(input)

        scale_x = lin
        scale_y = self.scale_orth(input)

        freq_term = torch.exp(1j * self.omega_0 * lin)

        if self.mode_3d:
            scale_z = self.scale_orth_z(input)
            arg = (
                scale_x.abs().square() + scale_y.abs().square() + scale_z.abs().square()
            )
        else:
            arg = scale_x.abs().square() + scale_y.abs().square()

        gauss_term = torch.exp(-self.scale_0 * self.scale_0 * arg)

        return freq_term * gauss_term


class INR2D(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=10,
        hidden_omega_0=10.0,
        scale=10.0,
        pos_encode=False,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
        mode_3d=False,
    ):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer2D

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 4
        hidden_features = int(hidden_features / 2)
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = "gabor"

        # Legacy parameter
        self.pos_encode = False

        self.net = []
        self.net.append(
            self.nonlin(
                in_features,
                hidden_features,
                omega0=first_omega_0,
                sigma0=scale,
                is_first=True,
                trainable=False,
                mode_3d=mode_3d,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    omega0=hidden_omega_0,
                    sigma0=scale,
                    mode_3d=mode_3d,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input

        if self.wavelet == "gabor":
            return output.real, coords

        return output, coords

    def forward_until_g(self, coords):
        """Forward pass until penultimate layer for RLoss"""
        g = self.net[:-2]
        return g(coords)


def get_INR(
    nonlin,
    in_features,
    hidden_features,
    hidden_layers,
    out_features,
    outermost_linear=True,
    first_omega_0=30,
    hidden_omega_0=30,
    scale=10,
    pos_encode=False,
    sidelength=512,
    fn_samples=None,
    use_nyquist=True,
):
    """
    Function to get a class instance for a given type of implicit neural representation

    Inputs:
        nonlin: One of 'gauss', 'mfn', 'posenc', 'siren', 'wire', 'wire2d'
        in_features: Number of input features. 2 for image, 3 for volume and so on.
        hidden_features: Number of features per hidden layer
        hidden_layers: Number of hidden layers
        out_features; Number of outputs features. 3 for colorimage, 1 for grayscale or volume and so on
        outermost_linear (True): If True, do not apply nonlin just before output
        first_omega0 (30): For siren and wire only: Omega for first layer
        hidden_omega0 (30): For siren and wire only: Omega for hidden layers
        scale (10): For wire and gauss only: Scale for Gaussian window
        pos_encode (False): If True apply positional encoding
        sidelength (512): if pos_encode is true, use this for side length parameter
        fn_samples (None): Redundant parameter
        use_nyquist (True): if True, use nyquist sampling for positional encoding
    Output: An INR class instance
    """

    if nonlin == "wire":
        return INR(
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
            scale,
            pos_encode,
            sidelength,
            fn_samples,
            use_nyquist,
        )
    elif nonlin == "wire2d":
        return INR2D(
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
            scale,
            pos_encode,
            sidelength,
            fn_samples,
            use_nyquist,
            mode_3d=False,
        )
    elif nonlin == "wire3d":
        return INR2D(
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
            scale,
            pos_encode,
            sidelength,
            fn_samples,
            use_nyquist,
            mode_3d=True,
        )
    else:
        return NotImplementedError(f"{nonlin} not implemented.")
