"""Flory-Huggins free energy for binary polymer blends.

The Flory-Huggins free energy density for a binary polymer blend is:

    f(φ) = (φ / N_A) ln(φ) + ((1 - φ) / N_B) ln(1 - φ) + χ φ (1 - φ)

where:
    φ   : volume fraction of polymer A
    N_A : degree of polymerization of polymer A
    N_B : degree of polymerization of polymer B
    χ   : Flory-Huggins interaction parameter
"""

from __future__ import annotations

import torch
from torch import Tensor


class FloryHuggins:
    """Flory-Huggins free energy for binary polymer blends.

    Parameters
    ----------
    chi : float
        Flory-Huggins interaction parameter.
    n_a : float
        Degree of polymerization of polymer A.
    n_b : float
        Degree of polymerization of polymer B.
    """

    def __init__(self, chi: float, n_a: float = 100.0, n_b: float = 100.0) -> None:
        self.chi = chi
        self.n_a = n_a
        self.n_b = n_b

    @property
    def chi_critical(self) -> float:
        """Critical χ value for the blend (spinodal instability onset)."""
        return 0.5 * (1.0 / self.n_a**0.5 + 1.0 / self.n_b**0.5) ** 2

    @property
    def phi_critical(self) -> float:
        """Critical composition."""
        na_sqrt = self.n_a**0.5
        nb_sqrt = self.n_b**0.5
        return nb_sqrt / (na_sqrt + nb_sqrt)

    def free_energy_density(self, phi: Tensor) -> Tensor:
        """Compute Flory-Huggins free energy density f(φ).

        Parameters
        ----------
        phi : Tensor
            Volume fraction field of polymer A.

        Returns
        -------
        Tensor
            Free energy density at each grid point.
        """
        phi_c = phi.clamp(1e-8, 1.0 - 1e-8)
        return (
            (phi_c / self.n_a) * phi_c.log()
            + ((1.0 - phi_c) / self.n_b) * (1.0 - phi_c).log()
            + self.chi * phi_c * (1.0 - phi_c)
        )

    def chemical_potential(self, phi: Tensor) -> Tensor:
        """Compute the bulk chemical potential df/dφ.

        Parameters
        ----------
        phi : Tensor
            Volume fraction field of polymer A.

        Returns
        -------
        Tensor
            Bulk chemical potential at each grid point.
        """
        phi_c = phi.clamp(1e-8, 1.0 - 1e-8)
        return (
            (1.0 / self.n_a) * (phi_c.log() + 1.0)
            - (1.0 / self.n_b) * ((1.0 - phi_c).log() + 1.0)
            + self.chi * (1.0 - 2.0 * phi_c)
        )

    def second_derivative(self, phi: Tensor) -> Tensor:
        """Compute d²f/dφ² (used for spinodal analysis).

        Parameters
        ----------
        phi : Tensor
            Volume fraction field of polymer A.

        Returns
        -------
        Tensor
            Second derivative of free energy density.
        """
        phi_c = phi.clamp(1e-8, 1.0 - 1e-8)
        return (
            1.0 / (self.n_a * phi_c)
            + 1.0 / (self.n_b * (1.0 - phi_c))
            - 2.0 * self.chi
        )
