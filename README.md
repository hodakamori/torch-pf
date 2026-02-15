# torch-pf

A PyTorch-based phase-field simulation library for 2D/3D polymer systems.
GPU-accelerated spectral (FFT) solvers for Cahn-Hilliard and Ohta-Kawasaki equations.

<p align="center">
  <img src="examples/results/block_copolymer_2d.png" width="90%" />
</p>

## Installation

```bash
uv sync                    # basic
uv sync --extra vis3d      # 3D isosurface rendering (scikit-image)
```

## Features

| Solver | Equation | Use case |
|---|---|---|
| `CahnHilliardSolver` | Cahn-Hilliard | Spinodal decomposition of binary blends |
| `OhtaKawasakiSolver` | Ohta-Kawasaki | Block-copolymer microphase separation |

| Free energy model | Formula |
|---|---|
| `FloryHuggins` | Regularized Flory-Huggins with safe log |
| `DoubleWell` | $W\,\phi^2(1-\phi)^2$ |

| Initializer | Description |
|---|---|
| `random_uniform` | Mean $\phi_0$ + small noise (for spinodal decomposition) |
| `droplet` | tanh-profiled circular/spherical droplet |

## Equations

### Cahn-Hilliard equation

$$\frac{\partial \phi}{\partial t} = M \nabla^2 \mu, \qquad \mu = f'(\phi) - \kappa \nabla^2 \phi$$

Solved with a stabilized semi-implicit spectral scheme:

$$\hat{\phi}^{n+1} = \frac{\hat{\phi}^n - \Delta t\, M\, k^2\, \hat{g}^n}{1 + \Delta t\, M\, k^2\, (C + \kappa\, k^2)}, \qquad g^n = f'(\phi^n) - C\,\phi^n$$

The stabilization constant $C$ is automatically determined from $\max|f''|$.

### Ohta-Kawasaki equation

Extends Cahn-Hilliard with a long-range interaction term:

$$\frac{\partial \phi}{\partial t} = M \nabla^2 \mu - \alpha\,(\phi - \bar\phi)$$

The parameter $\alpha$ controls the strength of the long-range penalty.
Larger $\alpha$ narrows the band of unstable modes, producing well-ordered
periodic structures (lamellae, hexagonal cylinders).

### Flory-Huggins free energy

$$f(\phi) = \frac{\phi}{N_A} \ln \phi + \frac{1-\phi}{N_B} \ln(1-\phi) + \chi\,\phi(1-\phi)$$

- Critical point: $\chi_c = \frac{1}{2}\left(\frac{1}{\sqrt{N_A}} + \frac{1}{\sqrt{N_B}}\right)^2$
- The log terms are smoothly replaced by quadratic extensions for $\phi < \varepsilon$ to avoid numerical divergence.

## Quick start

```python
from torch_pf import CahnHilliardSolver, FloryHuggins, SimulationParams, random_uniform

fe = FloryHuggins(chi=0.1, n_a=100, n_b=100)
params = SimulationParams(shape=(128, 128), dx=1.0, dt=0.5, mobility=1.0, kappa=0.5)
solver = CahnHilliardSolver(params, fe)
phi0 = random_uniform(128, 128, phi_mean=0.5, noise_amplitude=0.05, seed=42)
snapshots = solver.run(phi0, n_steps=5000, save_interval=1000)
```

## Examples

| Script | Description |
|---|---|
| `examples/spinodal_decomposition.py` | 2D spinodal decomposition (Flory-Huggins) |
| `examples/spinodal_3d.py` | 3D spinodal decomposition with isosurface rendering |
| `examples/block_copolymer_2d.py` | 2D Ohta-Kawasaki: lamellae and cylinder morphologies |
| `examples/block_copolymer_3d.py` | 3D Ohta-Kawasaki: gyroid / lamellar structures |

## Project structure

```
src/torch_pf/
  cahn_hilliard.py    Cahn-Hilliard solver (spectral FFT)
  ohta_kawasaki.py    Ohta-Kawasaki solver (extends CH)
  free_energy.py      Free energy models
  initializers.py     Initial condition generators
examples/             Example scripts
tests/                Tests
```

## Tests

```bash
uv run pytest
```
