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

## Architecture

The library separates **thermodynamics** (what is the equilibrium) from
**kinetics** (how the system evolves) and **numerics** (how the equations
are solved):

```
Thermodynamics                      Kinetics + Numerics
──────────────                      ───────────────────
FreeEnergy (ABC)                    GridParams
├─ FloryHuggins                       shape, dx, dt
└─ DoubleWell
                                    SpectralSolver
FreeEnergyFunctional                  mobility
  local: FreeEnergy   ──────────►     functional
  kappa: float                        device
  alpha: float                        wall: WallCondition

                                    WallCondition (ABC)
                                    ├─ SurfaceEnergyWall
                                    └─ VolumePenaltyWall
```

- **`FreeEnergyFunctional`** bundles `f(φ)`, κ, and α into a single thermodynamic object.
- **`SpectralSolver`** handles the dynamics and numerical scheme.
  A single solver class covers both CH and OK — the `alpha` parameter
  controls which regime:

| Condition | Physical model | Typical use case |
|---|---|---|
| `alpha = 0` (default) | **Cahn-Hilliard** — local + gradient only | Spinodal decomposition, nucleation and growth |
| `alpha > 0` | **Ohta-Kawasaki** — adds long-range repulsion | Block-copolymer microphase separation (lamellae, cylinders, spheres) |

Solid walls can be added via pluggable **wall conditions**:

```python
from torch_pf import SurfaceEnergyWall, VolumePenaltyWall, channel_walls

mask = channel_walls(256, 64, wall_thickness=5, axis=1)

# Surface energy method (recommended) — γ = σ cos(θ) directly controls contact angle
wall = SurfaceEnergyWall(mask, gamma=0.5, dx=1.0)

# Volume-penalty method (legacy) — penalty drives φ → φ_wall inside the wall
wall = VolumePenaltyWall(mask, phi_wall=1.0, penalty=10.0)

solver = SpectralSolver(functional, grid, mobility=1.0, wall=wall)
```

| Wall method | Chemical potential contribution | Contact angle control |
|---|---|---|
| `SurfaceEnergyWall(mask, γ)` | $-\gamma\,\lvert\nabla\Omega\rvert$ | Direct: $\gamma = \sigma\cos\theta$ |
| `VolumePenaltyWall(mask, φ_w, λ)` | $\lambda\,\Omega(\phi - \phi_w)$ | Indirect via $\phi_w$ and $\lambda$ |

## Equations

### Free energy functional

$$F[\phi] = \int \!\Big[ f(\phi) + \frac{\kappa}{2}|\nabla\phi|^2 \Big] d\mathbf{r} \;+\; \frac{\alpha}{2}\int\!\!\int G(\mathbf{r}-\mathbf{r}')(\phi-\bar\phi)(\phi'-\bar\phi)\, d\mathbf{r}\, d\mathbf{r}'$$

The first integral contains the local free energy density `f(φ)` and
the gradient energy (controlled by κ). The second integral is the
Ohta-Kawasaki long-range interaction (controlled by α; set α = 0 for
standard Cahn-Hilliard).

### Conserved dynamics (Model B)

$$\frac{\partial \phi}{\partial t} = M \nabla^2 \frac{\delta F}{\delta \phi}$$

Solved with a stabilised semi-implicit spectral scheme:

$$\hat{\phi}^{n+1} = \frac{\hat{\phi}^n - \Delta t\, M\, k^2\, \hat{g}^n}{D_k}, \qquad g^n = f'(\phi^n) - C\,\phi^n$$

where $D_k = 1 + \Delta t\, M\, k^2(C + \kappa k^2) + \Delta t\, M\, \alpha\, \mathbb{1}_{k\neq 0}$.
The stabilisation constant $C$ is automatically determined from $\max|f''|$.

### Flory-Huggins free energy

$$f(\phi) = \frac{\phi}{N_A} \ln \phi + \frac{1-\phi}{N_B} \ln(1-\phi) + \chi\,\phi(1-\phi)$$

- Critical point: $\chi_c = \frac{1}{2}\left(\frac{1}{\sqrt{N_A}} + \frac{1}{\sqrt{N_B}}\right)^2$
- The log terms are smoothly replaced by quadratic extensions for $\phi < \varepsilon$ to avoid numerical divergence.

### Double-well free energy

$$f(\phi) = W\,\phi^2(1-\phi)^2$$

## Quick start

```python
from torch_pf import FloryHuggins, FreeEnergyFunctional, GridParams, SpectralSolver, random_uniform

# Thermodynamics
fe = FloryHuggins(chi=0.1, n_a=100, n_b=100)
functional = FreeEnergyFunctional(local=fe, kappa=0.5)

# Grid + solver
grid = GridParams(shape=(128, 128), dx=1.0, dt=0.5)
solver = SpectralSolver(functional, grid, mobility=1.0)

# Run
phi0 = random_uniform(128, 128, phi_mean=0.5, noise_amplitude=0.05, seed=42)
snapshots = solver.run(phi0, n_steps=5000, save_interval=1000)
```

For block-copolymer microphase separation, just set α > 0:

```python
from torch_pf import DoubleWell, FreeEnergyFunctional, GridParams, SpectralSolver, random_uniform

functional = FreeEnergyFunctional(local=DoubleWell(W=1.0), kappa=0.5, alpha=0.3)
grid = GridParams(shape=(128, 128), dx=1.0, dt=0.5)
solver = SpectralSolver(functional, grid, mobility=1.0)
```

## Examples

| Script | Model | Description |
|---|---|---|
| `examples/spinodal_decomposition.py` | CH (α=0) | 2D spinodal decomposition (Flory-Huggins) |
| `examples/spinodal_3d.py` | CH (α=0) | 3D spinodal decomposition with isosurface rendering |
| `examples/nucleation.py` | CH (α=0) | Nucleation: super- vs sub-critical nucleus in metastable state |
| `examples/block_copolymer_2d.py` | OK (α>0) | 2D Ohta-Kawasaki: lamellae and cylinder morphologies |
| `examples/block_copolymer_3d.py` | OK (α>0) | 3D Ohta-Kawasaki: gyroid / lamellar structures |
| `examples/wetting.py` | CH + wall | Spinodal decomposition in a channel with wetting walls |

## Project structure

```
src/torch_pf/
  free_energy.py      Free energy models + FreeEnergyFunctional
  solver.py           SpectralSolver + GridParams
  wall.py             Wall conditions (SurfaceEnergyWall, VolumePenaltyWall)
  initializers.py     Initial condition generators
examples/             Example scripts
tests/                Tests
```

## Tests

```bash
uv run pytest
```
