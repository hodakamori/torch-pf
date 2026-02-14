# torch-pf

PyTorch ベースのフェーズフィールド (Cahn-Hilliard) シミュレーションライブラリ。2D / 3D 対応。

## インストール

```bash
uv sync            # 基本
uv sync --extra vis3d  # 3D 等値面描画 (scikit-image)
```

## プロジェクト構成

```
src/torch_pf/         パッケージ本体
  cahn_hilliard.py    Cahn-Hilliard ソルバー
  free_energy.py      自由エネルギーモデル
  initializers.py     初期条件生成
examples/             実行例スクリプト
tests/                テスト
```

## 実装されている式

### Cahn-Hilliard 方程式

$$\frac{\partial \phi}{\partial t} = M \nabla^2 \mu, \qquad \mu = f'(\phi) - \kappa \nabla^2 \phi$$

スペクトル法 (FFT) による安定化半陰解法で解く:

$$\hat{\phi}^{n+1} = \frac{\hat{\phi}^n - \Delta t\, M\, k^2\, \hat{g}^n}{1 + \Delta t\, M\, k^2\, (C + \kappa\, k^2)}, \qquad g^n = f'(\phi^n) - C\,\phi^n$$

$C$ は $|f''|$ の最大値から自動決定される安定化定数。

### 自由エネルギーモデル

**Flory-Huggins** (正則化済み):

$$f(\phi) = \frac{\phi}{N_A} \ln \phi + \frac{1-\phi}{N_B} \ln(1-\phi) + \chi\,\phi(1-\phi)$$

- 臨界点: $\chi_c = \frac{1}{2}\left(\frac{1}{\sqrt{N_A}} + \frac{1}{\sqrt{N_B}}\right)^2$
- 対数項は $\phi < \varepsilon$ で二次関数に滑らかに接続し、数値発散を回避

**Double-Well**:

$$f(\phi) = W\,\phi^2(1-\phi)^2$$

### 初期条件

| 関数 | 説明 |
|---|---|
| `random_uniform(*shape)` | 平均 $\phi_0$ + 微小ノイズ (スピノーダル分解用) |
| `droplet(*shape)` | tanh 型の円形/球状ドロプレット |

## 使い方

```python
from torch_pf import CahnHilliardSolver, FloryHuggins, SimulationParams, random_uniform

fe = FloryHuggins(chi=0.1, n_a=100, n_b=100)
params = SimulationParams(shape=(128, 128), dx=1.0, dt=0.5, mobility=1.0, kappa=0.5)
solver = CahnHilliardSolver(params, fe)
phi0 = random_uniform(128, 128, phi_mean=0.5, noise_amplitude=0.05, seed=42)
snapshots = solver.run(phi0, n_steps=5000, save_interval=1000)
```

## テスト

```bash
uv run pytest
```
