# 🛰️ Angles-Only Initial Orbit Determination (IOD) Using Neural Networks


> Angles-only Initial Orbit Determination framework comparing Traditional Methods (Gauss, Laplace)
> against Neural Network approaches (MLP, LSTM) for space object tracking and Space Situational Awareness (SSA).

**Authors:**  Dishant Adesara 
**Institution:** Julius-Maximilians-Universität Würzburg  


***

## 📌 Project Overview

Initial Orbit Determination (IOD) plays a vital role in building and maintaining Space Object Catalogues
such as ESA DISCOS for **Space Situational Awareness (SSA)**. With over **650+ fragmentation events**
recorded in orbit, efficient and accurate tracking of space objects is critical for space safety.

**The Problem:**
- Traditional methods (Gauss, Laplace) are **sensitive to noise** and require a good initial guess
- Optical observations provide **angles-only** data (Azimuth, Elevation, Time) — no range information
- Goal: perform IOD of space objects using angles-only measurements with well-trained neural networks

**This project implements and compares:**
1. **Traditional Methods** — Gauss & Laplace (baseline)
2. **MLP** (Multi-Layer Perceptron) with Optuna hyperparameter tuning
3. **LSTM** (Long Short-Term Memory) with Optuna hyperparameter tuning

***

## 🗂️ Repository Structure

```
Initial_Orbit_Determination/
│
├── Angles-Only-IOD/                        # ⭐ Traditional Methods + Evaluation
│   ├── groundtruth/                        # Ground truth orbit data
│   ├── input/                              # Input observation data
│   ├── output/                             # Method output files
│   ├── angles_only_traditional_iod.py      # ⭐ MAIN: Run Gauss / Laplace IOD
│   ├── conversion_utils.py                 # Unit conversion utilities
│   ├── gauss.py                            # Gauss IOD method implementation
│   ├── herrick_gibbs.py                    # Herrick-Gibbs method
│   ├── laplace.py                          # Laplace IOD method implementation
│   ├── space_objects.py                    # Space object data structures
│   ├── tracklets.py                        # Tracklet data handling
│   ├── EvaluationTraditional.ipynb         # ⭐ MAIN: Evaluate Traditional Methods
│   └── all_parsed_data.csv                 # Parsed output (generated after running IOD)
│
├── NN/                                     # ⭐ Neural Network Methods (MLP + LSTM)
│   ├── iod_mlp_v2_plot.ipynb               # ⭐ MAIN: MLP model (v2) + results plots
│   ├── lstm_optuna.ipynb                   # ⭐ MAIN: LSTM model with Optuna tuning
│   └── LSTM_without_optuna.ipynb           # LSTM baseline (no hyperparameter tuning)
│
├── FP_Angles_only_IOD_Using_Neural_...pptx # Final presentation slides
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

***

## 🔄 Pipeline Overview

```
Simulated Ground-Based Observations
(ESA MASTER Space Object Population)
        │
        ▼
  Angles-Only Data: [Azimuth, Elevation, Time (MJD)]
        │
        ├──────────────────────┬────────────────────────┐
        ▼                      ▼                        ▼
Traditional Methods         MLP Model               LSTM Model
(Gauss & Laplace)      (iod_mlp_v2_plot)        (lstm_optuna)
        │                      │                        │
        ▼                      ▼                        ▼
  6D State Vector         6D State Vector          6D State Vector
  [rx, ry, rz,            [rx, ry, rz,             [rx, ry, rz,
   vx, vy, vz]             vx, vy, vz]              vx, vy, vz]
        │                      │                        │
        └──────────────────────┴────────────────────────┘
                               │
                               ▼
                  Evaluation vs Ground Truth
                      (MAE + RMSE Metrics)
```

***

## 🧠 Model Details

### MLP — Multi-Layer Perceptron (`iod_mlp_v2_plot.ipynb`)

| Parameter | Value |
|---|---|
| Input Features | 9 — `start_time, start_el, start_az, mid_time, mid_el, mid_az, end_time, end_el, end_az` |
| Output | 6D State Vector — `rx, ry, rz` (km) + `vx, vy, vz` (km/s) |
| Data Split | 80% Train / 20% Test |
| Normalization | Z-Score Normalization |
| Activation | ReLU |
| Optimizer | Adam |
| Loss Function | MSE |
| Hidden Layers | 2 (Optuna tuned) |
| Hidden Units | 176 (Optuna tuned) |
| Learning Rate | 0.00962 (Optuna tuned) |
| Dropout Rate | 0.158 (Optuna tuned) |
| Batch Size | 32 |

### LSTM — Long Short-Term Memory (`lstm_optuna.ipynb`)

| Parameter | Value |
|---|---|
| Input Features | 3 per timestep — `Azimuth, Elevation, Δt` |
| Output | 6D State Vector — `rx, ry, rz` (km) + `vx, vy, vz` (km/s) |
| Architecture | Bidirectional LSTM (Forget gate, Input gate, Output gate) |
| Output Timestep | Middle time step only |
| Normalization | Min-Max Normalization |
| Activation | ReLU |
| Optimizer | Adam |
| Loss Function | MSE |
| Hidden Layers | 1 (Optuna tuned) |
| Hidden Units | 111 (Optuna tuned) |
| Learning Rate | 0.0001 – 0.002 (Optuna tuned) |
| Dropout Rate | 0.08 |

***

## 📊 Results

### MAE Comparison (Mean Absolute Error)

| Method | r_x (km) | r_y (km) | r_z (km) | v_x (km/s) | v_y (km/s) | v_z (km/s) |
|--------|----------|----------|----------|------------|------------|------------|
| **MLP** | 209.7 | 235.27 | 45.13 | 0.03 | 0.03 | 0.08 |
| LSTM | 1665.62 | 3047.47 | 72.69 | 0.311 | 0.88 | 2.508 |
| Gauss | 123.91 | 55.22 | 10.8 | 0.01 | 0.03 | 0.02 |
| Laplace | 124.00 | 55.26 | 10.8 | 0.01 | 0.03 | 0.02 |

### RMSE Comparison (Root Mean Square Error)

| Method | r_x (km) | r_y (km) | r_z (km) | v_x (km/s) | v_y (km/s) | v_z (km/s) |
|--------|----------|----------|----------|------------|------------|------------|
| **MLP** | 622.63 | 398.88 | 85.26 | 0.06 | 0.06 | 0.18 |
| LSTM | 2869.66 | 3373.77 | 210.68 | 0.378 | 0.894 | 2.514 |
| Gauss | 673.66 | 300.12 | 57.54 | 0.06 | 0.12 | 0.08 |
| Laplace | 673.68 | 300.12 | 57.54 | 0.06 | 0.12 | 0.08 |

### Key Findings

- **Traditional Methods (Gauss & Laplace)** — Very close to ground truth; MAE ~55–124 km, RMSE ~300–674 km
- **MLP** — Moderate accuracy; position MAE 45–235 km, velocity errors under 0.1 km/s. Comparable to traditional methods
- **LSTM** — Lowest accuracy; position errors >3000 km RMSE. Needs further optimization
- **MLP shows the most promise** among NN methods — acceptable velocity estimates, potential for further improvement

***

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Dishant860/Initial_Orbit_Determination.git
cd Initial_Orbit_Determination
```

### 2. Create Virtual Environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

***

## ▶️ How to Run

### Part 1 — Traditional Methods (Gauss & Laplace)

**Step 1: Run Traditional IOD**
```bash
cd Angles-Only-IOD

# Run with Gauss method
python angles_only_traditional_iod.py --method GAUSS

# OR run with Laplace method
python angles_only_traditional_iod.py --method LAPLACE
```
Generates: `all_parsed_data.csv` in the `Angles-Only-IOD/` folder

**Step 2: Evaluate Traditional Results**
```
Open:   Angles-Only-IOD/EvaluationTraditional.ipynb
Kernel: your venv
Run All Cells → MAE and RMSE results for Gauss & Laplace
```

> ⚠️ Run `angles_only_traditional_iod.py` **before** `EvaluationTraditional.ipynb` —
> the notebook reads `all_parsed_data.csv` generated by the Python script.

***

### Part 2 — Neural Network Methods (MLP & LSTM)

**MLP v2 (Recommended — best NN results):**
```
Open:   NN/iod_mlp_v2_plot.ipynb
Kernel: your venv
Run All Cells → trains MLP, shows True vs Predicted plots + MAE/RMSE metrics
```

**LSTM with Optuna tuning:**
```
Open:   NN/lstm_optuna.ipynb
Kernel: your venv
Run All Cells → trains LSTM with hyperparameter search, shows metrics
```

**LSTM baseline (no tuning):**
```
Open:   NN/LSTM_without_optuna.ipynb
Run All Cells → baseline LSTM without Optuna
```

***

## 📦 Key Dependencies

```
torch              # MLP model
tensorflow / keras # LSTM model
optuna             # Hyperparameter tuning for both models
numpy              # Numerical operations
pandas             # Data loading and processing
scikit-learn       # Preprocessing, metrics
matplotlib         # Plots and training history
scipy              # Numerical methods
ipykernel          # Jupyter kernel for venv
```

Install all: `pip install -r requirements.txt`

***

## ⚠️ Common Issues & Fixes

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated and requirements are installed |
| `FileNotFoundError` for `all_parsed_data.csv` | Run `angles_only_traditional_iod.py` first |
| CUDA / GPU not detected | Models run on CPU by default — no GPU required |
| Optuna taking too long | Reduce `n_trials` parameter in the notebook |

***

## 🔮 Future Scope

- **Bayesian NNs / Monte Carlo Dropout** — estimate uncertainty in orbit predictions for SSA
- **Physics-Informed Neural Networks (PINNs)** — enforce orbital dynamics equations within NN training
- **Integration with Tracking Systems** — connect model to ground station tracking software
- **Improved LSTM architecture** — attention mechanisms, deeper sequence modelling

***

## 👨‍💻 Developers

**Dishant Nileshkumar Adesara**  
M.Sc. Computer Science — Julius-Maximilians-Universität Würzburg  
🌐 [Portfolio](https://dishant860.github.io/portfolio) -  💼 [LinkedIn](https://www.linkedin.com/in/dishant-adesara) -  🐙 [GitHub](https://github.com/Dishant860)

**Shridevi Kerakalamatti** · **Yashkumar Lukhi**  
**Supervisor:** M.Sc.-Ing. Loay Gouda — JMU Würzburg

***

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

***

> **Note:** Dataset based on simulated ground-based observations of ESA's MASTER space object population.
> This project is part of the Aerospace Informatics Praktikum at JMU Würzburg.
