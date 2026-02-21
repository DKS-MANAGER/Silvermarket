# 🏦 Silver Market Quant Trading System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regime%20Filter-189AB4?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**A full institutional-grade quantitative trading system for Silver Futures (SI=F)**  
*CNN · BiLSTM · GRU · Multi-Head Attention · XGBoost Regime Filter · Monte Carlo Dropout*

</div>

---

## 📊 Performance Summary

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Sharpe Ratio | > 1.2 | **1.470** | ✅ |
| Max Drawdown | < 15% | **-9.03%** | ✅ |
| Directional Accuracy | > 56% | **85.16%** | ✅ |
| Win Rate | > 52% | **55.15%** | ✅ |
| Total Return (10y) | — | **+112.40%** | ✅ |
| Buy & Hold Benchmark | — | +459.48% | 📊 |
| 30-Day Forecast | — | **+13.39%** | 🔮 |
| Kill Switches | 4 active | **4/4 GREEN** | ✅ |

> The strategy targets **risk-adjusted** outperformance — Sharpe 1.47 vs Buy & Hold 0.84 — by trading selectively (437 trades over 10 years) with strict risk controls.

---

## 🏗️ System Architecture

```
 Yahoo Finance (batched download)
  Silver │  Gold │  DXY │  VIX
         │
         ▼
  Feature Engineering (18 alpha factors)
         │
    ┌────┴────┐
    ▼         ▼
VOTER A      VOTER B
Deep Learning  XGBoost
CNN→BiLSTM    Regime Classifier
→GRU→Attention (Bull/Bear/Neutral)
    │         │
    └────┬────┘
         ▼
  Two-Voter Ensemble Signal
         ▼
  Volatility-Targeted Position Size
  (15% annual vol · max 1.5× leverage)
         ▼
  Realistic Backtest (T+1 fill · 15 bps cost)
         ▼
  Kill Switches → MC Dropout 30-Day Forecast
```

---

## 🧠 Model: CNN-BiLSTM-GRU + Attention

```
Input: 60 timesteps × 18 features
  │
  ├─ Dilated Conv1D (rate=1) → BatchNorm ──────────────┐ (residual)
  ├─ Dilated Conv1D (rate=2) → BatchNorm               │
  ├─ Bidirectional LSTM (128) → LayerNorm → Dropout    │
  ├─ GRU (64) → Dropout                                │
  ├─ MultiHeadAttention (4 heads) → Add + Norm         │
  ├─ GlobalAveragePool                                  │
  └─ Concatenate ◄───────────────────────────────────── ┘
       │
       ├─ Dense(64) → BN → Dropout
       └─ Dense(1) → Predicted Log Return
```

**Loss:** Directional Huber = `Huber(δ=0.01) + 0.5 × ReLU(−y·ŷ)`  
**Parameters:** ~303,745

---

## ⚙️ Features (18 Alpha Factors)

| Category | Features |
|----------|----------|
| Momentum | `Return_Lag_1/5/10/20` |
| Volatility | `Realized_Vol_20/60`, `Vol_Ratio` |
| Trend | `Dist_SMA_20/50` |
| Oscillators | `RSI_14`, `BB_PctB` |
| Cross-Asset | `GSR_Zscore`, `DXY_Return`, `VIX_Zscore` |
| Interactions | `DXY_Vol_Interaction`, `VIX_GSR_Interaction` |
| Volume | `Volume_Zscore`, `OBV_Return` |

All features verified **stationary** via ADF test (p < 0.05).

---

## 🔒 Risk Engine — 4 Kill Switches

| Switch | Trigger | Action |
|--------|---------|--------|
| Drawdown | Portfolio DD > 15% | Liquidate → 100% cash |
| Sharpe | Rolling Sharpe < 1.0 | Halt new positions |
| VIX Spike | VIX > 35 | Halve all positions |
| Correlation | Silver/SPX corr > 0.8 | Reduce size 50% |

---

## 🔮 30-Day Forecast (Monte Carlo Dropout)

- Loads `checkpoints/fold5_best.keras` (no retraining)
- 30 forward passes with dropout active → mean + uncertainty
- Daily returns clipped at ±1.5%
- 95% CI via propagated MC standard deviation (±2σ√t)

---

## 📦 Quick Start

```bash
git clone https://github.com/DKS-MANAGER/Silvermarket.git
cd Silvermarket
pip install yfinance pandas numpy scikit-learn tensorflow plotly xgboost joblib statsmodels
jupyter notebook Silver_Price_Prediction_PRODUCTION.ipynb
```

Run cells **top to bottom**. Walk-forward CV (Cell 5) is the longest step (~20–40 min).

---

## 📁 Repository Structure

```
Silvermarket/
├── Silver_Price_Prediction_PRODUCTION.ipynb  ← Full trading system
├── README.md                                 ← This file
└── checkpoints/                              ← Saved model weights
    ├── fold1_best.keras ... fold5_best.keras
    └── scaler_fold5.pkl
```

---

## 🛠️ Tech Stack

`yfinance` · `pandas` · `numpy` · `scikit-learn` · `tensorflow` · `xgboost` · `statsmodels` · `plotly` · `joblib`

---

## 🗺️ Deployment Roadmap

- [x] Walk-forward validated model checkpoints
- [ ] FastAPI real-time prediction endpoint
- [ ] IBKR TWS API integration
- [ ] Live monitoring dashboard
- [ ] Weekly automated retraining pipeline

---

<div align="center">
Made by <a href="https://github.com/DKS-MANAGER">Divyansh Kumar Singh</a> · Silver Futures Quant Research · 2026
</div>
