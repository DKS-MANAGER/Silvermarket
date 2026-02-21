# 🏦 Silver Futures Quant Trading System — v3

**Notebook**: `Silver_Price_Prediction_PRODUCTION.ipynb`  
**Asset**: Silver Futures (`SI=F`) | **Data**: 10 years (~2,514 bars)  
**Last Executed**: 2026-02-21

---

## Overview

This notebook implements a full institutional-grade quantitative trading system for Silver Futures. It combines deep learning (CNN-BiLSTM-GRU with Multi-Head Attention) and machine learning (XGBoost regime classifier) in a two-voter ensemble, validated with zero-leakage walk-forward cross-validation.

---

## System Architecture

```
Market Data (yfinance)
       │
       ▼
Feature Engineering (18 alpha factors)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
VOTER A: Deep Learning Model          VOTER B: XGBoost Regime
CNN → BiLSTM → GRU → Attention        Bullish / Bearish / Neutral
  (price direction + magnitude)          (macro environment)
       │                                      │
       └──────────────┬───────────────────────┘
                      ▼
            Two-Voter Ensemble Signal
                      │
                      ▼
          Volatility-Targeted Position Size
                      │
                      ▼
            Realistic Backtest (T+1 fill, 15 bps cost)
                      │
                      ▼
          Kill Switches → Risk Dashboard → Forecast
```

---

## Pipeline Stages

### 1. Data Pipeline

Data is fetched from Yahoo Finance using a **single batched download** (4× faster than sequential calls) for four assets:

| Ticker | Asset |
|--------|-------|
| `SI=F` | Silver Futures (primary) |
| `GC=F` | Gold Futures |
| `DX-Y.NYB` | US Dollar Index (DXY) |
| `^VIX` | CBOE Volatility Index |

All series are aligned to Silver's trading days and forward-filled.

---

### 2. Feature Engineering (18 Alpha Factors)

All features are engineered to be **stationary** (verified via ADF test at p < 0.05). `Log_Return` is intentionally excluded from `feature_cols` to prevent data leakage — it is only used as the target variable.

| Category | Features |
|----------|----------|
| **Momentum Lags** | `Return_Lag_1`, `Return_Lag_5`, `Return_Lag_10`, `Return_Lag_20` |
| **Volatility** | `Realized_Vol_20`, `Realized_Vol_60`, `Vol_Ratio` |
| **Trend** | `Dist_SMA_20`, `Dist_SMA_50` |
| **Oscillators** | `RSI_14`, `BB_PctB` (Bollinger Band %B) |
| **Cross-Asset** | `GSR_Zscore` (Gold/Silver ratio z-score), `DXY_Return`, `VIX_Zscore` |
| **Interactions** | `DXY_Vol_Interaction`, `VIX_GSR_Interaction` |
| **Volume** | `Volume_Zscore`, `OBV_Return` |

> **Stationarity**: An ADF test is printed for all features. `GoldSilver_Ratio` (raw level) is confirmed non-stationary and excluded from the model — only its z-score (`GSR_Zscore`) is used.

---

### 3. Model Architecture (Voter A — Deep Learning)

Built using **Keras Functional API**, enabling residual skip connections:

```
Input (60 timesteps × 18 features)
  │
  ├─ Dilated Conv1D (kernel=3, rate=1) → BN ─────────────────────── (CNN residual)
  ├─ Dilated Conv1D (kernel=3, rate=2) → BN ─────────────────────── │
  │                                                                  │
  ├─ Bidirectional LSTM (128 units) → LayerNorm → Dropout(0.3)      │
  ├─ GRU (64 units) → Dropout(0.2)                                  │
  ├─ MultiHeadAttention (4 heads, key_dim=16) → Add+Norm            │
  ├─ GlobalAveragePooling1D                                          │
  └─ Concatenate (with CNN residual via GlobalAvgPool) ─────────────┘
       │
       ├─ Dense(64, relu) → BN → Dropout(0.2)
       ├─ Dense(32, relu)
       └─ Dense(1, linear) → Predicted Log Return
```

**Loss Function — Directional Huber**:

$$\mathcal{L} = \text{Huber}_{\delta=0.01}(y, \hat{y}) + 0.5 \times \mathbb{E}[\text{ReLU}(-y \cdot \hat{y})]$$

- Huber loss is robust to fat-tail silver returns
- The directional penalty penalises sign errors (wrong direction predictions)

**Total Parameters**: ~303,745

---

### 4. Walk-Forward Cross-Validation (5 Folds)

Uses `TimeSeriesSplit` (no shuffling) to prevent lookahead bias:

```
Fold 1: [====Train====][Test]
Fold 2: [====Train========][Test]
Fold 3: [====Train============][Test]
Fold 4: [====Train================][Test]
Fold 5: [====Train====================][Test]
```

**Per-fold**:
- `RobustScaler` fitted on **train data only** → applied to test
- Regime classifier (XGBoost) trained on **fold train data only**
- `ModelCheckpoint` saves best validation weights (`checkpoints/fold{N}_best.keras`)
- `EarlyStopping` (patience=10) + `ReduceLROnPlateau` (patience=5)

---

### 5. Regime Classifier (Voter B — XGBoost)

Classifies the macro environment each day into **Bullish / Bearish / Neutral** using 8 macro features:

```
GoldSilver_Ratio, GSR_Zscore, DXY_Return, VIX_Zscore,
DXY_Vol_Interaction, VIX_GSR_Interaction, Realized_Vol_20, Vol_Ratio
```

- **Threshold**: ±0.3% next-day return to define Bull/Bear (vs. Neutral)
- **Class weights**: Balanced (handles class imbalance)
- **Parameters**: 200 estimators, max_depth=4, learning_rate=0.05

---

### 6. Two-Voter Ensemble Signal

A trade is entered **only when both voters agree**:

| Condition | Result |
|-----------|--------|
| DL and XGBoost directions disagree | Signal = 0 (no trade) |
| \|DL predicted return\| < 0.3% threshold | Signal = 0 (no trade) |
| XGBoost Neutral probability > 50% | Signal = 0 (no trade) |
| Both agree (Long or Short) | Signal = +1 or −1 |

---

### 7. Position Sizing (Volatility Targeting)

$$\text{Position Size} = \text{Equity} \times \min\!\left(\frac{\sigma_{\text{target}}}{\sigma_{\text{realised}}}, \text{MaxLeverage}\right)$$

| Parameter | Value |
|-----------|-------|
| Target Annual Volatility | 15% |
| Max Leverage | 1.5× |
| Execution | T+1 Open price (no lookahead) |
| Transaction Cost | 5 bps |
| Slippage | 10 bps |
| **Total round-trip cost** | **15 bps** |

---

### 8. Kill Switch Risk Engine

Four automated kill switches halt or reduce trading in real time:

| Switch | Trigger | Action |
|--------|---------|--------|
| **Drawdown** | Portfolio DD > 15% | Liquidate all — 100% cash |
| **Sharpe** | Rolling Sharpe < 1.0 (after 60d) | Halt new positions |
| **VIX Spike** | VIX > 35 | Halve all position sizes |
| **Correlation** | Silver/SPX corr > 0.8 | Reduce positions 50% |

---

### 9. 30-Day Price Forecast (Monte Carlo Dropout)

Uses **fold-5 checkpoint model** (largest training window). No retraining is done at inference time.

**Method**:
1. Load `checkpoints/fold5_best.keras` and `checkpoints/scaler_fold5.pkl`
2. Run **30 forward passes** with `training=True` (dropout active) → distribution of predictions
3. Each step: predicted log return clipped to ±1.5%/day
4. Roll the prediction back as `Return_Lag_1` for the next step
5. Convert cumulative log returns → price levels
6. 95% confidence interval: $\pm 2\sigma\sqrt{t}$ (propagated MC standard deviation)

---

## Backtest Results (10-Year, 5-Fold WF-CV)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Sharpe Ratio | > 1.2 | **1.470** | ✅ |
| Max Drawdown | < 15% | **-9.03%** | ✅ |
| Directional Accuracy | > 56% | **85.16%** | ✅ |
| Win Rate | > 52% | **55.15%** | ✅ |
| Total Return (strategy) | — | **+112.40%** | ✅ |
| Buy & Hold (10y reference) | — | +459.48% | 📊 |
| 30-Day Forecast | — | +13.39% ($93.30) | 🔮 |

> The lower strategy return vs. buy-and-hold is expected — the system trades selectively (437 trades over 10y) and targets **risk-adjusted** performance (Sharpe 1.47 vs B&H 0.84).

---

## File Structure

```
Silver_Price_Prediction_PRODUCTION.ipynb  ← Main notebook
checkpoints/
  fold1_best.keras   ← Best model weights per fold
  fold2_best.keras
  fold3_best.keras
  fold4_best.keras
  fold5_best.keras   ← Used for production forecasting
  scaler_fold5.pkl   ← Feature scaler for production inference
silver_30day_forecast.csv                 ← Exported forecast output
```

---

## Configuration (`CFG` class)

All hyperparameters are centralised in the `CFG` class at the top of the notebook:

```python
class CFG:
    PERIOD       = '10y'     # Data lookback period
    SEQ_LEN      = 60        # Sequence window (days)
    N_SPLITS     = 5         # Walk-forward CV folds
    EPOCHS       = 100       # Max training epochs
    BATCH_SIZE   = 32
    LR           = 0.001     # Adam learning rate
    PATIENCE_ES  = 10        # EarlyStopping patience
    TARGET_VOL   = 0.15      # Annual volatility target
    MAX_LEVERAGE = 1.5
    SIGNAL_THR   = 0.003     # Minimum signal magnitude (0.3%)
    TC_BPS       = 5.0       # Transaction cost (bps)
    SLIP_BPS     = 10.0      # Slippage (bps)
    DD_THR       = -0.15     # Drawdown kill switch
    VIX_THR      = 35.0      # VIX kill switch
    CORR_THR     = 0.8       # Correlation kill switch
    SHARPE_FLOOR = 1.0       # Minimum Sharpe for trading
```

---

## Dependencies

```
yfinance       — market data
pandas/numpy   — data processing
scikit-learn   — preprocessing, metrics, TimeSeriesSplit
tensorflow     — deep learning (CNN-BiLSTM-GRU-Attention)
xgboost        — regime classification
statsmodels    — ADF stationarity test
plotly/kaleido — interactive visualisation
joblib         — model/scaler persistence
scipy          — statistical utilities
```

---

## How to Run

1. Open `Silver_Price_Prediction_PRODUCTION.ipynb` in VS Code or JupyterLab
2. Run cells **top to bottom** in order — each cell depends on variables from previous cells
3. Walk-forward validation (Cell 5) takes the longest (~20–40 min depending on hardware)
4. The forecast cell (Cell 8) loads `fold5_best.keras` automatically — no retraining needed

---

## Production Deployment Roadmap

| Step | Status |
|------|--------|
| Model checkpoints saved | ✅ Done |
| FastAPI prediction endpoint | 🔄 Next milestone |
| IBKR TWS API integration | ⏳ Pending |
| Live monitoring dashboard | ⏳ Pending |
| Weekly retraining pipeline | ⏳ Pending |

**Overall Verdict**: ✅ **APPROVED FOR PRODUCTION** (paper trading phase)
