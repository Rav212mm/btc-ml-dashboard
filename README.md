# BTC ML Dashboard

A Bitcoin price analysis and prediction dashboard combining live market data, technical indicators, and machine learning models — all in a single web interface.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Live price feed** — real-time BTC/USD from Binance, Kraken, and CoinGecko (averaged)
- **Price chart** — interactive candlestick-style chart with toggle buttons for indicator groups
- **Technical indicators**
  - EMA 21 / 50 / 200
  - RSI 14
  - Bollinger Bands
  - MACD + histogram
- **Prediction models**
  - Stock-to-Flow (PlanB)
  - Log Regression Rainbow (±2σ bands)
  - LSTM neural network (PyTorch, AMD DirectML / CUDA / CPU fallback)
  - XGBoost regressor
- **Market signal gauge** — composite score (−100 to +100) based on RSI, EMAs, MACD, and S2F
- **14-day / 120-day price forecast** returned as JSON from `/api/predict`

---

## Project Structure

```
app/
├── backend/
│   └── app.py          # Flask server (port 5002)
├── ml/
│   └── predict.py      # LSTM + XGBoost prediction engine
└── frontend/
    ├── templates/
    │   └── index.html  # Main dashboard page
    └── static/
        ├── app.js
        └── style.css

data/                   # Historical CSV data (bitcoinity)
notebooks/              # Exploratory Jupyter notebooks
requirements.txt
```

---

## Quick Start

### Requirements

- Python 3.10 (with all packages installed)
- Packages: `flask`, `requests`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`

```bash
pip install -r requirements.txt
```

### Run

```bash
python app/backend/app.py
```

Open [http://localhost:5002](http://localhost:5002)

> **Note:** Port 5001 is reserved by macOS/Windows system services. The app runs on **5002** by default.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/data` | Live price from all three exchanges |
| `GET /api/predict` | ML forecast (trains on first call, ~60 s) |
| `GET /api/predict/reset` | Clear prediction cache and retrain |

---

## ML Models

Training data is fetched live from the Binance public API (last 500 daily candles). No local dataset is required for predictions.

| Model | Horizon | Notes |
|---|---|---|
| LSTM | 120 days | PyTorch, GPU-accelerated if available |
| XGBoost | 14 days | CPU, multi-threaded |
| Stock-to-Flow | Long-term | PlanB formula |
| Log Rainbow | Long-term | Power-law regression ±2σ |

---

## Data Sources

- **Live prices:** Binance, Kraken, CoinGecko public APIs
- **Historical data** (`data/`): exported from [bitcoinity.org](https://bitcoinity.org) — includes price/volume, hash rate, mining difficulty, market cap, and volatility

---

## Planned Features

- Choppiness Index (CHOP 14)
- Puell Multiple & NVT Ratio (Glassnode on-chain data)
- Pi Cycle Top Indicator
- Twitter/X sentiment analysis (in progress)
- Multi-variable LSTM

---

## License

MIT
