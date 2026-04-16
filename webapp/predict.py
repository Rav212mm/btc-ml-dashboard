"""
Prediction module — LSTM (PyTorch + AMD GPU via DirectML) + XGBoost (CPU multi-thread).
Trains on Binance daily close prices fetched at runtime.
"""

import math
import os
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── device selection ──────────────────────────────────────────────────────────

def _get_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        print(f"[predict] GPU: AMD via DirectML ({dev})")
        return dev
    except Exception:
        pass
    if torch.cuda.is_available():
        print("[predict] GPU: CUDA")
        return torch.device("cuda")
    cpu = torch.device("cpu")
    # use all CPU threads (AMD 9800X3D = 16 threads)
    torch.set_num_threads(os.cpu_count() or 8)
    print(f"[predict] CPU ({torch.get_num_threads()} threads)")
    return cpu


DEVICE = _get_device()

LOOK_BACK    = 30
PREDICT_DAYS = 120
HISTORY_DAYS = 500


# ── data ──────────────────────────────────────────────────────────────────────

def fetch_binance_history(days: int = HISTORY_DAYS):
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": days},
        timeout=15,
    )
    r.raise_for_status()
    raw = r.json()
    dates  = [pd.Timestamp(d[0], unit="ms").strftime("%Y-%m-%d") for d in raw]
    prices = [float(d[4]) for d in raw]
    return dates, prices


def future_dates(last_date_str: str, n: int):
    base = pd.Timestamp(last_date_str)
    return [(base + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(n)]


# ── LSTM model (PyTorch) ──────────────────────────────────────────────────────
# Uses manual gate implementation — nn.LSTM's fused kernel is not supported on
# DirectML (AMD GPU). All ops here (Linear, sigmoid, tanh) are DirectML-native.

class _LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        gates = self.W(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new


class LSTMModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.cell1  = _LSTMCell(1, hidden)
        self.cell2  = _LSTMCell(hidden, hidden)
        self.drop   = nn.Dropout(0.2)
        self.fc     = nn.Linear(hidden, 1)

    def forward(self, x):
        B, T, _ = x.shape
        dev = x.device
        h1 = torch.zeros(B, self.hidden, device=dev)
        c1 = torch.zeros(B, self.hidden, device=dev)
        h2 = torch.zeros(B, self.hidden, device=dev)
        c2 = torch.zeros(B, self.hidden, device=dev)
        for t in range(T):
            h1, c1 = self.cell1(x[:, t, :], h1, c1)
            h2, c2 = self.cell2(self.drop(h1), h2, c2)
        return self.fc(h2)


def _make_sequences(scaled: np.ndarray):
    X, y = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X.append(scaled[i - LOOK_BACK:i])
        y.append(scaled[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_lstm(prices: list) -> dict:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()

    X, y = _make_sequences(scaled)
    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # tensors → GPU/CPU
    Xt = torch.from_numpy(X_train).unsqueeze(-1).to(DEVICE)
    yt = torch.from_numpy(y_train).unsqueeze(-1).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    model = LSTMModel(hidden=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss, patience, no_improve = float("inf"), 6, 0
    for epoch in range(50):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # validation on test set
        model.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_test).unsqueeze(-1).to(DEVICE)
            val_loss = criterion(model(Xv), torch.from_numpy(y_test).unsqueeze(-1).to(DEVICE)).item()

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # test RMSE
    with torch.no_grad():
        Xv = torch.from_numpy(X_test).unsqueeze(-1).to(DEVICE)
        test_pred = model(Xv).cpu().numpy().flatten()
    test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    test_real_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    rmse = math.sqrt(mean_squared_error(test_real_inv, test_pred_inv))

    # multi-step future prediction
    window = list(scaled[-LOOK_BACK:])
    future = []
    model.eval()
    with torch.no_grad():
        for _ in range(PREDICT_DAYS):
            inp = torch.tensor(window[-LOOK_BACK:], dtype=torch.float32).view(1, LOOK_BACK, 1).to(DEVICE)
            pred = model(inp).item()
            future.append(pred)
            window.append(pred)

    future_prices = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten().tolist()

    return {"predictions": future_prices, "rmse": round(rmse, 2)}


# ── XGBoost (all CPU threads) ─────────────────────────────────────────────────

LAG_DAYS = [1, 2, 3, 5, 7, 14, 21, 30, 60]


def _build_lag_df(prices: list) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices})
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df["price"].shift(lag)
    return df.dropna().reset_index(drop=True)


def run_xgboost(prices: list) -> dict:
    df = _build_lag_df(prices)
    X = df.drop("price", axis=1).values
    y = df["price"].values

    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "tree_method": "hist",          # fastest CPU method
        "nthread": os.cpu_count() or 8, # use all cores (AMD 9800X3D = 16)
        "seed": 123,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtest, "test")],
        verbose_eval=False,
        early_stopping_rounds=20,
    )

    test_pred = model.predict(dtest)
    rmse = math.sqrt(mean_squared_error(y_test, test_pred))

    # rolling multi-step prediction
    history = list(prices)
    future = []
    for _ in range(PREDICT_DAYS):
        row = [history[-lag] for lag in LAG_DAYS]
        pred = float(model.predict(xgb.DMatrix(np.array(row).reshape(1, -1)))[0])
        future.append(pred)
        history.append(pred)

    return {"predictions": future, "rmse": round(rmse, 2)}


# ── public entry point ────────────────────────────────────────────────────────

def build_prediction_payload() -> dict:
    dates, prices = fetch_binance_history()

    lstm_result    = run_lstm(prices)
    xgboost_result = run_xgboost(prices)

    pred_dates = future_dates(dates[-1], PREDICT_DAYS)

    return {
        "history": {
            "dates":  dates[-180:],
            "prices": prices[-180:],
        },
        "predictions": {
            "dates": pred_dates,
            "lstm": {
                "prices": lstm_result["predictions"],
                "rmse":   lstm_result["rmse"],
            },
            "xgboost": {
                "prices": xgboost_result["predictions"],
                "rmse":   xgboost_result["rmse"],
            },
        },
        "trained_on_days": len(prices),
        "device": str(DEVICE),
    }
