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


# ── technical indicators ──────────────────────────────────────────────────────

def calc_ema(prices: list, period: int) -> list:
    """Exponential Moving Average. Returns list of same length as prices (None-padded)."""
    result = [None] * (period - 1)
    sma = float(np.mean(prices[:period]))
    result.append(round(sma, 2))
    alpha = 2.0 / (period + 1)
    for p in prices[period:]:
        sma = alpha * p + (1 - alpha) * sma
        result.append(round(sma, 2))
    return result


def calc_rsi(prices: list, period: int = 14) -> list:
    """Wilder's RSI. Returns list of same length as prices (None-padded)."""
    deltas = np.diff(prices)
    gains  = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    result = [None] * (period + 1)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        result.append(round(100.0 - 100.0 / (1 + rs), 2))
    return result


def calc_bb(prices: list, period: int = 20) -> tuple:
    """Bollinger Bands (2σ). Returns (upper, middle, lower) — each same length as prices."""
    upper, middle, lower = [], [], []
    for i in range(len(prices)):
        if i < period - 1:
            upper.append(None); middle.append(None); lower.append(None)
        else:
            window = prices[i - period + 1 : i + 1]
            m = float(np.mean(window))
            s = float(np.std(window, ddof=0))
            middle.append(round(m, 2))
            upper.append(round(m + 2 * s, 2))
            lower.append(round(m - 2 * s, 2))
    return upper, middle, lower


def calc_macd(prices: list, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD. Returns (macd_line, signal_line, histogram) — each same length as prices."""
    ema_f = calc_ema(prices, fast)
    ema_s = calc_ema(prices, slow)

    macd_line = [None] * (slow - 1)
    for i in range(slow - 1, len(prices)):
        macd_line.append(round(ema_f[i] - ema_s[i], 2))

    valid = [v for v in macd_line if v is not None]   # n - slow + 1 values
    sig = [None] * (slow - 1 + signal - 1)
    s = float(np.mean(valid[:signal]))
    sig.append(round(s, 2))
    alpha = 2.0 / (signal + 1)
    for v in valid[signal:]:
        s = alpha * v + (1 - alpha) * s
        sig.append(round(s, 2))

    histogram = []
    for i in range(len(prices)):
        if macd_line[i] is not None and i < len(sig) and sig[i] is not None:
            histogram.append(round(macd_line[i] - sig[i], 2))
        else:
            histogram.append(None)
    return macd_line, sig, histogram


# ── Stock-to-Flow model (PlanB) ───────────────────────────────────────────────

_HALVING_SCHEDULE = [
    (pd.Timestamp("2009-01-03"), 50.0),
    (pd.Timestamp("2012-11-28"), 25.0),
    (pd.Timestamp("2016-07-09"), 12.5),
    (pd.Timestamp("2020-05-11"), 6.25),
    (pd.Timestamp("2024-04-20"), 3.125),
]
_GENESIS        = _HALVING_SCHEDULE[0][0]
_BLOCKS_PER_DAY = 144.0


def _btc_state_at(ts: pd.Timestamp) -> tuple:
    """Returns (block_reward, approx_circulating_supply) at given date."""
    reward  = 50.0
    supply  = 0.0
    prev_ts = _GENESIS
    for halving_ts, new_reward in _HALVING_SCHEDULE[1:]:
        if ts >= halving_ts:
            supply += (halving_ts - prev_ts).days * _BLOCKS_PER_DAY * reward
            prev_ts = halving_ts
            reward  = new_reward
        else:
            break
    supply += (ts - prev_ts).days * _BLOCKS_PER_DAY * reward
    return reward, min(supply, 21_000_000.0)


def calc_s2f(dates: list) -> list:
    """PlanB S2F: Market Cap = exp(14.6) * SF^3.3. Returns model price per date."""
    result = []
    for d in dates:
        ts = pd.Timestamp(d)
        reward, supply = _btc_state_at(ts)
        if supply <= 0 or reward <= 0:
            result.append(None)
            continue
        flow_annual = reward * _BLOCKS_PER_DAY * 365.25
        sf          = supply / flow_annual
        mcap        = math.exp(14.6) * (sf ** 3.3)
        result.append(round(mcap / supply, 2))
    return result


# ── Log-Regression Rainbow ────────────────────────────────────────────────────

def calc_log_regression(dates: list, prices: list, future_dates_list: list = None) -> tuple:
    """Power-law regression on historical prices, extended to future dates.
    Returns (reg, upper_2σ, lower_2σ) for hist+future combined."""
    base      = pd.Timestamp(dates[0])
    days_hist = [(pd.Timestamp(d) - base).days + 1 for d in dates]
    log_d     = np.log(days_hist)
    log_p     = np.log(np.maximum(prices, 1.0))

    coeffs             = np.polyfit(log_d, log_p, 1)
    slope, intercept   = float(coeffs[0]), float(coeffs[1])
    residuals          = log_p - (slope * log_d + intercept)
    std                = float(np.std(residuals))

    all_days = days_hist + (
        [(pd.Timestamp(d) - base).days + 1 for d in (future_dates_list or [])]
    )

    def _p(d): return math.exp(slope * math.log(d) + intercept)

    reg   = [round(_p(d), 2)                    for d in all_days]
    upper = [round(_p(d) * math.exp(2 * std), 2) for d in all_days]
    lower = [round(_p(d) * math.exp(-2 * std), 2) for d in all_days]
    return reg, upper, lower


# ── Signal score (−100 … +100) ────────────────────────────────────────────────

def calc_signal_score(
    prices: list,
    rsi: list,
    ema50: list,
    ema200: list,
    macd_line: list,
    macd_sig: list,
    s2f_hist: list,
) -> int:
    def last_val(lst):
        return next((v for v in reversed(lst) if v is not None), None)

    cur   = prices[-1]
    score = 0

    # RSI (±25 pts)
    r = last_val(rsi)
    if r is not None:
        if   r <= 30: score += 25
        elif r >= 70: score -= 25
        else:         score += round(25 * (50 - r) / 20)

    # EMA 200 (±20 pts): price above long MA = bullish
    e200 = last_val(ema200)
    if e200 is not None:
        score += 20 if cur > e200 else -20

    # EMA 50 (±15 pts)
    e50 = last_val(ema50)
    if e50 is not None:
        score += 15 if cur > e50 else -15

    # MACD (±25 pts)
    ml = last_val(macd_line)
    ms = last_val(macd_sig)
    if ml is not None and ms is not None:
        if   ml > 0 and ml > ms: score += 25
        elif ml < 0 and ml < ms: score -= 25
        elif ml > ms:             score += 10
        else:                     score -= 10

    # S2F (±15 pts): price vs fair-value model
    sf = last_val(s2f_hist)
    if sf and sf > 0:
        ratio = cur / sf
        if   ratio < 0.5: score += 15
        elif ratio < 0.8: score += 8
        elif ratio < 1.5: score += 0
        elif ratio < 2.5: score -= 8
        else:             score -= 15

    return max(-100, min(100, score))


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

    Xt = torch.from_numpy(X_train).unsqueeze(-1).to(DEVICE)
    yt = torch.from_numpy(y_train).unsqueeze(-1).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    model = LSTMModel(hidden=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss, patience, no_improve = float("inf"), 6, 0
    best_state = None
    for epoch in range(50):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            Xv = torch.from_numpy(X_test).unsqueeze(-1).to(DEVICE)
            val_loss = criterion(model(Xv), torch.from_numpy(y_test).unsqueeze(-1).to(DEVICE)).item()

        if val_loss < best_loss - 1e-5:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        Xv = torch.from_numpy(X_test).unsqueeze(-1).to(DEVICE)
        test_pred = model(Xv).cpu().numpy().flatten()
    test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    test_real_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    rmse = math.sqrt(mean_squared_error(test_real_inv, test_pred_inv))

    window = list(scaled[-LOOK_BACK:])
    future = []
    model.eval()
    with torch.no_grad():
        for _ in range(PREDICT_DAYS):
            inp  = torch.tensor(window[-LOOK_BACK:], dtype=torch.float32).view(1, LOOK_BACK, 1).to(DEVICE)
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
    X  = df.drop("price", axis=1).values
    y  = df["price"].values

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
        "tree_method": "hist",
        "nthread": os.cpu_count() or 8,
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

    history = list(prices)
    future  = []
    for _ in range(PREDICT_DAYS):
        row  = [history[-lag] for lag in LAG_DAYS]
        pred = float(model.predict(xgb.DMatrix(np.array(row).reshape(1, -1)))[0])
        future.append(pred)
        history.append(pred)

    return {"predictions": future, "rmse": round(rmse, 2)}


# ── public entry point ────────────────────────────────────────────────────────

def build_prediction_payload() -> dict:
    dates, prices = fetch_binance_history()

    lstm_result    = run_lstm(prices)
    xgboost_result = run_xgboost(prices)
    pred_dates     = future_dates(dates[-1], PREDICT_DAYS)

    # Technical indicators (all computed on full history length)
    ema21  = calc_ema(prices, 21)
    ema50  = calc_ema(prices, 50)
    ema200 = calc_ema(prices, 200)
    rsi    = calc_rsi(prices)
    bb_up, bb_mid, bb_low = calc_bb(prices)
    macd_line, macd_sig, macd_hist = calc_macd(prices)

    # S2F: history + future period
    all_dates = dates + pred_dates
    s2f_all   = calc_s2f(all_dates)
    s2f_hist  = s2f_all[:len(dates)]
    s2f_pred  = s2f_all[len(dates):]

    # Log regression: history + future period
    lr, lr_up, lr_dn = calc_log_regression(dates, prices, pred_dates)
    lr_hist    = lr[:len(dates)]
    lr_up_hist = lr_up[:len(dates)]
    lr_dn_hist = lr_dn[:len(dates)]
    lr_pred    = lr[len(dates):]
    lr_up_pred = lr_up[len(dates):]
    lr_dn_pred = lr_dn[len(dates):]

    signal = calc_signal_score(prices, rsi, ema50, ema200, macd_line, macd_sig, s2f_hist)

    SHOW = 180

    return {
        "history": {
            "dates":  dates[-SHOW:],
            "prices": prices[-SHOW:],
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
        "indicators": {
            "ema21":           ema21[-SHOW:],
            "ema50":           ema50[-SHOW:],
            "ema200":          ema200[-SHOW:],
            "rsi":             rsi[-SHOW:],
            "bb_upper":        bb_up[-SHOW:],
            "bb_mid":          bb_mid[-SHOW:],
            "bb_lower":        bb_low[-SHOW:],
            "macd_line":       macd_line[-SHOW:],
            "macd_signal":     macd_sig[-SHOW:],
            "macd_hist":       macd_hist[-SHOW:],
            "s2f_hist":        s2f_hist[-SHOW:],
            "s2f_pred":        s2f_pred,
            "log_reg_hist":    lr_hist[-SHOW:],
            "log_reg_up_hist": lr_up_hist[-SHOW:],
            "log_reg_dn_hist": lr_dn_hist[-SHOW:],
            "log_reg_pred":    lr_pred,
            "log_reg_up_pred": lr_up_pred,
            "log_reg_dn_pred": lr_dn_pred,
        },
        "signal_score":    signal,
        "trained_on_days": len(prices),
        "device":          str(DEVICE),
    }
