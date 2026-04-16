from datetime import datetime, timezone
from threading import Thread

import sys
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "ml"))

app = Flask(
    __name__,
    template_folder=str(_ROOT / "frontend" / "templates"),
    static_folder=str(_ROOT / "frontend" / "static"),
)

# prediction cache — avoid retraining on every request
_prediction_cache: dict | None = None
_prediction_running = False


# ── live price fetchers ───────────────────────────────────────────────────────

def fetch_binance():
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=10)
        d = r.json()
        return {
            "source": "Binance",
            "price": float(d["lastPrice"]),
            "change_24h": float(d["priceChangePercent"]),
            "volume_24h": float(d["volume"]),
            "high_24h": float(d["highPrice"]),
            "low_24h": float(d["lowPrice"]),
        }
    except Exception as e:
        return {"source": "Binance", "error": str(e)}


def fetch_kraken():
    try:
        r = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=10)
        d = r.json()
        ticker = d["result"]["XXBTZUSD"]
        open_price = float(ticker["o"])
        last_price = float(ticker["c"][0])
        change_pct = ((last_price - open_price) / open_price) * 100
        return {
            "source": "Kraken",
            "price": last_price,
            "change_24h": round(change_pct, 2),
            "volume_24h": float(ticker["v"][1]),
            "high_24h": float(ticker["h"][1]),
            "low_24h": float(ticker["l"][1]),
        }
    except Exception as e:
        return {"source": "Kraken", "error": str(e)}


def fetch_coingecko():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin"
            "?localization=false&tickers=false&community_data=false&developer_data=false",
            timeout=10,
        )
        d = r.json()
        m = d["market_data"]
        return {
            "source": "CoinGecko",
            "price": m["current_price"]["usd"],
            "change_24h": round(m["price_change_percentage_24h"], 2),
            "volume_24h": m["total_volume"]["usd"],
            "high_24h": m["high_24h"]["usd"],
            "low_24h": m["low_24h"]["usd"],
        }
    except Exception as e:
        return {"source": "CoinGecko", "error": str(e)}


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    results = [fetch_binance(), fetch_kraken(), fetch_coingecko()]
    prices = [r["price"] for r in results if "price" in r]
    avg_price = sum(prices) / len(prices) if prices else None
    return jsonify({
        "sources": results,
        "avg_price": avg_price,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


@app.route("/api/predict")
def api_predict():
    global _prediction_cache, _prediction_running

    if _prediction_cache is not None:
        return jsonify({**_prediction_cache, "status": "ready"})

    if _prediction_running:
        return jsonify({"status": "training", "message": "Modele są trenowane, spróbuj za chwilę…"})

    # start training in background thread so request returns immediately
    def train():
        global _prediction_cache, _prediction_running
        _prediction_running = True
        try:
            from predict import build_prediction_payload
            _prediction_cache = build_prediction_payload()
        except Exception as e:
            _prediction_cache = {"error": str(e)}
        finally:
            _prediction_running = False

    Thread(target=train, daemon=True).start()
    return jsonify({"status": "training", "message": "Rozpoczęto trening modeli (LSTM + XGBoost). Zajmie ~60 s."})


@app.route("/api/predict/reset")
def api_predict_reset():
    global _prediction_cache
    _prediction_cache = None
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=False, port=5001)
