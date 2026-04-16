function fmt(num) {
    if (num == null) return "—";
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(num);
}

function fmtVol(num) {
    if (num == null) return "—";
    if (num >= 1e9) return (num / 1e9).toFixed(2) + "B";
    if (num >= 1e6) return (num / 1e6).toFixed(2) + "M";
    if (num >= 1e3) return (num / 1e3).toFixed(2) + "K";
    return num.toFixed(2);
}

function renderCards(sources) {
    const container = document.getElementById("cards");
    container.innerHTML = sources.map(s => {
        if (s.error) {
            return `<div class="card">
                <div class="card-header">
                    <span class="card-source">${s.source}</span>
                    <span class="card-badge badge-error">BŁĄD</span>
                </div>
                <div class="card-error">${s.error}</div>
            </div>`;
        }
        const changeClass = s.change_24h >= 0 ? "positive" : "negative";
        const changeSign = s.change_24h >= 0 ? "+" : "";
        return `<div class="card">
            <div class="card-header">
                <span class="card-source">${s.source}</span>
                <span class="card-badge badge-ok">LIVE</span>
            </div>
            <div class="card-price">${fmt(s.price)}</div>
            <div class="card-change ${changeClass}">${changeSign}${s.change_24h}% (24h)</div>
            <div class="card-stats">
                <div class="stat">
                    <div class="stat-label">Najwyższy 24h</div>
                    <div class="stat-value">${fmt(s.high_24h)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Najniższy 24h</div>
                    <div class="stat-value">${fmt(s.low_24h)}</div>
                </div>
                <div class="stat" style="grid-column: span 2">
                    <div class="stat-label">Wolumen 24h</div>
                    <div class="stat-value">${fmtVol(s.volume_24h)} BTC</div>
                </div>
            </div>
        </div>`;
    }).join("");
}

async function fetchData() {
    const btn = document.getElementById("refresh-btn");
    const icon = document.getElementById("btn-icon");
    const errorMsg = document.getElementById("error-msg");

    btn.disabled = true;
    icon.classList.add("spinning");
    errorMsg.classList.add("hidden");

    try {
        const res = await fetch("/api/data");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        document.getElementById("avg-price").textContent = data.avg_price ? fmt(data.avg_price) : "—";
        document.getElementById("updated-at").textContent = "Zaktualizowano: " + data.updated_at;
        renderCards(data.sources);
    } catch (err) {
        errorMsg.textContent = "Błąd pobierania danych: " + err.message;
        errorMsg.classList.remove("hidden");
    } finally {
        btn.disabled = false;
        icon.classList.remove("spinning");
    }
}

fetchData();
