# -*- coding: utf-8 -*-
"""
AI 相場分析ダッシュボード（ドル円 / 株）
Streamlit + LightGBM + Plotly

- データ: Yahoo Finance (yfinance) / FRED公開CSV（APIキー・Secrets設定は不要）
- モデル: LightGBM（予測期間ごとに個別学習）
- 本アプリは教育・情報提供目的であり、投資助言ではありません。
"""
import io
import math
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI 相場分析ダッシュボード", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# 定数
# ----------------------------------------------------------------------------
# 暦日 -> 営業日（学習時の先読み日数）
HORIZONS = {30: 21, 90: 63, 180: 126, 365: 252}

FEATURE_LABELS = {
    "rate_diff_policy": "日米政策金利差",
    "rate_diff_2y": "日米2年金利差",
    "rate_diff_10y": "日米10年金利差",
    "cpi_yoy": "CPI（前年比%）",
    "pce_yoy": "PCE（前年比%）",
    "payems_chg": "雇用統計（非農業部門雇用者数 前月差）",
    "vix": "VIX",
    "vix_chg20": "VIX 20日変化",
    "rsi14": "RSI(14)",
    "macd": "MACD",
    "macd_signal": "MACDシグナル",
    "macd_hist": "MACDヒストグラム",
    "atr_pct": "ATR(14)（価格比%）",
    "ret_5": "5日リターン",
    "ret_20": "20日リターン",
    "vol_20": "20日ボラティリティ",
}

FRED_IDS = {
    "us_policy": "FEDFUNDS",          # 米フェデラルファンド実効金利（月次）
    "jp_policy": "IRSTCI01JPM156N",   # 日本 短期（コール）金利（月次・政策金利の代理）
    "us2y": "DGS2",                   # 米2年債利回り（日次）
    "us10y": "DGS10",                 # 米10年債利回り（日次）
    "jp10y": "IRLTLT01JPM156N",       # 日本10年債利回り（月次）
    "cpi": "CPIAUCSL",                # 米CPI（月次）
    "pce": "PCEPI",                   # 米PCEデフレーター（月次）
    "payems": "PAYEMS",               # 米非農業部門雇用者数（月次）
}

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=2,
    verbose=-1,
)

STAR_LABELS = {5: "強気", 4: "やや強気", 3: "中立", 2: "やや弱気", 1: "弱気"}


# ----------------------------------------------------------------------------
# ユーティリティ
# ----------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def rating_from_prob(p_up: float) -> int:
    if p_up >= 0.62:
        return 5
    if p_up >= 0.54:
        return 4
    if p_up >= 0.46:
        return 3
    if p_up >= 0.38:
        return 2
    return 1


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


# ----------------------------------------------------------------------------
# データ取得
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, period: str = "20y") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"])
    for col in ("High", "Low"):
        if col not in df.columns:
            df[col] = df["Close"]
    df = df[["High", "Low", "Close"]].astype(float)
    return _clean_index(df)


@st.cache_data(ttl=1800, show_spinner=False)
def load_fred(series_id: str) -> pd.Series:
    """FRED公開CSV（APIキー不要）から取得。失敗時は空のSeriesを返す（アプリは止めない）。"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; fx-dashboard/1.0)"}
    for _ in range(3):
        try:
            r = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={"id": series_id},
                headers=headers,
                timeout=25,
            )
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            s = pd.Series(
                pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                index=pd.to_datetime(df.iloc[:, 0]),
            )
            s = s[s.index >= "2000-01-01"].dropna().sort_index()
            if not s.empty:
                return s.astype(float)
        except Exception:
            time.sleep(1)
    return pd.Series(dtype=float)


def _to_daily(s: pd.Series, index: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """公表ラグ分だけ遅らせて日次へ前方補完（未来情報の混入を防ぐ）。"""
    if s.empty:
        return pd.Series(np.nan, index=index)
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def build_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    raw = {name: load_fred(sid) for name, sid in FRED_IDS.items()}
    # FRED取得失敗時は Yahoo Finance の米10年債利回り(^TNX)で代替
    if raw["us10y"].empty:
        tnx = load_prices("^TNX")
        if not tnx.empty:
            raw["us10y"] = tnx["Close"]

    us_policy = _to_daily(raw["us_policy"], index, 30)
    jp_policy = _to_daily(raw["jp_policy"], index, 30)
    us2y = _to_daily(raw["us2y"], index, 1)
    us10y = _to_daily(raw["us10y"], index, 1)
    jp10y = _to_daily(raw["jp10y"], index, 30)
    # 日本2年金利は公的な無料系列が乏しいため、短期金利と10年金利の中間値で近似
    jp2y = (jp_policy + jp10y) / 2.0

    cpi = raw["cpi"]
    pce = raw["pce"]
    payems = raw["payems"]
    cpi_yoy = _to_daily(cpi.pct_change(12) * 100 if not cpi.empty else cpi, index, 45)
    pce_yoy = _to_daily(pce.pct_change(12) * 100 if not pce.empty else pce, index, 45)
    pay_chg = _to_daily(payems.diff() if not payems.empty else payems, index, 45)

    return pd.DataFrame(
        {
            "rate_diff_policy": us_policy - jp_policy,
            "rate_diff_2y": us2y - jp2y,
            "rate_diff_10y": us10y - jp10y,
            "cpi_yoy": cpi_yoy,
            "pce_yoy": pce_yoy,
            "payems_chg": pay_chg,
        },
        index=index,
    )


# ----------------------------------------------------------------------------
# 特徴量
# ----------------------------------------------------------------------------
def technicals(px: pd.DataFrame) -> pd.DataFrame:
    close, high, low = px["Close"], px["High"], px["Low"]
    out = pd.DataFrame(index=px.index)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / close * 100
    out["macd_signal"] = signal / close * 100
    out["macd_hist"] = (macd - signal) / close * 100

    prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    out["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / close * 100

    logret = np.log(close).diff()
    out["ret_5"] = np.log(close / close.shift(5))
    out["ret_20"] = np.log(close / close.shift(20))
    out["vol_20"] = logret.rolling(20).std() * math.sqrt(252)
    return out


def build_features(px: pd.DataFrame):
    """特徴量行列と、取得できなかった特徴量名のリストを返す。"""
    X = technicals(px)

    vix = load_prices("^VIX")
    if not vix.empty:
        v = vix["Close"].reindex(px.index).ffill()
        X["vix"] = v
        X["vix_chg20"] = v.diff(20)
    else:
        X["vix"] = np.nan
        X["vix_chg20"] = np.nan

    macro = build_macro(px.index)
    X = pd.concat([X, macro], axis=1)

    all_nan = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=all_nan)
    X = X.iloc[60:]  # テクニカル指標のウォームアップ分を除外
    return X, [FEATURE_LABELS.get(c, c) for c in all_nan]


# ----------------------------------------------------------------------------
# モデル（LightGBM）
# ----------------------------------------------------------------------------
def _fit_horizon(X: pd.DataFrame, close: pd.Series, h_td: int, shrink: float) -> dict:
    y = np.log(close.shift(-h_td) / close).reindex(X.index)
    mask = y.notna()
    Xtr, ytr = X[mask], y[mask]

    resid, hits = [], []
    tscv = TimeSeriesSplit(n_splits=4, gap=h_td)
    for tr, te in tscv.split(Xtr):
        if len(tr) < 400:
            continue
        m = LGBMRegressor(**LGBM_PARAMS).fit(Xtr.iloc[tr], ytr.iloc[tr])
        p = shrink * m.predict(Xtr.iloc[te])
        yt = ytr.iloc[te].values
        resid.append(yt - p)
        hits.append(np.sign(p) == np.sign(yt))

    sigma = float(np.std(np.concatenate(resid))) if resid else float(ytr.std())
    sigma = max(sigma, 0.005)
    hit = float(np.mean(np.concatenate(hits))) if hits else float("nan")

    final = LGBMRegressor(**LGBM_PARAMS).fit(Xtr, ytr)
    mu = shrink * float(final.predict(X.iloc[[-1]])[0])

    gain = final.booster_.feature_importance(importance_type="gain").astype(float)
    total = gain.sum()
    imp = pd.Series(gain / total * 100 if total > 0 else gain, index=X.columns)
    return {"mu": mu, "sigma": sigma, "hit": hit, "imp": imp, "n": int(len(Xtr))}


@st.cache_data(ttl=3600, show_spinner=False)
def run_model(X: pd.DataFrame, close: pd.Series, shrink: float) -> dict:
    return {h: _fit_horizon(X, close, td, shrink) for h, td in HORIZONS.items()}


def forecast_table(s0: float, res: dict, fmt: str) -> pd.DataFrame:
    rows = []
    for h, r in res.items():
        mu, sg = r["mu"], r["sigma"]
        center = s0 * math.exp(mu)
        p_up = norm_cdf(mu / sg)
        lo50, hi50 = s0 * math.exp(mu - 0.6745 * sg), s0 * math.exp(mu + 0.6745 * sg)
        lo95, hi95 = s0 * math.exp(mu - 1.96 * sg), s0 * math.exp(mu + 1.96 * sg)
        rows.append(
            {
                "期間": f"{h}日後",
                "予測値": fmt.format(center),
                "変化": f"{center - s0:+,.2f}",
                "上昇確率": f"{p_up * 100:.1f}%",
                "下落確率": f"{(1 - p_up) * 100:.1f}%",
                "予想レンジ(50%)": f"{fmt.format(lo50)} 〜 {fmt.format(hi50)}",
                "95%信頼区間": f"{fmt.format(lo95)} 〜 {fmt.format(hi95)}",
                "OOS方向的中率": "-" if math.isnan(r["hit"]) else f"{r['hit'] * 100:.0f}%",
            }
        )
    return pd.DataFrame(rows)


def overall_rating(res: dict) -> tuple:
    probs = [norm_cdf(r["mu"] / r["sigma"]) for r in res.values()]
    avg = float(np.mean(probs))
    n = rating_from_prob(avg)
    return n, avg


# ----------------------------------------------------------------------------
# 描画
# ----------------------------------------------------------------------------
def fan_chart(px: pd.DataFrame, res: dict, title: str) -> go.Figure:
    close = px["Close"]
    s0, d0 = float(close.iloc[-1]), close.index[-1]
    hist = close.tail(400)

    xs = [d0] + [d0 + pd.Timedelta(days=h) for h in res]
    center = [s0] + [s0 * math.exp(r["mu"]) for r in res.values()]
    hi95 = [s0] + [s0 * math.exp(r["mu"] + 1.96 * r["sigma"]) for r in res.values()]
    lo95 = [s0] + [s0 * math.exp(r["mu"] - 1.96 * r["sigma"]) for r in res.values()]
    hi50 = [s0] + [s0 * math.exp(r["mu"] + 0.6745 * r["sigma"]) for r in res.values()]
    lo50 = [s0] + [s0 * math.exp(r["mu"] - 0.6745 * r["sigma"]) for r in res.values()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="実績", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=xs, y=hi95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=lo95, fill="tonexty", fillcolor="rgba(255,127,14,0.15)",
                             line=dict(width=0), name="95%信頼区間"))
    fig.add_trace(go.Scatter(x=xs, y=hi50, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=lo50, fill="tonexty", fillcolor="rgba(255,127,14,0.30)",
                             line=dict(width=0), name="50%レンジ"))
    fig.add_trace(go.Scatter(x=xs, y=center, name="予測（中心）", mode="lines+markers",
                             line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def importance_chart(imp: pd.Series) -> go.Figure:
    imp = imp.sort_values()
    labels = [FEATURE_LABELS.get(i, i) for i in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=labels, orientation="h", marker_color="#2ca02c"))
    fig.update_layout(title="特徴量重要度（gain, %）", height=max(320, 28 * len(imp) + 100),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_rating(res: dict):
    n, avg = overall_rating(res)
    st.markdown("### 総合判定")
    st.markdown(
        f"<div style='font-size:2.2rem;'>{stars(n)} "
        f"<span style='font-size:1.6rem;'>{STAR_LABELS[n]}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"30/90/180/365日の上昇確率の平均: {avg * 100:.1f}%  "
               "（62%以上=強気 / 54%以上=やや強気 / 46%以上=中立 / 38%以上=やや弱気 / それ未満=弱気）")


def render_importance(res: dict):
    st.subheader("🧠 特徴量重要度（LightGBM）")
    h = st.selectbox("予測期間", list(res.keys()), index=1, format_func=lambda d: f"{d}日予測モデル")
    st.plotly_chart(importance_chart(res[h]["imp"]), use_container_width=True)


# ----------------------------------------------------------------------------
# モンテカルロ
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def monte_carlo(s0, units, swap_day, mu_annual, sigma_daily, n=10000, steps=252, seed=42):
    rng = np.random.default_rng(seed)
    # 裾の厚い分布（t分布 自由度5, 分散1に正規化）
    z = rng.standard_t(5, size=(n, steps)) / math.sqrt(5 / 3)
    r = mu_annual / steps + sigma_daily * z
    S = s0 * np.exp(np.cumsum(r, axis=1))
    swap_step = swap_day * 365.0 / steps
    t = np.arange(1, steps + 1)
    pnl = units * (S - s0) + swap_step * t
    pnl = np.concatenate([np.zeros((n, 1)), pnl], axis=1)

    final = pnl[:, -1]
    dd = (np.maximum.accumulate(pnl, axis=1) - pnl).max(axis=1)
    pct = np.percentile(pnl, [5, 50, 95], axis=0)
    return {"final": final, "dd": dd, "pct": pct, "sample": pnl[:80], "steps": steps}


def render_monte_carlo(s0, units, swap_day, res, close):
    st.subheader("🎲 モンテカルロシミュレーション（10,000回 / 1年）")
    sigma_daily = float(np.log(close).diff().tail(250).std())
    mu_annual = res[365]["mu"]
    mc = monte_carlo(s0, float(units), float(swap_day), mu_annual, sigma_daily)

    final, dd = mc["final"], mc["dd"]
    notional = units * s0 if units > 0 else float("nan")
    loss_p = float((final < 0).mean())
    c = st.columns(4)
    c[0].metric("損失確率", f"{loss_p * 100:.1f}%")
    c[1].metric("利益確率", f"{(1 - loss_p) * 100:.1f}%")
    c[2].metric("1年後 期待収益", f"{final.mean():,.0f} 円",
                None if math.isnan(notional) else f"元本比 {final.mean() / notional * 100:+.1f}%")
    c[3].metric("最大ドローダウン予測", f"{dd.mean():,.0f} 円",
                f"95%タイル {np.percentile(dd, 95):,.0f} 円", delta_color="off")
    st.caption(f"前提: 年率ドリフト={mu_annual * 100:+.2f}%（365日モデル予測）、日次ボラ={sigma_daily * 100:.2f}%（直近250日）、"
               "スワップは日割りで累積。最大ドローダウンは損益曲線の山から谷までの最大下落額。")

    x = np.linspace(0, 365, mc["steps"] + 1)
    fig = go.Figure()
    for path in mc["sample"]:
        fig.add_trace(go.Scatter(x=x, y=path, mode="lines", line=dict(width=0.6, color="rgba(100,100,100,0.25)"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mc["pct"][2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mc["pct"][0], fill="tonexty", fillcolor="rgba(31,119,180,0.18)",
                             line=dict(width=0), name="5〜95%帯"))
    fig.add_trace(go.Scatter(x=x, y=mc["pct"][1], name="中央値", line=dict(color="#d62728", width=2.5)))
    fig.update_layout(title="損益パス（円）", xaxis_title="経過日数", yaxis_title="累積損益（円）",
                      height=420, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    h1 = go.Figure(go.Histogram(x=final, nbinsx=80, marker_color="#1f77b4"))
    h1.add_vline(x=0, line_dash="dash", line_color="red")
    h1.update_layout(title="1年後損益の分布（円）", height=350, margin=dict(l=10, r=10, t=50, b=10))
    c1.plotly_chart(h1, use_container_width=True)
    h2 = go.Figure(go.Histogram(x=dd, nbinsx=80, marker_color="#ff7f0e"))
    h2.update_layout(title="最大ドローダウンの分布（円）", height=350, margin=dict(l=10, r=10, t=50, b=10))
    c2.plotly_chart(h2, use_container_width=True)


# ----------------------------------------------------------------------------
# スワップ分析
# ----------------------------------------------------------------------------
def render_swap(s0):
    st.subheader("💰 スワップ分析")
    c1, c2 = st.columns(2)
    units = c1.number_input("保有通貨数", min_value=0, value=100000, step=1000)
    swap_day = c2.number_input("スワップ/日（円）", min_value=0, value=1500, step=100)

    annual = swap_day * 365
    breakeven = annual / units if units > 0 else float("nan")
    m = st.columns(3)
    m[0].metric("年間スワップ", f"{annual:,.0f} 円")
    m[1].metric("損益分岐円高幅（1年保有）", "-" if math.isnan(breakeven) else f"{breakeven:.2f} 円")
    m[2].metric("損益分岐レート", "-" if math.isnan(breakeven) else f"{s0 - breakeven:.2f}")

    rows = []
    for d in (1, 3, 5, 10, 20):
        fx_pl = -d * units
        rows.append(
            {
                "現在レートから": f"-{d}円",
                "下落後レート": f"{s0 - d:.2f}",
                "為替損益": f"{fx_pl:+,.0f} 円",
                "年間スワップ": f"{annual:+,.0f} 円",
                "合計損益（1年）": f"{fx_pl + annual:+,.0f} 円",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    return units, swap_day


# ----------------------------------------------------------------------------
# 画面: ドル円
# ----------------------------------------------------------------------------
def fx_page(shrink: float):
    st.title("💴 ドル円（USD/JPY）分析")
    with st.spinner("データ取得・モデル学習中…（初回は数十秒かかります）"):
        px = load_prices("USDJPY=X")
        if px.empty or len(px) < 800:
            st.error("Yahoo Financeからドル円データを取得できませんでした。しばらくしてから再読込してください。")
            return
        X, missing = build_features(px)
        res = run_model(X, px["Close"], shrink)

    if missing:
        st.warning("取得できず除外した特徴量: " + "、".join(missing) +
                   "（外部データ提供元に一時的に接続できない場合に発生します。しばらくして「キャッシュをクリア」を押すと再取得します）")

    s0 = float(px["Close"].iloc[-1])
    st.caption(f"最終データ日: {px.index[-1].date()}  /  学習サンプル数(30日モデル): {res[30]['n']:,}")

    cols = st.columns(5)
    cols[0].metric("現在レート", f"{s0:.2f} 円")
    for col, (h, r) in zip(cols[1:], res.items()):
        pred = s0 * math.exp(r["mu"])
        col.metric(f"{h}日予測", f"{pred:.2f} 円", f"{pred - s0:+.2f}")

    render_rating(res)
    st.subheader("📊 予測サマリー")
    st.dataframe(forecast_table(s0, res, "{:.2f}"), use_container_width=True, hide_index=True)
    st.plotly_chart(fan_chart(px, res, "USD/JPY 実績と予測レンジ"), use_container_width=True)

    st.divider()
    units, swap_day = render_swap(s0)
    st.divider()
    render_monte_carlo(s0, units, swap_day, res, px["Close"])
    st.divider()
    render_importance(res)


# ----------------------------------------------------------------------------
# 画面: 株
# ----------------------------------------------------------------------------
def stock_page(shrink: float):
    st.title("🏢 株分析")
    ticker = st.text_input("ティッカー（例: 7203.T, 6758.T, AAPL, ^N225）", value="7203.T").strip()
    if not ticker:
        return
    with st.spinner("データ取得・モデル学習中…"):
        px = load_prices(ticker)
        if px.empty or len(px) < 800:
            st.error("データを取得できないか、学習に必要な履歴（約3年以上）が不足しています。ティッカーを確認してください。")
            return
        X, missing = build_features(px)
        res = run_model(X, px["Close"], shrink)

    if missing:
        st.warning("取得できず除外した特徴量: " + "、".join(missing))

    s0 = float(px["Close"].iloc[-1])
    st.caption(f"最終データ日: {px.index[-1].date()}  /  マクロ特徴量は米国・日本の共通指標を市場環境として使用")
    cols = st.columns(5)
    cols[0].metric("現在値", f"{s0:,.2f}")
    for col, (h, r) in zip(cols[1:], res.items()):
        pred = s0 * math.exp(r["mu"])
        col.metric(f"{h}日予測", f"{pred:,.2f}", f"{(pred / s0 - 1) * 100:+.1f}%")

    render_rating(res)
    st.subheader("📊 予測サマリー")
    st.dataframe(forecast_table(s0, res, "{:,.2f}"), use_container_width=True, hide_index=True)
    st.plotly_chart(fan_chart(px, res, f"{ticker} 実績と予測レンジ"), use_container_width=True)
    st.divider()
    render_importance(res)


# ----------------------------------------------------------------------------
# エントリポイント
# ----------------------------------------------------------------------------
def main():
    st.sidebar.title("📈 AI 相場分析")
    mode = st.sidebar.radio("分析モード", ["株分析モード", "ドル円分析モード"], index=1)
    shrink = st.sidebar.slider(
        "モデル予測の縮小係数", 0.0, 1.0, 0.5, 0.05,
        help="為替・株の予測はノイズが大きいため、LightGBMの予測値を縮小して過信を防ぎます。1.0で縮小なし。",
    )
    if st.sidebar.button("🔄 キャッシュをクリアして再取得"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("データ: Yahoo Finance / FRED\nモデル: LightGBM")

    if mode == "ドル円分析モード":
        fx_page(shrink)
    else:
        stock_page(shrink)

    st.divider()
    st.caption("⚠️ 本アプリは教育・情報提供を目的としたもので、投資助言ではありません。"
               "予測は将来の結果を保証せず、投資判断は自己責任で行ってください。")


main()
