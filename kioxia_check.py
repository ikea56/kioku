#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半導体4銘柄 売られ過ぎ／買われ過ぎ 判定ツール
  キオクシア(285A) / 東京エレクトロン(8035) / アドバンテスト(6857) / ディスコ(6146)
--------------------------------------------------------------
目的は「未来予測」ではなく、
「海外の動きを受けて、各銘柄がどれだけ織り込んだか(予測とのズレ)」を測ること。

  予測 = 前日の海外指標から計算した、その銘柄の寄付き予想(前日比%)
  気配 = 寄り前の気配(前日比%)  ※スマホの画面に自分で入力
  差異 = 気配 − 予測   → マイナスが大きいほど「売られ過ぎ」

★4銘柄は同じモデルを使わず、銘柄ごとに参考にする指標を変えています(TARGETS を参照)。

実行:   python kioxia_check.py
        python kioxia_check.py --actual 285A=-9.4 8035=-2.0   # 気配も指定して判定
        python kioxia_check.py --demo                         # 架空データで動作確認
出力:   kioxia_judge.html (スマホで開く画面) / predictions.csv
--------------------------------------------------------------
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

TZ = "Asia/Tokyo"

# 参考にする海外・為替の指標: 内部名: (Yahoo Financeのコード, 画面に出す名前)
INDICATORS = {
    "MU": ("MU", "マイクロン"),
    "HYNIX": ("000660.KS", "SKハイニックス"),
    "SAMSUNG": ("005930.KS", "サムスン"),
    "SOX": ("^SOX", "SOX指数"),
    "NASDAQ": ("^IXIC", "NASDAQ"),
    "USDJPY": ("JPY=X", "ドル円"),
    "TSM": ("TSM", "TSMC"),
    "ASML": ("ASML", "ASML"),
    "NVDA": ("NVDA", "エヌビディア"),
    "AMD": ("AMD", "AMD"),
    "AVGO": ("AVGO", "ブロードコム"),
}

# 予測する銘柄。feats は「重要な順」。銘柄ごとに別々のモデルで予測します。
# strong / normal は判定のしきい値(予測と気配の差, %)。変えたいときはここを直す。
TARGETS = {
    "KIOXIA": dict(ticker="285A.T", code="285A", name="キオクシア",
                   purpose="メモリ(NAND)市況との連動を見ています",
                   feats=["MU", "HYNIX", "SAMSUNG", "SOX", "USDJPY", "NASDAQ"],
                   strong=4.0, normal=2.0),
    "TEL": dict(ticker="8035.T", code="8035", name="東京エレクトロン",
                purpose="半導体の設備投資の動きとの連動を見ています",
                feats=["SOX", "TSM", "ASML", "NVDA", "NASDAQ", "USDJPY"],
                strong=4.0, normal=2.0),
    "ADV": dict(ticker="6857.T", code="6857", name="アドバンテスト",
                purpose="AI向け半導体の需要との連動を見ています",
                feats=["NVDA", "SOX", "AMD", "AVGO", "NASDAQ", "USDJPY"],
                strong=4.0, normal=2.0),
    "DISCO": dict(ticker="6146.T", code="6146", name="ディスコ",
                  purpose="半導体の製造全体の動きとの連動を見ています",
                  feats=["SOX", "NVDA", "TSM", "ASML", "HYNIX", "NASDAQ"],
                  strong=4.0, normal=2.0),
}
ALPHAS = [1.0, 30.0, 300.0]   # 指標が多いとき、偏りを抑える強さの候補

SKIP_FIRST_DAYS = 5     # 上場直後(値付けが不安定)は分析から除外
MAX_STALE_DAYS = 5      # 指標のデータが5日より古い日は使わない
MIN_TRAIN = 120         # 過去検証で最初に学習に使う日数
REFIT_EVERY = 20        # 過去検証で学び直す間隔(営業日)
CONF_MIN_PRED = 0.3     # 信頼度の計算で、予測がこれより小さい日(±0.3%未満)は除く

BINS = [   # (下限以上, 上限未満, 表示名)
    (10, np.inf, "+10%以上"),
    (5, 10, "+5%〜+10%"),
    (2, 5, "+2%〜+5%"),
    (-2, 2, "-2%〜+2%"),
    (-5, -2, "-5%〜-2%"),
    (-10, -5, "-10%〜-5%"),
    (-np.inf, -10, "-10%以下"),
]


def stars(diff, th):
    """予測と気配の差(%) → ★の数(5=強い買い 〜 1=強い売り)"""
    if diff <= -th["strong"]: return 5
    if diff <= -th["normal"]: return 4
    if diff < th["normal"]: return 3
    if diff < th["strong"]: return 2
    return 1


STAR_NAME = {5: "強い買い", 4: "買い", 3: "中立", 2: "売り", 1: "強い売り"}


# ---------------------------------------------------------------- データ取得
def fetch(years):
    import yfinance as yf
    start = (pd.Timestamp.now(tz=TZ) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    tickers = {**{k: v[0] for k, v in INDICATORS.items()},
               **{k: v["ticker"] for k, v in TARGETS.items()}}
    out = {}
    for name, ticker in tickers.items():
        df = None
        for _ in range(2):   # 取得失敗に備えて2回まで試す
            try:
                df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
                if df is not None and not df.empty:
                    break
            except Exception:
                df = None
        if df is None or df.empty:
            print(f"[注意] {name}({ticker})は取得できなかったので、使わずに進めます。")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out[name] = df.sort_index()    # 始値・高値・安値・終値・出来高が入っています
    return out


def demo_data():
    """動作確認用の架空データ。実際の相場とは無関係です。"""
    rng = np.random.default_rng(11)
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    days = pd.bdate_range(end=today, periods=1300)
    n = len(days)
    f = {k: rng.normal(0, 0.015, n) for k in ("mkt", "semi", "mem", "ai", "eq")}
    nz = lambda s: rng.normal(0, s, n)
    R = {
        "MU": 1.2 * f["mem"] + 0.6 * f["semi"] + nz(.015),
        "HYNIX": 1.0 * f["mem"] + 0.5 * f["semi"] + nz(.015),
        "SAMSUNG": 0.6 * f["mem"] + 0.4 * f["semi"] + nz(.012),
        "SOX": 1.0 * f["semi"] + 0.3 * f["ai"] + 0.3 * f["eq"] + nz(.006),
        "NASDAQ": 0.5 * f["mkt"] + 0.3 * f["semi"] + nz(.005),
        "USDJPY": -0.1 * f["mkt"] + nz(.005),
        "TSM": 0.9 * f["semi"] + 0.5 * f["eq"] + nz(.012),
        "ASML": 0.8 * f["semi"] + 0.8 * f["eq"] + nz(.012),
        "NVDA": 1.2 * f["ai"] + 0.6 * f["semi"] + nz(.015),
        "AMD": 1.0 * f["ai"] + 0.6 * f["semi"] + nz(.015),
        "AVGO": 0.8 * f["ai"] + 0.5 * f["semi"] + nz(.012),
    }

    def frame(r, base, idx=days):
        c = base * np.cumprod(1 + r)
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, len(c)))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, len(c))))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, len(c))))
        return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c,
                             "Volume": rng.integers(1e6, 5e6, len(c))}, index=idx)

    out = {k: frame(r, 100) for k, r in R.items()}
    out["USDJPY"] = frame(R["USDJPY"], 150)

    # 各銘柄の「寄付きの動き」は、前日の指標の動きの組み合わせ + ノイズ(銘柄ごとに効く指標が違う)
    recipes = {
        "KIOXIA": ({"MU": .55, "HYNIX": .25, "SAMSUNG": .1, "SOX": .2, "USDJPY": -.3}, .02, 470),
        "TEL": ({"SOX": .7, "TSM": .15, "ASML": .15, "NVDA": .05, "NASDAQ": .1}, .011, 1300),
        "ADV": ({"NVDA": .45, "SOX": .4, "AMD": .15, "AVGO": .1}, .014, 1300),
        "DISCO": ({"SOX": .6, "NVDA": .1, "TSM": .15, "ASML": .15, "HYNIX": .08}, .013, 1300),
    }
    for key, (w, s, length) in recipes.items():
        gap = np.zeros(n)
        for name, coef in w.items():
            gap[1:] += coef * R[name][:-1]
        noise = rng.normal(0, s, n)
        gap += noise
        intra = rng.normal(0, s * 0.8, n) - 0.2 * noise
        o_, c_, prev = [], [], 3000.0
        for g, it in zip(gap, intra):
            o = prev * (1 + g)
            c = o * (1 + it)
            o_.append(o); c_.append(c); prev = c
        df = pd.DataFrame({"Open": o_, "Close": c_}, index=days)
        df["High"] = df[["Open", "Close"]].max(axis=1) * 1.01
        df["Low"] = df[["Open", "Close"]].min(axis=1) * 0.99
        df["Volume"] = 1_000_000
        out[key] = df.iloc[-length:]
    return out


# ---------------------------------------------------------------- 表の作成
def build_one(stock, data, feats):
    """日本の各営業日に、その直前の各指標の騰落率を結びつける(寄り前に分かる情報だけ使う)。"""
    k = stock[["Open", "Close"]].copy()
    k["prev_close"] = k["Close"].shift(1)
    k["k_open"] = (k["Open"] / k["prev_close"] - 1) * 100     # 寄付きの前日比
    k["k_close"] = (k["Close"] / k["prev_close"] - 1) * 100   # 終値の前日比
    k["k_intra"] = (k["Close"] / k["Open"] - 1) * 100         # 寄付き→終値
    k.index.name = "date"
    k = k.iloc[SKIP_FIRST_DAYS:].dropna().reset_index()
    k["date"] = pd.to_datetime(k["date"])

    for name in feats:
        c = data[name]["Close"].dropna()
        r = pd.DataFrame({f"{name}_date": pd.to_datetime(c.index),
                          name: c.pct_change().values * 100}).dropna()
        k = pd.merge_asof(k.sort_values("date"), r.sort_values(f"{name}_date"),
                          left_on="date", right_on=f"{name}_date",
                          direction="backward", allow_exact_matches=False)
        stale = (k["date"] - k[f"{name}_date"]).dt.days > MAX_STALE_DAYS
        k.loc[stale, name] = np.nan
    return k


def bin_table(df, col):
    rows = []
    for lo, hi, label in BINS:
        s = df[(df[col] >= lo) & (df[col] < hi)]
        n = len(s)
        row = {"label": label, "n": n}
        for key, c in (("open", "k_open"), ("close", "k_close")):
            up, down = int((s[c] > 0).sum()), int((s[c] < 0).sum())
            row[f"{key}_avg"] = s[c].mean() if n else None
            row[f"{key}_max"] = s[c].max() if n else None
            row[f"{key}_min"] = s[c].min() if n else None
            row[f"{key}_up"], row[f"{key}_down"] = up, down
            row[f"{key}_win"] = round(up / (up + down) * 100) if up + down else None
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 予測モデル(銘柄ごとに別々に作る)
def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    b = w / sd
    return float(ym - (b * mu).sum()), b


def walk_forward(base, cols, alpha):
    """過去を1日ずつ進めながら、『その日より前のデータだけ』で予測する(ズルなしの検証)。"""
    X, y = base[cols].values.astype(float), base["k_open"].values.astype(float)
    n = len(base)
    pred = np.full(n, np.nan)
    i = min(MIN_TRAIN, max(30, n // 3))
    while i < n:
        j = min(i + REFIT_EVERY, n)
        a, b = ridge_fit(X[:i], y[:i], alpha)
        pred[i:j] = a + X[i:j] @ b
        i = j
    return pd.Series(pred, index=base.index)


def choose_model(base, sets):
    """指標の使い方(1つだけ/上位3つ/全部)と偏りの強さを試し、過去検証で一番当たったものを選ぶ。"""
    y = base["k_open"].astype(float)
    results, seen = [], set()
    for label, cols in sets.items():
        cols = tuple(cols)
        if not cols or cols in seen:
            continue
        seen.add(cols)
        for alpha in ([ALPHAS[0]] if len(cols) == 1 else ALPHAS):
            pred = walk_forward(base, list(cols), alpha)
            m = pred.notna()
            if m.sum() < 30:
                continue
            err = y[m] - pred[m]
            sse = float((err ** 2).sum())
            sst = float(((y[m] - y[m].mean()) ** 2).sum())
            results.append({"label": label, "cols": list(cols), "alpha": alpha,
                            "rmse": float(np.sqrt((err ** 2).mean())),
                            "r2": 1 - sse / sst, "pred": pred})
    if not results:
        return None
    best = min(results, key=lambda r: r["rmse"])
    a, b = ridge_fit(base[best["cols"]].values.astype(float), y.values, best["alpha"])
    return {**best, "a": a, "coef": dict(zip(best["cols"], b))}


# ---------------------------------------------------------------- 過去検証(バックテスト)
def bt_stats(r):
    n = len(r)
    if n == 0:
        return None
    w, l = r[r > 0], r[r < 0]
    return {"n": n, "win": round((r > 0).sum() / n * 100),
            "avg_win": w.mean() if len(w) else None,
            "avg_loss": l.mean() if len(l) else None,
            "worst": r.min(), "expect": r.mean()}


def backtest(d):
    """『買い(★4以上)』の日に寄付きで買い、その日の終値で売っていたら？(手数料なし)"""
    last = d["date"].max()
    out, prev_n = [], None
    for label, days in (("過去1年", 365), ("過去3年", 1095)):
        g = d[d["date"] > last - pd.Timedelta(days=days)]
        if g.empty or len(g) == prev_n:
            continue
        prev_n = len(g)
        span = (last - g["date"].min()).days / 365.25
        if span < days / 365.25 - 0.1:
            label = f"全期間(約{span:.1f}年分)"
        out.append({"label": label, "buy": bt_stats(g.loc[g["star"] >= 4, "k_intra"])})
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (np.isnan(o) or np.isinf(o)) else round(float(o), 2)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y/%m/%d")
    return o


# ---------------------------------------------------------------- 1銘柄ぶんの分析
def analyze(key, cfg, data, target, actual):
    note = []
    feats = [f for f in cfg["feats"] if f in data]
    for f in cfg["feats"]:
        if f not in data:
            note.append(f"{INDICATORS[f][1]}のデータが取れませんでした")
    if key not in data or not feats:
        return None, [f"{cfg['name']}のデータが取れなかったので判定できません"]
    df = build_one(data[key], data, feats)
    feats = [f for f in feats if df[f].notna().mean() > 0.9]
    if not feats:
        return None, [f"{cfg['name']}: 使える指標がありません"]
    base = df.dropna(subset=feats).reset_index(drop=True)
    if len(base) < 60:
        return None, [f"{cfg['name']}: 分析できる日数が少なすぎます({len(base)}日)"]

    sets = {"最重要の1つ": feats[:1], "上位3つ": feats[:3], "全部": feats}
    best = choose_model(base, sets)
    if best is None:
        return None, [f"{cfg['name']}: 過去検証に使えるデータが足りません"]

    th = {"strong": cfg["strong"], "normal": cfg["normal"]}
    d = base.assign(pred=best["pred"]).dropna(subset=["pred"]).copy()
    d["diff"] = d["k_open"] - d["pred"]
    d["star"] = d["diff"].apply(lambda v: stars(v, th))

    # 信頼度: 過去の検証で、予測の向き(上か下か)が当たった日の割合
    m = d["pred"].abs() >= CONF_MIN_PRED
    sub = d[m] if m.sum() >= 30 else d
    acc = float((np.sign(sub["pred"]) == np.sign(sub["k_open"])).mean())

    def grp(g):
        return {"n": len(g), "up": int((g["k_intra"] > 0).sum()),
                "down": int((g["k_intra"] < 0).sum()),
                "avg": g["k_intra"].mean() if len(g) else None}
    by_star = {s: grp(d[d["star"] == s]) for s in (1, 2, 3, 4, 5)}
    by_side = {"buy": grp(d[d["star"] >= 4]), "sell": grp(d[d["star"] <= 2])}

    # 今日の入力値: 対象日より前の最新の1日分の騰落率
    latest = {}
    for n_ in best["cols"]:
        c = data[n_]["Close"].dropna()
        c = c[c.index < target]
        if len(c) >= 2 and (target - c.index[-1]).days <= MAX_STALE_DAYS + 2:
            latest[n_] = {"pct": float((c.iloc[-1] / c.iloc[-2] - 1) * 100), "date": c.index[-1]}
    for n_ in best["cols"]:
        if n_ not in latest:
            note.append(f"{INDICATORS[n_][1]}の最新データが取れず、動きなしとして計算")
    x = {n_: (latest[n_]["pct"] if n_ in latest else 0.0) for n_ in best["cols"]}
    contrib = {n_: best["coef"][n_] * x[n_] for n_ in best["cols"]}
    pred_today = best["a"] + sum(contrib.values())

    effects = sorted(
        [{"name": INDICATORS[n_][1], "pct": x[n_], "contrib": contrib[n_],
          "date": (f"{latest[n_]['date'].month}/{latest[n_]['date'].day}" if n_ in latest else "-")}
         for n_ in best["cols"]], key=lambda e: -abs(e["contrib"]))
    top = effects[0]
    if abs(top["contrib"]) > 0.05:
        why = f"{top['name']}の前日の動き({top['pct']:+.1f}%)が、予測に一番大きく効いています。"
    else:
        why = "海外の動きが小さく、予測もほぼ動きなしです。"

    top_col = feats[0]
    res = {
        "key": key, "code": cfg["code"], "name": cfg["name"], "purpose": cfg["purpose"],
        "pred": pred_today, "actual": actual, "th": th,
        "conf": round(acc * 100), "n_valid": int(len(sub)),
        "used": [INDICATORS[c][1] for c in best["cols"]],
        "rmse": best["rmse"], "why": why, "effects": effects,
        "by_star": by_star, "by_side": by_side, "backtest": backtest(d),
        "table": bin_table(df, top_col), "table_name": INDICATORS[top_col][1],
        "n": len(base), "start": base["date"].min(), "end": base["date"].max(),
        "notes": note,
    }
    return res, note


# ---------------------------------------------------------------- 画面(HTML)
HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>半導体4銘柄判定</title>
<style>
:root{--bg:#f1f2f5;--ink:#1c1f24;--mut:#777}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;-webkit-text-size-adjust:100%}
.wrap{max-width:480px;margin:0 auto;padding:12px 12px 40px}
.top{display:flex;justify-content:space-between;align-items:baseline;font-size:13px;color:#555;margin-bottom:8px}
.demo{background:#fff3cd;border:1px solid #e0c36a;border-radius:10px;padding:8px 10px;font-size:13px;margin-bottom:10px}
.warn{background:#fff3cd;border-radius:12px;padding:9px 12px;font-size:13px;margin-bottom:10px;line-height:1.5}
.s0{background:#a3a9b2}.s5{background:#0b6e3a}.s4{background:#2e9e55}.s3{background:#8a8f98}.s2{background:#ee8a1c}.s1{background:#d73027}
.hero{border-radius:20px;padding:20px 14px;text-align:center;color:#fff}
.hero .hs{font-size:13px;opacity:.9}
.hero .hm{font-size:30px;font-weight:800;margin-top:4px;line-height:1.25}
.hero .hst{font-size:22px;letter-spacing:3px;margin-top:2px}
.hero .hsub{font-size:14px;margin-top:6px;line-height:1.5;opacity:.95}
.box{background:#fff;border-radius:16px;margin-top:12px;padding:12px 14px}
.box h3{font-size:15px;margin:0 0 6px}
.rk{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #eee;text-decoration:none;color:inherit;font-size:16px;font-weight:700}
.rk:last-child{border-bottom:0}
.rk .p{font-size:12px;color:var(--mut);font-weight:400;display:block}
.chip{color:#fff;border-radius:999px;padding:4px 10px;font-size:14px;font-weight:700;white-space:nowrap}
.stock{background:#fff;border-radius:18px;margin-top:14px;overflow:hidden}
.band{color:#fff;padding:14px 14px 12px;display:flex;justify-content:space-between;align-items:center}
.band .nm{font-size:21px;font-weight:800}
.band .code{font-size:12px;opacity:.85}
.band .br{text-align:right}
.band .stars{font-size:26px;letter-spacing:2px;line-height:1.1}
.band .lab{font-size:22px;font-weight:800}
.band .stt{font-size:12px;opacity:.9}
.body{padding:12px 14px 14px}
.nums{display:flex;gap:8px}
.nums .n{flex:1;text-align:center;background:#f6f7f9;border-radius:12px;padding:8px 2px}
.nums .k{display:block;font-size:12px;color:var(--mut)}
.nums .v{display:block;font-size:24px;font-weight:800;margin-top:2px}
.nums input{width:92%;font-size:22px;font-weight:800;text-align:center;border:2px solid #ccd;border-radius:10px;padding:3px 2px;background:#fff;margin-top:2px}
.nums .sg{font-size:14px;font-weight:800;border:1px solid #ccd;border-radius:8px;background:#fff;padding:1px 8px;margin-left:4px}
.conf{margin-top:10px;font-size:15px;display:flex;align-items:center;gap:8px}
.conf .bar{flex:1;height:8px;background:#e6e8ec;border-radius:4px;max-width:140px}
.conf .bar i{display:block;height:8px;border-radius:4px;background:#4a6cf7}
.cm{font-size:15px;line-height:1.6;margin-top:8px}.cm p{margin:0 0 3px}
.why{width:100%;margin-top:10px;font-size:15px;font-weight:700;padding:11px;border:0;border-radius:12px;background:#1c1f24;color:#fff}
.dt{margin-top:10px;border-top:1px solid #eee;padding-top:8px}
.dt h4{font-size:14px;margin:12px 0 4px}
.mut{font-size:12px;color:var(--mut);line-height:1.6}
.eff{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #eee;font-size:15px}
.eff b{font-weight:800}
.tiles{display:flex;gap:4px;margin:6px 0}
.tile{flex:1;border-radius:8px;padding:9px 0;text-align:center;font-size:12px;color:#fff;line-height:1.3}
.tile small{display:block;font-size:10px;opacity:.9}
table{width:100%;border-collapse:collapse;margin-top:6px;font-size:12px}
th,td{padding:5px 3px;border-bottom:1px solid #eee;text-align:center;vertical-align:top}
th{color:#666;font-weight:600}
td small{display:block;color:#777;font-size:10px}
details{margin-top:10px}summary{font-size:14px;font-weight:700;cursor:pointer}
.foot{font-size:12px;color:#888;margin-top:18px;line-height:1.6}
</style></head><body><div class="wrap">
<div class="top"><span>半導体4銘柄判定 <b id="target"></b></span><span id="asof"></span></div>
<div id="demo" class="demo" style="display:none">⚠ これは動作確認用の<b>架空データ</b>です。実際の相場の結果ではありません。</div>
<div id="warn" class="warn" style="display:none"></div>

<div id="hero" class="hero s0"><div class="hs">今日の注目</div><div class="hm" id="hm">--</div><div class="hst" id="hst"></div><div class="hsub" id="hsub"></div></div>

<div class="box"><h3>今日の注目度ランキング</h3><div id="rank"></div></div>

<div id="cards"></div>

<div class="foot">※ 過去の傾向から計算した目安で、将来の値動きを保証するものではありません。売買の最終判断はご自身で行ってください。<br>※ 信頼度は、過去の検証で「予測の向き(上か下か)」が当たった日の割合です。50%はコイン投げと同じ水準です。</div>
</div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const fmt=(x,d)=>x===null||x===undefined||isNaN(x)?"-":(x>0?"+":"")+x.toFixed(d===undefined?1:d)+"%";
const starsFor=(f,th)=>f<=-th.strong?5:f<=-th.normal?4:f<th.normal?3:f<th.strong?2:1;
const LABEL={5:"強い買い",4:"買い",3:"中立",2:"売り",1:"強い売り"};
const STATE={5:"売られ過ぎ",4:"売られ過ぎ",3:"想定どおり",2:"買われ過ぎ",1:"買われ過ぎ"};
const star=s=>"★".repeat(s)+"☆".repeat(5-s);
const heat=(share,n)=>(n<5||share===null||share===undefined)?"#d9dce1":share>=0.6?"#2e9e55":share<=0.4?"#d73027":"#9aa0a8";
const KEY="kq4_"+D.target_iso;
let saved={};try{saved=JSON.parse(localStorage.getItem(KEY)||"{}")||{}}catch(e){saved={}}
const persist=()=>{try{localStorage.setItem(KEY,JSON.stringify(saved))}catch(e){}};

function calc(i){
  const s=D.stocks[i];
  const raw=String($("act"+i).value||"").replace(/[−－ー]/g,"-").replace(",",".").trim();
  const act=parseFloat(raw);
  if(isNaN(act))return null;
  const diff=act-s.pred;
  return {act,diff,star:starsFor(diff,s.th)};
}

function comments(s,r){
  let c1,c2;
  if(r.star===5)c1="通常よりかなり強く売られています。";
  else if(r.star===4)c1="通常より強く売られています。";
  else if(r.star===2)c1="通常より強く買われています。";
  else if(r.star===1)c1="通常よりかなり強く買われています。";
  else c1="ほぼ想定どおりの動きです。";
  if(r.star===3)return [c1,"無理に動かなくて大丈夫です。"];
  const buy=r.star>=4;
  let g=s.by_star[String(r.star)];
  if(!g||g.n<5)g=buy?s.by_side.buy:s.by_side.sell;
  if(!g||g.n<5)return [c1,"過去に似た日が少なく、参考程度です。"];
  const k=buy?g.up:g.down,rate=k/g.n,t=`(${g.n}回中${k}回)`;
  if(buy)c2=rate>=0.6?`過去の類似パターンでは反発傾向があります${t}。`:rate<=0.4?`過去の類似パターンでは反発しにくい傾向です${t}。`:"過去の類似パターンでは、はっきりした傾向はありません。";
  else c2=rate>=0.6?`過去の類似パターンでは下がりやすい傾向です${t}。`:rate<=0.4?`過去の類似パターンでは下がりにくい傾向です${t}。`:"過去の類似パターンでは、はっきりした傾向はありません。";
  return [c1,c2];
}

function renderCard(i){
  const s=D.stocks[i],r=calc(i);
  $("pred"+i).textContent=fmt(s.pred);
  const band=$("band"+i);
  if(!r){
    band.className="band s0";$("stars"+i).textContent="☆☆☆☆☆";$("lab"+i).textContent="気配を入力";$("stt"+i).textContent="";
    $("diff"+i).textContent="--";$("dk"+i).textContent="差異";
    $("c1_"+i).textContent="海外の動きから、今日の予想を計算しました。";
    $("c2_"+i).textContent="寄り前の気配を入れると判定します。";
    return null;
  }
  band.className="band s"+r.star;
  $("stars"+i).textContent=star(r.star);$("lab"+i).textContent=LABEL[r.star];$("stt"+i).textContent=STATE[r.star];
  $("diff"+i).textContent=fmt(r.diff);$("dk"+i).textContent=r.diff<0?"売られ過ぎ度":r.diff>0?"買われ過ぎ度":"差異";
  const c=comments(s,r);$("c1_"+i).textContent=c[0];$("c2_"+i).textContent=c[1];
  return r;
}

function renderAll(){
  const rs=D.stocks.map((s,i)=>({i,s,r:renderCard(i)}));
  const done=rs.filter(x=>x.r).sort((a,b)=>a.r.diff-b.r.diff);
  const wait=rs.filter(x=>!x.r);
  let h="";
  done.forEach((x,k)=>{h+=`<a class="rk" href="#st${x.i}"><span>${k+1}位 ${x.s.name}</span><span class="chip s${x.r.star}">${star(x.r.star)} ${LABEL[x.r.star]}</span></a>`;});
  wait.forEach(x=>{h+=`<a class="rk" href="#st${x.i}"><span>${x.s.name}<span class="p">予測 ${fmt(x.s.pred)}</span></span><span class="chip s0">気配待ち</span></a>`;});
  $("rank").innerHTML=h;
  const hero=$("hero");
  if(!done.length){
    hero.className="hero s0";$("hm").textContent="気配を入れてください";$("hst").textContent="";
    $("hsub").textContent="4銘柄の寄り前の気配(前日比%)を入れると、今日一番チャンスのある銘柄が分かります。";return;
  }
  const top=done[0];
  const sells=done.filter(x=>x.r.star<=2).map(x=>x.s.name);
  let sub=`入力済み ${done.length}/4`;
  if(top.r.star>=4){
    hero.className="hero s"+top.r.star;$("hm").textContent=`${top.s.name}\n${LABEL[top.r.star]}`.replace("\n"," ");
    $("hst").textContent=star(top.r.star);
    sub=`予測 ${fmt(top.s.pred)} / 気配 ${fmt(top.r.act)} / 差異 ${fmt(top.r.diff)}　信頼度 ${top.s.conf}%`;
  }else{
    hero.className="hero s3";$("hm").textContent="今日は買い候補なし";$("hst").textContent="";
    sub+="　様子見でOKです。";
  }
  if(sells.length)sub+=`<br>買われ過ぎ(売り注意): ${sells.join("、")}`;
  if(done.length<4)sub+="<br>気配を入れていない銘柄は、まだ判定に入っていません。";
  $("hsub").innerHTML=sub;
}

function btTable(s){
  const rows=s.backtest;if(!rows||!rows.length)return '<div class="mut">検証できる期間がありません。</div>';
  const g=(o,f)=>o?f(o):"-";
  const col=o=>[g(o,x=>x.n+"回"),g(o,x=>x.win+"%"),g(o,x=>fmt(x.avg_win)),g(o,x=>fmt(x.avg_loss)),g(o,x=>fmt(x.worst)),g(o,x=>fmt(x.expect))];
  const names=["回数","勝率","平均利益","平均損失","最大損失","1回あたり平均(期待値)"];
  const cs=rows.map(w=>col(w.buy));
  let h="<table><tr><th></th>"+rows.map(w=>`<th>${w.label}</th>`).join("")+"</tr>";
  names.forEach((n,i)=>{h+=`<tr><td>${n}</td>${cs.map(c=>`<td>${c[i]}</td>`).join("")}</tr>`;});
  return h+"</table>";
}

function detailHTML(s){
  let h=`<div class="mut">${s.purpose}</div><h4>なぜこの判定？ 影響した要因ランキング</h4><div class="mut">${s.why}</div>`;
  h+=s.effects.map((e,k)=>`<div class="eff"><span>${k+1}位 ${e.name} <b>${e.pct>0?"▲":e.pct<0?"▼":"－"}${fmt(e.pct)}</b></span><span class="mut">予測を ${fmt(e.contrib)}</span></div>`).join("");
  h+=`<div class="mut">各指標の前日の動きが、予測をどれだけ上下させたかです。</div>`;
  h+=`<h4>過去の検証(買い判定の日に寄付きで買い、終値で売った場合)</h4>`+btTable(s);
  h+=`<div class="mut">手数料なし。気配の代わりに実際の寄付き値を使った計算です。</div>`;
  h+=`<h4>ヒートマップ(判定別に、寄付き後どうなりやすいか)</h4><div class="tiles">`;
  h+=[5,4,3,2,1].map(k=>{const t=s.by_star[String(k)];const sh=t&&(t.up+t.down)?t.up/(t.up+t.down):null;const c=heat(sh,t?t.n:0);return `<div class="tile" style="background:${c};${c==="#d9dce1"?"color:#666":""}">★${k}<small>${t?t.n:0}回</small></div>`;}).join("")+`</div>`;
  h+=`<div class="mut">緑＝寄付き後に上がりやすい(反発傾向)／赤＝下がり続けやすい／灰＝はっきりしない／薄い灰＝回数が少ない</div>`;
  let t="<tr><th>"+s.table_name+"</th><th>回数</th><th>寄付き平均</th><th>終値平均</th><th>勝率</th></tr>";
  s.table.forEach(r=>{t+=`<tr><td>${r.label}</td><td>${r.n}${r.n<5?"※":""}</td><td>${r.n?fmt(r.open_avg):"-"}</td><td>${r.n?fmt(r.close_avg):"-"}<small>最大${fmt(r.close_max)}／${fmt(r.close_min)}</small></td><td>${r.close_win===null?"-":r.close_win+"%"}<small>上${r.close_up}・下${r.close_down}</small></td></tr>`;});
  h+=`<details><summary>${s.table_name}の動き別の集計表</summary><table>${t}</table><div class="mut">勝率＝前日より上がった日の割合。回数が5回未満(※)は参考程度に。</div></details>`;
  h+=`<details><summary>この判定のしくみ</summary><div class="mut" style="margin-top:6px">分析した日数: ${s.n}日(${s.start}〜${s.end})<br>使った指標: ${s.used.join("、")}<br>予測のふだんのブレ: 約±${s.rmse.toFixed(1)}%<br>信頼度の元になった検証日数: ${s.n_valid}日`+(s.n_valid<60?"<br>※ 検証日数が少ないため、信頼度は参考程度です。":"")+`<br>判定の目安(予測と気配の差): −${s.th.normal}%以下で買い、−${s.th.strong}%以下で強い買い／+${s.th.normal}%以上で売り、+${s.th.strong}%以上で強い売り`+(s.notes.length?"<br>※ "+s.notes.join("／"):"")+`</div></details>`;
  return h;
}

function init(){
  $("asof").textContent=D.asof+" 更新";
  $("target").textContent="(対象日 "+D.target+")";
  if(D.demo)$("demo").style.display="block";
  if(D.warns.length){const w=$("warn");w.style.display="block";w.innerHTML=D.warns.map(x=>"※ "+x).join("<br>");}
  $("cards").innerHTML=D.stocks.map((s,i)=>`
<section class="stock" id="st${i}">
  <div class="band s0" id="band${i}">
    <div><div class="nm">${s.name}</div><div class="code">${s.code}</div></div>
    <div class="br"><div class="stars" id="stars${i}">☆☆☆☆☆</div><div class="lab" id="lab${i}">--</div><div class="stt" id="stt${i}"></div></div>
  </div>
  <div class="body">
    <div class="nums">
      <div class="n"><span class="k">予測</span><span class="v" id="pred${i}">--</span></div>
      <div class="n"><span class="k">気配 <button class="sg" type="button" data-i="${i}">±</button></span><input id="act${i}" type="text" inputmode="decimal" placeholder="例 -9.4"></div>
      <div class="n"><span class="k" id="dk${i}">差異</span><span class="v" id="diff${i}">--</span></div>
    </div>
    <div class="conf">信頼度 <b>${s.conf}%</b><span class="bar"><i style="width:${s.conf}%"></i></span></div>
    <div class="cm"><p id="c1_${i}"></p><p id="c2_${i}"></p></div>
    <button class="why" type="button" data-i="${i}">なぜこの判定？ ▼</button>
    <div class="dt" id="dt${i}" style="display:none">${detailHTML(s)}</div>
  </div>
</section>`).join("");
  D.stocks.forEach((s,i)=>{
    const inp=$("act"+i);
    if(s.actual!==null&&s.actual!==undefined)inp.value=s.actual;
    else if(saved[s.code]!==undefined)inp.value=saved[s.code];
    inp.addEventListener("input",()=>{saved[s.code]=inp.value;persist();renderAll();});
  });
  document.querySelectorAll(".sg").forEach(b=>b.addEventListener("click",()=>{
    const i=+b.dataset.i,inp=$("act"+i),t=String(inp.value||"").trim();
    inp.value=t.startsWith("-")?t.slice(1):"-"+t;saved[D.stocks[i].code]=inp.value;persist();renderAll();}));
  document.querySelectorAll(".why").forEach(b=>b.addEventListener("click",()=>{
    const i=+b.dataset.i,d=$("dt"+i),o=d.style.display==="none";
    d.style.display=o?"block":"none";b.textContent=o?"閉じる ▲":"なぜこの判定？ ▼";}));
  renderAll();
}
init();
</script></body></html>
"""


def make_html(payload, path):
    text = json.dumps(clean(payload), ensure_ascii=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", text))


# ---------------------------------------------------------------- メイン
def parse_actual(items):
    """--actual 285A=-9.4 8035=-2.0 → {'KIOXIA': -9.4, 'TEL': -2.0}"""
    out = {}
    code2key = {v["code"]: k for k, v in TARGETS.items()}
    for it in items or []:
        try:
            c, v = it.split("=", 1)
            c = c.strip().upper().replace(".T", "")
            key = code2key.get(c) or (c if c in TARGETS else None)
            if key is None:
                raise ValueError
            out[key] = float(v.replace("−", "-"))
        except ValueError:
            sys.exit(f"--actual の書き方が違います: {it}  (例: 285A=-9.4 8035=-2.0)")
    return out


def main():
    ap = argparse.ArgumentParser(description="半導体4銘柄 判定ツール")
    ap.add_argument("--actual", nargs="*", default=[],
                    help="寄り前の気配(前日比%%)。例: 285A=-9.4 8035=-2.0 6857=1.5 6146=-3.0")
    ap.add_argument("--demo", action="store_true", help="架空データで動作確認")
    ap.add_argument("--years", type=int, default=5, help="取得する年数(既定5)")
    ap.add_argument("--out", default="kioxia_judge.html")
    args = ap.parse_args()
    actuals = parse_actual(args.actual)

    data = demo_data() if args.demo else fetch(args.years)

    # 対象日: 日本の15:30(大引け)より前なら今日、後なら次の営業日(土日をとばす)
    now = pd.Timestamp.now(tz=TZ)
    today_n = now.tz_localize(None).normalize()
    start_day = today_n + pd.Timedelta(days=1) if now.hour * 60 + now.minute >= 15 * 60 + 30 else today_n
    target = pd.Timestamp(np.busday_offset(np.datetime64(start_day.date()), 0, roll="forward"))
    wd = "月火水木金土日"[target.weekday()]

    results, warns = [], []
    for key, cfg in TARGETS.items():
        res, notes = analyze(key, cfg, data, target, actuals.get(key))
        if res is None:
            warns += notes
            continue
        results.append(res)
        for n_ in notes:
            if "取れませんでした" not in n_ or True:
                warns.append(f"{cfg['name']}: {n_}")
    if not results:
        sys.exit("どの銘柄も分析できませんでした。ネット接続を確認してください。")

    # 最新データが前営業日分まで入っているか(米国などの休場日なら問題ありません)
    expected = pd.Timestamp(np.busday_offset(np.datetime64(target.date()), -1, roll="backward"))
    stale = []
    for k_, v in INDICATORS.items():
        if k_ in data and data[k_]["Close"].dropna().index[data[k_]["Close"].dropna().index < target].max() < expected:
            last = data[k_]["Close"].dropna().index[data[k_]["Close"].dropna().index < target].max()
            stale.append(f"{v[1]}({last.month}/{last.day}分まで)")
    if stale:
        warns.append("最新の動きがまだ反映されていない指標があります: " + "、".join(stale)
                     + "。休場日の場合は問題ありません。")

    payload = {
        "demo": args.demo, "asof": now.strftime("%m/%d %H:%M"),
        "target": f"{target.month}/{target.day}({wd})", "target_iso": target.strftime("%Y%m%d"),
        "warns": warns, "stocks": results,
    }
    make_html(payload, args.out)
    pd.DataFrame([{"銘柄": r["name"], "コード": r["code"], "予測(%)": round(r["pred"], 2),
                   "信頼度(%)": r["conf"]} for r in results]).to_csv(
        "predictions.csv", index=False, encoding="utf-8-sig")

    print("=" * 46)
    print(f"対象日 {payload['target']}   (更新 {payload['asof']})")
    ranked = []
    for r in results:
        line = f"{r['name']:<8} 予測 {r['pred']:+.1f}%  信頼度 {r['conf']}%  (指標: {'、'.join(r['used'])})"
        a = r["actual"]
        if a is not None:
            diff = a - r["pred"]
            s = stars(diff, r["th"])
            line += f"\n         気配 {a:+.1f}% 差異 {diff:+.1f}% → {'★' * s}{'☆' * (5 - s)} {STAR_NAME[s]}"
            ranked.append((diff, r["name"], s))
        print(line)
        print(f"         予測のふだんのブレ ±{r['rmse']:.1f}% / 分析 {r['n']}日 ({r['start']:%Y/%m/%d}〜{r['end']:%Y/%m/%d})")
    if ranked:
        print("-" * 46)
        print("今日の注目度ランキング:")
        for k, (diff, name, s) in enumerate(sorted(ranked), 1):
            print(f"  {k}位 {name} {'★' * s}{'☆' * (5 - s)} {STAR_NAME[s]}")
    for w in warns:
        print("[注意]", w)
    print(f"\n画面を作りました → {args.out}  (スマホで開いてください)")


if __name__ == "__main__":
    main()
