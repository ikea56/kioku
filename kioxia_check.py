#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
キオクシア(285A) 売られ過ぎ／買われ過ぎ 判定ツール
--------------------------------------------------------------
目的は「未来予測」ではなく、
「海外の動きを受けて、キオクシアがどれだけ織り込んだか(予測とのズレ)」を測ること。

  予測 = 前日の海外指標(MU・SOX・NASDAQ・ドル円など)から計算した、キオクシアの寄付き予想
  実際 = 今日の気配(前日比%)  ※スマホの画面に自分で入力
  差   = 実際 − 予測   → マイナスが大きいほど「売られ過ぎ」

実行:   python kioxia_check.py                  # 実データで分析 → kioxia_judge.html
        python kioxia_check.py --actual -9.5    # 今日の気配(前日比%)も指定
        python kioxia_check.py --demo           # 架空データで動作確認(実際の結果ではありません)
--------------------------------------------------------------
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

TZ = "Asia/Tokyo"
TARGET = "285A.T"

# 内部名: (Yahoo Financeのコード, 画面に出す名前)
INDICATORS = {
    "MU": ("MU", "マイクロン"),
    "SOX": ("^SOX", "SOX指数"),
    "NASDAQ": ("^IXIC", "NASDAQ"),
    "USDJPY": ("JPY=X", "ドル円"),
    "HYNIX": ("000660.KS", "SKハイニックス"),
    "SAMSUNG": ("005930.KS", "サムスン"),
    "NVDA": ("NVDA", "エヌビディア"),
    "AMD": ("AMD", "AMD"),
    "INTC": ("INTC", "インテル"),
}
# 予測に使う指標の組み合わせ候補。過去検証で一番当たったものを自動で選びます。
FEATURE_SETS = {
    "マイクロンだけ": ["MU"],
    "主要4つ": ["MU", "SOX", "NASDAQ", "USDJPY"],
    "全部": list(INDICATORS),
}
ALPHAS = [1.0, 30.0, 300.0]   # 指標が多いとき、偏りを抑える強さの候補

SKIP_FIRST_DAYS = 5     # 上場直後(値付けが不安定)は分析から除外
MAX_STALE_DAYS = 5      # 指標のデータが5日より古い日は使わない
MIN_TRAIN = 120         # 過去検証で最初に学習に使う日数
REFIT_EVERY = 20        # 過去検証で学び直す間隔(営業日)

# 判定のしきい値(予測と実際の差, %)  ← 指示書どおり。変えたいときはここだけ直す
T_STRONG = 4.0
T_NORMAL = 2.0

BINS = [   # (下限以上, 上限未満, 表示名)
    (10, np.inf, "+10%以上"),
    (5, 10, "+5%〜+10%"),
    (2, 5, "+2%〜+5%"),
    (-2, 2, "-2%〜+2%"),
    (-5, -2, "-5%〜-2%"),
    (-10, -5, "-10%〜-5%"),
    (-np.inf, -10, "-10%以下"),
]


def stars(diff):
    """予測と実際の差(%) → ★の数(5=強い買い 〜 1=強い売り)"""
    if diff <= -T_STRONG: return 5
    if diff <= -T_NORMAL: return 4
    if diff < T_NORMAL: return 3
    if diff < T_STRONG: return 2
    return 1


# ---------------------------------------------------------------- データ取得
def fetch(years):
    import yfinance as yf
    start = (pd.Timestamp.now(tz=TZ) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    tickers = {"KIOXIA": TARGET, **{k: v[0] for k, v in INDICATORS.items()}}
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
            if name in ("KIOXIA", "MU"):
                sys.exit(f"{name}({ticker})のデータが取得できませんでした。")
            print(f"[注意] {name}({ticker})は取得できなかったので、使わずに進めます。")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out[name] = df.sort_index()    # 始値・高値・安値・終値・出来高が入っています
    return out


def demo_data():
    """動作確認用の架空データ。実際の相場とは無関係です。"""
    rng = np.random.default_rng(7)
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    days = pd.bdate_range(end=today, periods=520)
    n = len(days)
    semi = rng.normal(0, 0.02, n)   # 半導体全体の動き(共通要因)

    def frame(r, base):
        c = base * np.cumprod(1 + r)
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, n))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
        return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c,
                             "Volume": rng.integers(1e6, 5e6, n)}, index=days)

    betas = {"MU": (1.3, .02), "SOX": (1.0, .007), "NASDAQ": (.5, .007),
             "HYNIX": (1.1, .015), "SAMSUNG": (.7, .012), "NVDA": (1.3, .02),
             "AMD": (1.2, .018), "INTC": (.7, .015)}
    out = {k: frame(b * semi + rng.normal(0, s, n), 100) for k, (b, s) in betas.items()}
    usd = rng.normal(0, 0.005, n) - 0.1 * semi
    out["USDJPY"] = frame(usd, 150)

    noise = rng.normal(0, 0.02, n)
    k_open = np.zeros(n)
    k_open[1:] = 0.9 * semi[:-1] - 0.5 * usd[:-1] + noise[1:]
    k_intra = rng.normal(0, 0.02, n) - 0.2 * noise
    opens, closes, prev = [], [], 3000.0
    for i in range(n):
        o = prev * (1 + k_open[i])
        c = o * (1 + k_intra[i])
        opens.append(o); closes.append(c); prev = c
    kio = pd.DataFrame({"Open": opens, "Close": closes}, index=days)
    kio["High"] = kio[["Open", "Close"]].max(axis=1) * 1.01
    kio["Low"] = kio[["Open", "Close"]].min(axis=1) * 0.99
    kio["Volume"] = 1_000_000
    out["KIOXIA"] = kio
    return out


# ---------------------------------------------------------------- 表の作成
def build(data):
    """日本の各営業日に、その直前の各指標の騰落率を結びつける(朝8:30に分かる情報だけ使う)。"""
    k = data["KIOXIA"][["Open", "Close"]].copy()
    k["prev_close"] = k["Close"].shift(1)
    k["k_open"] = (k["Open"] / k["prev_close"] - 1) * 100     # 寄付きの前日比
    k["k_close"] = (k["Close"] / k["prev_close"] - 1) * 100   # 終値の前日比
    k["k_intra"] = (k["Close"] / k["Open"] - 1) * 100         # 寄付き→終値
    k.index.name = "date"
    k = k.iloc[SKIP_FIRST_DAYS:].dropna().reset_index()
    k["date"] = pd.to_datetime(k["date"])

    for name in INDICATORS:
        if name not in data:
            continue
        c = data[name]["Close"].dropna()
        r = pd.DataFrame({f"{name}_date": pd.to_datetime(c.index),
                          name: c.pct_change().values * 100}).dropna()
        k = pd.merge_asof(k.sort_values("date"), r.sort_values(f"{name}_date"),
                          left_on="date", right_on=f"{name}_date",
                          direction="backward", allow_exact_matches=False)
        stale = (k["date"] - k[f"{name}_date"]).dt.days > MAX_STALE_DAYS
        k.loc[stale, name] = np.nan
    return k.dropna(subset=["MU"]).reset_index(drop=True)


def bin_table(df):
    rows = []
    for lo, hi, label in BINS:
        s = df[(df["MU"] >= lo) & (df["MU"] < hi)]
        n = len(s)
        row = {"label": label, "n": n}
        for key, col in (("open", "k_open"), ("close", "k_close")):
            up, down = int((s[col] > 0).sum()), int((s[col] < 0).sum())
            row[f"{key}_avg"] = s[col].mean() if n else None
            row[f"{key}_max"] = s[col].max() if n else None
            row[f"{key}_min"] = s[col].min() if n else None
            row[f"{key}_up"], row[f"{key}_down"] = up, down
            row[f"{key}_win"] = round(up / (up + down) * 100) if up + down else None
        iu, idn = int((s["k_intra"] > 0).sum()), int((s["k_intra"] < 0).sum())
        row["rebound"] = iu / (iu + idn) if iu + idn else None   # 寄付き後に上がった割合
        rows.append(row)
    return rows


def influence(df, names):
    """各指標が、キオクシアの寄付きとどれくらい同じように動くか(0〜100%)"""
    res = []
    for n in names:
        s = df[[n, "k_open"]].dropna()
        if len(s) < 30:
            continue
        c = float(s[n].corr(s["k_open"]))
        res.append({"key": n, "name": INDICATORS[n][1], "score": round(abs(c) * 100),
                    "dir": "same" if c >= 0 else "opposite"})
    return sorted(res, key=lambda r: -r["score"])


# ---------------------------------------------------------------- 予測モデル
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


def choose_model(base, names):
    y = base["k_open"].astype(float)
    results, seen = [], set()
    for label, cols in FEATURE_SETS.items():
        cols = tuple(c for c in cols if c in names)
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
        sys.exit("過去検証に使えるデータが足りません。")
    best = min(results, key=lambda r: r["rmse"])
    mu_only = next((r for r in results if r["cols"] == ["MU"]), best)
    a, b = ridge_fit(base[best["cols"]].values.astype(float), y.values, best["alpha"])
    best = {**best, "a": a, "coef": dict(zip(best["cols"], b))}
    return best, mu_only


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
            label = f"上場来(約{span:.1f}年分)"
        out.append({"label": label,
                    "period": f"{g['date'].min():%Y/%m/%d}〜{g['date'].max():%Y/%m/%d}",
                    "buy": bt_stats(g.loc[g["star"] >= 4, "k_intra"]),
                    "strong": bt_stats(g.loc[g["star"] == 5, "k_intra"])})
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


# ---------------------------------------------------------------- 画面(HTML)
HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>キオクシア判定</title>
<style>
:root{--bg:#f4f5f7;--ink:#1c1f24;--mut:#777}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;-webkit-text-size-adjust:100%}
.wrap{max-width:480px;margin:0 auto;padding:14px 14px 40px}
.top{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;color:#555;margin-bottom:8px}
.demo{background:#fff3cd;border:1px solid #e0c36a;border-radius:10px;padding:8px 10px;font-size:13px;margin-bottom:10px}
.card{border-radius:20px;padding:22px 16px;text-align:center;color:#fff;transition:background .2s}
.s0{background:#9aa0a8}.s5{background:#0b6e3a}.s4{background:#2e9e55}.s3{background:#8a8f98}.s2{background:#ee8a1c}.s1{background:#d73027}
.stars{font-size:44px;letter-spacing:4px;line-height:1.1}
.label{font-size:40px;font-weight:800;margin-top:6px}
.head{font-size:17px;margin-top:8px;font-weight:600}
.box{background:#fff;border-radius:16px;margin-top:12px;padding:6px 16px}
.row{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid #eee}
.row:last-child{border-bottom:0}
.row .k{font-size:16px;color:#555}
.row .v{font-size:30px;font-weight:800}
.inp{display:flex;gap:6px;align-items:center}
.inp input{width:110px;font-size:28px;font-weight:800;text-align:right;border:2px solid #ccd;border-radius:10px;padding:4px 8px;background:#fafbff}
.inp button{font-size:20px;font-weight:800;border:2px solid #ccd;border-radius:10px;background:#fff;padding:6px 10px}
.sub{font-size:12px;color:var(--mut);display:block}
.cm{font-size:16px;line-height:1.65;padding:10px 0}
.cm b{display:block;font-size:13px;color:var(--mut);margin-bottom:4px}
.cm p{margin:0 0 6px}
.warn{background:#fff3cd;border-radius:12px;padding:10px 12px;font-size:14px;margin-top:12px}
.more{width:100%;margin-top:14px;font-size:18px;font-weight:700;padding:14px;border:0;border-radius:14px;background:#1c1f24;color:#fff}
h3{font-size:16px;margin:18px 0 6px}
.mut{font-size:12px;color:var(--mut)}
.list .it{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee;font-size:15px}
.rank{padding:7px 0;border-bottom:1px solid #eee;font-size:15px}
.rank .bar{height:8px;background:#e6e8ec;border-radius:4px;margin-top:4px}
.rank .bar i{display:block;height:8px;border-radius:4px;background:#4a6cf7}
.tiles{display:flex;gap:4px;margin:6px 0}
.tile{flex:1;border-radius:8px;padding:10px 0;text-align:center;font-size:11px;color:#fff;line-height:1.3}
.tile small{display:block;font-size:10px;opacity:.85}
table{width:100%;border-collapse:collapse;margin-top:6px;font-size:13px}
th,td{padding:6px 3px;border-bottom:1px solid #eee;text-align:center;vertical-align:top}
th{color:#666;font-weight:600}
td small{display:block;color:#777;font-size:11px}
details{margin-top:10px}summary{font-size:15px;font-weight:700;cursor:pointer}
.foot{font-size:12px;color:#888;margin-top:18px;line-height:1.6}
</style></head><body><div class="wrap">
<div class="top"><span>キオクシア判定</span><span id="asof"></span></div>
<div id="demo" class="demo" style="display:none">⚠ これは動作確認用の<b>架空データ</b>です。実際の相場の結果ではありません。</div>

<div id="card" class="card s0">
  <div class="stars" id="stars">☆☆☆☆☆</div>
  <div class="label" id="label">--</div>
  <div class="head" id="head"></div>
</div>

<div class="box">
  <div class="row"><span class="k">予測<span class="sub">海外の動きから見た、今日の予想</span></span><span class="v" id="pred">--</span></div>
  <div class="row"><span class="k">実際<span class="sub">気配の前日比(%)を入力</span></span>
    <span class="inp"><button id="sign" type="button">±</button><input id="act" type="text" inputmode="decimal" placeholder="例 -9.5"></span></div>
  <div class="row"><span class="k" id="difflabel">売られ過ぎ度</span><span class="v" id="diff">--</span></div>
</div>

<div class="box"><div class="cm"><b>コメント</b><p id="c1"></p><p id="c2"></p><p id="c3"></p></div></div>
<div id="warn" class="warn" style="display:none"></div>

<button id="more" class="more" type="button">くわしく見る ▼</button>
<div id="detail" style="display:none">
  <h3>各指標の前日の動き</h3><div class="list" id="ind"></div>
  <h3>キオクシアへの影響度ランキング</h3><div id="rank"></div>
  <div class="mut">同じ向きに動きやすい度合いです。100%に近いほど連動します。</div>
  <h3>ヒートマップ</h3>
  <div class="mut">マイクロンの前日の動き別：キオクシアは寄付き後どうなりやすい？</div><div class="tiles" id="heat1"></div>
  <div class="mut">今回の判定(★)別：寄付き後どうなりやすい？</div><div class="tiles" id="heat2"></div>
  <div class="mut">緑＝寄付き後に上がりやすい(反発傾向)／赤＝下がり続けやすい／灰＝はっきりしない／薄い灰＝回数が少なく判断できない</div>
  <h3>バックテスト(過去の検証)</h3><div id="bt"></div>
  <details><summary>マイクロンの動き別の集計表</summary><table id="tbl"></table>
    <div class="mut">勝率＝前日より上がった日の割合。回数が5回未満(※)は参考程度に。</div></details>
  <details><summary>この判定のしくみ</summary><div class="mut" id="basis" style="font-size:13px;line-height:1.7;margin-top:6px"></div></details>
</div>

<div class="foot">※ 過去の傾向から計算した目安で、将来の値動きを保証するものではありません。売買の最終判断はご自身で行ってください。</div>
</div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const fmt=(x,d)=>x===null||x===undefined||isNaN(x)?"-":(x>0?"+":"")+x.toFixed(d===undefined?1:d)+"%";
const starsFor=f=>f<=-D.th.strong?5:f<=-D.th.normal?4:f<D.th.normal?3:f<D.th.strong?2:1;
const LABEL={5:"強い買い",4:"買い",3:"中立",2:"売り",1:"強い売り"};
const HEAD={5:"かなり売られ過ぎ。買い候補です",4:"売られ過ぎ。買い候補です",3:"想定どおり。様子見でOK",2:"買われ過ぎ。売り寄りです",1:"かなり買われ過ぎ。売り注意"};
const heat=(share,n)=>(n<5||share===null||share===undefined)?"#d9dce1":share>=0.6?"#2e9e55":share<=0.4?"#d73027":"#9aa0a8";

function render(){
  const raw=String($("act").value||"").replace("−","-").replace("－","-").replace(",",".").trim();
  const act=parseFloat(raw);
  const card=$("card");
  $("pred").textContent=fmt(D.pred);
  if(isNaN(act)){
    card.className="card s0";$("stars").textContent="☆☆☆☆☆";$("label").textContent="気配を入力";
    $("head").textContent="「実際」に今日のキオクシアの前日比(%)を入れてください";
    $("diff").textContent="--";$("difflabel").textContent="売られ過ぎ度";
    $("c1").textContent="海外の動きから、キオクシアの予想される動きを計算しました。";
    $("c2").textContent="実際の気配を入れると、売られ過ぎ/買われ過ぎを判定します。";
    $("c3").textContent=D.reason;return;
  }
  const diff=act-D.pred,s=starsFor(diff);
  card.className="card s"+s;
  $("stars").textContent="★".repeat(s)+"☆".repeat(5-s);
  $("label").textContent=LABEL[s];$("head").textContent=HEAD[s];
  $("diff").textContent=fmt(diff);
  $("difflabel").textContent=diff<0?"売られ過ぎ度":"買われ過ぎ度";
  let c1,c2;
  if(s>=4)c1="海外の動きに対して、今日のキオクシアは通常より大きく売られています。";
  else if(s<=2)c1="海外の動きに対して、今日のキオクシアは通常より大きく買われています。";
  else c1="海外の動きから見て、ほぼ想定どおりの動きです。";
  const st=D.by_star[String(s)];
  if(s===3||!st||st.n<5)c2=(s===3?"特別な偏りはないので、無理に動かなくて大丈夫です。":"過去に同じ判定の日が少ないため、参考程度にしてください。");
  else if(s>=4){
    const r=st.up/st.n;
    c2=`過去の同じ判定の日は${st.n}回中${st.up}回、寄付きより終値が高くなりました。`+(r>=0.6?"反発しやすい傾向です。":r<=0.4?"反発は期待しにくいです。":"はっきりした傾向はありません。");
  }else{
    const r=st.down/st.n;
    c2=`過去の同じ判定の日は${st.n}回中${st.down}回、寄付きより終値が安くなりました。`+(r>=0.6?"下がりやすい傾向です。":r<=0.4?"下がりにくい傾向です。":"はっきりした傾向はありません。");
  }
  $("c1").textContent=c1;$("c2").textContent=c2;$("c3").textContent=D.reason;
}

function btTable(w){
  const g=(o,f)=>o?f(o):"-";
  const col=o=>[g(o,x=>x.n+"回"),g(o,x=>x.win+"%"),g(o,x=>fmt(x.avg_win)),g(o,x=>fmt(x.avg_loss)),g(o,x=>fmt(x.worst)),g(o,x=>fmt(x.expect))];
  const a=col(w.buy),b=col(w.strong),names=["回数","勝率","平均利益","平均損失","最大損失","1回あたり平均(期待値)"];
  let h=`<div style="margin-top:8px"><b>${w.label}</b> <span class="mut">${w.period}</span></div><table><tr><th></th><th>買い以上(★4〜5)</th><th>強い買いだけ(★5)</th></tr>`;
  names.forEach((n,i)=>{h+=`<tr><td>${n}</td><td>${a[i]}</td><td>${b[i]}</td></tr>`;});
  return h+"</table>";
}

function init(){
  $("asof").textContent=D.asof+" 更新";
  if(D.demo)$("demo").style.display="block";
  if(D.actual!==null&&D.actual!==undefined)$("act").value=D.actual;
  if(D.rel.r2<0.1){const w=$("warn");w.style.display="block";w.textContent="※ 海外の動きとの連動が弱いため、判定は参考程度にしてください。";}
  if(D.missing&&D.missing.length){const w=$("warn");w.style.display="block";w.textContent="※ "+D.missing.join("・")+"の最新データが取れなかったため、動きなしとして計算しています。";}
  $("ind").innerHTML=D.today.map(t=>`<div class="it"><span>${t.name}<span class="mut"> ${t.date}</span></span><b>${t.pct>0?"▲":t.pct<0?"▼":"－"} ${fmt(t.pct)}</b></div>`).join("");
  $("rank").innerHTML=D.influence.map((r,i)=>`<div class="rank">${i+1}位 ${r.name}　影響度 ${r.score}%<span class="mut"> ${r.dir==="same"?"(同じ向き)":"(逆向き)"}</span><div class="bar"><i style="width:${r.score}%"></i></div></div>`).join("");
  const short=["+10↑","+5〜10","+2〜5","±2","-2〜-5","-5〜-10","-10↓"];
  $("heat1").innerHTML=D.table.map((r,i)=>`<div class="tile" style="background:${heat(r.rebound,r.n)};${heat(r.rebound,r.n)==="#d9dce1"?"color:#666":""}">${short[i]}<small>${r.n}回</small></div>`).join("");
  $("heat2").innerHTML=[5,4,3,2,1].map(s=>{const t=D.by_star[String(s)];const sh=t&&(t.up+t.down)?t.up/(t.up+t.down):null;const c=heat(sh,t?t.n:0);return `<div class="tile" style="background:${c};${c==="#d9dce1"?"color:#666":""}">★${s}<small>${t?t.n:0}回</small></div>`;}).join("");
  $("bt").innerHTML=D.backtest.map(btTable).join("")+`<div class="mut" style="margin-top:6px">「買い」の日に寄付きで買い、その日の終値で売っていたら、という計算です(手数料なし)。判定は、その日より前のデータだけで作った予測で行っています。</div>`;
  let h="<tr><th>マイクロン</th><th>回数</th><th>寄付き平均</th><th>終値平均</th><th>勝率</th></tr>";
  D.table.forEach(r=>{h+=`<tr><td>${r.label}</td><td>${r.n}${r.n<5?"※":""}</td><td>${r.n?fmt(r.open_avg):"-"}</td><td>${r.n?fmt(r.close_avg):"-"}<small>最大${fmt(r.close_max)}／${fmt(r.close_min)}</small></td><td>${r.close_win===null?"-":r.close_win+"%"}<small>上${r.close_up}・下${r.close_down}</small></td></tr>`;});
  $("tbl").innerHTML=h;
  $("basis").innerHTML=`分析した日数: ${D.n}日(${D.start}〜${D.end})<br>使った指標: ${D.rel.used.join("、")}<br>`+
    `予測の当たりやすさ: ${D.rel.text}(過去を1日ずつ検証した結果、約${Math.round(Math.max(D.rel.r2,0)*100)}%を説明。マイクロンだけだと約${Math.round(Math.max(D.rel.mu_r2,0)*100)}%)<br>`+
    `予測のふだんのブレ: 約±${D.rel.rmse.toFixed(1)}%<br>`+
    `判定の目安(予測と実際の差): −${D.th.normal}%以下で買い、−${D.th.strong}%以下で強い買い／+${D.th.normal}%以上で売り、+${D.th.strong}%以上で強い売り`;
  $("more").onclick=()=>{const d=$("detail");const o=d.style.display==="none";d.style.display=o?"block":"none";$("more").textContent=o?"閉じる ▲":"くわしく見る ▼";};
  $("sign").onclick=()=>{const t=String($("act").value||"").trim();$("act").value=t.startsWith("-")?t.slice(1):"-"+t;render();};
  $("act").addEventListener("input",render);
  render();
}
init();
</script></body></html>
"""


def make_html(payload, path):
    text = json.dumps(clean(payload), ensure_ascii=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", text))


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser(description="キオクシア判定ツール")
    ap.add_argument("--actual", type=float, default=None,
                    help="今日のキオクシアの気配/現在値の前日比(%%)。例: -9.5")
    ap.add_argument("--demo", action="store_true", help="架空データで動作確認")
    ap.add_argument("--years", type=int, default=5, help="取得する年数(既定5)")
    ap.add_argument("--out", default="kioxia_judge.html")
    args = ap.parse_args()

    data = demo_data() if args.demo else fetch(args.years)
    df = build(data)
    if len(df) < 60:
        sys.exit(f"分析できる日数が少なすぎます({len(df)}日)。")

    names = [n for n in INDICATORS if n in df.columns and df[n].notna().mean() > 0.9]
    base = df.dropna(subset=names).reset_index(drop=True)

    best, mu_only = choose_model(base, names)
    d = base.assign(pred=best["pred"]).dropna(subset=["pred"]).copy()
    d["diff"] = d["k_open"] - d["pred"]
    d["star"] = d["diff"].apply(stars)

    by_star = {}
    for s in (1, 2, 3, 4, 5):
        g = d[d["star"] == s]
        by_star[s] = {"n": len(g), "up": int((g["k_intra"] > 0).sum()),
                      "down": int((g["k_intra"] < 0).sum()),
                      "avg": g["k_intra"].mean() if len(g) else None}

    # 今日の入力値: 「今日(日本時間)より前」の最新の1日分の騰落率
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    latest, missing = {}, []
    for n in names:
        c = data[n]["Close"].dropna()
        c = c[c.index < today]
        if len(c) >= 2 and (today - c.index[-1]).days <= MAX_STALE_DAYS + 2:
            latest[n] = {"pct": float((c.iloc[-1] / c.iloc[-2] - 1) * 100), "date": c.index[-1]}
    if "MU" not in latest:
        sys.exit("マイクロンの最新データがありません。")
    for n in best["cols"]:
        if n not in latest:
            missing.append(INDICATORS[n][1])
    x = {n: (latest[n]["pct"] if n in latest else 0.0) for n in best["cols"]}
    effects = {n: best["coef"][n] * x[n] for n in best["cols"]}
    pred_today = best["a"] + sum(effects.values())

    top = max(effects, key=lambda n: abs(effects[n]))
    reason = (f"今日の予測は、{INDICATORS[top][1]}の前日の動き({x[top]:+.1f}%)が一番大きく効いています。"
              if abs(effects[top]) > 0.05 else "今日は海外の動きが小さく、予測もほぼ動きなしです。")

    r2 = best["r2"]
    rel_text = "高め" if r2 >= 0.3 else "ふつう" if r2 >= 0.1 else "低め"

    table = bin_table(df)
    pd.DataFrame(table).to_csv("kioxia_table.csv", index=False, encoding="utf-8-sig")
    bt = backtest(d)

    payload = {
        "demo": args.demo, "actual": args.actual,
        "asof": pd.Timestamp.now(tz=TZ).strftime("%m/%d %H:%M"),
        "start": df["date"].min(), "end": df["date"].max(), "n": len(df),
        "pred": pred_today, "reason": reason, "missing": missing,
        "th": {"strong": T_STRONG, "normal": T_NORMAL},
        "today": [{"name": INDICATORS[n][1], "pct": latest[n]["pct"],
                   "date": f"{latest[n]['date'].month}/{latest[n]['date'].day}"}
                  for n in names if n in latest],
        "influence": influence(df, names),
        "table": table, "by_star": by_star, "backtest": bt,
        "rel": {"r2": r2, "mu_r2": mu_only["r2"], "rmse": best["rmse"], "text": rel_text,
                "used": [INDICATORS[n][1] for n in best["cols"]]},
    }
    make_html(payload, args.out)

    print("=" * 44)
    print(f"分析日数 {len(df)}日 ({df['date'].min():%Y/%m/%d}〜{df['date'].max():%Y/%m/%d})")
    print(f"採用した予測方式: {best['label']}  使った指標: {'、'.join(payload['rel']['used'])}")
    print(f"過去検証での当たりやすさ: {r2 * 100:.0f}% (マイクロンだけだと {mu_only['r2'] * 100:.0f}%) / ふだんのブレ ±{best['rmse']:.1f}%")
    print(f"今日の予測(キオクシアの寄付き): {pred_today:+.1f}%   {reason}")
    if args.actual is not None:
        diff = args.actual - pred_today
        s = stars(diff)
        name = {5: "強い買い", 4: "買い", 3: "中立", 2: "売り", 1: "強い売り"}[s]
        print(f"実際 {args.actual:+.1f}% / 差 {diff:+.1f}% → {'★' * s}{'☆' * (5 - s)} {name}")
    print("-" * 44)
    print("影響度ランキング: " + " > ".join(f"{r['name']}{r['score']}%" for r in payload["influence"]))
    for w in bt:
        b = w["buy"]
        if b:
            print(f"バックテスト[{w['label']}] 買い判定 {b['n']}回 勝率{b['win']}% 1回あたり平均{b['expect']:+.2f}%")
    print("-" * 44)
    print(pd.DataFrame(table)[["label", "n", "open_avg", "close_avg", "close_win", "close_max", "close_min"]]
          .round(2).to_string(index=False))
    print(f"\n画面を作りました → {args.out}  (スマホで開いてください)")


if __name__ == "__main__":
    main()
