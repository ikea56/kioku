#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
キオクシア判定ツール
--------------------------------------------------------------
マイクロン(MU)の前日の動きから、次の日本の取引日のキオクシア(285A)の
動きを予測し、「今日は売られ過ぎか？買いか？」をスマホ用HTMLで表示します。

準備:   pip install yfinance pandas numpy
実行:   python kioxia_check.py                  # 実データで分析
        python kioxia_check.py --actual -8.0    # 今日のキオクシア気配(前日比%)も指定
        python kioxia_check.py --demo           # 架空データで動作確認(実際の結果ではありません)

出力:   kioxia_judge.html  (スマホで開く画面)
        kioxia_table.csv   (集計表)
--------------------------------------------------------------
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

TICKERS = {"MU": "MU", "KIOXIA": "285A.T", "SOX": "^SOX",
           "NASDAQ": "^IXIC", "USDJPY": "JPY=X"}
SKIP_FIRST_DAYS = 5   # 上場直後(値付けが不安定)の日数は分析から除外
MAX_STALE_DAYS = 5    # MUのデータが5日より古い日は使わない

# (下限以上, 上限未満, 表示名)
BINS = [
    (10, np.inf, "+10%以上"),
    (5, 10, "+5%〜+10%"),
    (2, 5, "+2%〜+5%"),
    (-2, 2, "-2%〜+2%"),
    (-5, -2, "-5%〜-2%"),
    (-10, -5, "-10%〜-5%"),
    (-np.inf, -10, "-10%以下"),
]


# ---------------------------------------------------------------- データ取得
def fetch(years=3):
    import yfinance as yf
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    out = {}
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df is None or df.empty:
            if name in ("MU", "KIOXIA"):
                sys.exit(f"{name}({ticker})のデータが取得できませんでした。ネット接続を確認してください。")
            print(f"[注意] {name}({ticker})は取得できなかったので、参考分析から外します。")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        out[name] = df.sort_index()
    return out


def demo_data():
    """動作確認用の架空データ。実際の相場とは無関係です。"""
    rng = np.random.default_rng(42)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=470)
    n = len(days)
    mu_r = rng.normal(0.002, 0.035, n)
    sox_r = 0.6 * mu_r + rng.normal(0, 0.012, n)
    nas_r = 0.4 * mu_r + rng.normal(0, 0.008, n)
    fx_r = rng.normal(0, 0.005, n)

    def frame(r, base):
        c = base * np.cumprod(1 + r)
        return pd.DataFrame({"Open": c, "Close": c}, index=days)

    k_open_r = np.zeros(n)
    k_open_r[1:] = 0.45 * mu_r[:-1] + rng.normal(0, 0.025, n - 1)
    pred_part = np.r_[0, 0.45 * mu_r[:-1]]
    k_intra = rng.normal(0, 0.02, n) - 0.15 * (k_open_r - pred_part)
    opens, closes, prev = [], [], 3000.0
    for i in range(n):
        o = prev * (1 + k_open_r[i])
        c = o * (1 + k_intra[i])
        opens.append(o)
        closes.append(c)
        prev = c
    return {
        "MU": frame(mu_r, 100), "SOX": frame(sox_r, 5000),
        "NASDAQ": frame(nas_r, 15000), "USDJPY": frame(fx_r, 150),
        "KIOXIA": pd.DataFrame({"Open": opens, "Close": closes}, index=days),
    }


# ---------------------------------------------------------------- 表の作成
def build(data):
    """日本の各営業日に、その直前の米国取引日の騰落率を結びつける。"""
    k = data["KIOXIA"][["Open", "Close"]].copy()
    k["prev_close"] = k["Close"].shift(1)
    k["k_open"] = (k["Open"] / k["prev_close"] - 1) * 100     # 寄付きの前日比
    k["k_close"] = (k["Close"] / k["prev_close"] - 1) * 100   # 終値の前日比
    k["k_intra"] = (k["Close"] / k["Open"] - 1) * 100         # 寄付き→終値
    k.index.name = "date"
    k = k.iloc[SKIP_FIRST_DAYS:].dropna().reset_index()
    k["date"] = pd.to_datetime(k["date"])

    for name in ("MU", "SOX", "NASDAQ", "USDJPY"):
        if name not in data:
            continue
        d = data[name]
        r = pd.DataFrame({f"{name}_date": pd.to_datetime(d.index),
                          name: d["Close"].pct_change().values * 100}).dropna()
        k = pd.merge_asof(k.sort_values("date"), r.sort_values(f"{name}_date"),
                          left_on="date", right_on=f"{name}_date",
                          direction="backward", allow_exact_matches=False)
    stale = (k["date"] - k["MU_date"]).dt.days > MAX_STALE_DAYS
    return k[~stale].dropna(subset=["MU"]).reset_index(drop=True)


def pct(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def bin_table(df):
    rows = []
    for lo, hi, label in BINS:
        s = df[(df["MU"] >= lo) & (df["MU"] < hi)]
        n = len(s)
        row = {"label": label, "n": n}
        for key, col in (("open", "k_open"), ("close", "k_close")):
            up = int((s[col] > 0).sum())
            down = int((s[col] < 0).sum())
            row[f"{key}_avg"] = pct(s[col].mean()) if n else None
            row[f"{key}_up"] = up
            row[f"{key}_down"] = down
            row[f"{key}_win"] = round(up / (up + down) * 100) if (up + down) else None
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 予測モデル
def fit(df):
    x, y = df["MU"].values, df["k_open"].values
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    sd = resid.std(ddof=2)
    r2 = 1 - resid.var() / y.var()

    # 参考: SOX・NASDAQ・ドル円も加えたらどれくらい当たりやすくなるか
    cols = [c for c in ("MU", "SOX", "NASDAQ", "USDJPY") if c in df.columns]
    sub = df.dropna(subset=cols)
    X = np.column_stack([np.ones(len(sub))] + [sub[c] for c in cols])
    yy = sub["k_open"].values
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r2_multi = 1 - (yy - X @ coef).var() / yy.var()
    corr = {c: round(float(sub[c].corr(sub["k_open"])), 2) for c in cols}

    z = resid / sd
    d = df.assign(z=z)

    def group(mask, up_if_positive=True):
        g = d[mask]
        n = len(g)
        return {"n": n,
                "up": int((g["k_intra"] > 0).sum()),
                "down": int((g["k_intra"] < 0).sum()),
                "avg": pct(g["k_intra"].mean()) if n else None}

    return {
        "a": float(a), "b": float(b), "sd": float(sd),
        "r2": float(r2), "r2_multi": float(r2_multi), "corr": corr,
        "low": group(d["z"] <= -0.75),    # 予測より下がり過ぎた日
        "high": group(d["z"] >= 0.75),    # 予測より上がり過ぎた日
    }


def stars(z):
    if z <= -1.5: return 5
    if z <= -0.75: return 4
    if z < 0.75: return 3
    if z < 1.5: return 2
    return 1


# ---------------------------------------------------------------- 画面(HTML)
HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>キオクシア判定</title>
<style>
:root{--buy:#1a9850;--hold:#8a8f98;--sell:#d73027;--bg:#f4f5f7;--ink:#1c1f24}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;-webkit-text-size-adjust:100%}
.wrap{max-width:480px;margin:0 auto;padding:14px 14px 40px}
.top{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;color:#555;margin-bottom:8px}
.demo{background:#fff3cd;border:1px solid #e0c36a;border-radius:10px;padding:8px 10px;font-size:13px;margin-bottom:10px}
.card{border-radius:20px;padding:22px 16px;text-align:center;color:#fff;transition:background .2s}
.card.buy{background:var(--buy)}.card.hold{background:var(--hold)}.card.sell{background:var(--sell)}
.stars{font-size:44px;letter-spacing:4px;line-height:1.1}
.label{font-size:40px;font-weight:800;margin-top:6px}
.head{font-size:18px;margin-top:8px;font-weight:600}
.box{background:#fff;border-radius:16px;margin-top:12px;padding:6px 16px}
.row{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid #eee}
.row:last-child{border-bottom:0}
.row .k{font-size:16px;color:#555}
.row .v{font-size:30px;font-weight:800}
.row input{width:130px;font-size:28px;font-weight:800;text-align:right;border:2px solid #ccd;border-radius:10px;padding:4px 8px;background:#fafbff}
.row .sub{font-size:12px;color:#777;display:block}
.cm{font-size:17px;line-height:1.6}
.cm b{display:block;font-size:13px;color:#777;margin-bottom:4px}
.warn{background:#fff3cd;border-radius:12px;padding:10px 12px;font-size:14px;margin-top:12px}
details{background:#fff;border-radius:16px;margin-top:12px;padding:12px 16px}
summary{font-size:16px;font-weight:700;cursor:pointer}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
th,td{padding:7px 4px;border-bottom:1px solid #eee;text-align:center;vertical-align:top}
th{color:#666;font-weight:600}
td small{display:block;color:#777;font-size:11px}
.note{font-size:12px;color:#777;line-height:1.6;margin-top:8px}
.foot{font-size:12px;color:#888;margin-top:16px;line-height:1.6}
</style></head><body><div class="wrap">
<div class="top"><span>キオクシア判定</span><span id="asof"></span></div>
<div id="demo" class="demo" style="display:none">⚠ これは動作確認用の<b>架空データ</b>です。実際の相場の結果ではありません。</div>

<div id="card" class="card hold">
  <div class="stars" id="stars">☆☆☆☆☆</div>
  <div class="label" id="label">--</div>
  <div class="head" id="head"></div>
</div>

<div class="box">
  <div class="row"><span class="k">マイクロン前日<span class="sub" id="mudate"></span></span>
    <span><input id="mu" type="number" step="0.1" inputmode="decimal"></span></div>
  <div class="row"><span class="k">予測</span><span class="v" id="pred">--</span></div>
  <div class="row"><span class="k">実際<span class="sub">気配/現在値の前日比(%)</span></span>
    <span><input id="act" type="number" step="0.1" inputmode="decimal" placeholder="例 -8.0"></span></div>
  <div class="row"><span class="k">差<span class="sub">実際 − 予測</span></span><span class="v" id="diff">--</span></div>
</div>

<div class="box"><div class="cm"><b>コメント</b><span id="c1"></span><br><span id="c2"></span></div></div>
<div id="warn" class="warn" style="display:none"></div>

<details><summary>くわしい根拠(タップで開く)</summary>
  <div class="note" id="basis"></div>
  <table id="tbl"></table>
  <div class="note">勝率＝キオクシアが前日より上がった日の割合。回数が5回未満の行(※)は偶然の影響が大きいので参考程度に。</div>
</details>

<div class="foot">※ 過去の傾向から計算した目安で、将来の値動きを保証するものではありません。売買の最終判断はご自身で行ってください。</div>
</div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const fmt=x=>(x>0?"+":"")+x.toFixed(1)+"%";
const starsFor=z=>z<=-1.5?5:z<=-0.75?4:z<0.75?3:z<1.5?2:1;
const LABEL={5:"強い買い",4:"買い",3:"中立",2:"売り",1:"強い売り"};
const HEAD={5:"かなり売られ過ぎ。買い候補です",4:"売られ過ぎ。買い候補です",3:"予測どおり。様子見でOK",2:"買われ過ぎ。売り寄りです",1:"かなり買われ過ぎ。売り注意"};
const COL={5:"buy",4:"buy",3:"hold",2:"sell",1:"sell"};

function render(){
  const mu=parseFloat($("mu").value), act=parseFloat($("act").value);
  if(isNaN(mu))return;
  const pred=D.a+D.b*mu;
  $("pred").textContent=fmt(pred);
  const card=$("card");
  if(isNaN(act)){
    card.className="card hold";$("stars").textContent="☆☆☆☆☆";$("label").textContent="気配を入力";
    $("head").textContent="「実際」に今日のキオクシアの前日比(%)を入れてください";
    $("diff").textContent="--";$("c1").textContent="マイクロンの動きから、キオクシアの予想される動きを計算しました。";
    $("c2").textContent="実際の気配を入れると、売られ過ぎ/買われ過ぎを判定します。";return;
  }
  const diff=act-pred, z=diff/D.sd, s=starsFor(z);
  card.className="card "+COL[s];
  $("stars").textContent="★".repeat(s)+"☆".repeat(5-s);
  $("label").textContent=LABEL[s];$("head").textContent=HEAD[s];
  $("diff").textContent=fmt(diff);
  let c1,c2,g=null,low=z<=-0.75,high=z>=0.75;
  if(low){c1="マイクロンの動きに対して、キオクシアは通常より売られています。";g=D.low;}
  else if(high){c1="マイクロンの動きに対して、キオクシアは通常より買われています。";g=D.high;}
  else c1="マイクロンの動きに対して、ほぼ想定どおりの動きです。";
  if(!g)c2="特別な偏りはないので、無理に動かなくて大丈夫です。";
  else if(g.n<5)c2="過去に似た日が少ないため、参考程度にしてください。";
  else{
    const cnt=low?g.up:g.down, rate=cnt/g.n;
    if(low)c2=`過去の似た日は${g.n}回中${cnt}回、その日のうちに寄付きより上がりました。`+(rate>=0.6?"反発しやすい傾向です。":rate<=0.4?"反発は期待しにくいです。":"はっきりした傾向はありません。");
    else c2=`過去の似た日は${g.n}回中${cnt}回、その日のうちに寄付きより下がりました。`+(rate>=0.6?"上がった後は下がりやすい傾向です。":rate<=0.4?"下がりにくい傾向です。":"はっきりした傾向はありません。");
  }
  $("c1").textContent=c1;$("c2").textContent=c2;
}

function cell(avg,up,down,win){
  if(avg===null)return "-";
  return fmt(avg)+`<small>上${up}・下${down}・勝率${win===null?"-":win+"%"}</small>`;
}
function init(){
  $("asof").textContent=D.asof+" 時点";
  if(D.demo)$("demo").style.display="block";
  $("mu").value=D.mu_today.toFixed(1);
  $("mudate").textContent="米国 "+D.mu_date+" の終値";
  if(D.actual!==null)$("act").value=D.actual;
  const rel=D.r2>=0.3?"高め":D.r2>=0.1?"ふつう":"低め";
  $("basis").innerHTML=`分析した日数: ${D.n}日(${D.start}〜${D.end})<br>`+
    `マイクロンの動きだけで、キオクシアの寄付きの動きの約${Math.round(D.r2*100)}%を説明できます(予測の当たりやすさ: ${rel})。<br>`+
    `SOX・NASDAQ・ドル円も加えると約${Math.round(D.r2_multi*100)}%です。<br>`+
    `マイクロンが1%動くと、キオクシアは平均${D.b.toFixed(2)}%同じ向きに動きます。`;
  if(D.r2<0.1){const w=$("warn");w.style.display="block";w.textContent="※ マイクロンとの連動が弱いため、判定は参考程度にしてください。";}
  let h="<tr><th>マイクロン</th><th>回数</th><th>翌朝の寄付き</th><th>終値</th></tr>";
  D.table.forEach(r=>{
    h+=`<tr><td>${r.label}</td><td>${r.n}${r.n<5?"※":""}</td><td>${cell(r.open_avg,r.open_up,r.open_down,r.open_win)}</td><td>${cell(r.close_avg,r.close_up,r.close_down,r.close_win)}</td></tr>`;
  });
  $("tbl").innerHTML=h;
  $("mu").addEventListener("input",render);$("act").addEventListener("input",render);
  render();
}
init();
</script></body></html>
"""


def make_html(payload, path):
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser(description="キオクシア判定ツール")
    ap.add_argument("--actual", type=float, default=None,
                    help="今日のキオクシアの気配/現在値の前日比(%%)。例: -8.0")
    ap.add_argument("--demo", action="store_true", help="架空データで動作確認")
    ap.add_argument("--years", type=int, default=3, help="分析する年数(既定3)")
    ap.add_argument("--out", default="kioxia_judge.html")
    args = ap.parse_args()

    data = demo_data() if args.demo else fetch(args.years)
    df = build(data)
    if len(df) < 30:
        sys.exit(f"分析できる日数が少なすぎます({len(df)}日)。")

    table = bin_table(df)
    pd.DataFrame(table).to_csv("kioxia_table.csv", index=False, encoding="utf-8-sig")
    m = fit(df)

    mu_close = data["MU"]["Close"]
    mu_today = float(mu_close.pct_change().iloc[-1] * 100)
    mu_date = mu_close.index[-1]
    pred_today = m["a"] + m["b"] * mu_today

    payload = {
        **m, "table": table, "demo": args.demo,
        "mu_today": round(mu_today, 2), "mu_date": f"{mu_date.month}/{mu_date.day}",
        "actual": args.actual, "n": len(df),
        "start": df["date"].min().strftime("%Y/%m/%d"),
        "end": df["date"].max().strftime("%Y/%m/%d"),
        "asof": pd.Timestamp.today().strftime("%m/%d %H:%M"),
    }
    make_html(payload, args.out)

    # コンソールにも要点を表示
    print("=" * 40)
    print(f"分析日数 {len(df)}日 ({payload['start']}〜{payload['end']})")
    print(f"マイクロン直近 {mu_today:+.1f}% → キオクシア予測 {pred_today:+.1f}%")
    if args.actual is not None:
        diff = args.actual - pred_today
        s = stars(diff / m["sd"])
        name = {5: "強い買い", 4: "買い", 3: "中立", 2: "売り", 1: "強い売り"}[s]
        print(f"実際 {args.actual:+.1f}% / 差 {diff:+.1f}% → {'★' * s}{'☆' * (5 - s)} {name}")
    print(f"予測の当たりやすさ(MUだけ) {m['r2'] * 100:.0f}% / 他指標も加えると {m['r2_multi'] * 100:.0f}%")
    print("=" * 40)
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\n画面を作りました → {args.out}  (スマホに送って開いてください)")


if __name__ == "__main__":
    main()
