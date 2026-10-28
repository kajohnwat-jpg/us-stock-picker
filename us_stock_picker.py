#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Stock Picker — โปรแกรมคัดเลือกหุ้น USA (S&P 500 + Nasdaq-100)
=====================================================================
คัด Top 20 หุ้นสำหรับ 3 ช่วงเวลา: รายสัปดาห์ / รายเดือน / รายปี
โดยรวมคะแนนจาก
  • Technical  : SMA20/50/200, RSI, MACD, ADX, Bollinger, ATR, Volume, Relative Strength vs S&P500
  • Fundamental: ROE, Margin, FCF, หนี้สิน, การเติบโต, P/E, PEG, EV/EBITDA, ปันผล, มุมมองนักวิเคราะห์
  • News       : ข่าวล่าสุดจาก Yahoo Finance + วิเคราะห์ทิศทางข่าว (sentiment) แบบคำสำคัญ
ข้อมูลดึงสดจาก Yahoo Finance ทุกครั้งที่รัน (ราคาอาจดีเลย์ ~15 นาทีระหว่างตลาดเปิด)

วิธีใช้:
    pip install -r requirements.txt
    python us_stock_picker.py              # รันเต็ม แล้วเปิดรายงานในเบราว์เซอร์
    python us_stock_picker.py --quick      # ทดลองเร็ว (เฉพาะ ~150 บริษัทใหญ่)
    python us_stock_picker.py --help       # ดูตัวเลือกทั้งหมด

คำเตือน: เป็นเครื่องมือช่วยคัดกรองข้อมูล ไม่ใช่คำแนะนำการลงทุน
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import math
import os
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    sys.exit("กรุณาติดตั้งไลบรารีก่อน:  pip install -r requirements.txt")

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
REPORT_DIR = APP_DIR / "reports"

# ---------------------------------------------------------------------------
# 1) รายชื่อหุ้น (Universe)
# ---------------------------------------------------------------------------
# รายชื่อสำรอง (ใช้เมื่อดึงรายชื่อจาก Wikipedia ไม่ได้ หรือใช้โหมด --quick)
FALLBACK_TICKERS = """
AAPL MSFT NVDA AMZN GOOGL GOOG META AVGO TSLA BRK-B LLY JPM V UNH XOM MA JNJ PG HD COST
ABBV MRK ORCL CVX KO PEP BAC WMT ADBE CRM NFLX AMD TMO ACN LIN MCD CSCO ABT DIS WFC
INTU DHR TXN VZ QCOM CMCSA PM AMGN IBM NOW CAT GE UNP SPGI AMAT ISRG NEE RTX HON
LOW GS PFE BKNG T UBER ELV AXP SYK MS BLK PLD VRTX LMT TJX MDT DE C SCHW ADP
MMC CB PGR BSX ADI REGN LRCX PANW MU KLAC SNPS CDNS ANET GILD SBUX BMY MO SO DUK
ZTS CL MDLZ ICE CME EQIX TT SHW WM ITW APH MCO AON CTAS PH ORLY AZO MAR HLT ROP
NOC GD FDX UPS CSX NSC MSI ECL APD EMR ADSK FTNT CRWD MRVL ASML PYPL ABNB MELI
KDP MNST PCAR PAYX FAST ODFL VRSK IDXX CPRT ROST DXCM LULU CTSH EA TTWO WDAY
TEAM DDOG ZS MDB CEG VST KMB GIS HSY KHC SYY KR TGT DG DLTR CVS CI HUM MCK COR
CAH HCA EW RMD A WAT MTD IQV PSA O SPG WELL AMT CCI DLR AVB EQR AFL MET PRU
TRV ALL AIG HIG PNC USB TFC COF DFS FI FIS GPN WMB KMI OKE EOG COP PSX MPC VLO
SLB HAL OXY FANG NUE STLD FCX NEM DOW DD PPG ALB MLM VMC CARR OTIS JCI ETN ROK
URI PWR GWW XYL DOV IR AME HUBB TDG HWM LHX TXT BA
""".split()

# หุ้นใน Nasdaq-100 ที่ไม่อยู่ใน S&P 500 (เติมให้เสมอ เผื่อดึงรายชื่อ Nasdaq-100 ไม่สำเร็จ)
NASDAQ_EXTRA = "ASML AZN MELI PDD ARM SHOP CCEP TEAM DDOG ZS MDB GFS TRI".split()

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _read_wiki_tables(url: str) -> list[pd.DataFrame]:
    if requests is None:
        return pd.read_html(url)
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def _symbols_from_tables(tables: list[pd.DataFrame], min_rows: int) -> list[str]:
    for t in tables:
        if len(t) < min_rows:
            continue
        for c in t.columns:
            name = " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
            if "symbol" in name.lower() or "ticker" in name.lower():
                vals = [str(s).strip() for s in t[c].dropna().tolist()]
                vals = [v for v in vals if 0 < len(v) <= 6 and v.replace(".", "").replace("-", "").isalpha()]
                if len(vals) >= min_rows * 0.9:
                    return vals
    return []


WIKI_META: dict[str, dict] = {}  # ชื่อบริษัท/กลุ่มธุรกิจจาก Wikipedia (ใช้เมื่อ Yahoo ไม่ให้ข้อมูล)
GICS_TO_YAHOO = {
    "Information Technology": "Technology", "Health Care": "Healthcare", "Financials": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical", "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials", "Communication Services": "Communication Services",
    "Industrials": "Industrials", "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate",
}


def _collect_meta(tables: list[pd.DataFrame]) -> None:
    for t in tables:
        cols = {str(c).lower(): c for c in t.columns}
        sym = next((cols[k] for k in cols if "symbol" in k or "ticker" in k), None)
        name = next((cols[k] for k in cols if k in ("security", "company", "company name")), None)
        sec = next((cols[k] for k in cols if "gics sector" in k), None)
        ind = next((cols[k] for k in cols if "sub-industry" in k or "sub industry" in k), None)
        if sym is None or (name is None and sec is None):
            continue
        for _, row in t.iterrows():
            tk = str(row[sym]).strip().replace(".", "-").upper()
            m = WIKI_META.setdefault(tk, {})
            if name is not None and isinstance(row[name], str):
                m.setdefault("longName", row[name])
            if sec is not None and isinstance(row[sec], str):
                m.setdefault("sector", GICS_TO_YAHOO.get(row[sec], row[sec]))
            if ind is not None and isinstance(row[ind], str):
                m.setdefault("industry", row[ind])


def get_universe(quick: bool = False) -> tuple[list[str], str]:
    """คืนรายชื่อหุ้น S&P 500 + Nasdaq-100 (ดึงสดจาก Wikipedia)"""
    if quick:
        return sorted(set(FALLBACK_TICKERS)), "Quick mode (~200 บริษัทขนาดใหญ่)"
    tickers: set[str] = set()
    src = []
    try:
        tabs = _read_wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp = _symbols_from_tables(tabs, 400)
        _collect_meta(tabs)
        if sp:
            tickers.update(sp)
            src.append(f"S&P 500 ({len(sp)})")
    except Exception as e:
        print(f"  ! ดึงรายชื่อ S&P 500 ไม่ได้: {e}")
    try:
        tabs = _read_wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        nd = _symbols_from_tables(tabs, 90)
        _collect_meta(tabs)
        if nd:
            tickers.update(nd)
            src.append(f"Nasdaq-100 ({len(nd)})")
    except Exception as e:
        print(f"  ! ดึงรายชื่อ Nasdaq-100 ไม่ได้: {e}")
    if len(tickers) < 300:
        tickers.update(FALLBACK_TICKERS)
        src.append("รายชื่อสำรอง")
    if not any(x.startswith("Nasdaq") for x in src):
        tickers.update(NASDAQ_EXTRA)
        src.append("Nasdaq-100 ส่วนเพิ่ม")
    clean = sorted({t.replace(".", "-").upper() for t in tickers if t and t != "nan"})
    return clean, " + ".join(src)


# ---------------------------------------------------------------------------
# 2) ดึงราคาและคำนวณ Technical
# ---------------------------------------------------------------------------
def download_prices(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    chunk = 100
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        print(f"  ดาวน์โหลดราคา {i + 1}-{i + len(part)} / {len(tickers)} ...")
        for attempt in range(3):
            try:
                df = yf.download(part, period=period, interval="1d", group_by="ticker",
                                 auto_adjust=True, threads=True, progress=False)
                break
            except Exception as e:  # pragma: no cover
                print(f"    retry ({e})")
                time.sleep(3 * (attempt + 1))
        else:
            continue
        for t in part:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    lv0 = df.columns.get_level_values(0)
                    sub = df[t] if t in lv0 else df.xs(t, axis=1, level=1)
                else:
                    sub = df
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(sub) >= 220:
                    out[t] = sub.copy()
            except Exception:
                pass
    return out


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx, pdi, mdi, atr


def compute_technicals(df: pd.DataFrame, spy_close: pd.Series) -> dict:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    rsi = _rsi(c)
    macd = _ema(c, 12) - _ema(c, 26)
    sig = _ema(macd, 9)
    hist = macd - sig
    adx, pdi, mdi, atr = _adx(h, l, c)
    std20 = c.rolling(20).std()
    bb_up, bb_lo = sma20 + 2 * std20, sma20 - 2 * std20
    rets = c.pct_change()

    def ret(n):
        return float(c.iloc[-1] / c.iloc[-n - 1] - 1) if len(c) > n else np.nan

    last = c.iloc[-1]
    hi52 = c.iloc[-252:].max()
    lo52 = c.iloc[-252:].min()
    window = c.iloc[-252:]
    dd = float((window / window.cummax() - 1).min())
    vol20 = v.iloc[-20:].mean()
    vol50 = v.iloc[-50:].mean()
    up_days = rets.iloc[-20:] > 0
    up_vol = v.iloc[-20:][up_days].mean() if up_days.any() else np.nan
    dn_vol = v.iloc[-20:][~up_days].mean() if (~up_days).any() else np.nan

    spy = spy_close.reindex(c.index).ffill()

    def rs(n):
        if len(c) <= n or pd.isna(spy.iloc[-n - 1]):
            return np.nan
        return float((c.iloc[-1] / c.iloc[-n - 1]) / (spy.iloc[-1] / spy.iloc[-n - 1]) - 1)

    sma200_slope = float(sma200.iloc[-1] / sma200.iloc[-21] - 1) if len(sma200.dropna()) > 21 else np.nan
    sma20_slope = float(sma20.iloc[-1] / sma20.iloc[-6] - 1)

    return {
        "price": float(last),
        "chg_1d": ret(1), "ret_1w": ret(5), "ret_1m": ret(21), "ret_3m": ret(63),
        "ret_6m": ret(126), "ret_1y": ret(252), "ret_3y": ret(min(len(c) - 1, 756)),
        "mom_12_1": float(c.iloc[-22] / c.iloc[-253] - 1) if len(c) > 253 else np.nan,
        "sma20": float(sma20.iloc[-1]), "sma50": float(sma50.iloc[-1]), "sma200": float(sma200.iloc[-1]),
        "sma20_slope": sma20_slope, "sma200_slope": sma200_slope,
        "rsi": float(rsi.iloc[-1]),
        "macd": float(macd.iloc[-1]), "macd_sig": float(sig.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]), "macd_hist_prev": float(hist.iloc[-4]),
        "macd_cross_up": bool((hist.iloc[-5:] > 0).iloc[-1] and (hist.iloc[-5:] <= 0).any()),
        "adx": float(adx.iloc[-1]), "pdi": float(pdi.iloc[-1]), "mdi": float(mdi.iloc[-1]),
        "atr": float(atr.iloc[-1]), "atr_pct": float(atr.iloc[-1] / last),
        "bb_pos": float((last - bb_lo.iloc[-1]) / (bb_up.iloc[-1] - bb_lo.iloc[-1]))
        if bb_up.iloc[-1] != bb_lo.iloc[-1] else 0.5,
        "hi52": float(hi52), "lo52": float(lo52), "from_hi52": float(last / hi52 - 1),
        "max_dd_1y": dd, "vol_1y": float(rets.iloc[-252:].std() * math.sqrt(252)),
        "vol_ratio": float(vol20 / vol50) if vol50 else np.nan,
        "updown_vol": float(up_vol / dn_vol) if dn_vol and not pd.isna(dn_vol) else np.nan,
        "dollar_vol": float((c.iloc[-20:] * v.iloc[-20:]).mean()),
        "rs_1m": rs(21), "rs_3m": rs(63), "rs_6m": rs(126), "rs_1y": rs(252),
        "spark_dates": [x.strftime("%Y-%m-%d") for x in c.index[-252:]],
        "spark": [round(float(x), 2) for x in c.iloc[-252:].tolist()],
        "spark50": [None if pd.isna(x) else round(float(x), 2) for x in sma50.iloc[-252:].tolist()],
        "spark200": [None if pd.isna(x) else round(float(x), 2) for x in sma200.iloc[-252:].tolist()],
    }


# ---------------------------------------------------------------------------
# 3) Fundamental (Yahoo Finance .info) — มี cache วันละครั้ง
# ---------------------------------------------------------------------------
INFO_KEYS = [
    "longName", "shortName", "sector", "industry", "country", "website", "fullTimeEmployees",
    "longBusinessSummary", "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
    "pegRatio", "trailingPegRatio", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "returnOnEquity", "returnOnAssets", "profitMargins",
    "operatingMargins", "grossMargins", "ebitdaMargins", "revenueGrowth", "earningsGrowth",
    "earningsQuarterlyGrowth", "debtToEquity", "currentRatio", "quickRatio", "totalCash",
    "totalDebt", "freeCashflow", "operatingCashflow", "totalRevenue", "trailingEps",
    "forwardEps", "dividendYield", "payoutRatio", "fiveYearAvgDividendYield", "beta",
    "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
    "dividendRate", "trailingAnnualDividendYield", "targetMeanPrice", "targetHighPrice", "targetLowPrice", "heldPercentInstitutions",
    "earningsTimestamp", "earningsTimestampStart", "exchange", "quoteType",
]


_RATE = {"pause_until": 0.0, "fails": 0, "blocked": False, "deadline": float("inf")}
INFO_TIME_BUDGET = 420  # วินาที: ถ้าดึงงบนานเกินนี้ ตัวที่เหลือใช้ข้อมูลแบบเบาไปก่อน


def _is_rate_limit(e: Exception) -> bool:
    m = f"{type(e).__name__} {e}".lower()
    return "rate" in m or "too many" in m or "429" in m


def _fast_info(t: str) -> dict:
    """ข้อมูลพื้นฐานแบบเบา (มูลค่าตลาด) จาก fast_info — ใช้เมื่อ .info ถูกบล็อก"""
    try:
        fi = yf.Ticker(t).fast_info
        mc = None
        for k in ("market_cap", "marketCap"):
            try:
                mc = fi[k]
                break
            except Exception:
                mc = getattr(fi, k, None)
                if mc:
                    break
        if mc and not (isinstance(mc, float) and math.isnan(mc)):
            return {"marketCap": float(mc), "quoteType": "EQUITY", "_partial": True}
    except Exception:
        pass
    return {}


def _fetch_info(t: str) -> dict:
    for attempt in range(3):
        if _RATE["blocked"] or time.time() > _RATE["deadline"]:
            break  # ถูกบล็อกต่อเนื่อง: ข้ามไปใช้ข้อมูลแบบเบาเลย ไม่เสียเวลารอ
        wait = _RATE["pause_until"] - time.time()
        if wait > 0:
            time.sleep(min(wait, 20))
        try:
            info = yf.Ticker(t).info or {}
            if info and (info.get("marketCap") or info.get("longName") or info.get("sector")):
                _RATE["fails"] = 0
                return {k: info.get(k) for k in INFO_KEYS}
        except Exception as e:  # noqa: BLE001
            if _is_rate_limit(e):
                _RATE["fails"] += 1
                if _RATE["fails"] >= 15:
                    _RATE["blocked"] = True
                _RATE["pause_until"] = max(_RATE["pause_until"], time.time() + 5 * (attempt + 1))
                continue
            time.sleep(0.5 * (attempt + 1))
    return _fast_info(t)


def fetch_fundamentals(tickers: list[str], workers: int = 4, progress=None) -> dict[str, dict]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "info_cache.json"
    max_age = 7 * 86400  # งบการเงินเปลี่ยนทุกไตรมาส: เก็บไว้ 7 วัน ลดการเรียก Yahoo
    cache: dict[str, dict] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    # ดึงใหม่ทั้งตัวที่ยังไม่มี และตัวที่ได้ข้อมูลไม่ครบรอบก่อน
    now_ts = time.time()
    todo = [t for t in tickers if not cache.get(t) or cache[t].get("_partial")
            or now_ts - cache[t].get("_ts", 0) > max_age]
    _RATE.update(pause_until=0.0, fails=0, blocked=False, deadline=time.time() + INFO_TIME_BUDGET)
    if todo:
        print(f"  ดึงงบการเงิน/ข้อมูลบริษัท {len(todo)} ตัว (cache แล้ว {len(tickers) - len(todo)}) ...")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch_info, t): t for t in todo}
            for f in as_completed(futs):
                t = futs[f]
                res = f.result()
                if res:
                    res["_ts"] = time.time()
                if res and (not res.get("_partial") or not cache.get(t) or cache[t].get("_partial")):
                    cache[t] = res
                elif not cache.get(t):
                    cache[t] = {}
                done += 1
                if progress and (done % 10 == 0 or done == len(todo)):
                    progress(done, len(todo))
                if done % 50 == 0 or done == len(todo):
                    print(f"    {done}/{len(todo)}")
                    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    for old in CACHE_DIR.glob("info_2*.json"):  # ไฟล์ cache รูปแบบเก่า
        if old != cache_file:
            try:
                old.unlink()
            except Exception:
                pass
    return {t: cache.get(t, {}) for t in tickers}


# ---------------------------------------------------------------------------
# 4) ข่าว + Sentiment
# ---------------------------------------------------------------------------
POS_WORDS = {
    "beat": 2, "beats": 2, "tops": 2, "exceeds": 2, "surpass": 2, "record": 1.5, "surge": 2,
    "surges": 2, "soar": 2, "soars": 2, "jump": 1.5, "jumps": 1.5, "rally": 1.5, "rallies": 1.5,
    "gain": 1, "gains": 1, "rise": 1, "rises": 1, "climb": 1, "climbs": 1, "upgrade": 2.5,
    "upgrades": 2.5, "upgraded": 2.5, "outperform": 2, "overweight": 1.5, "buy": 1,
    "raises": 2, "raised": 1.5, "boost": 1.5, "boosts": 1.5, "strong": 1.5, "stronger": 1.5,
    "growth": 1, "profit": 1, "profitable": 1.5, "wins": 1.5, "win": 1, "award": 1,
    "approval": 2, "approved": 2, "approves": 2, "partnership": 1, "expands": 1, "expansion": 1,
    "bullish": 2, "buyback": 1.5, "repurchase": 1.5, "dividend": 0.5, "hike": 1, "rebound": 1,
    "optimistic": 1.5, "upbeat": 1.5, "breakthrough": 2, "launch": 0.5, "deal": 0.5,
    "accelerate": 1, "accelerates": 1, "momentum": 0.5, "top": 0.5, "best": 1, "higher": 0.5,
}
NEG_WORDS = {
    "miss": 2, "misses": 2, "missed": 2, "plunge": 2.5, "plunges": 2.5, "plummet": 2.5,
    "tumble": 2, "tumbles": 2, "sink": 2, "sinks": 2, "slump": 2, "slumps": 2, "fall": 1,
    "falls": 1, "drop": 1, "drops": 1, "decline": 1, "declines": 1, "slide": 1, "slides": 1,
    "downgrade": 2.5, "downgrades": 2.5, "downgraded": 2.5, "underperform": 2,
    "underweight": 1.5, "sell": 1, "cut": 1.5, "cuts": 1.5, "lowers": 1.5, "lowered": 1.5,
    "weak": 1.5, "weaker": 1.5, "loss": 1.5, "losses": 1.5, "lawsuit": 2, "sued": 2, "sues": 2,
    "probe": 2, "investigation": 2, "subpoena": 2, "recall": 2, "fraud": 3, "layoffs": 1.5,
    "layoff": 1.5, "warning": 2, "warns": 2, "bearish": 2, "antitrust": 1.5, "fine": 1,
    "fined": 2, "penalty": 1.5, "delay": 1.5, "delays": 1.5, "halt": 2, "halts": 2,
    "concern": 1, "concerns": 1, "risk": 0.5, "risks": 0.5, "bankruptcy": 3, "default": 2,
    "selloff": 2, "sell-off": 2, "crash": 2.5, "disappoint": 2, "disappointing": 2,
    "short": 0.5, "lower": 0.5, "worst": 1.5, "slowdown": 1.5, "tariff": 1, "tariffs": 1,
}


def headline_sentiment(text: str) -> float:
    words = [w.strip(".,:;!?\"'()[]").lower() for w in text.replace("-", " - ").split()]
    words += [w.strip(".,:;!?\"'()[]").lower() for w in text.split()]  # คงคำมีขีด เช่น sell-off
    pos = sum(POS_WORDS.get(w, 0) for w in words)
    neg = sum(NEG_WORDS.get(w, 0) for w in words)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _parse_news_item(n: dict) -> dict | None:
    c = n.get("content") if isinstance(n.get("content"), dict) else None
    if c:
        title = c.get("title")
        url = ((c.get("canonicalUrl") or {}).get("url")
               or (c.get("clickThroughUrl") or {}).get("url"))
        pub = (c.get("provider") or {}).get("displayName", "")
        ts = c.get("pubDate") or c.get("displayTime")
        try:
            when = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            when = None
        summary = c.get("summary") or ""
    else:
        title = n.get("title")
        url = n.get("link")
        pub = n.get("publisher", "")
        ts = n.get("providerPublishTime")
        when = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc) if ts else None
        summary = ""
    if not title:
        return None
    return {"title": title, "url": url or "", "publisher": pub, "when": when, "summary": summary}


def fetch_news(t: str) -> dict:
    try:
        raw = yf.Ticker(t).news or []
    except Exception:
        raw = []
    items = [x for x in (_parse_news_item(n) for n in raw) if x]
    now = dt.datetime.now(dt.timezone.utc)
    items.sort(key=lambda x: x["when"] or dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc), reverse=True)
    wsum = ssum = 0.0
    for it in items:
        s = headline_sentiment(it["title"] + " " + it["summary"][:300])
        it["sent"] = s
        age_days = (now - it["when"]).total_seconds() / 86400 if it["when"] else 7
        it["age_days"] = age_days
        w = math.exp(-age_days / 5)  # ข่าวใหม่มีน้ำหนักมากกว่า
        wsum += w
        ssum += w * s
    score = ssum / wsum if wsum else 0.0
    recent = sum(1 for it in items if it["age_days"] <= 7)
    for it in items:
        it["when"] = it["when"].strftime("%Y-%m-%d %H:%M") if it["when"] else ""
    return {"items": items[:8], "sentiment": score, "count_7d": recent}


def translate_th(texts: dict[str, str]) -> dict[str, str]:
    """แปลคำอธิบายธุรกิจเป็นภาษาไทย (ถ้าติดตั้ง deep-translator ไว้)"""
    try:
        from deep_translator import GoogleTranslator
    except Exception:
        return {}
    out = {}

    def one(k, txt):
        try:
            return k, GoogleTranslator(source="en", target="th").translate(txt[:4500])
        except Exception:
            return k, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, v in ex.map(lambda kv: one(*kv), texts.items()):
            if v:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# 5) ระบบให้คะแนน
# ---------------------------------------------------------------------------
def pct(s: pd.Series, higher_better: bool = True) -> pd.Series:
    """แปลงเป็นคะแนนเปอร์เซ็นไทล์ 0-100 (ค่าที่ไม่มีข้อมูล = 40 กลางค่อนต่ำ)"""
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    r = s.rank(pct=True, ascending=higher_better) * 100
    return r.fillna(40)


def avg(*cols: pd.Series, weights=None) -> pd.Series:
    df = pd.concat(cols, axis=1)
    w = np.array(weights if weights else [1] * len(cols), dtype=float)
    return (df * w).sum(axis=1) / w.sum()


def rsi_score(r: float, above200: bool) -> float:
    if pd.isna(r):
        return 50
    if 50 <= r <= 65:
        return 100
    if 45 <= r < 50 or 65 < r <= 70:
        return 80
    if 40 <= r < 45:
        return 65
    if 70 < r <= 75:
        return 50
    if r > 75:
        return 20          # Overbought — เสี่ยงย่อ
    if 30 <= r < 40:
        return 55 if above200 else 35
    return 60 if above200 else 15   # Oversold: ถ้ายังอยู่ในขาขึ้นใหญ่ = โอกาสเด้ง


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)

    above200 = d["price"] > d["sma200"]
    above50 = d["price"] > d["sma50"]
    above20 = d["price"] > d["sma20"]
    golden = d["sma50"] > d["sma200"]

    # ---------- Technical ระยะสั้น ----------
    trend_short = (above20.astype(float) * 35 + (d["sma20"] > d["sma50"]).astype(float) * 35
                   + (d["sma20_slope"] > 0).astype(float) * 30)
    macd_sc = ((d["macd_hist"] > 0).astype(float) * 50
               + (d["macd_hist"] > d["macd_hist_prev"]).astype(float) * 30
               + (d["macd"] > 0).astype(float) * 10 + d["macd_cross_up"].astype(float) * 10)
    rsi_sc = pd.Series([rsi_score(r, a) for r, a in zip(d["rsi"], above200)], index=d.index)
    adx_sc = np.where(d["pdi"] > d["mdi"], np.clip(d["adx"] * 2.5, 30, 100), np.clip(40 - d["adx"], 0, 40))
    ext = (d["price"] / d["sma20"] - 1)
    ext_sc = np.where(ext > 0.10, 20, np.where(ext > 0.06, 55, np.where(ext < -0.08, 35, 90)))
    vol_sc = pct(d["updown_vol"])
    d["s_tech_short"] = avg(trend_short, macd_sc, rsi_sc, pd.Series(adx_sc, index=d.index),
                            pd.Series(ext_sc, index=d.index), vol_sc,
                            pct(d["ret_1w"]), pct(d["rs_1m"]),
                            weights=[15, 15, 15, 10, 10, 10, 10, 15])

    # ---------- Technical ระยะกลาง-ยาว (แนวโน้ม) ----------
    trend_long = (above200.astype(float) * 40 + golden.astype(float) * 30
                  + (d["sma200_slope"] > 0).astype(float) * 20 + above50.astype(float) * 10)
    d["s_trend"] = avg(trend_long, pct(d["mom_12_1"]), pct(d["rs_6m"]), pct(d["rs_3m"]),
                       pct(d["from_hi52"]), pct(d["max_dd_1y"]),
                       weights=[30, 20, 15, 10, 15, 10])

    # ---------- Fundamental ----------
    roe = num("returnOnEquity").clip(-1, 1.5)
    fcf_margin = num("freeCashflow") / num("totalRevenue")
    fcf_yield = num("freeCashflow") / num("marketCap")
    de = num("debtToEquity")
    d["fcf_yield"] = fcf_yield
    d["fcf_margin"] = fcf_margin
    d["s_quality"] = avg(pct(roe), pct(num("operatingMargins")), pct(num("grossMargins")),
                         pct(fcf_margin), pct(de, higher_better=False), pct(num("currentRatio")),
                         pct(num("returnOnAssets")), weights=[20, 15, 10, 20, 15, 5, 15])
    d["s_growth"] = avg(pct(num("revenueGrowth").clip(-1, 2)), pct(num("earningsGrowth").clip(-1, 3)),
                        pct(num("earningsQuarterlyGrowth").clip(-1, 3)),
                        pct((num("forwardEps") / num("trailingEps") - 1).where(num("trailingEps") > 0).clip(-1, 2)),
                        weights=[35, 25, 15, 25])

    # Value เทียบภายใน Sector เดียวกัน (ยุติธรรมกว่าเทียบข้ามกลุ่ม)
    fpe = num("forwardPE").where(num("forwardPE") > 0)
    peg = num("trailingPegRatio").fillna(num("pegRatio")).where(lambda x: x > 0)
    eve = num("enterpriseToEbitda").where(num("enterpriseToEbitda") > 0)
    sector = d["sector"].fillna("Other")
    sec_pct = lambda s, hb: s.groupby(sector).rank(pct=True, ascending=hb).mul(100).fillna(30)
    d["s_value"] = avg(sec_pct(fpe, False), sec_pct(peg, False), sec_pct(eve, False),
                       pct(fcf_yield), weights=[35, 20, 20, 25])

    beta = num("beta")
    d["s_safety"] = avg(pct(np.log(num("marketCap"))), pct(d["vol_1y"], False),
                        pct((beta - 0.9).abs(), False), pct(d["max_dd_1y"]),
                        pct(de, higher_better=False), weights=[30, 25, 15, 20, 10])

    upside = num("targetMeanPrice") / d["price"] - 1
    d["upside"] = upside
    n_an = num("numberOfAnalystOpinions").fillna(0)
    conf = np.clip(n_an / 15, 0, 1)
    raw_an = avg(pct(num("recommendationMean"), False), pct(upside.clip(-0.5, 1)), weights=[55, 45])
    d["s_analyst"] = raw_an * conf + 50 * (1 - conf)

    # คำนวณ yield เองจากเงินปันผลต่อหุ้น (เลี่ยงปัญหาหน่วย % ที่ต่างกันในแต่ละเวอร์ชันของ yfinance)
    div = (num("dividendRate") / d["price"]).fillna(num("trailingAnnualDividendYield")).fillna(0)
    d["div_yield"] = div
    payout = num("payoutRatio")
    d["s_dividend"] = np.where((div > 0) & (payout.fillna(0) < 0.85), pct(div.where(div > 0)), 35)

    # ---------- News ----------
    sent = num("news_sent").fillna(0)
    d["s_news"] = (50 + sent * 50).clip(0, 100)

    # ---------- คะแนนรวมแต่ละช่วงเวลา ----------
    d["score_week"] = (d["s_tech_short"] * 0.45 + d["s_trend"] * 0.15 + d["s_news"] * 0.20
                       + d["s_quality"] * 0.10 + d["s_safety"] * 0.10)
    d["score_month"] = (d["s_tech_short"] * 0.25 + d["s_trend"] * 0.25 + d["s_news"] * 0.10
                        + d["s_quality"] * 0.15 + d["s_growth"] * 0.10 + d["s_value"] * 0.05
                        + d["s_analyst"] * 0.10)
    d["score_year"] = (d["s_quality"] * 0.25 + d["s_growth"] * 0.20 + d["s_value"] * 0.15
                       + d["s_safety"] * 0.12 + d["s_trend"] * 0.10 + d["s_analyst"] * 0.10
                       + d["s_dividend"] * 0.03 + d["s_news"] * 0.05)

    # ปรับลด: งบประกาศภายใน 7 วัน (ความผันผวนสูงสำหรับรายสัปดาห์)
    d["score_week"] = d["score_week"] - np.where(d["earn_days"].between(0, 7), 5, 0)
    # ปรับลด: หุ้นขาดทุน / หนี้สูงมาก สำหรับระยะยาว
    d["score_year"] = d["score_year"] - np.where(num("profitMargins") <= 0, 15, 0) \
        - np.where(de > 300, 5, 0)
    d["score_month"] = d["score_month"] - np.where(num("profitMargins") <= 0, 8, 0)

    # ---------- ⭐ อันดับน่าซื้อสะสมตอนนี้ (มองไปข้างหน้า) ----------
    fwd_eps = (num("forwardEps") / num("trailingEps") - 1).where(num("trailingEps") > 0).clip(-1, 2)
    d["fwd_eps_growth"] = fwd_eps
    d["s_fwd"] = avg(pct(fwd_eps), pct(num("revenueGrowth").clip(-1, 2)), pct(num("earningsGrowth").clip(-1, 3)),
                     weights=[45, 30, 25])
    d["s_mom"] = pct(d["mom_12_1"])
    ind = (d["industry"].fillna("") if "industry" in d else pd.Series("", index=d.index)).str.lower()
    d["is_cyclical"] = ind.str.contains("|".join(CYCLICAL_KEYWORDS), regex=True)
    d["score_buy"] = (d["s_quality"] * 0.25 + d["s_fwd"] * 0.20 + d["s_mom"] * 0.20 + d["s_value"] * 0.15
                      + d["s_analyst"] * 0.10 + d["s_safety"] * 0.10)
    d["score_buy"] = (d["score_buy"] - np.where(num("profitMargins") <= 0, 15, 0)
                      - np.where(d["price"] < d["sma200"], 5, 0)
                      - np.where(d["is_cyclical"], 5, 0)
                      - np.where(de > 300, 5, 0))
    return d


# อุตสาหกรรมที่กำไรขึ้นกับราคาสินค้าโภคภัณฑ์ (ดูดีที่สุดในสูตรตอนกำไรพีค)
CYCLICAL_KEYWORDS = ["oil & gas", "gold", "silver", "copper", "steel", "metals & mining", "aluminum",
                     "agricultural", "fertilizer", "coal", "uranium"]


def pick_top(d: pd.DataFrame, col: str, n: int, max_per_sector: int,
             max_per_industry: int | None = None) -> pd.DataFrame:
    ranked = d.sort_values(col, ascending=False)
    chosen, count, icount = [], {}, {}
    for idx, row in ranked.iterrows():
        sec = sv(row.get("sector"), None) or f"_{idx}"  # ไม่รู้กลุ่ม = ไม่นับรวมกับตัวอื่น
        ind = sv(row.get("industry"), None) or f"_{idx}"
        if count.get(sec, 0) >= max_per_sector:
            continue
        if max_per_industry and icount.get(ind, 0) >= max_per_industry:
            continue
        chosen.append(idx)
        count[sec] = count.get(sec, 0) + 1
        icount[ind] = icount.get(ind, 0) + 1
        if len(chosen) >= n:
            break
    if max_per_industry and len(chosen) < n:  # ข้อมูลอุตสาหกรรมไม่พอ: เติมโดยไม่จำกัดอุตสาหกรรม
        extra = [i for i in ranked.index if i not in chosen][: n - len(chosen)]
        chosen += extra
        return ranked.loc[chosen].sort_values(col, ascending=False)
    return ranked.loc[chosen]


# ---------------------------------------------------------------------------
# 6) เหตุผลประกอบ (ภาษาไทย)
# ---------------------------------------------------------------------------
SECTOR_TH = {
    "Technology": "เทคโนโลยี", "Healthcare": "สุขภาพ/การแพทย์", "Financial Services": "การเงิน",
    "Consumer Cyclical": "สินค้าฟุ่มเฟือย", "Consumer Defensive": "สินค้าจำเป็น",
    "Communication Services": "สื่อสาร/อินเทอร์เน็ต", "Industrials": "อุตสาหกรรม",
    "Energy": "พลังงาน", "Utilities": "สาธารณูปโภค", "Real Estate": "อสังหาริมทรัพย์",
    "Basic Materials": "วัตถุดิบพื้นฐาน",
}
REC_TH = {"strong_buy": "ซื้อแข็งแกร่ง", "buy": "ซื้อ", "hold": "ถือ", "underperform": "ต่ำกว่าตลาด",
          "sell": "ขาย", "strongbuy": "ซื้อแข็งแกร่ง", "none": "-"}


def sv(x, default=""):
    """ค่าที่ปลอดภัย: แปลง None/NaN เป็นค่า default"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return default
    return x


def fmt_pct(x, digits=1, sign=True):
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def fmt_num(x, digits=2):
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"


def fmt_cap(x):
    if x is None or pd.isna(x):
        return "-"
    if x >= 1e12:
        return f"${x / 1e12:.2f}T"
    if x >= 1e9:
        return f"${x / 1e9:.1f}B"
    return f"${x / 1e6:.0f}M"


def reasons(r: pd.Series, horizon: str) -> tuple[list[str], list[str]]:
    g, b = [], []  # จุดเด่น, ข้อควรระวัง
    if r["price"] > r["sma200"] and r["sma50"] > r["sma200"]:
        g.append("แนวโน้มหลักเป็นขาขึ้น (ราคา > SMA200, SMA50 > SMA200)")
    elif r["price"] < r["sma200"]:
        b.append("ราคาต่ำกว่าเส้น SMA200 (แนวโน้มหลักยังไม่ขาขึ้น)")
    if horizon in ("week", "month"):
        if r["macd_cross_up"]:
            g.append("MACD เพิ่งตัดขึ้นเหนือ Signal (สัญญาณซื้อระยะสั้น)")
        elif r["macd_hist"] > 0 and r["macd_hist"] > r["macd_hist_prev"]:
            g.append("MACD Histogram เป็นบวกและกำลังขยายตัว (โมเมนตัมเพิ่ม)")
        if 50 <= r["rsi"] <= 65:
            g.append(f"RSI {r['rsi']:.0f} อยู่โซนแข็งแรงแต่ยังไม่ Overbought")
        if r["adx"] > 25 and r["pdi"] > r["mdi"]:
            g.append(f"ADX {r['adx']:.0f} แนวโน้มขาขึ้นมีความแข็งแรง")
        if not pd.isna(r["updown_vol"]) and r["updown_vol"] > 1.3:
            g.append("วอลุ่มวันขึ้นมากกว่าวันลง (มีแรงซื้อสะสม)")
    if r["rsi"] > 72:
        b.append(f"RSI {r['rsi']:.0f} Overbought — อาจย่อตัวระยะสั้น ควรทยอยซื้อ")
    if r["rsi"] < 32:
        b.append(f"RSI {r['rsi']:.0f} Oversold — ราคาอ่อนตัวแรง")
    if r["price"] / r["sma20"] - 1 > 0.08:
        b.append("ราคาวิ่งห่างเส้น SMA20 มาก (ยืดเกิน)")
    if not pd.isna(r.get("rs_6m")) and r["rs_6m"] > 0.10:
        g.append(f"ชนะดัชนี S&P 500 ใน 6 เดือน {fmt_pct(r['rs_6m'])}")
    roe = r.get("returnOnEquity")
    if roe is not None and not pd.isna(roe) and roe > 0.20:
        g.append(f"ROE สูง {fmt_pct(roe, 0, False)} (ใช้ทุนได้มีประสิทธิภาพ)")
    om = r.get("operatingMargins")
    if om is not None and not pd.isna(om) and om > 0.25:
        g.append(f"อัตรากำไรจากการดำเนินงาน {fmt_pct(om, 0, False)}")
    rg = r.get("revenueGrowth")
    if rg is not None and not pd.isna(rg):
        if rg > 0.12:
            g.append(f"รายได้โต {fmt_pct(rg, 0)} YoY")
        elif rg < 0:
            b.append(f"รายได้หดตัว {fmt_pct(rg, 0)} YoY")
    fe = r.get("fwd_eps_growth")
    if fe is not None and not pd.isna(fe) and fe > 0.12:
        g.append(f"นักวิเคราะห์คาดกำไรต่อหุ้นปีหน้าโต {fmt_pct(fe, 0)}")
    if bool(sv(r.get("is_cyclical"), False)):
        b.append("ธุรกิจวัฏจักร: กำไรขึ้นกับราคาสินค้าโภคภัณฑ์ อาจดูดีที่สุดตอนกำไรพีค")
    eg = r.get("earningsGrowth")
    if eg is not None and not pd.isna(eg) and eg > 0.15:
        g.append(f"กำไรโต {fmt_pct(eg, 0)} YoY")
    if not pd.isna(r.get("fcf_yield")) and r["fcf_yield"] > 0.04:
        g.append(f"Free Cash Flow Yield {fmt_pct(r['fcf_yield'], 1, False)} (กระแสเงินสดแข็ง)")
    if not pd.isna(r.get("fcf_yield")) and r["fcf_yield"] < 0:
        b.append("Free Cash Flow ติดลบ")
    de = r.get("debtToEquity")
    if de is not None and not pd.isna(de):
        if de > 200:
            b.append(f"หนี้สินต่อทุนสูง (D/E {de / 100:.1f} เท่า)")
        elif de < 50:
            g.append(f"หนี้ต่ำ (D/E {de / 100:.2f} เท่า)")
    fpe = r.get("forwardPE")
    if fpe is not None and not pd.isna(fpe) and fpe > 45:
        b.append(f"Forward P/E สูง {fpe:.0f} เท่า (ราคาแพง คาดหวังการเติบโตสูง)")
    if r["s_value"] > 70:
        g.append("Valuation ถูกเมื่อเทียบกับหุ้นใน Sector เดียวกัน")
    if not pd.isna(r.get("upside")) and r["upside"] > 0.15 and sv(r.get("numberOfAnalystOpinions"), 0) >= 8:
        g.append(f"ราคาเป้าหมายนักวิเคราะห์ (เฉลี่ย) สูงกว่าราคาปัจจุบัน {fmt_pct(r['upside'], 0)}")
    if not pd.isna(r.get("upside")) and r["upside"] < -0.03:
        b.append("ราคาปัจจุบันสูงกว่าราคาเป้าหมายเฉลี่ยของนักวิเคราะห์แล้ว")
    if r.get("news_sent", 0) > 0.25:
        g.append("ข่าวช่วงล่าสุดโทนบวก")
    elif r.get("news_sent", 0) < -0.25:
        b.append("ข่าวช่วงล่าสุดโทนลบ — ควรอ่านรายละเอียด")
    if 0 <= r.get("earn_days", -1) <= 14:
        b.append(f"จะประกาศงบใน {int(r['earn_days'])} วัน — ราคาอาจผันผวนแรง")
    if r["max_dd_1y"] < -0.35:
        b.append(f"เคยร่วงหนักสุด {fmt_pct(r['max_dd_1y'], 0)} ในรอบ 1 ปี")
    return g[:7], b[:5]


# ---------------------------------------------------------------------------
# 7) สร้างรายงาน HTML
# ---------------------------------------------------------------------------
def sparkline_svg(r: pd.Series, w: int = 600, h: int = 170) -> str:
    y = r["spark"]
    if not y:
        return ""
    s50, s200 = r["spark50"], r["spark200"]
    allv = [v for v in y + s50 + s200 if v is not None]
    lo, hi = min(allv), max(allv)
    rng = (hi - lo) or 1
    pad = 6
    n = len(y)

    def pts(arr):
        segs, cur = [], []
        for i, v in enumerate(arr):
            if v is None:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            x = pad + i * (w - 2 * pad) / max(n - 1, 1)
            yy = pad + (hi - v) * (h - 2 * pad) / rng
            cur.append(f"{x:.1f},{yy:.1f}")
        if cur:
            segs.append(cur)
        return segs

    col = "var(--up)" if y[-1] >= y[0] else "var(--down)"
    out = [f'<svg viewBox="0 0 {w} {h}" class="spark" preserveAspectRatio="none" role="img" '
           f'aria-label="กราฟราคา 1 ปี">']
    for seg in pts(s200):
        out.append(f'<polyline points="{" ".join(seg)}" class="l200"/>')
    for seg in pts(s50):
        out.append(f'<polyline points="{" ".join(seg)}" class="l50"/>')
    for seg in pts(y):
        out.append(f'<polyline points="{" ".join(seg)}" style="stroke:{col}" class="lp"/>')
    out.append(f'<text x="{w - pad}" y="14" text-anchor="end" class="svgt">สูง {hi:,.2f}</text>')
    out.append(f'<text x="{w - pad}" y="{h - 4}" text-anchor="end" class="svgt">ต่ำ {lo:,.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def bar(v: float) -> str:
    v = 0 if pd.isna(v) else max(0, min(100, v))
    cls = "hi" if v >= 70 else ("mid" if v >= 45 else "lo")
    return f'<span class="bar"><span class="fill {cls}" style="width:{v:.0f}%"></span></span><b>{v:.0f}</b>'


def stock_card(r: pd.Series, horizon: str, rank: int, score_col: str, desc_th: dict) -> str:
    e = html.escape
    t = r.name
    name = sv(r.get("longName")) or sv(r.get("shortName")) or t
    sec = sv(r.get("sector"), "-")
    good, bad = reasons(r, horizon)
    summary = sv(r.get("longBusinessSummary"))
    th = desc_th.get(t)
    news_html = ""
    items = r.get("news_items")
    for it in (items if isinstance(items, list) else []):
        s = it.get("sent", 0)
        tag = ('<span class="tag pos">บวก</span>' if s > 0.2 else
               '<span class="tag neg">ลบ</span>' if s < -0.2 else '<span class="tag neu">กลาง</span>')
        news_html += (f'<li>{tag} <a href="{e(it["url"])}" target="_blank" rel="noopener">{e(it["title"])}</a>'
                      f' <span class="muted">— {e(it.get("publisher") or "")} {e(it.get("when") or "")}</span></li>')
    if not news_html:
        news_html = '<li class="muted">ไม่พบข่าวล่าสุด</li>'
    rk = str(sv(r.get("recommendationKey"), "none"))
    rec = REC_TH.get(rk.lower(), rk)
    earn = "-"
    if not pd.isna(r.get("earn_days")) and sv(r.get("earn_date")):
        earn = f'{r["earn_date"]} ({int(r["earn_days"])} วัน)' if r["earn_days"] >= 0 else r["earn_date"]

    def kv(k, v):
        return f"<div><span>{k}</span><b>{v}</b></div>"

    fund = "".join([
        kv("มูลค่าตลาด", fmt_cap(r.get("marketCap"))),
        kv("P/E (ย้อนหลัง)", fmt_num(r.get("trailingPE"), 1)),
        kv("P/E (คาดการณ์)", fmt_num(r.get("forwardPE"), 1)),
        kv("PEG", fmt_num(sv(r.get("trailingPegRatio"), None) or r.get("pegRatio"), 2)),
        kv("P/B", fmt_num(r.get("priceToBook"), 1)),
        kv("EV/EBITDA", fmt_num(r.get("enterpriseToEbitda"), 1)),
        kv("ROE", fmt_pct(r.get("returnOnEquity"), 1, False)),
        kv("ROA", fmt_pct(r.get("returnOnAssets"), 1, False)),
        kv("Gross Margin", fmt_pct(r.get("grossMargins"), 1, False)),
        kv("Operating Margin", fmt_pct(r.get("operatingMargins"), 1, False)),
        kv("Net Margin", fmt_pct(r.get("profitMargins"), 1, False)),
        kv("รายได้โต (YoY)", fmt_pct(r.get("revenueGrowth"))),
        kv("กำไรโต (YoY)", fmt_pct(r.get("earningsGrowth"))),
        kv("FCF Yield", fmt_pct(r.get("fcf_yield"), 1, False)),
        kv("D/E", fmt_num(sv(r.get("debtToEquity"), np.nan) / 100, 2)),
        kv("Current Ratio", fmt_num(r.get("currentRatio"), 2)),
        kv("ปันผล", fmt_pct(r.get("div_yield"), 2, False)),
        kv("Payout Ratio", fmt_pct(r.get("payoutRatio"), 0, False)),
        kv("Beta", fmt_num(r.get("beta"), 2)),
        kv("นักวิเคราะห์", f'{rec} ({int(sv(r.get("numberOfAnalystOpinions"), 0))} ราย)'),
        kv("เป้าหมายเฉลี่ย", f'${fmt_num(r.get("targetMeanPrice"))} ({fmt_pct(r.get("upside"), 0)})'),
        kv("ประกาศงบครั้งถัดไป", earn),
    ])
    tech = "".join([
        kv("ราคา", f'${fmt_num(r["price"])} ({fmt_pct(r["chg_1d"], 2)})'),
        kv("SMA20 / 50 / 200", f'{fmt_num(r["sma20"])} / {fmt_num(r["sma50"])} / {fmt_num(r["sma200"])}'),
        kv("RSI(14)", fmt_num(r["rsi"], 1)),
        kv("MACD / Signal", f'{fmt_num(r["macd"])} / {fmt_num(r["macd_sig"])}'),
        kv("ADX (+DI/-DI)", f'{fmt_num(r["adx"], 0)} ({fmt_num(r["pdi"], 0)}/{fmt_num(r["mdi"], 0)})'),
        kv("ATR(14)", f'${fmt_num(r["atr"])} ({fmt_pct(r["atr_pct"], 1, False)})'),
        kv("ตำแหน่งใน Bollinger", fmt_pct(r["bb_pos"], 0, False)),
        kv("ห่างจุดสูงสุด 52 สัปดาห์", fmt_pct(r["from_hi52"])),
        kv("ผลตอบแทน 1W / 1M", f'{fmt_pct(r["ret_1w"])} / {fmt_pct(r["ret_1m"])}'),
        kv("ผลตอบแทน 3M / 6M", f'{fmt_pct(r["ret_3m"])} / {fmt_pct(r["ret_6m"])}'),
        kv("ผลตอบแทน 1Y / 3Y", f'{fmt_pct(r["ret_1y"])} / {fmt_pct(r["ret_3y"])}'),
        kv("เทียบ S&P500 (6M)", fmt_pct(r["rs_6m"])),
        kv("ร่วงหนักสุดใน 1 ปี", fmt_pct(r["max_dd_1y"])),
        kv("ความผันผวนต่อปี", fmt_pct(r["vol_1y"], 0, False)),
    ])
    zone = (f'แนวรับเชิงเทคนิคใกล้ ๆ: SMA50 ≈ <b>${fmt_num(r["sma50"])}</b>, '
            f'SMA200 ≈ <b>${fmt_num(r["sma200"])}</b> · ความผันผวนรายวันเฉลี่ย (ATR) ≈ '
            f'<b>${fmt_num(r["atr"])}</b>')
    subs = "".join(f'<div class="sub"><span>{lbl}</span>{bar(r[c])}</div>' for lbl, c in [
        ("Technical สั้น", "s_tech_short"), ("แนวโน้ม", "s_trend"), ("คุณภาพ", "s_quality"),
        ("การเติบโต", "s_growth"), ("ความคุ้มค่า", "s_value"), ("ความมั่นคง", "s_safety"),
        ("นักวิเคราะห์", "s_analyst"), ("ข่าว", "s_news")])
    desc_block = ""
    if th:
        desc_block += f'<p>{e(th)}</p>'
    if summary:
        desc_block += (f'<details class="en"><summary>คำอธิบายภาษาอังกฤษ (Yahoo Finance)</summary>'
                       f'<p>{e(summary)}</p></details>' if th else f'<p>{e(summary)}</p>')
    if not desc_block:
        desc_block = '<p class="muted">ไม่มีข้อมูลคำอธิบายธุรกิจ</p>'
    good_html = "".join(f"<li>{e(x)}</li>" for x in good) or '<li class="muted">-</li>'
    bad_html = "".join(f"<li>{e(x)}</li>" for x in bad) or '<li class="muted">ไม่พบจุดเสี่ยงเด่นชัด</li>'
    chg_cls = "up" if r["chg_1d"] >= 0 else "down"
    return f"""
<details class="card" data-search="{e((t + ' ' + str(name) + ' ' + sec).lower())}">
  <summary>
    <span class="rank">{rank}</span>
    <span class="tk">{e(t)}</span>
    <span class="nm">{e(str(name))}<small>{e(SECTOR_TH.get(sec, sec))} · {e(str(sv(r.get("industry"))))}</small></span>
    <span class="px">${fmt_num(r["price"])}<small class="{chg_cls}">{fmt_pct(r["chg_1d"], 2)}</small></span>
    <span class="cap">{fmt_cap(r.get("marketCap"))}</span>
    <span class="sc">{bar(r[score_col])}</span>
  </summary>
  <div class="body">
    <div class="grid2">
      <section><h4>ธุรกิจของบริษัท</h4>{desc_block}
        <p class="muted small">{e(str(sv(r.get("country"))))}
          {(" · พนักงาน " + format(int(r["fullTimeEmployees"]), ",") + " คน") if sv(r.get("fullTimeEmployees")) else ""}
          {(' · <a href="' + e(str(r["website"])) + '" target="_blank" rel="noopener">เว็บไซต์</a>') if sv(r.get("website")) else ""}</p>
      </section>
      <section><h4>กราฟราคา 1 ปี <span class="lg"><i class="k50"></i>SMA50 <i class="k200"></i>SMA200</span></h4>
        {sparkline_svg(r)}
        <p class="small">ดูกราฟสด: <a href="https://www.tradingview.com/chart/?symbol={e(t.replace('-', '.'))}" target="_blank" rel="noopener">TradingView</a> ·
        <a href="https://finance.yahoo.com/quote/{e(t)}" target="_blank" rel="noopener">Yahoo Finance</a> ·
        <a href="https://finviz.com/quote.ashx?t={e(t)}" target="_blank" rel="noopener">Finviz</a></p>
      </section>
    </div>
    <div class="grid2">
      <section><h4>จุดเด่น</h4><ul class="good">{good_html}</ul></section>
      <section><h4>ข้อควรระวัง</h4><ul class="bad">{bad_html}</ul></section>
    </div>
    <section><h4>คะแนนย่อย (0-100)</h4><div class="subs">{subs}</div></section>
    <div class="grid2">
      <section><h4>Fundamental</h4><div class="kv">{fund}</div></section>
      <section><h4>Technical</h4><div class="kv">{tech}</div><p class="small">{zone}</p></section>
    </div>
    <section><h4>ข่าวล่าสุด</h4><ul class="news">{news_html}</ul></section>
  </div>
</details>"""


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--muted:#6b7280;--line:#e5e7eb;--acc:#2563eb;
--up:#16a34a;--down:#dc2626;--hi:#16a34a;--mid:#d97706;--lo:#dc2626;--chip:#eef2ff}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e6e8ec;--muted:#9aa3b2;
--line:#2a2f3a;--acc:#60a5fa;--up:#22c55e;--down:#f87171;--hi:#22c55e;--mid:#f59e0b;--lo:#f87171;--chip:#1e293b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,"Segoe UI","Noto Sans Thai","Sarabun",Tahoma,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:24px;margin:0 0 4px}h4{margin:0 0 8px;font-size:14px;color:var(--muted);font-weight:600}
.muted{color:var(--muted)}.small{font-size:13px}a{color:var(--acc)}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 16px}
.chip{background:var(--chip);border-radius:999px;padding:4px 12px;font-size:13px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0}
.tabs button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:9px 16px;
border-radius:10px;font:inherit;cursor:pointer}.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.pane{display:none}.pane.on{display:block}
.note{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:12px;font-size:14px}
input.q{width:100%;max-width:360px;padding:9px 12px;border-radius:10px;border:1px solid var(--line);
background:var(--card);color:var(--ink);font:inherit;margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:8px 0;overflow:hidden}
.card>summary{list-style:none;display:grid;grid-template-columns:34px 70px 1fr 110px 80px 170px;
gap:10px;align-items:center;padding:12px 14px;cursor:pointer}
.card>summary::-webkit-details-marker{display:none}
.rank{width:28px;height:28px;border-radius:50%;background:var(--chip);display:grid;place-items:center;font-weight:700;font-size:13px}
.tk{font-weight:800;letter-spacing:.3px}.nm{display:flex;flex-direction:column;min-width:0}
.nm small,.px small{color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.px{display:flex;flex-direction:column;text-align:right;font-variant-numeric:tabular-nums}
.px small.up{color:var(--up)}.px small.down{color:var(--down)}.cap{color:var(--muted);text-align:right}
.bar{display:inline-block;width:110px;height:8px;background:var(--line);border-radius:5px;vertical-align:middle;margin-right:8px;overflow:hidden}
.fill{display:block;height:100%;border-radius:5px}.fill.hi{background:var(--hi)}.fill.mid{background:var(--mid)}.fill.lo{background:var(--lo)}
.body{padding:4px 16px 16px;border-top:1px solid var(--line)}.body section{margin-top:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:13.5px}
.kv div{display:flex;justify-content:space-between;border-bottom:1px dashed var(--line);padding:3px 0;gap:8px}
.kv span{color:var(--muted)}.kv b{font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
.subs{display:grid;grid-template-columns:repeat(4,1fr);gap:6px 18px;font-size:13px}
.sub{display:flex;justify-content:space-between;align-items:center}.sub .bar{width:70px}
ul{margin:0;padding-left:18px}ul.good li::marker{color:var(--up)}ul.bad li::marker{color:var(--down)}
ul.news{list-style:none;padding:0}ul.news li{padding:5px 0;border-bottom:1px dashed var(--line);font-size:14px}
.tag{font-size:11px;padding:1px 7px;border-radius:6px;margin-right:4px}
.tag.pos{background:color-mix(in srgb,var(--up) 18%,transparent);color:var(--up)}
.tag.neg{background:color-mix(in srgb,var(--down) 18%,transparent);color:var(--down)}
.tag.neu{background:var(--chip);color:var(--muted)}
svg.spark{width:100%;height:170px;background:color-mix(in srgb,var(--line) 25%,transparent);border-radius:8px}
.lp{fill:none;stroke-width:1.8}.l50{fill:none;stroke:#f59e0b;stroke-width:1.2}.l200{fill:none;stroke:#8b5cf6;stroke-width:1.2;stroke-dasharray:4 3}
.svgt{fill:var(--muted);font-size:11px}.lg{font-weight:400;font-size:12px;margin-left:8px}
.lg i{display:inline-block;width:14px;height:3px;margin:0 4px 3px 8px}.k50{background:#f59e0b}.k200{background:#8b5cf6}
details.en summary{cursor:pointer;color:var(--muted);font-size:13px}
table.sum{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);border-radius:12px;overflow:hidden}
table.sum th,table.sum td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}
table.sum th{color:var(--muted);font-weight:600}
.disc{margin-top:28px;font-size:13px;color:var(--muted)}
@media (max-width:820px){.card>summary{grid-template-columns:28px 54px 1fr 76px 30px;gap:8px}.cap,.sc .bar{display:none}
.grid2{grid-template-columns:1fr}.subs{grid-template-columns:1fr 1fr}.kv{grid-template-columns:1fr}}
"""

JS = """
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id===b.dataset.t));});
document.querySelectorAll('input.q').forEach(i=>i.oninput=()=>{const q=i.value.toLowerCase();
 i.closest('.pane').querySelectorAll('.card').forEach(c=>c.style.display=c.dataset.search.includes(q)?'':'none');});
"""

HORIZONS = [
    ("buy", "score_buy", "⭐ น่าซื้อสะสมตอนนี้",
     "จัดอันดับหุ้นสำหรับ **ซื้อเก็บระยะยาว** จากข้อมูลปัจจุบัน + แนวโน้มข้างหน้า: คุณภาพกิจการ 25% · "
     "การเติบโตที่คาดการณ์ (กำไรต่อหุ้นปีหน้า, รายได้, กำไร) 20% · โมเมนตัมราคา 12 เดือน 20% · "
     "ความคุ้มค่าเทียบกลุ่มเดียวกัน 15% · มุมมองนักวิเคราะห์ 10% · ความมั่นคง 10% · "
     "หักคะแนน: ขาดทุน, ราคาต่ำกว่า SMA200, หุ้นวัฏจักรสินค้าโภคภัณฑ์ (น้ำมัน/โลหะ/ปุ๋ย), หนี้สูงมาก · "
     "จำกัดอุตสาหกรรมเดียวกันไม่เกิน 2 ตัว"),
    ("week", "score_week", "รายสัปดาห์",
     "เน้นสัญญาณ Technical ระยะสั้น (MACD, RSI, ADX, วอลุ่ม, โมเมนตัม 1 สัปดาห์–1 เดือน) 45% · แนวโน้มหลัก 15% · "
     "ข่าวล่าสุด 20% · คุณภาพกิจการ 10% · ความมั่นคง 10% · หักคะแนนถ้าประกาศงบภายใน 7 วัน"),
    ("month", "score_month", "รายเดือน",
     "Technical ระยะสั้น 25% · แนวโน้ม/Relative Strength 25% · คุณภาพ 15% · การเติบโต 10% · "
     "นักวิเคราะห์ 10% · ข่าว 10% · Valuation 5%"),
    ("year", "score_year", "รายปี (สะสมระยะยาว)",
     "คุณภาพกิจการ (ROE, Margin, FCF, หนี้) 25% · การเติบโต 20% · ความคุ้มค่าเทียบ Sector 15% · ความมั่นคง 12% · "
     "แนวโน้มราคา 10% · นักวิเคราะห์ 10% · ข่าว 5% · ปันผล 3% · หักคะแนนหนักถ้าบริษัทขาดทุน"),
]


def build_report(d: pd.DataFrame, picks: dict, meta: dict, desc_th: dict) -> str:
    e = html.escape
    tabs, panes = [], []
    for i, (key, col, label, method) in enumerate(HORIZONS):
        top = picks[key]
        rows = "".join(
            f"<tr><td>{n}</td><td><b>{e(t)}</b></td><td>{e(str(sv(r.get('longName')) or sv(r.get('shortName')) or t))}</td>"
            f"<td>{e(SECTOR_TH.get(sv(r.get('sector'), '-'), sv(r.get('sector'), '-')))}</td>"
            f"<td>${fmt_num(r['price'])}</td><td>{r[col]:.0f}</td></tr>"
            for n, (t, r) in enumerate(top.iterrows(), 1))
        cards = "".join(stock_card(r, key, n, col, desc_th) for n, (t, r) in enumerate(top.iterrows(), 1))
        sec_mix = top["sector"].fillna("Other").map(lambda s: SECTOR_TH.get(s, s)).value_counts()
        mix = " · ".join(f"{k} {v}" for k, v in sec_mix.items())
        tabs.append(f'<button data-t="p-{key}" class="{"on" if i == 0 else ""}">{label}</button>')
        panes.append(f"""
<div class="pane {'on' if i == 0 else ''}" id="p-{key}">
  <div class="note"><b>วิธีให้คะแนน {label}:</b> {method}<br><span class="muted">สัดส่วนกลุ่มธุรกิจ: {mix}</span></div>
  <details class="note"><summary><b>ตารางสรุป Top {len(top)}</b></summary>
  <table class="sum"><tr><th>#</th><th>หุ้น</th><th>บริษัท</th><th>กลุ่ม</th><th>ราคา</th><th>คะแนน</th></tr>{rows}</table></details>
  <input class="q" placeholder="ค้นหาหุ้น / บริษัท / กลุ่ม ...">
  {cards}
</div>""")
    regime = meta["regime"]
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>US Stock Picker {meta['date']}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>US Stock Picker — Top {meta['top']} หุ้น USA</h1>
<div class="muted">ข้อมูล ณ {meta['date']} (เวลาไทย) · แหล่งข้อมูล: Yahoo Finance · รายชื่อหุ้น: {e(meta['universe_src'])}</div>
<div class="meta">
  <span class="chip">หุ้นที่สแกน {meta['n_scanned']} ตัว</span>
  <span class="chip">ผ่านเกณฑ์ความน่าเชื่อถือ {meta['n_pass']} ตัว</span>
  <span class="chip">มูลค่าตลาดขั้นต่ำ {fmt_cap(meta['min_cap'])}</span>
  <span class="chip">S&amp;P 500 (SPY) {fmt_pct(meta['spy_1m'])} 1 เดือน · {fmt_pct(meta['spy_1y'])} 1 ปี</span>
  <span class="chip">ภาวะตลาด: {regime}</span>
</div>
<div class="tabs">{''.join(tabs)}</div>
{''.join(panes)}
<div class="disc"><b>คำเตือน:</b> รายงานนี้สร้างจากการคำนวณอัตโนมัติด้วยข้อมูลสาธารณะ เพื่อช่วยคัดกรองหุ้นที่น่าศึกษาต่อเท่านั้น
ไม่ใช่คำแนะนำการลงทุน ข้อมูลอาจคลาดเคลื่อนหรือดีเลย์ การวิเคราะห์ข่าวเป็นแบบคำสำคัญอย่างง่าย ผลตอบแทนในอดีตไม่ได้รับประกันอนาคต
ควรศึกษางบการเงินและข่าวด้วยตนเอง กระจายความเสี่ยง และทยอยลงทุน (DCA) ตามความเสี่ยงที่รับได้</div>
</div><script>{JS}</script></body></html>"""


# ---------------------------------------------------------------------------
# 8) Main
# ---------------------------------------------------------------------------
class ScanError(RuntimeError):
    pass


def run_scan(quick: bool = False, top: int = 20, min_cap: float = 10e9, min_dollar_vol: float = 30e6,
             max_per_sector: int = 4, workers: int = 8, translate: bool = True,
             log=print, progress=None) -> dict:
    """สแกนหุ้นทั้งหมดแล้วคืนผล {scored, picks, meta, desc_th}  (ใช้ร่วมกันทั้ง CLI และเว็บแอป)"""
    step = progress or (lambda frac, msg: None)
    t0 = time.time()
    step(0.02, "โหลดรายชื่อหุ้น S&P 500 + Nasdaq-100 ...")
    log("[1/6] โหลดรายชื่อหุ้น S&P 500 + Nasdaq-100 ...")
    universe, src = get_universe(quick)
    log(f"  ได้ {len(universe)} ตัว ({src})")

    step(0.08, f"ดาวน์โหลดราคาย้อนหลัง 5 ปีของ {len(universe)} ตัว ...")
    log("[2/6] ดาวน์โหลดราคาย้อนหลัง 5 ปี + คำนวณ Technical ...")
    prices = download_prices(universe + ["SPY"])
    if "SPY" not in prices:
        raise ScanError("ดึงข้อมูลราคาจาก Yahoo Finance ไม่ได้ — อาจถูกจำกัดการเรียกชั่วคราว ลองใหม่ในอีกสักครู่")
    spy = prices.pop("SPY")["Close"]
    import backtest as _bt  # ตารางราคาไว้ใช้ทดสอบย้อนหลัง
    panels = _bt.panels_from_prices(prices, spy)
    tech = {}
    for t, df in prices.items():
        try:
            tech[t] = compute_technicals(df, spy)
        except Exception:
            pass
    d = pd.DataFrame.from_dict(tech, orient="index")
    d = d[(d["price"] >= 5) & (d["dollar_vol"] >= min_dollar_vol)]
    log(f"  คำนวณได้ {len(tech)} ตัว · ผ่านเกณฑ์สภาพคล่อง {len(d)} ตัว")

    step(0.35, f"ดึงงบการเงิน/ข้อมูลบริษัท {len(d)} ตัว (ขั้นตอนนี้นานที่สุด) ...")
    log("[3/6] ดึงข้อมูลพื้นฐาน / งบการเงิน ...")
    info = fetch_fundamentals(
        d.index.tolist(), workers,
        progress=lambda i, n: step(0.35 + 0.38 * i / n, f"ดึงงบการเงิน/ข้อมูลบริษัท {i}/{n} ตัว ..."))
    fdf = pd.DataFrame.from_dict(info, orient="index").reindex(d.index)
    for k in INFO_KEYS + ["_partial"]:
        if k not in fdf:
            fdf[k] = np.nan
    d = d.join(fdf)
    d["has_fund"] = (d[["returnOnEquity", "operatingMargins", "forwardPE", "revenueGrowth"]]
                     .apply(pd.to_numeric, errors="coerce").notna().any(axis=1))
    # เติมชื่อบริษัท/กลุ่มธุรกิจจาก Wikipedia ให้ตัวที่ Yahoo ไม่ให้ข้อมูล
    for k in ("longName", "sector", "industry"):
        wiki = pd.Series({t: WIKI_META.get(t, {}).get(k) for t in d.index}, dtype=object)
        d[k] = d[k].where(d[k].notna(), wiki)
    mc = pd.to_numeric(d["marketCap"], errors="coerce")
    n_fund, n_liq = int(d["has_fund"].sum()), len(d)
    log(f"  ได้งบการเงินครบ {n_fund}/{n_liq} ตัว · มีมูลค่าตลาด {int(mc.notna().sum())} ตัว")
    if mc.notna().mean() < 0.5:
        # ข้อมูลมูลค่าตลาดขาดเยอะ: ใช้มูลค่าซื้อขายต่อวันสูง (หุ้นใหญ่) เป็นเกณฑ์แทนสำหรับตัวที่ขาดข้อมูล
        keep = (mc >= min_cap) | (mc.isna() & (d["dollar_vol"] >= 150e6))
    else:
        keep = mc >= min_cap
    d = d[keep]
    d = d[d["quoteType"].fillna("EQUITY") == "EQUITY"]
    log(f"  ผ่านเกณฑ์ความน่าเชื่อถือ (มูลค่าตลาด >= {fmt_cap(min_cap)}) {len(d)} ตัว")
    if len(d) == 0:
        raise ScanError("ไม่มีหุ้นผ่านเกณฑ์ — ลองลดมูลค่าตลาดขั้นต่ำ หรือ Yahoo อาจจำกัดการเรียกข้อมูลชั่วคราว")

    now = dt.datetime.now(dt.timezone.utc)
    ets = pd.to_numeric(d["earningsTimestampStart"].fillna(d["earningsTimestamp"]), errors="coerce")
    d["earn_days"] = ((ets - now.timestamp()) / 86400).round()
    d["earn_date"] = ets.map(lambda x: dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d") if not pd.isna(x) else "")

    step(0.75, "ให้คะแนนรอบแรก + ดึงข่าวล่าสุด ...")
    log("[4/6] ให้คะแนนรอบแรก + ดึงข่าวล่าสุดของผู้เข้ารอบ ...")
    d["news_sent"] = 0.0
    scored = build_scores(d)
    shortlist = set()
    for key, col, *_ in HORIZONS:
        shortlist.update(scored.sort_values(col, ascending=False).head(top * 2 + 10).index)
    news = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_news, t): t for t in shortlist}
        for f in as_completed(futs):
            news[futs[f]] = f.result()
    d["news_sent"] = pd.Series({t: v["sentiment"] for t, v in news.items()}, dtype=float)
    d["news_sent"] = d["news_sent"].fillna(0.0)
    d["news_items"] = pd.Series({t: v["items"] for t, v in news.items()}, dtype=object).reindex(d.index)
    log(f"  ดึงข่าว {len(news)} บริษัท")

    step(0.88, "ให้คะแนนรอบสุดท้าย + คัด Top ...")
    log("[5/6] ให้คะแนนรอบสุดท้าย + คัด Top ...")
    scored = build_scores(d)
    # รายเดือน/รายปี ใช้งบการเงินเป็นส่วนสำคัญ: เลือกเฉพาะหุ้นที่ดึงงบได้ (ถ้ามีพอ)
    with_fund = scored[scored["has_fund"]]
    picks = {}
    for key, col, *_ in HORIZONS:
        base = with_fund if (key != "week" and len(with_fund) >= top * 3) else scored
        picks[key] = pick_top(base, col, top, max_per_sector, 2 if key == "buy" else None)

    desc_th = {}
    if translate:
        final = set().union(*[set(p.index) for p in picks.values()])
        texts = {t: scored.at[t, "longBusinessSummary"] for t in final
                 if isinstance(scored.at[t, "longBusinessSummary"], str)}
        if texts:
            step(0.92, f"แปลคำอธิบายธุรกิจเป็นภาษาไทย {len(texts)} บริษัท ...")
            log(f"  แปลคำอธิบายธุรกิจเป็นภาษาไทย {len(texts)} บริษัท ...")
            desc_th = translate_th(texts)

    spy_sma200 = spy.rolling(200).mean().iloc[-1]
    regime = ("ขาขึ้น (S&P 500 อยู่เหนือ SMA200)" if spy.iloc[-1] > spy_sma200
              else "อ่อนตัว (S&P 500 ต่ำกว่า SMA200) — ระวังความเสี่ยง")
    meta = {
        "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%Y-%m-%d %H:%M"), "top": top,
        "universe_src": src, "n_scanned": len(universe), "n_pass": len(scored), "min_cap": min_cap,
        "spy_price": float(spy.iloc[-1]), "spy_1d": float(spy.iloc[-1] / spy.iloc[-2] - 1),
        "spy_1m": float(spy.iloc[-1] / spy.iloc[-22] - 1), "spy_1y": float(spy.iloc[-1] / spy.iloc[-253] - 1),
        "regime": regime, "seconds": round(time.time() - t0),
        "n_fund": int(scored["has_fund"].sum()),
    }
    step(1.0, "เสร็จแล้ว")
    return {"scored": scored, "picks": picks, "meta": meta, "desc_th": desc_th, "panels": panels, "spy": spy}


EXPORT_COLS = ["longName", "sector", "industry", "price", "marketCap", "score_week", "score_month", "score_year",
               "s_tech_short", "s_trend", "s_quality", "s_growth", "s_value", "s_safety", "s_analyst", "s_news",
               "rsi", "adx", "ret_1m", "ret_1y", "rs_6m", "forwardPE", "returnOnEquity", "revenueGrowth",
               "earningsGrowth", "fcf_yield", "debtToEquity", "div_yield", "upside", "recommendationKey"]


def main():
    ap = argparse.ArgumentParser(description="US Stock Picker — Top 20 หุ้น USA รายสัปดาห์/เดือน/ปี")
    ap.add_argument("--top", type=int, default=20, help="จำนวนหุ้นที่คัดต่อช่วงเวลา (default 20)")
    ap.add_argument("--min-cap", type=float, default=10e9, help="มูลค่าตลาดขั้นต่ำ USD (default 10e9 = $10B)")
    ap.add_argument("--min-dollar-vol", type=float, default=30e6, help="มูลค่าซื้อขายเฉลี่ยต่อวันขั้นต่ำ (default $30M)")
    ap.add_argument("--max-per-sector", type=int, default=4, help="จำกัดหุ้นต่อกลุ่มธุรกิจ เพื่อกระจายความเสี่ยง (default 4)")
    ap.add_argument("--workers", type=int, default=8, help="จำนวน thread ดึงข้อมูล (default 8)")
    ap.add_argument("--quick", action="store_true", help="สแกนเฉพาะ ~200 บริษัทใหญ่ (เร็วกว่า)")
    ap.add_argument("--no-translate", action="store_true", help="ไม่แปลคำอธิบายธุรกิจเป็นภาษาไทย")
    ap.add_argument("--no-open", action="store_true", help="ไม่ต้องเปิดเบราว์เซอร์อัตโนมัติ")
    ap.add_argument("--out", default=None, help="ตำแหน่งไฟล์รายงาน HTML")
    ap.add_argument("--backtest", action="store_true", help="ทดสอบย้อนหลังสูตรคะแนน Technical แล้วพิมพ์ผล")
    ap.add_argument("--cost-bps", type=float, default=10, help="ค่าธรรมเนียม+สเปรดต่อการซื้อหรือขาย (basis points, default 10 = 0.10%%)")
    args = ap.parse_args()

    try:
        res = run_scan(args.quick, args.top, args.min_cap, args.min_dollar_vol, args.max_per_sector,
                       args.workers, not args.no_translate)
    except ScanError as e:
        sys.exit(str(e))
    scored, picks, meta, desc_th = res["scored"], res["picks"], res["meta"], res["desc_th"]

    print("[6/6] สร้างรายงาน ...")
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out) if args.out else REPORT_DIR / f"us_stock_picks_{stamp}.html"
    out.write_text(build_report(scored, picks, meta, desc_th), encoding="utf-8")
    scored[[c for c in EXPORT_COLS if c in scored]].sort_values("score_year", ascending=False) \
        .to_csv(out.with_suffix(".csv"), encoding="utf-8-sig", float_format="%.4f")

    print(f"\nเสร็จแล้วใน {meta['seconds']} วินาที")
    for key, col, label, _ in HORIZONS:
        print(f"  {label:<22}: " + ", ".join(picks[key].index[:20]))
    print(f"\nรายงาน: {out}\nคะแนนทั้งหมด (CSV): {out.with_suffix('.csv')}")
    if args.backtest:
        import backtest as bt
        print("\n===== ทดสอบย้อนหลัง (เฉพาะคะแนน Technical) =====")
        for key in bt.STRATS:
            r = bt.run_backtest(res["panels"], res["spy"], key, args.top, args.cost_bps, args.min_dollar_vol)
            st_ = r["stats"]
            print(f"\n[{r['label']}] {r['desc']}")
            print(f"  ช่วงทดสอบ {st_['start']:%Y-%m-%d} ถึง {st_['end']:%Y-%m-%d} · {st_['n']} รอบ · "
                  f"หุ้นที่ผ่านเกณฑ์เฉลี่ย {st_['universe']:.0f} ตัว")
            m = r["metrics"].copy().astype(object)
            for idx_ in m.index:
                m.loc[idx_] = [f"{x:.2f}" if idx_ == "Sharpe Ratio" else f"{x:+.1%}" for x in r["metrics"].loc[idx_]]
            print(m.to_string())
            print(f"  ชนะ SPY {st_['win_spy']:.0%} ของรอบ · ชนะ equal-weight {st_['win_ew']:.0%} · "
                  f"IC เฉลี่ย {st_['ic_mean']:+.3f} (t={st_['ic_t']:.1f}) · Q5-Q1 {st_['q_spread']:+.1%}/ปี")
            print("  สรุป:", bt.verdict(r)[1])
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
