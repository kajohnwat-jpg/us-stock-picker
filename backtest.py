# -*- coding: utf-8 -*-
"""
Backtest — ทดสอบย้อนหลังสูตรคะแนน Technical ของ US Stock Picker
=================================================================
หลักการ (กันการ "แอบดูอนาคต"):
  • ทุกรอบปรับพอร์ต ณ วันที่ t ใช้ข้อมูลราคาถึงสิ้นวัน t เท่านั้น
  • ซื้อที่ราคาปิดของวันถัดไป (t+1) และถือจนถึงรอบถัดไป
  • หักค่าธรรมเนียม/สเปรดตามสัดส่วนหุ้นที่เปลี่ยน (turnover)
  • เทียบกับ SPY และ "ซื้อทุกตัวเท่ากัน" (equal-weight) จากกลุ่มหุ้นเดียวกัน

ข้อจำกัดที่ต้องรู้:
  • ทดสอบได้เฉพาะคะแนน Technical (Yahoo ฟรีไม่มีงบการเงินย้อนหลังรายวัน)
  • ใช้รายชื่อหุ้น "ปัจจุบัน" ย้อนไปในอดีต (survivorship bias) → ผลจะดูดีเกินจริง
    จึงควรดูว่า "ชนะ equal-weight" หรือไม่ มากกว่าดูว่าชนะ SPY
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATS = {
    "week": {"label": "รายสัปดาห์", "hold": 5, "w_short": 0.75, "w_trend": 0.25,
             "desc": "ปรับพอร์ตทุก 5 วันทำการ · คะแนน = Technical สั้น 75% + แนวโน้ม 25%"},
    "month": {"label": "รายเดือน", "hold": 21, "w_short": 0.50, "w_trend": 0.50,
              "desc": "ปรับพอร์ตทุก 21 วันทำการ · คะแนน = Technical สั้น 50% + แนวโน้ม 50%"},
    "year": {"label": "ระยะยาว (แนวโน้ม)", "hold": 63, "w_short": 0.0, "w_trend": 1.0,
             "desc": "ปรับพอร์ตทุก 3 เดือน · คะแนน = แนวโน้มระยะยาว 100% (ส่วน Fundamental ทดสอบย้อนหลังไม่ได้)"},
    # ---- สูตรทางเลือก (กำหนดไว้ล่วงหน้าจากงานวิจัย ไม่ได้ปรับให้เข้ากับผลทดสอบ) ----
    "mom": {"label": "ทางเลือก A: โมเมนตัม 12 เดือน", "hold": 21, "kind": "mom", "buffer": True,
            "desc": "เลือกหุ้นที่ราคาขึ้นมากสุดในรอบ 12 เดือน (ไม่นับเดือนล่าสุด) และราคาอยู่เหนือ SMA200 · "
                    "ปรับพอร์ตทุกเดือน · ถือตัวเดิมต่อถ้ายังอยู่ใน Top 2 เท่า (ลดการซื้อขาย)"},
    "mom_lv": {"label": "ทางเลือก B: โมเมนตัม + ผันผวนต่ำ", "hold": 21, "kind": "mom_lv", "buffer": True,
               "desc": "คะแนน = โมเมนตัม 12 เดือน 60% + ความผันผวนต่ำ 40% · เฉพาะหุ้นที่อยู่เหนือ SMA200 · "
                       "ปรับพอร์ตทุกเดือน · ถือตัวเดิมต่อถ้ายังอยู่ใน Top 2 เท่า"},
    "big_mom": {"label": "ทางเลือก C: หุ้นใหญ่ 100 ตัว + โมเมนตัม", "hold": 21, "kind": "big_mom", "buffer": True,
                "desc": "คัดเฉพาะ 100 ตัวที่ซื้อขายมูลค่าสูงสุด (หุ้นยักษ์) แล้วเลือกตามโมเมนตัม 12 เดือน · "
                        "เฉพาะหุ้นที่อยู่เหนือ SMA200 · ปรับพอร์ตทุกเดือน · ถือตัวเดิมต่อถ้ายังอยู่ใน Top 2 เท่า"},
}
WARMUP = 273  # ต้องมีข้อมูล ~13 เดือนก่อนเริ่มคำนวณโมเมนตัม 12-1 และ SMA200


# ---------------------------------------------------------------------------
# เตรียมข้อมูล
# ---------------------------------------------------------------------------
def panels_from_prices(prices: dict[str, pd.DataFrame], spy: pd.Series) -> dict[str, pd.DataFrame]:
    """แปลง dict ราคารายตัว -> ตาราง (วันที่ x หุ้น) โดยใช้ปฏิทินวันทำการของ SPY"""
    idx = spy.index
    out = {}
    for col, name in [("Close", "close"), ("High", "high"), ("Low", "low"), ("Volume", "volume")]:
        out[name] = pd.DataFrame({t: df[col] for t, df in prices.items()}).reindex(idx).astype(float)
    return out


def _pct(df: pd.DataFrame, elig: pd.DataFrame, higher: bool = True) -> pd.DataFrame:
    r = df.where(elig).rank(axis=1, pct=True, ascending=higher) * 100
    return r.fillna(40).where(elig)


def _wavg(parts, weights):
    w = np.asarray(weights, dtype=float)
    tot = parts[0] * w[0]
    for p, wi in zip(parts[1:], w[1:]):
        tot = tot + p * wi
    return tot / w.sum()


def compute_scores(P: dict, spy: pd.Series, dates: pd.DatetimeIndex, min_dollar_vol: float = 30e6):
    """คำนวณคะแนน Technical สั้น / แนวโน้ม ณ วันที่ปรับพอร์ต — สูตรเดียวกับ build_scores()"""
    c, h, l, v = P["close"], P["high"], P["low"], P["volume"]
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()

    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    cross_up = (hist > 0) & (hist.rolling(5).min() <= 0)

    upm, dnm = h.diff(), -l.diff()
    plus_dm = upm.where((upm > dnm) & (upm > 0), 0.0)
    minus_dm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    tr = pd.DataFrame(np.fmax(np.fmax((h - l).values, (h - c.shift()).abs().values),
                              (l - c.shift()).abs().values), index=c.index, columns=c.columns)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False).mean()

    ret = c.pct_change()
    upv = v.where(ret > 0).rolling(20, min_periods=1).mean()
    dnv = v.where(ret <= 0).rolling(20, min_periods=1).mean()
    updown = upv / dnv
    dollar_vol = (c * v).rolling(20).mean()

    spy_a = spy.reindex(c.index)
    rs = lambda n: (c / c.shift(n)).div(spy_a / spy_a.shift(n), axis=0) - 1
    mom_12_1 = c.shift(21) / c.shift(252) - 1
    from_hi = c / c.rolling(252).max() - 1
    dd = (c / c.rolling(252, min_periods=1).max() - 1).rolling(252).min()

    # ---- ตัดเฉพาะวันที่ปรับพอร์ต ----
    at = lambda x: x.loc[dates]
    C, S20, S50, S200 = at(c), at(sma20), at(sma50), at(sma200)
    elig = C.notna() & S200.notna() & at(mom_12_1).notna() & (C >= 5) & (at(dollar_vol) >= min_dollar_vol)

    above200, above50, above20 = C > S200, C > S50, C > S20
    trend_short = (above20 * 35.0 + (S20 > S50) * 35.0 + (at(sma20 / sma20.shift(5) - 1) > 0) * 30.0)
    H, Hp = at(hist), at(hist.shift(3))
    macd_sc = (H > 0) * 50.0 + (H > Hp) * 30.0 + (at(macd) > 0) * 10.0 + at(cross_up) * 10.0
    R = at(rsi)
    rsi_sc = pd.DataFrame(np.select(
        [R.isna(), (R >= 50) & (R <= 65), ((R >= 45) & (R < 50)) | ((R > 65) & (R <= 70)), (R >= 40) & (R < 45),
         (R > 70) & (R <= 75), R > 75, (R >= 30) & (R < 40)],
        [50, 100, 80, 65, 50, 20, np.where(above200, 55, 35)],
        default=np.where(above200, 60, 15)), index=R.index, columns=R.columns)
    A, PD, MD = at(adx), at(pdi), at(mdi)
    adx_sc = pd.DataFrame(np.where(PD > MD, np.clip(A * 2.5, 30, 100), np.clip(40 - A, 0, 40)),
                          index=A.index, columns=A.columns)
    ext = C / S20 - 1
    ext_sc = pd.DataFrame(np.where(ext > 0.10, 20, np.where(ext > 0.06, 55, np.where(ext < -0.08, 35, 90))),
                          index=C.index, columns=C.columns)
    tech_short = _wavg([trend_short, macd_sc, rsi_sc, adx_sc, ext_sc, _pct(at(updown), elig),
                        _pct(at(c / c.shift(5) - 1), elig), _pct(at(rs(21)), elig)],
                       [15, 15, 15, 10, 10, 10, 10, 15])

    trend_long = (above200 * 40.0 + (S50 > S200) * 30.0 + (at(sma200 / sma200.shift(20)) > 1) * 20.0
                  + above50 * 10.0)
    trend = _wavg([trend_long, _pct(at(mom_12_1), elig), _pct(at(rs(126)), elig), _pct(at(rs(63)), elig),
                   _pct(at(from_hi), elig), _pct(at(dd), elig)], [30, 20, 15, 10, 15, 10])
    extra = {"mom": at(mom_12_1).where(elig), "vol": at(ret.rolling(252).std()).where(elig),
             "dv60": at((c * v).rolling(60).mean()).where(elig), "above200": above200}
    return tech_short.where(elig), trend.where(elig), elig, extra


# ---------------------------------------------------------------------------
# จำลองพอร์ต
# ---------------------------------------------------------------------------
def _metrics(r: pd.Series, ppy: float) -> dict:
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / ppy
    total = eq.iloc[-1] - 1
    sd = r.std()
    return {
        "ผลตอบแทนรวม": total,
        "ผลตอบแทนเฉลี่ยต่อปี (CAGR)": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan,
        "ความผันผวนต่อปี": sd * np.sqrt(ppy),
        "Sharpe Ratio": (r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan,
        "ขาดทุนสะสมหนักสุด (Max DD)": (eq / eq.cummax().clip(lower=1) - 1).min(),
    }


def run_backtest(P: dict, spy: pd.Series, key: str, top: int = 20, cost_bps: float = 10,
                 min_dollar_vol: float = 30e6) -> dict:
    s = STRATS[key]
    hold = s["hold"]
    close = P["close"]
    n = len(close.index)
    reb = list(range(WARMUP, n - hold - 1, hold))
    if len(reb) < 4:
        raise ValueError("ข้อมูลย้อนหลังไม่พอสำหรับทดสอบ (ต้องมีอย่างน้อย ~1.5 ปี)")
    dates = close.index[reb]
    short, trend, elig, ex = compute_scores(P, spy, dates, min_dollar_vol)
    kind = s.get("kind", "orig")
    up = elig & ex["above200"]
    if kind == "mom":
        score = ex["mom"].where(up)
    elif kind == "mom_lv":
        score = 0.6 * _pct(ex["mom"], up) + 0.4 * _pct(ex["vol"], up, higher=False)
    elif kind == "big_mom":
        big = ex["dv60"].rank(axis=1, ascending=False) <= 100
        score = ex["mom"].where(up & big)
    else:
        score = short * s["w_short"] + trend * s["w_trend"]

    rows, prev = [], set()
    for k, i in enumerate(reb):
        if int(elig.iloc[k].sum()) < top * 2:
            continue
        sc = score.iloc[k].dropna()
        entry, exit_ = i + 1, i + 1 + hold
        fwd = close.iloc[exit_] / close.iloc[entry] - 1
        fwd_el = fwd.reindex(sc.index)
        if s.get("buffer") and prev:
            rk = sc.rank(ascending=False)
            keep = [t for t in prev if t in rk.index and rk[t] <= top * 2]
            fill = [t for t in sc.sort_values(ascending=False).index if t not in keep][: top - len(keep)]
            picks = pd.Index(keep + fill)
        else:
            picks = sc.nlargest(top).index
        # ถ้าหุ้นผ่านเกณฑ์น้อยกว่า top ส่วนที่เหลือถือเป็นเงินสด (ผลตอบแทน 0)
        gross = fwd_el[picks].fillna(0).sum() / top
        turnover = len(set(picks) - prev) / top
        cost = turnover * (1 if not prev else 2) * cost_bps / 1e4
        valid = fwd_el.dropna()
        ew = fwd[elig.iloc[k]].dropna().mean()
        if len(valid) >= 10:
            q = pd.qcut(valid.index.map(sc).to_series(index=valid.index).rank(method="first"), 5,
                        labels=[1, 2, 3, 4, 5])
            qmean = valid.groupby(q, observed=False).mean()
            ic = sc[valid.index].rank().corr(valid.rank())
        else:
            qmean, ic = pd.Series(dtype=float), np.nan
        rows.append({
            "เริ่ม": close.index[entry], "สิ้นสุด": close.index[exit_],
            "กลยุทธ์": gross - cost, "ก่อนหักค่าธรรมเนียม": gross,
            "SPY": spy.iloc[exit_] / spy.iloc[entry] - 1, "ซื้อทุกตัวเท่ากัน": ew,
            "turnover": turnover, "IC": ic, "จำนวนหุ้นที่ผ่านเกณฑ์": len(sc),
            **{f"Q{j}": qmean.get(j, np.nan) for j in range(1, 6)},
            "หุ้นที่เลือก": ", ".join(picks) if len(picks) else "(เงินสด)",
        })
        prev = set(picks)

    per = pd.DataFrame(rows).set_index("สิ้นสุด")
    ppy = 252 / hold
    cols = ["กลยุทธ์", "SPY", "ซื้อทุกตัวเท่ากัน"]
    metrics = pd.DataFrame({c: _metrics(per[c], ppy) for c in cols})
    excess_spy = per["กลยุทธ์"] - per["SPY"]
    excess_ew = per["กลยุทธ์"] - per["ซื้อทุกตัวเท่ากัน"]
    first = per["เริ่ม"].iloc[0]
    equity = (1 + per[cols]).cumprod() * 100
    equity = pd.concat([pd.DataFrame({c: [100.0] for c in cols}, index=[first]), equity])
    yearly = (1 + per[cols]).groupby(per.index.year).prod() - 1
    yearly.index.name = "ปี"
    quint = per[[f"Q{j}" for j in range(1, 6)]].mean()
    q_ann = (1 + quint) ** ppy - 1
    return {
        "key": key, "label": s["label"], "desc": s["desc"], "hold": hold, "periods": per,
        "equity": equity, "metrics": metrics, "yearly": yearly,
        "quintile": pd.DataFrame({"เฉลี่ยต่อรอบ": quint, "คิดเป็นต่อปี": q_ann}),
        "stats": {
            "start": first, "end": per.index[-1], "n": len(per),
            "win_spy": float((excess_spy > 0).mean()), "win_ew": float((excess_ew > 0).mean()),
            "ic_mean": float(per["IC"].mean()),
            "ic_t": float(per["IC"].mean() / per["IC"].std() * np.sqrt(len(per))) if per["IC"].std() > 0 else np.nan,
            "turnover": float(per["turnover"].iloc[1:].mean()) if len(per) > 1 else np.nan,
            "universe": float(per["จำนวนหุ้นที่ผ่านเกณฑ์"].mean()),
            "q_spread": float(q_ann.iloc[-1] - q_ann.iloc[0]),
            "monotonic": bool(quint.is_monotonic_increasing),
        },
    }


def verdict(bt: dict) -> tuple[str, str]:
    """สรุปผลเป็นภาษาไทยแบบตรงไปตรงมา -> (ระดับ, ข้อความ)"""
    st_ = bt["stats"]
    m = bt["metrics"]
    cagr_s = m.at["ผลตอบแทนเฉลี่ยต่อปี (CAGR)", "กลยุทธ์"]
    cagr_ew = m.at["ผลตอบแทนเฉลี่ยต่อปี (CAGR)", "ซื้อทุกตัวเท่ากัน"]
    beat = cagr_s - cagr_ew
    t = st_["ic_t"]
    sig = (f"การจัดอันดับมีนัยสำคัญทางสถิติ (t-stat ของ IC = {t:.1f})" if t > 2
           else f"การจัดอันดับยังไม่มีนัยสำคัญทางสถิติ (t-stat ของ IC = {t:.1f}, ควรเกิน 2)")
    if beat > 0.02 and t > 2 and st_["q_spread"] > 0:
        return "good", (f"สัญญาณมีประสิทธิภาพ: ชนะการซื้อทุกตัวเท่ากัน {beat * 100:+.1f}% ต่อปีหลังหักค่าธรรมเนียม "
                        f"และหุ้นคะแนนสูงทำผลงานดีกว่าหุ้นคะแนนต่ำ — {sig}")
    if (beat > 0 and t > 1) or (beat > -0.01 and t > 2 and st_["q_spread"] > 0):
        return "mixed", (f"ช่วยได้บ้างแต่ไม่ชัดเจน: เทียบการซื้อทุกตัวเท่ากัน {beat * 100:+.1f}% ต่อปีหลังหักค่าธรรมเนียม — {sig}")
    return "bad", (f"ยังไม่พบหลักฐานว่าสัญญาณนี้ช่วยเพิ่มผลตอบแทน: เทียบการซื้อทุกตัวเท่ากัน {beat * 100:+.1f}% ต่อปี — {sig} "
                   "จึงไม่ควรใช้สัญญาณช่วงเวลานี้เป็นเหตุผลหลักในการซื้อ")
