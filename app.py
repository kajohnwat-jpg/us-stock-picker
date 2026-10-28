# -*- coding: utf-8 -*-
"""
US Stock Picker — เว็บแอป (Streamlit)
รัน:  streamlit run app.py        หรือ deploy ฟรีบน Streamlit Community Cloud
"""
import datetime as dt
import threading
import time

import numpy as np
import pandas as pd
import streamlit as st

import backtest as bt
import us_stock_picker as sp

st.set_page_config(page_title="US Stock Picker", page_icon="📈", layout="wide")

CACHE_HOURS = 4  # ผลสแกนใช้ร่วมกันทุกคนที่เปิดเว็บ และสแกนใหม่อัตโนมัติเมื่อเก่ากว่านี้


# ---------------------------------------------------------------------------
# ที่เก็บผลสแกน (ใช้ร่วมกันทุก session บนเซิร์ฟเวอร์)
# ---------------------------------------------------------------------------
@st.cache_resource
def store():
    return {"lock": threading.Lock(), "results": {}, "jobs": {}}


def md(x) -> str:
    """กัน Markdown ตีความ $ เป็นสูตรคณิต"""
    return str(x).replace("$", "\\$")


def _scan_job(key: tuple, job: dict):
    """รันสแกนใน background thread — ไม่หยุดแม้ผู้ใช้กดปุ่ม/เปลี่ยนหน้า"""
    quick, min_cap, max_sector, translate = key
    try:
        res = sp.run_scan(quick=quick, top=20, min_cap=min_cap, max_per_sector=max_sector, workers=4,
                          translate=translate, progress=lambda f, m: job.update(frac=f, msg=m))
        res["ts"] = time.time()
        store()["results"][key] = res
    except Exception as e:  # noqa: BLE001
        job["error"] = str(e)
    finally:
        job["done"] = True


def start_job(key: tuple):
    s = store()
    with s["lock"]:
        job = s["jobs"].get(key)
        if job and not job.get("done"):
            return job
        job = {"frac": 0.0, "msg": "เริ่มต้น ...", "done": False, "error": None, "started": time.time()}
        s["jobs"][key] = job
        threading.Thread(target=_scan_job, args=(key, job), daemon=True).start()
        return job


def get_results(key: tuple, force: bool):
    s = store()
    cached = s["results"].get(key)
    stale = (not cached) or (time.time() - cached["ts"] > CACHE_HOURS * 3600)
    if force and cached and time.time() - cached["ts"] < 120:
        force = False  # เพิ่งสแกนไปไม่ถึง 2 นาที
    job = s["jobs"].get(key)
    running = bool(job and not job.get("done"))
    if (stale or force) and not running:
        if not (job and job.get("error") and time.time() - job.get("started", 0) < 60 and not force):
            job = start_job(key)
            running = True
    if cached:
        if running:
            st.toast(f"กำลังอัปเดตข้อมูลใหม่เบื้องหลัง ({job['frac'] * 100:.0f}%) — รีเฟรชหน้าอีกครั้งในไม่กี่นาที")
        return cached
    # ยังไม่มีผลเลย: แสดงความคืบหน้าแล้วรีเฟรชเอง
    if job and job.get("error") and not running:
        st.error(f"สแกนไม่สำเร็จ: {job['error']}")
        if st.button("ลองใหม่"):
            start_job(key)
            st.rerun()
        st.stop()
    st.title("📈 US Stock Picker")
    st.info("กำลังสแกนหุ้นด้วยข้อมูลล่าสุดจาก Yahoo Finance — ครั้งแรกใช้เวลาประมาณ 3–8 นาที "
            f"หลังจากนั้นทุกคนจะเห็นผลทันที และระบบจะสแกนใหม่อัตโนมัติทุก {CACHE_HOURS} ชั่วโมง "
            "(ปิดหน้านี้แล้วกลับมาทีหลังได้ การสแกนยังทำงานต่อ)")
    st.progress(min(max(job["frac"], 0.0), 1.0), text=f"{job['msg']}  ·  ผ่านไป {int(time.time() - job['started'])} วินาที")
    time.sleep(3)
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    universe = st.radio("กลุ่มหุ้นที่สแกน", ["S&P 500 + Nasdaq-100 (~520 ตัว)", "บริษัทใหญ่ ~200 ตัว (เร็วกว่า)"])
    min_cap_b = st.select_slider("มูลค่าตลาดขั้นต่ำ (พันล้าน USD)", options=[2, 5, 10, 20, 50, 100], value=10)
    max_sector = st.slider("จำกัดหุ้นต่อกลุ่มธุรกิจ (กระจายความเสี่ยง)", 2, 10, 4)
    translate = st.toggle("แปลคำอธิบายธุรกิจเป็นภาษาไทย", value=True)
    force = st.button("🔄 สแกนใหม่ตอนนี้", width="stretch")
    st.caption(f"ผลสแกนเก็บไว้ {CACHE_HOURS} ชั่วโมงแล้วอัปเดตอัตโนมัติ · "
               "ราคา Yahoo ดีเลย์ประมาณ 15 นาทีระหว่างตลาดเปิด (ตลาด US เปิด ~20:30–03:00 น. เวลาไทย)")

key = (universe.startswith("บริษัทใหญ่"), float(min_cap_b) * 1e9, int(max_sector), bool(translate))
res = get_results(key, force)
scored, picks, meta, desc_th = res["scored"], res["picks"], res["meta"], res["desc_th"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 US Stock Picker — Top 20 หุ้น USA")
age_min = int((time.time() - res["ts"]) / 60)
st.caption(f"อัปเดตล่าสุด {meta['date']} น. (เวลาไทย, {age_min} นาทีที่แล้ว) · แหล่งข้อมูล Yahoo Finance · "
           f"สแกน {meta['n_scanned']} ตัว ผ่านเกณฑ์ความน่าเชื่อถือ {meta['n_pass']} ตัว "
           f"(มูลค่าตลาด ≥ {sp.fmt_cap(meta['min_cap'])}) · {md(meta['universe_src'])}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("S&P 500 (SPY)", f"${meta.get('spy_price', 0):,.2f}", sp.fmt_pct(meta.get("spy_1d"), 2))
c2.metric("SPY 1 เดือน", sp.fmt_pct(meta["spy_1m"]))
c3.metric("SPY 1 ปี", sp.fmt_pct(meta["spy_1y"]))
c4.metric("ภาวะตลาด", "ขาขึ้น" if meta["regime"].startswith("ขาขึ้น") else "อ่อนตัว",
          meta["regime"].split(" ", 1)[1] if " " in meta["regime"] else "", delta_color="off")


# ---------------------------------------------------------------------------
# ส่วนแสดงหุ้น
# ---------------------------------------------------------------------------
def kv_table(rows):
    return pd.DataFrame(rows, columns=["รายการ", "ค่า"])


def show_stock(r: pd.Series, horizon: str, rank: int, col: str):
    t = r.name
    name = sp.sv(r.get("longName")) or sp.sv(r.get("shortName")) or t
    sec = sp.sv(r.get("sector"), "-")
    label = (f"**{rank}. {t}** — {name}  ·  ${r['price']:,.2f} ({sp.fmt_pct(r['chg_1d'], 2)})  ·  "
             f"คะแนน {r[col]:.0f}")
    with st.expander(md(label)):
        st.caption(md(f"{sp.SECTOR_TH.get(sec, sec)} · {sp.sv(r.get('industry'))} · "
                      f"มูลค่าตลาด {sp.fmt_cap(r.get('marketCap'))}"))
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("##### 🏢 ธุรกิจของบริษัท")
            summary = sp.sv(r.get("longBusinessSummary"))
            if desc_th.get(t):
                st.write(desc_th[t])
                if summary:
                    with st.popover("ภาษาอังกฤษ (ต้นฉบับ)"):
                        st.write(summary)
            elif summary:
                st.write(summary)
            else:
                st.caption("ไม่มีข้อมูลคำอธิบายธุรกิจ")
            emp = sp.sv(r.get("fullTimeEmployees"))
            extra = []
            if sp.sv(r.get("country")):
                extra.append(str(r["country"]))
            if emp:
                extra.append(f"พนักงาน {int(emp):,} คน")
            if extra:
                st.caption(" · ".join(extra))
        with right:
            st.markdown("##### 📊 กราฟราคา 1 ปี")
            dates = r.get("spark_dates")
            if isinstance(dates, list) and dates:
                ch = pd.DataFrame({"ราคา": r["spark"], "SMA50": r["spark50"], "SMA200": r["spark200"]},
                                  index=pd.to_datetime(dates))
                st.line_chart(ch, color=["#2563eb", "#f59e0b", "#8b5cf6"], height=230)
            b1, b2, b3 = st.columns(3)
            b1.link_button("TradingView", f"https://www.tradingview.com/chart/?symbol={t.replace('-', '.')}",
                           width="stretch")
            b2.link_button("Yahoo", f"https://finance.yahoo.com/quote/{t}", width="stretch")
            b3.link_button("Finviz", f"https://finviz.com/quote.ashx?t={t}", width="stretch")

        good, bad = sp.reasons(r, horizon)
        g, b = st.columns(2)
        with g:
            st.markdown("##### ✅ จุดเด่น")
            st.markdown("\n".join(f"- {md(x)}" for x in good) or "-")
        with b:
            st.markdown("##### ⚠️ ข้อควรระวัง")
            st.markdown("\n".join(f"- {md(x)}" for x in bad) or "- ไม่พบจุดเสี่ยงเด่นชัด")

        st.markdown("##### คะแนนย่อย (0–100)")
        subs = [("Technical สั้น", "s_tech_short"), ("แนวโน้ม", "s_trend"), ("คุณภาพ", "s_quality"),
                ("การเติบโต", "s_growth"), ("ความคุ้มค่า", "s_value"), ("ความมั่นคง", "s_safety"),
                ("นักวิเคราะห์", "s_analyst"), ("ข่าว", "s_news")]
        cols = st.columns(4)
        for i, (lbl, c) in enumerate(subs):
            v = float(r[c]) if not pd.isna(r[c]) else 0.0
            cols[i % 4].progress(min(max(v / 100, 0.0), 1.0), text=f"{lbl}: {v:.0f}")

        rk = str(sp.sv(r.get("recommendationKey"), "none"))
        earn = "-"
        if not pd.isna(r.get("earn_days")) and sp.sv(r.get("earn_date")):
            earn = f"{r['earn_date']} ({int(r['earn_days'])} วัน)" if r["earn_days"] >= 0 else r["earn_date"]
        fund = kv_table([
            ("มูลค่าตลาด", sp.fmt_cap(r.get("marketCap"))),
            ("P/E ย้อนหลัง / คาดการณ์", f"{sp.fmt_num(r.get('trailingPE'), 1)} / {sp.fmt_num(r.get('forwardPE'), 1)}"),
            ("PEG", sp.fmt_num(sp.sv(r.get("trailingPegRatio"), None) or r.get("pegRatio"), 2)),
            ("P/B · EV/EBITDA", f"{sp.fmt_num(r.get('priceToBook'), 1)} · {sp.fmt_num(r.get('enterpriseToEbitda'), 1)}"),
            ("ROE / ROA", f"{sp.fmt_pct(r.get('returnOnEquity'), 1, False)} / {sp.fmt_pct(r.get('returnOnAssets'), 1, False)}"),
            ("Gross / Operating / Net Margin", f"{sp.fmt_pct(r.get('grossMargins'), 0, False)} / "
             f"{sp.fmt_pct(r.get('operatingMargins'), 0, False)} / {sp.fmt_pct(r.get('profitMargins'), 0, False)}"),
            ("รายได้โต / กำไรโต (YoY)", f"{sp.fmt_pct(r.get('revenueGrowth'))} / {sp.fmt_pct(r.get('earningsGrowth'))}"),
            ("FCF Yield", sp.fmt_pct(r.get("fcf_yield"), 1, False)),
            ("หนี้สิน/ทุน (D/E) · Current Ratio", f"{sp.fmt_num(sp.sv(r.get('debtToEquity'), np.nan) / 100, 2)} · "
             f"{sp.fmt_num(r.get('currentRatio'), 2)}"),
            ("ปันผล · Payout", f"{sp.fmt_pct(r.get('div_yield'), 2, False)} · {sp.fmt_pct(r.get('payoutRatio'), 0, False)}"),
            ("Beta", sp.fmt_num(r.get("beta"), 2)),
            ("นักวิเคราะห์", f"{sp.REC_TH.get(rk.lower(), rk)} ({int(sp.sv(r.get('numberOfAnalystOpinions'), 0))} ราย)"),
            ("ราคาเป้าหมายเฉลี่ย", f"${sp.fmt_num(r.get('targetMeanPrice'))} ({sp.fmt_pct(r.get('upside'), 0)})"),
            ("ประกาศงบครั้งถัดไป", earn),
        ])
        tech = kv_table([
            ("ราคา", f"${sp.fmt_num(r['price'])} ({sp.fmt_pct(r['chg_1d'], 2)})"),
            ("SMA20 / 50 / 200", f"{sp.fmt_num(r['sma20'])} / {sp.fmt_num(r['sma50'])} / {sp.fmt_num(r['sma200'])}"),
            ("RSI(14)", sp.fmt_num(r["rsi"], 1)),
            ("MACD / Signal", f"{sp.fmt_num(r['macd'])} / {sp.fmt_num(r['macd_sig'])}"),
            ("ADX (+DI / -DI)", f"{sp.fmt_num(r['adx'], 0)} ({sp.fmt_num(r['pdi'], 0)} / {sp.fmt_num(r['mdi'], 0)})"),
            ("ATR(14)", f"${sp.fmt_num(r['atr'])} ({sp.fmt_pct(r['atr_pct'], 1, False)})"),
            ("ตำแหน่งใน Bollinger Band", sp.fmt_pct(r["bb_pos"], 0, False)),
            ("ห่างจุดสูงสุด 52 สัปดาห์", sp.fmt_pct(r["from_hi52"])),
            ("ผลตอบแทน 1W / 1M / 3M", f"{sp.fmt_pct(r['ret_1w'])} / {sp.fmt_pct(r['ret_1m'])} / {sp.fmt_pct(r['ret_3m'])}"),
            ("ผลตอบแทน 6M / 1Y / 3Y", f"{sp.fmt_pct(r['ret_6m'])} / {sp.fmt_pct(r['ret_1y'])} / {sp.fmt_pct(r['ret_3y'])}"),
            ("ชนะ S&P 500 (6 เดือน)", sp.fmt_pct(r["rs_6m"])),
            ("ร่วงหนักสุดใน 1 ปี", sp.fmt_pct(r["max_dd_1y"])),
            ("ความผันผวนต่อปี", sp.fmt_pct(r["vol_1y"], 0, False)),
            ("แนวรับใกล้ (SMA50 / SMA200)", f"${sp.fmt_num(r['sma50'])} / ${sp.fmt_num(r['sma200'])}"),
        ])
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("##### 📑 Fundamental")
            st.dataframe(fund, hide_index=True, width="stretch")
        with f2:
            st.markdown("##### 📉 Technical")
            st.dataframe(tech, hide_index=True, width="stretch")

        st.markdown("##### 📰 ข่าวล่าสุด")
        items = r.get("news_items")
        if isinstance(items, list) and items:
            lines = []
            for it in items:
                s = it.get("sent", 0)
                tag = "🟢 บวก" if s > 0.2 else ("🔴 ลบ" if s < -0.2 else "⚪ กลาง")
                title = md(it["title"]).replace("[", "(").replace("]", ")")
                link = f"[{title}]({it['url']})" if it.get("url") else title
                lines.append(f"- {tag} · {link} — *{md(it.get('publisher') or '')} {it.get('when') or ''}*")
            st.markdown("\n".join(lines))
        else:
            st.caption("ไม่พบข่าวล่าสุด")


def summary_table(top: pd.DataFrame, col: str):
    rows = []
    for n, (t, r) in enumerate(top.iterrows(), 1):
        rows.append({
            "#": n, "หุ้น": t,
            "บริษัท": sp.sv(r.get("longName")) or sp.sv(r.get("shortName")) or t,
            "กลุ่ม": sp.SECTOR_TH.get(sp.sv(r.get("sector"), "-"), sp.sv(r.get("sector"), "-")),
            "ราคา ($)": r["price"], "วันนี้": r["chg_1d"] * 100, "1 ปี": r["ret_1y"] * 100,
            "คะแนน": r[col], "Technical": r["s_tech_short"], "คุณภาพ": r["s_quality"],
            "เติบโต": r["s_growth"], "ข่าว": r["s_news"],
        })
    df = pd.DataFrame(rows)
    pc = lambda lbl: st.column_config.ProgressColumn(lbl, min_value=0, max_value=100, format="%.0f")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "ราคา ($)": st.column_config.NumberColumn(format="%.2f"),
        "วันนี้": st.column_config.NumberColumn("วันนี้ %", format="%+.2f"),
        "1 ปี": st.column_config.NumberColumn("1 ปี %", format="%+.1f"),
        "คะแนน": pc("คะแนนรวม"), "Technical": pc("Technical"), "คุณภาพ": pc("คุณภาพ"),
        "เติบโต": pc("เติบโต"), "ข่าว": pc("ข่าว"),
    })


tab_labels = [f"📅 {label}" for _, _, label, _ in sp.HORIZONS] + ["🔎 หุ้นทั้งหมด", "🧪 ทดสอบย้อนหลัง"]
tabs = st.tabs(tab_labels)
for tab, (key_h, col, label, method) in zip(tabs[:3], sp.HORIZONS):
    with tab:
        st.info(f"**วิธีให้คะแนน{label}:** {md(method)}")
        top = picks[key_h]
        mix = top["sector"].fillna("Other").map(lambda s: sp.SECTOR_TH.get(s, s)).value_counts()
        st.caption("สัดส่วนกลุ่มธุรกิจ: " + " · ".join(f"{k} {v}" for k, v in mix.items()))
        summary_table(top, col)
        q = st.text_input("ค้นหาในรายการนี้", key=f"q_{key_h}", placeholder="เช่น AAPL, Microsoft, เทคโนโลยี")
        st.caption("กดที่ชื่อหุ้นเพื่อดูรายละเอียด ธุรกิจ กราฟ งบการเงิน และข่าว")
        for n, (t, r) in enumerate(top.iterrows(), 1):
            hay = f"{t} {sp.sv(r.get('longName'))} {sp.sv(r.get('sector'))} " \
                  f"{sp.SECTOR_TH.get(sp.sv(r.get('sector')), '')}".lower()
            if q and q.lower() not in hay:
                continue
            show_stock(r, key_h, n, col)

with tabs[3]:
    st.caption("คะแนนของหุ้นทุกตัวที่ผ่านเกณฑ์ — กดหัวคอลัมน์เพื่อเรียงลำดับ")
    allq = st.text_input("ค้นหาหุ้น", key="q_all", placeholder="เช่น NVDA หรือ Coca-Cola")
    cols = [c for c in sp.EXPORT_COLS if c in scored]
    full = scored[cols].copy()
    full.insert(0, "หุ้น", full.index)
    if allq:
        m = full["หุ้น"].str.lower().str.contains(allq.lower(), regex=False) | \
            full["longName"].fillna("").str.lower().str.contains(allq.lower(), regex=False)
        full = full[m]
    pc = lambda lbl: st.column_config.ProgressColumn(lbl, min_value=0, max_value=100, format="%.0f")
    st.dataframe(full.sort_values("score_year", ascending=False), hide_index=True, width="stretch",
                 height=600, column_config={
                     "longName": "บริษัท", "sector": "Sector", "industry": "Industry",
                     "price": st.column_config.NumberColumn("ราคา", format="%.2f"),
                     "marketCap": st.column_config.NumberColumn("มูลค่าตลาด", format="compact"),
                     "score_week": pc("สัปดาห์"), "score_month": pc("เดือน"), "score_year": pc("ปี"),
                 })
    st.download_button("⬇️ ดาวน์โหลดคะแนนทั้งหมด (CSV เปิดด้วย Excel)",
                       scored[cols].to_csv(float_format="%.4f").encode("utf-8-sig"),
                       file_name=f"us_stock_scores_{dt.date.today():%Y%m%d}.csv", mime="text/csv")



# ---------------------------------------------------------------------------
# แท็บทดสอบย้อนหลัง
# ---------------------------------------------------------------------------
def fmt_metrics(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy().astype(object)
    for i in m.index:
        out.loc[i] = [("-" if pd.isna(x) else f"{x:.2f}") if i == "Sharpe Ratio"
                      else ("-" if pd.isna(x) else f"{x * 100:+.1f}%") for x in m.loc[i]]
    return out


def get_backtest(key_s: str, top: int, cost: float):
    cache = store().setdefault("bt", {})
    k = (key, res["ts"], key_s, top, cost)
    if k not in cache:
        cache[k] = bt.run_backtest(res["panels"], res["spy"], key_s, top=top, cost_bps=cost)
    return cache[k]


with tabs[4]:
    if "panels" not in res:
        st.warning("ข้อมูลรอบนี้ยังไม่มีราคาย้อนหลังสำหรับทดสอบ — กด 🔄 สแกนใหม่ตอนนี้ ที่แถบซ้าย")
    else:
        st.info(
            "**ทดสอบย้อนหลังทำงานอย่างไร** — ย้อนกลับไปในอดีตทุกรอบ (สัปดาห์ / เดือน / 3 เดือน) "
            "ให้คะแนนหุ้นด้วยสูตร Technical เดียวกับหน้าหลัก โดยใช้ข้อมูลถึงวันนั้นเท่านั้น "
            "แล้วซื้อ Top หุ้นที่ราคาปิดวันถัดไป ถือจนถึงรอบถัดไป หักค่าธรรมเนียม แล้ววัดผลเทียบกับ "
            "**SPY** (ดัชนี S&P 500) และ **ซื้อทุกตัวเท่ากัน** (ซื้อหุ้นทุกตัวในกลุ่มเดียวกันเท่า ๆ กัน)")
        c1, c2 = st.columns(2)
        top_n = c1.select_slider("จำนวนหุ้นในพอร์ต", options=[10, 15, 20, 30, 50], value=20)
        cost = c2.select_slider("ค่าธรรมเนียม + สเปรด ต่อการซื้อหรือขาย", options=[0, 5, 10, 20, 30, 50], value=10,
                                format_func=lambda x: f"{x / 100:.2f}%")
        try:
            results = {k_: get_backtest(k_, top_n, cost) for k_ in bt.STRATS}
        except Exception as e:  # noqa: BLE001
            st.error(f"ทดสอบไม่สำเร็จ: {e}")
            st.stop()

        st.markdown("#### สรุปผลทั้ง 3 แบบ")
        icon = {"good": "🟢 มีประสิทธิภาพ", "mixed": "🟡 ช่วยได้เล็กน้อย", "bad": "🔴 ยังไม่พบหลักฐาน"}
        rows = []
        for k_, r_ in results.items():
            m_ = r_["metrics"]
            cg = m_.loc["ผลตอบแทนเฉลี่ยต่อปี (CAGR)"]
            rows.append({
                "แบบ": r_["label"], "ช่วงทดสอบ": f"{r_['stats']['start']:%Y-%m} ถึง {r_['stats']['end']:%Y-%m}",
                "กลยุทธ์ / ปี": f"{cg['กลยุทธ์'] * 100:+.1f}%", "ซื้อทุกตัวเท่ากัน / ปี": f"{cg['ซื้อทุกตัวเท่ากัน'] * 100:+.1f}%",
                "SPY / ปี": f"{cg['SPY'] * 100:+.1f}%", "ชนะ (ทุกตัวเท่ากัน) กี่ % ของรอบ": f"{r_['stats']['win_ew']:.0%}",
                "ผลสรุป": icon[bt.verdict(r_)[0]],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        pick = st.radio("ดูรายละเอียด", list(bt.STRATS), horizontal=True,
                        format_func=lambda k_: bt.STRATS[k_]["label"], index=1)
        r_ = results[pick]
        stt = r_["stats"]
        level, msg = bt.verdict(r_)
        {"good": st.success, "mixed": st.warning, "bad": st.error}[level](msg)
        st.caption(f"{md(r_['desc'])} · ทดสอบ {stt['n']} รอบ · หุ้นที่ผ่านเกณฑ์เฉลี่ย {stt['universe']:.0f} ตัว · "
                   f"เปลี่ยนหุ้นเฉลี่ย {stt['turnover']:.0%} ต่อรอบ")

        a1, a2 = st.columns([1.3, 1])
        with a1:
            st.markdown("##### มูลค่าพอร์ต (เริ่มต้น 100)")
            st.line_chart(r_["equity"], color=["#2563eb", "#9ca3af", "#f59e0b"], height=320)
        with a2:
            st.markdown("##### ตัวชี้วัด")
            st.dataframe(fmt_metrics(r_["metrics"]), width="stretch")
            m1, m2 = st.columns(2)
            m1.metric("ชนะ SPY", f"{stt['win_spy']:.0%} ของรอบ")
            m2.metric("ชนะ ซื้อทุกตัวเท่ากัน", f"{stt['win_ew']:.0%} ของรอบ")

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### ผลตอบแทนแยกตามกลุ่มคะแนน (ต่อปี)")
            q = r_["quintile"]["คิดเป็นต่อปี"] * 100
            q.index = ["Q1 คะแนนต่ำสุด", "Q2", "Q3", "Q4", "Q5 คะแนนสูงสุด"]
            st.bar_chart(q.rename("ผลตอบแทน % ต่อปี"), height=260)
            st.caption("ถ้าสูตรใช้ได้จริง แท่งควรสูงขึ้นเรื่อย ๆ จาก Q1 ไป Q5 · "
                       f"ส่วนต่าง Q5–Q1 = {stt['q_spread'] * 100:+.1f}% ต่อปี · "
                       f"IC เฉลี่ย {stt['ic_mean']:+.3f} (t-stat {stt['ic_t']:.1f}; มากกว่า 2 ถือว่ามีนัยสำคัญ)")
        with b2:
            st.markdown("##### ผลตอบแทนรายปี")
            yr = (r_["yearly"] * 100).round(1)
            st.dataframe(yr.astype(str) + "%", width="stretch")

        with st.expander("ดูทุกรอบการซื้อขาย (หุ้นที่ระบบเลือกในแต่ละรอบ)"):
            per = r_["periods"][["เริ่ม", "กลยุทธ์", "SPY", "ซื้อทุกตัวเท่ากัน", "หุ้นที่เลือก"]].copy()
            for c_ in ["กลยุทธ์", "SPY", "ซื้อทุกตัวเท่ากัน"]:
                per[c_] = (per[c_] * 100).round(2)
            per["เริ่ม"] = per["เริ่ม"].dt.strftime("%Y-%m-%d")
            per.index = per.index.strftime("%Y-%m-%d")
            per.index.name = "สิ้นสุด"
            st.dataframe(per.iloc[::-1], width="stretch", height=400)

        st.warning(
            "**ข้อจำกัดที่ต้องรู้ก่อนเชื่อผล**\n\n"
            "- ใช้รายชื่อหุ้น **ปัจจุบัน** ย้อนไปในอดีต หุ้นที่เคยอยู่ในดัชนีแล้วถูกถอดออกหรือล้มละลายจะไม่ถูกนับ "
            "ทำให้ผลตอบแทนดูดีเกินจริง (survivorship bias) จึงควรดูว่า **ชนะการซื้อทุกตัวเท่ากัน** หรือไม่ มากกว่าดูว่าชนะ SPY\n"
            "- ทดสอบได้เฉพาะคะแนน **Technical** — ส่วน Fundamental และข่าวทดสอบย้อนหลังไม่ได้ เพราะไม่มีข้อมูลย้อนหลังฟรี\n"
            "- ไม่ได้จำกัดจำนวนหุ้นต่อกลุ่มธุรกิจเหมือนหน้าหลัก และช่วงทดสอบประมาณ 3–4 ปีถือว่าสั้น "
            "ผลอาจต่างออกไปในภาวะตลาดอื่น\n"
            "- ผลตอบแทนในอดีตไม่รับประกันอนาคต")

st.divider()
st.caption("**คำเตือน:** เครื่องมือนี้คำนวณอัตโนมัติจากข้อมูลสาธารณะ เพื่อช่วยคัดกรองหุ้นที่น่าศึกษาต่อเท่านั้น "
           "ไม่ใช่คำแนะนำการลงทุน ข้อมูลอาจคลาดเคลื่อนหรือดีเลย์ การวิเคราะห์ข่าวเป็นแบบคำสำคัญอย่างง่าย "
           "ผลตอบแทนในอดีตไม่รับประกันอนาคต ควรศึกษาข้อมูลเพิ่มเติม กระจายความเสี่ยง และทยอยลงทุน (DCA)")
