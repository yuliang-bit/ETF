"""
ETF 30分钟 K线分析 - Streamlit 网页版
数据源：东方财富 push2 / push2his（免费、免鉴权）
运行：streamlit run app.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="ETF 30分钟行情分析",
    page_icon="📈",
    layout="wide",
)

# ==================== 标的配置 ====================
ETF_LIST = [
    ("159299", "金融科技ETF易方达", "0"),
    ("159278", "机器人ETF鹏华", "0"),
    ("159663", "机床ETF华夏", "0"),
    ("159108", "工业软件ETF博时", "0"),
    ("159622", "创新药ETF东财", "0"),
    ("589850", "科创50ETF东财", "1"),
    ("159558", "半导体设备ETF易方达", "0"),
    ("159242", "创业板人工智能ETF大成", "0"),
    ("159283", "通用航空ETF南方", "0"),
    ("159599", "芯片ETF东财", "0"),
    ("159160", "电池ETF东财", "0"),
    ("512630", "卫星ETF广发", "1"),
    ("515880", "通信ETF国泰", "1"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}
REQUEST_INTERVAL = 1.2  # 东财限速红线约 1 次/秒，留余量


# ==================== 工具函数 ====================
def secid(market: str, code: str) -> str:
    return f"{market}.{code}"


def fetch_json(url: str, params: dict, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    """通达信/同花顺口径的 RSI，用 ewm(alpha=1/N) 模拟 SMA(…,N,1)"""
    delta = series.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def candle_shape(row: pd.Series) -> str:
    """判断单根 K 线形态：阴阳 + 上下影"""
    o, c, h, l = row["开盘"], row["收盘"], row["最高"], row["最低"]
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    parts = ["阳" if c >= o else "阴"]
    if upper > body * 0.5:
        parts.append("长上影")
    if lower > body * 0.5:
        parts.append("长下影")
    if body < (h - l) * 0.1:
        parts.append("十字")
    return " ".join(parts)


def status_from_rsi(rsi3: float, rsi12: float) -> str:
    if rsi3 > 95:
        return "🔴超买不追"
    if rsi3 < 30:
        return "🟡偏弱等支撑"
    if rsi3 > 50 and rsi12 > 50:
        return "🟢强势"
    return "🟡中性"


def strong_weak(rsi3: float) -> str:
    if rsi3 > 80:
        return "🟢强势"
    if rsi3 < 30:
        return "🔴偏弱"
    return "🟡中性"


# ==================== 数据拉取（单只，带缓存） ====================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_one(code: str, market: str) -> dict:
    """
    拉取单只 ETF 的实时行情 + 30分钟 K 线，返回 dict。
    ttl=60：60 秒内重复访问直接走缓存，不重复请求东财。
    """
    # --- 实时行情 ---
    fields = "f43,f44,f45,f46,f48,f60,f168"
    params = {"fltt": 2, "invt": 2, "fields": fields, "secid": secid(market, code)}
    rt = fetch_json("https://push2.eastmoney.com/api/qt/stock/get", params)["data"]

    last_price = rt.get("f43")
    prev_close = rt.get("f60")
    # 用最新价和昨收自己算涨跌幅，避免 f170 口径不一致
    if last_price and prev_close:
        change_pct = (last_price - prev_close) / prev_close * 100
    else:
        change_pct = 0.0

    realtime = {
        "最新价": last_price,
        "涨跌幅%": round(change_pct, 2),
        "今开": rt.get("f46"),
        "最高": rt.get("f44"),
        "最低": rt.get("f45"),
        "成交额(亿)": round((rt.get("f48") or 0) / 1e8, 2),
        "换手率%": rt.get("f168"),
    }

    time.sleep(REQUEST_INTERVAL)

    # --- 30 分钟 K 线 ---
    params = {
        "secid": secid(market, code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 30,
        "fqt": 1,
        "end": "20500101",
        "lmt": 80,
    }
    kd = fetch_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", params)
    klines = kd["data"]["klines"]
    rows = [line.split(",") for line in klines]
    df = pd.DataFrame(rows, columns=[
        "时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
        "振幅", "涨跌幅", "涨跌额", "换手率",
    ])
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["RSI3"] = calc_rsi(df["收盘"], 3)
    df["RSI12"] = calc_rsi(df["收盘"], 12)
    df["形态"] = df.apply(candle_shape, axis=1)

    time.sleep(REQUEST_INTERVAL)

    return {"realtime": realtime, "kline": df}


# ==================== 主流程 ====================
st.title("📈 ETF 30 分钟行情分析")
st.caption(f"数据源：东方财富 · 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

col_btn, col_info = st.columns([1, 4])
with col_btn:
    if st.button("🔄 强制刷新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_info:
    st.info("页面缓存 60 秒；点击「强制刷新」立即重新拉取。", icon="ℹ️")

# 拉取数据（带进度条）
all_data = {}
errors = {}
progress = st.progress(0, text="准备拉取 13 只 ETF ...")

for i, (code, name, market) in enumerate(ETF_LIST):
    progress.progress(
        (i + 1) / len(ETF_LIST),
        text=f"正在拉取 {code} {name}（{i+1}/{len(ETF_LIST)}）",
    )
    try:
        all_data[code] = {"name": name, "market": market, **fetch_one(code, market)}
    except Exception as e:
        errors[code] = f"{name}: {e}"

progress.empty()

if errors:
    with st.expander(f"⚠️ {len(errors)} 只标的拉取失败", expanded=False):
        for k, v in errors.items():
            st.write(f"- `{k}` {v}")

if not all_data:
    st.error("全部标的拉取失败，请检查网络或稍后重试。")
    st.stop()

# ---------- 一、实时行情 ----------
st.header("一、实时行情")

rt_rows = []
for code, info in all_data.items():
    rt = info["realtime"]
    last = info["kline"].iloc[-1]
    rt_rows.append({
        "代码": code,
        "名称": info["name"],
        "最新价": rt["最新价"],
        "涨跌幅%": rt["涨跌幅%"],
        "今开": rt["今开"],
        "最高": rt["最高"],
        "最低": rt["最低"],
        "成交额(亿)": rt["成交额(亿)"],
        "换手率%": rt["换手率%"],
        "RSI3": round(last["RSI3"], 2),
        "RSI12": round(last["RSI12"], 2),
        "状态": status_from_rsi(last["RSI3"], last["RSI12"]),
    })

rt_df = pd.DataFrame(rt_rows)
st.dataframe(rt_df, use_container_width=True, hide_index=True)

# ---------- 二、30 分钟 K 线分析 ----------
st.header("二、30 分钟 K 线分析")
st.caption("东财 30 分钟 K 线以结束时间标注；「当日」按日期前缀过滤。")

today_str = datetime.now().strftime("%Y-%m-%d")

for code, info in all_data.items():
    name = info["name"]
    df = info["kline"]
    today_df = df[df["时间"].astype(str).str.startswith(today_str)]

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    judgment = strong_weak(last["RSI3"])

    with st.expander(f"{code} {name} —— {judgment}  |  当前 {last['收盘']:.4f}", expanded=False):
        if today_df.empty:
            st.write("今日暂无 30 分钟 K 线（可能未开盘或数据延迟）。")
            continue

        show_df = today_df[["时间", "开盘", "最高", "最低", "收盘", "形态"]].copy()
        show_df = show_df.rename(columns={"时间": "K线时间"})
        show_df["K线时间"] = show_df["K线时间"].astype(str).str[-5:]  # 只显示 HH:MM
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        vol_ratio = (
            today_df["成交量"].iloc[-1] / today_df["成交量"].mean()
            if len(today_df) > 1 and today_df["成交量"].mean() > 0
            else 1.0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("量能对比（最新/均量）", f"{vol_ratio:.2f}")
        c2.metric("多空判断", judgment)
        c3.metric("上一根 → 当前形态", f"{prev['形态']} → {last['形态']}")

# ---------- 低吸信号表 ----------
st.header("低吸信号表")
st.caption("按 RSI3 由低到高排序；第一买点取当日 30 分钟 K 线最低点。")

low_rows = []
for code, info in all_data.items():
    df = info["kline"]
    today_df = df[df["时间"].astype(str).str.startswith(today_str)]
    if today_df.empty:
        continue
    last = df.iloc[-1]
    current = last["收盘"]
    first_buy = today_df["最低"].min()
    if not first_buy:
        continue
    distance = (current - first_buy) / first_buy * 100
    low_rows.append({
        "排名": 0,
        "代码": code,
        "标的": info["name"],
        "RSI3": round(last["RSI3"], 2),
        "第一买点": round(first_buy, 4),
        "确认条件": "RSI3 回升 + 阳线放量",
        "当前价": round(current, 4),
        "距买点%": round(distance, 2),
        "止损": round(first_buy * 0.985, 4),
        "目标": round(current * 1.03, 4),
    })

low_df = pd.DataFrame(low_rows)
low_df = low_df[low_df["RSI3"] < 30].sort_values("RSI3", ascending=True).reset_index(drop=True)
if low_df.empty:
    st.info("当前无 RSI3 < 30 的标的。")
else:
    low_df["排名"] = low_df.index + 1
    st.dataframe(low_df, use_container_width=True, hide_index=True)

# ---------- 高抛信号表 ----------
st.header("高抛信号表")
st.caption("RSI3 > 95 或「长上影 + RSI 偏高」触发；按 RSI3 由高到低排序。")

high_rows = []
for code, info in all_data.items():
    df = info["kline"]
    today_df = df[df["时间"].astype(str).str.startswith(today_str)]
    if today_df.empty:
        continue
    last = df.iloc[-1]
    current = last["收盘"]
    resistance = today_df["最高"].max()
    distance = (resistance - current) / current * 100 if current else 0

    if last["RSI3"] > 95 or ("长上影" in last["形态"] and last["RSI3"] > 70):
        reason = "RSI3>95 强超买" if last["RSI3"] > 95 else "长上影 + RSI 偏高"
        high_rows.append({
            "排名": 0,
            "代码": code,
            "标的": info["name"],
            "RSI3": round(last["RSI3"], 2),
            "高抛价(压力区)": round(resistance, 4),
            "确认条件": "触及压力区 + RSI3 拐头向下",
            "当前价": round(current, 4),
            "距压力%": round(distance, 2),
            "理由": reason,
        })

high_df = pd.DataFrame(high_rows)
if high_df.empty:
    st.info("当前无 RSI3 > 95 或冲高回落信号的标的。")
else:
    high_df = high_df.sort_values("RSI3", ascending=False).reset_index(drop=True)
    high_df["排名"] = high_df.index + 1
    st.dataframe(high_df, use_container_width=True, hide_index=True)

# ---------- 三、关键支撑/压力位 ----------
st.header("三、关键支撑 / 压力位")
st.caption("基于近 40 根 30 分钟 K 线的摆动高低点；RSI3>95 时强压力取当日最高点。")

sup_rows = []
for code, info in all_data.items():
    df = info["kline"]
    last = df.iloc[-1]
    current = last["收盘"]
    recent = df.tail(40)
    swing_high = recent["最高"].max()
    swing_low = recent["最低"].min()
    rsi3 = last["RSI3"]

    strong_pressure = swing_high if rsi3 > 95 else round(swing_high * 1.005, 4)
    sup_rows.append({
        "代码": code,
        "标的": info["name"],
        "当前价": round(current, 4),
        "强压力": round(strong_pressure, 4),
        "压力": round(swing_high * 0.995, 4),
        "支撑": round(swing_low * 1.005, 4),
        "强支撑": round(swing_low * 0.995, 4),
        "RSI3": round(rsi3, 2),
    })

sup_df = pd.DataFrame(sup_rows)
st.dataframe(sup_df, use_container_width=True, hide_index=True)

# ---------- 下载 ----------
st.header("导出")
d1, d2, d3 = st.columns(3)
d1.download_button(
    "下载实时行情 CSV",
    rt_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"etf_realtime_{today_str}.csv",
    mime="text/csv",
    use_container_width=True,
)
d2.download_button(
    "下载低吸信号 CSV",
    low_df.to_csv(index=False).encode("utf-8-sig") if not low_df.empty else "无信号".encode("utf-8-sig"),
    file_name=f"etf_low_{today_str}.csv",
    mime="text/csv",
    use_container_width=True,
)
d3.download_button(
    "下载高抛信号 CSV",
    high_df.to_csv(index=False).encode("utf-8-sig") if not high_df.empty else "无信号".encode("utf-8-sig"),
    file_name=f"etf_high_{today_str}.csv",
    mime="text/csv",
    use_container_width=True,
)