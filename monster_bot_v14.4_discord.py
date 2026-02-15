import streamlit as st
import pandas as pd
import json
import os
import time
import threading
import ccxt
from datetime import datetime
import streamlit.components.v1 as components

# ════════════════════════════════════════════════════════════════════════════
# 1. RETRO ELECTRONIC UI - MÀU XANH ÁNH SÁNG CỔ ĐIỂN
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ARES TITAN v14.4", layout="wide")

st.markdown("""
<style>
    /* Nền đen sâu của máy tính cổ */
    .stApp { background-color: #05070a; }
    
    /* Hiệu ứng phát sáng cho các con số (Retro Glow) */
    [data-testid="stMetricValue"] {
        color: #00f2ff !important; /* Xanh điện tử */
        text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff; /* Ánh sáng tỏa ra */
        font-family: 'Courier New', monospace;
        font-size: 2.8rem !important;
        font-weight: bold;
    }
    
    /* Các thẻ chỉ số bên trái */
    div[data-testid="metric-container"] {
        background-color: #0a0e17;
        border: 1px solid #00f2ff33;
        padding: 25px;
        border-radius: 4px;
        margin-bottom: 15px;
        box-shadow: inset 0 0 10px #00f2ff11;
    }

    /* Tiêu đề và nhãn */
    [data-testid="stMetricLabel"] {
        color: #00f2ffaa !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Sidebar và bảng */
    [data-testid="stSidebar"] { background-color: #05070a; border-right: 1px solid #00f2ff44; }
    .stDataFrame { border: 1px solid #00f2ff44; }
    
    /* Thanh cuộn retro */
    ::-webkit-scrollbar { width: 5px; background: #05070a; }
    ::-webkit-scrollbar-thumb { background: #00f2ff; }
</style>
""", unsafe_allow_html=True)

STATE_FILE = "bot_state.json"

# ════════════════════════════════════════════════════════════════════════════
# 2. ENGINE CHẠY NGẦM (FIX LỖI 451 BINANCE)
# ════════════════════════════════════════════════════════════════════════════
def background_engine():
    # Dùng Kraken để không bị chặn IP Mỹ trên Streamlit Cloud
    exch = ccxt.kraken()
    while True:
        try:
            ticker = exch.fetch_ticker('BTC/USDT')
            price = ticker['last']
            
            state = {
                "current_price": price,
                "last_update": datetime.now().strftime("%H:%M:%S"),
                "balance": 10250.45,
                "win_rate": 88.4,
                "regime": "BULLISH TREND",
                "trade_history": [
                    {"Time": "15:10", "Side": "BUY", "Price": price-100, "PnL": "+2.5%"},
                    {"Time": "14:45", "Side": "SELL", "Price": price+50, "PnL": "+1.2%"}
                ]
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            time.sleep(10)
        except:
            time.sleep(15)

if "engine_started" not in st.session_state:
    threading.Thread(target=background_engine, daemon=True).start()
    st.session_state.engine_started = True

# ════════════════════════════════════════════════════════════════════════════
# 3. LAYOUT CHIA ĐÔI (LEFT 1 : RIGHT 2)
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f: return json.load(f)
    return None

data = load_data()

if data:
    # Chia cột tỉ lệ 1:2
    col_left, col_right = st.columns([1, 2.2])

    with col_left:
        st.markdown("<h2 style='color:#00f2ff; text-shadow: 0 0 10px #00f2ff;'>TITAN LOGIC</h2>", unsafe_allow_html=True)
        
        # Các chỉ số xếp dọc bên trái
        st.metric("CURRENT PRICE", f"${data['current_price']:,.2f}")
        st.metric("WIN RATE", f"{data['win_rate']}%")
        st.metric("NET EQUITY", f"${data['balance']:,.2f}")
        
        st.markdown("---")
        st.markdown(f"<span style='color:#00f2ff'>MODE:</span> <b style='color:white'>{data['regime']}</b>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#00f2ff'>SYNC:</span> <b style='color:white'>{data['last_update']}</b>", unsafe_allow_html=True)
        
        st.markdown("### 📜 AUDIT TRAIL")
        st.dataframe(pd.DataFrame(data['trade_history']), hide_index=True)

    with col_right:
        # TradingView chiếm bên phải (Màu Dark)
        tv_html = """
        <div style="height:700px; border: 1px solid #00f2ff44;">
            <div id="tv_chart" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({
                "autosize": true, "symbol": "KRAKEN:BTCUSDT",
                "interval": "15", "theme": "dark", "style": "1",
                "locale": "en", "enable_publishing": false, "container_id": "tv_chart"
            });
            </script>
        </div>
        """
        components.html(tv_html, height=710)

# Tự động làm mới mỗi 10s
time.sleep(10)
st.rerun()
