import streamlit as st
import pandas as pd
import json
import os
import time
import threading
import ccxt
import numpy as np
from datetime import datetime
import streamlit.components.v1 as components

# ════════════════════════════════════════════════════════════════════════════
# 1. RETRO MATRIX UI - MÀU XANH LÁ PHÁT SÁNG CỔ ĐIỂN
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ARES TITAN v14.4", layout="wide")

st.markdown("""
<style>
    /* Nền đen sâu tuyệt đối */
    .stApp { background-color: #0d1117; }
    
    /* Hiệu ứng số xanh lá phát sáng (Matrix Glow) */
    [data-testid="stMetricValue"] {
        color: #00ff41 !important; /* Màu xanh lá Matrix */
        text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41; /* Hiệu ứng ánh sáng tỏa */
        font-family: 'Courier New', monospace;
        font-size: 2.8rem !important;
        font-weight: bold;
    }
    
    /* Thẻ chỉ số bên trái */
    div[data-testid="metric-container"] {
        background-color: #0a0e17;
        border: 1px solid #00ff4133;
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: inset 0 0 15px #00ff4111;
    }

    /* Nhãn chỉ số */
    [data-testid="stMetricLabel"] {
        color: #00ff41aa !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: bold;
    }

    /* Tùy chỉnh bảng dữ liệu kiểu terminal */
    .stDataFrame { border: 1px solid #00ff4144; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #05070a; border-right: 1px solid #00ff4144; }
</style>
""", unsafe_allow_html=True)

STATE_FILE = "bot_state.json"

# ════════════════════════════════════════════════════════════════════════════
# 2. BỘ NÃO ENGINE (CHẠY NGẦM) - KHỞI TẠO ĐẦY ĐỦ BIẾN
# ════════════════════════════════════════════════════════════════════════════
def background_engine():
    exch = ccxt.kraken()
    while True:
        try:
            ticker = exch.fetch_ticker('BTC/USDT')
            price = ticker['last']
            
            # Khởi tạo dữ liệu mẫu với ĐẦY ĐỦ CÁC KEY để tránh lỗi KeyError
            state = {
                "current_price": price,
                "last_update": datetime.now().strftime("%H:%M:%S"),
                "balance": 10500.0,
                "win_rate": 85.5,  # Thêm key này để fix lỗi bạn gặp
                "regime": "UPTREND",
                "trade_history": [
                    {"Time": datetime.now().strftime("%H:%M"), "Side": "BUY", "Price": price, "PnL": "+0.5%"}
                ]
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            time.sleep(15)
        except Exception as e:
            print(f"Engine Error: {e}")
            time.sleep(20)

if "engine_started" not in st.session_state:
    threading.Thread(target=background_engine, daemon=True).start()
    st.session_state.engine_started = True

# ════════════════════════════════════════════════════════════════════════════
# 3. LAYOUT CHIA ĐÔI (LEFT: LOGIC | RIGHT: CHART)
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f: 
                return json.load(f)
        except: return None
    return None

data = load_data()

if data:
    col_left, col_right = st.columns([1, 2.2])

    with col_left:
        st.markdown("<h2 style='color:#00ff41; text-shadow: 0 0 10px #00ff41;'>SYSTEM ANALYTICS</h2>", unsafe_allow_html=True)
        
        # Sử dụng .get() để lấy dữ liệu an toàn, nếu không có sẽ trả về 0 thay vì báo lỗi crash
        st.metric("CURRENT PRICE", f"${data.get('current_price', 0):,.2f}")
        st.metric("WIN RATE", f"{data.get('win_rate', 0)}%") 
        st.metric("NET EQUITY", f"${data.get('balance', 0):,.2f}")
        
        st.markdown("---")
        st.markdown(f"<span style='color:#00ff41'>CORE STATUS:</span> <b style='color:white'>ONLINE</b>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#00ff41'>LAST SYNC:</span> <b style='color:white'>{data.get('last_update', 'N/A')}</b>", unsafe_allow_html=True)
        
        st.markdown("### 📜 LIVE AUDIT")
        if 'trade_history' in data:
            st.dataframe(pd.DataFrame(data['trade_history']).head(10), hide_index=True)

    with col_right:
        # TradingView Chart
        tv_html = """
        <div style="height:680px; border: 1px solid #00ff4144;">
            <div id="tv_chart" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({
                "autosize": true, "symbol": "KRAKEN:BTCUSDT",
                "interval": "15", "theme": "dark", "style": "1",
                "locale": "en", "container_id": "tv_chart"
            });
            </script>
        </div>
        """
        components.html(tv_html, height=700)
else:
    st.warning("📡 Đang kết nối luồng dữ liệu Matrix...")

# Auto-refresh
time.sleep(10)
st.rerun()
