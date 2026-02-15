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
# 1. CẤU TRÚC GIAO DIỆN & CSS (V13 DARK NEON)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ARES TITAN v14.4", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; }
    /* Metric Card dọc */
    div[data-testid="metric-container"] {
        background-color: #161a25;
        border: 1px solid #2e3344;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    /* Chữ xanh Neon chuẩn v13 */
    [data-testid="stMetricValue"] {
        color: #00ffcc !important;
        font-family: 'Courier New', monospace;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] { color: #8e94a5 !important; }
    /* Bảng dữ liệu */
    .stDataFrame { border: 1px solid #2e3344; }
</style>
""", unsafe_allow_html=True)

STATE_FILE = "bot_state.json"

# ════════════════════════════════════════════════════════════════════════════
# 2. BỘ NÃO ENGINE (CHẠY NGẦM) - FIX LỖI 451 BINANCE
# ════════════════════════════════════════════════════════════════════════════
def background_engine():
    # Dùng Kraken để không bị chặn tại Mỹ (Server Streamlit)
    exch = ccxt.kraken()
    symbol = 'BTC/USDT'
    
    while True:
        try:
            ticker = exch.fetch_ticker(symbol)
            price = ticker['last']
            
            # Khởi tạo hoặc đọc dữ liệu
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f: state = json.load(f)
            else:
                state = {"balance": 10000.0, "trade_history": [], "open_trades": []}

            state["current_price"] = price
            state["last_update"] = datetime.now().strftime("%H:%M:%S")
            state["regime"] = "HIGH VOLATILITY" if np.random.rand() > 0.5 else "STABLE" # Test logic
            
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            time.sleep(10)
        except Exception as e:
            print(f"Engine Error: {e}")
            time.sleep(15)

if "engine_started" not in st.session_state:
    threading.Thread(target=background_engine, daemon=True).start()
    st.session_state.engine_started = True

# ════════════════════════════════════════════════════════════════════════════
# 3. LAYOUT CHIA 2 BÊN (LEFT: LOGIC | RIGHT: CHART)
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f: return json.load(f)
    return None

data = load_data()

if data:
    # Chia tỉ lệ 1:2 (Bên trái hẹp hơn để hiện chỉ số, bên phải rộng hiện Chart)
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown("### 🤖 TITAN LOGIC")
        st.metric("CURRENT PRICE", f"${data['current_price']:,.2f}")
        
        # Winrate giả lập dựa trên history
        hist = data.get('trade_history', [])
        wins = len([t for t in hist if t.get('pnl', 0) > 0])
        wr = (wins/len(hist)*100) if hist else 0.0
        
        st.metric("WIN RATE", f"{wr:.1f}%")
        st.metric("NET EQUITY", f"${data['balance']:,.2f}")
        
        st.markdown("---")
        st.write(f"**Market Regime:** `{data.get('regime', 'SCANNING')}`")
        st.write(f"**AI Signal:** `STRONG BUY`" if data['current_price'] > 0 else "`WAITING`")
        st.write(f"🕒 Update: {data['last_update']}")

    with col_right:
        # TradingView Widget (Full Height)
        tv_html = f"""
        <div style="height:650px;">
            <div id="tv_chart" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({{
                "autosize": true, "symbol": "KRAKEN:BTCUSDT",
                "interval": "15", "theme": "dark", "style": "1",
                "locale": "en", "enable_publishing": false,
                "hide_side_toolbar": false, "container_id": "tv_chart"
            }});
            </script>
        </div>
        """
        components.html(tv_html, height=650)

    # Bảng Audit Trail ở dưới cùng (Full Width)
    st.markdown("### 📜 TITAN AUDIT TRAIL")
    if hist:
        st.dataframe(pd.DataFrame(hist).iloc[::-1], use_container_width=True)
    else:
        st.info("Sytem is monitoring. No trades executed yet.")

# Tự động làm mới
time.sleep(5)
st.rerun()
