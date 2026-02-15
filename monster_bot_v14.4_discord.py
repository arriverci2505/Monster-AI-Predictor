import streamlit as st
import pandas as pd
import json
import os
import time
import threading
import ccxt
from datetime import datetime
import streamlit.components.v1 as components

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="ARES TITAN v14.4", layout="wide")

STATE_FILE = "bot_state.json"

# Hàm khởi tạo file dữ liệu mặc định nếu chưa có
def init_state():
    if not os.path.exists(STATE_FILE):
        data = {
            "current_price": 0.0,
            "last_update": "Initializing...",
            "balance": 10000.0,
            "trade_history": [],
            "open_trades": [],
            "regime": "Scanning..."
        }
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)

init_state()

# 2. ENGINE CHẠY NGẦM (Dùng Kraken để tránh bị chặn IP Mỹ)
def background_engine():
    # ĐỔI SANG KRAKEN ĐỂ CHẠY ĐƯỢC TRÊN STREAMLIT CLOUD
    exchange = ccxt.kraken() 
    symbol = 'BTC/USDT'
    
    while True:
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            # Đọc dữ liệu cũ
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            
            # Cập nhật dữ liệu mới
            state["current_price"] = price
            state["last_update"] = datetime.now().strftime("%H:%M:%S")
            
            # Ghi lại vào file
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
                
            time.sleep(15) 
        except Exception as e:
            print(f"Engine Error: {e}")
            time.sleep(20)

# Khởi chạy luồng ngầm (Chỉ chạy 1 lần duy nhất)
if "engine_started" not in st.session_state:
    thread = threading.Thread(target=background_engine, daemon=True)
    thread.start()
    st.session_state.engine_started = True

# 3. GIAO DIỆN ĐỌC DỮ LIỆU TỪ FILE
def load_data():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return None

state = load_data()

# --- HIỂN THỊ GIAO DIỆN V13 ---
if state:
    st.title("🤖 ARES TITAN AI - CLOUD VERSION")
    
    # 4 Cột Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CURRENT PRICE", f"${state['current_price']:,.2f}")
    c2.metric("WIN RATE", "92.5%") # Ví dụ
    c3.metric("STATUS", "ONLINE ✅")
    c4.metric("NET EQUITY", f"${state['balance']:,.2f}")

    # TradingView
    components.html(f"""
        <div style="height:500px;"><div id="tv"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"autosize":true,"symbol":"KRAKEN:BTCUSDT","interval":"15","theme":"dark","container_id":"tv"}});</script>
        </div>""", height=500)

st.write(f"Cập nhật lúc: {state['last_update'] if state else 'N/A'}")
time.sleep(10)
st.rerun()
