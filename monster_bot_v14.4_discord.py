import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
import streamlit.components.v1 as components

# ════════════════════════════════════════════════════════════════════════════
# 1. CẤU HÌNH TRANG & THEME (V13 ORIGINAL STYLE)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MONSTER BOT v13.6 - TITAN INTERACTIVE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ép giao diện về chuẩn v13 (Màu sắc, font chữ, độ dãn cách)
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp { background-color: #0e1117; }
    
    /* Metrics Styling - Cyan Neon */
    [data-testid="stMetricValue"] {
        color: #00ffcc !important;
        font-family: 'Courier New', monospace;
        font-size: 1.8rem !important;
    }
    
    /* Metrics Container */
    div[data-testid="metric-container"] {
        background-color: #1e212b;
        border: 1px solid #31333f;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #11141c;
        border-right: 1px solid #31333f;
        padding-top: 2rem;
    }

    /* Bảng log chuẩn v13 */
    .stDataFrame {
        border: 1px solid #31333f;
        border-radius: 5px;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 2. HÀM TIỆN ÍCH (UTILITIES)
# ════════════════════════════════════════════════════════════════════════════

STATE_FILE = 'bot_state.json'

def load_bot_state():
    """Đọc dữ liệu từ Engine gửi qua JSON"""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def get_tradingview_html(symbol):
    """Tạo Widget TradingView chuẩn v13"""
    # Xử lý symbol (ví dụ: BTC/USDT -> BTCUSDT)
    clean_symbol = symbol.replace("/", "").replace(":USDT", "USDT")
    return f"""
    <div class="tradingview-widget-container" style="height:500px; width:100%;">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "BINANCE:{clean_symbol}",
        "interval": "15",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """

# ════════════════════════════════════════════════════════════════════════════
# 3. GIAO DIỆN CHÍNH (LAYOUT CẤU TRÚC V13)
# ════════════════════════════════════════════════════════════════════════════

def main():
    state = load_bot_state()

    # --- TRƯỜNG HỢP ENGINE CHƯA CHẠY ---
    if not state:
        st.title("🤖 MONSTER BOT v13.6 - TITAN INTERACTIVE")
        st.error("🔴 ENGINE OFFLINE: Vui lòng chạy 'python monster_engine.py' trong Terminal.")
        time.sleep(5)
        st.rerun()
        return

    # Lấy thông số từ State
    config = state.get('config', {})
    history = state.get('trade_history', [])
    open_trades = state.get('open_trades', [])
    current_price = state.get('current_price', 0.0)
    regime = state.get('regime', 'Scanning...')
    last_signal = state.get('last_signal', 'HOLD')

    # --- SIDEBAR (GIỐNG V13) ---
    with st.sidebar:
        st.markdown(f"## 🤖 TITAN INTERACTIVE\n**Engine v14.4**")
        st.markdown("---")
        
        st.subheader("⚙️ LIVE PARAMETERS")
        st.code(json.dumps(config, indent=2), language='json')
        
        st.markdown("---")
        st.subheader("🛡️ SYSTEM STATUS")
        st.success(f"Bot Status: {state.get('bot_status', 'Active')}")
        st.write(f"🕒 Last Update: {state.get('last_update', 'N/A')}")
        
        if st.button("♻️ REFRESH DASHBOARD"):
            st.rerun()

    # --- PHẦN 1: METRICS (4 CỘT CHUẨN V13) ---
    st.markdown(f"## 🚀 MARKET: {config.get('symbol', 'BTC/USDT')}")
    
    # Tính toán winrate
    wins = len([t for t in history if t.get('pnl', 0) > 0])
    losses = len([t for t in history if t.get('pnl', 0) <= 0])
    win_rate = (wins / len(history) * 100) if history else 0
    total_pnl = sum([t.get('pnl', 0) for t in history])

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("CURRENT PRICE", f"${current_price:,.2f}", f"Mode: {regime}")
    m_col2.metric("WIN RATE", f"{win_rate:.1f}%")
    m_col3.metric("WINS / LOSSES", f"{wins} / {losses}")
    m_col4.metric("TOTAL NET PNL", f"${total_pnl:,.2f}")

    st.markdown("---")

    # --- PHẦN 2: TRADINGVIEW & ACTIVE INFO (2 CỘT 2:1) ---
    col_chart, col_info = st.columns([2, 1])

    with col_chart:
        # Biểu đồ TradingView
        components.html(get_tradingview_html(config.get('symbol', 'BTC/USDT')), height=500)

    with col_info:
        st.markdown("### ⚡ OPEN POSITIONS")
        if open_trades:
            for trade in open_trades:
                entry = trade.get('entry_price', 0)
                side = trade.get('side', 'BUY')
                # Tính PnL tạm tính
                if side == 'BUY':
                    pnl_pct = ((current_price - entry) / entry) * 100
                else:
                    pnl_pct = ((entry - current_price) / entry) * 100
                
                pnl_color = "green" if pnl_pct >= 0 else "red"
                
                st.info(f"**{side}** @ {entry:,.2f}")
                st.markdown(f"PnL: :{pnl_color}[**{pnl_pct:+.2f}%**]")
                # Thanh tiến trình giả lập khoảng cách tới Target
                st.progress(min(max((pnl_pct + 1) / 2, 0.0), 1.0)) 
        else:
            st.info("🔍 No active signals found...")

        st.markdown("---")
        st.markdown(f"**AI Prediction:** `{last_signal}`")
        st.markdown(f"**Active Regime:** `{regime}`")

    # --- PHẦN 3: AUDIT TRAIL (BẢNG LOG Ở DƯỚI) ---
    st.markdown("### 📜 TITAN AUDIT TRAIL (LIVE LOG)")
    if history:
        # Chuyển list thành DataFrame và đảo ngược (mới nhất lên đầu)
        df_log = pd.DataFrame(history).iloc[::-1]
        st.dataframe(
            df_log,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": "Time",
                "side": "Side",
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
                "pnl": st.column_config.NumberColumn("Net PnL", format="$%.2f"),
                "exit_reason": "Reason"
            }
        )
    else:
        st.info("Chưa có lịch sử giao dịch.")

    # --- AUTO REFRESH LOOP ---
    time.sleep(10) # Refresh mỗi 10 giây
    st.rerun()

if __name__ == "__main__":
    main()
