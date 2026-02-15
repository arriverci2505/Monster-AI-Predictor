"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER UI v14.4 - LIGHTWEIGHT STREAMLIT DASHBOARD                      ║
║  Read-Only Interface for Cloud Deployment                                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import time

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monster Bot v14.4 Titan",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom để giống v13 và fix lỗi giao diện
st.markdown("""
<style>
    /* Fix lỗi nhãn Sidebar bị đè */
    [data-testid="stSidebar"] { padding-top: 1rem; }
    
    /* Metrics Style */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    
    /* Bảng Log đẹp hơn */
    div[data-testid="stDataFrame"] {
        width: 100%;
    }
    
    /* Thanh Progress Bar màu xanh/đỏ */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4caf50, #8bc34a);
    }
</style>
""", unsafe_allow_html=True)

# File dữ liệu được tạo ra bởi monster_engine.py
STATE_FILE = 'bot_state.json'

def load_bot_state():
    """Load bot state from JSON file"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            return state
        else:
            return {
                'balance': 10000.0,
                'open_trades': [],
                'trade_history': [],
                'last_update_time': None,
                'bot_status': 'Waiting for Engine',
                'pending_orders': []
            }
    except Exception as e:
        st.error(f"Error loading state: {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def display_header(state):
    """Display main header with status"""
    status = state.get('bot_status', 'Unknown')
    
    # Determine status class
    if status == 'Running':
        status_class = 'status-running'
        status_emoji = '🟢'
    elif 'Error' in status:
        status_class = 'status-error'
        status_emoji = '🟡'
    else:
        status_class = 'status-stopped'
        status_emoji = '🔴'
    
    st.markdown(f"""
    <div class='main-header'>
        <h1>🤖 MONSTER BOT v14.4</h1>
        <p style='color: white; margin-top: 0.5rem;'>Cloud-Optimized Trading Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-container'>
            <span class='metric-label'>Status</span>
            <span class='status-badge {status_class}'>{status_emoji} {status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        last_update = state.get('last_update_time', 'Never')
        if last_update and last_update != 'Never':
            try:
                update_time = datetime.fromisoformat(last_update)
                time_str = update_time.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = str(last_update)
        else:
            time_str = 'Never'
        
        st.markdown(f"""
        <div class='metric-container'>
            <span class='metric-label'>Last Update</span>
            <span class='metric-value'>{time_str}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        balance = state.get('balance', 10000.0)
        st.markdown(f"""
        <div class='metric-container'>
            <span class='metric-label'>Balance</span>
            <span class='metric-value'>${balance:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

def display_active_trade(trade, current_price=None):
    """Display active trade card with progress visualization"""
    
    st.markdown("""
    <div class='trade-card'>
        <h3 style='color: #00ff41; margin-top: 0;'>📊 ACTIVE POSITION</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Trade details
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Side", trade['side'])
    
    with col2:
        st.metric("Entry Price", f"${trade['entry_price']:,.2f}")
    
    with col3:
        st.metric("Stop Loss", f"${trade['stop_loss']:,.2f}")
    
    with col4:
        st.metric("Take Profit", f"${trade['take_profit']:,.2f}")
    
    # Calculate PnL if current price available
    if current_price:
        entry_price = trade['entry_price']
        side = trade['side']
        
        if side == 'BUY':
            pnl = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl = ((entry_price - current_price) / entry_price) * 100
        
        # Subtract commission
        net_pnl = pnl - (2 * 0.075)  # 0.075% per trade
        
        pnl_class = 'profit-positive' if net_pnl > 0 else 'profit-negative'
        
        st.markdown(f"""
        <div class='metric-container'>
            <span class='metric-label'>Current PnL</span>
            <span class='metric-value {pnl_class}'>{net_pnl:+.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar visualization
        st.markdown("**Position Progress:**")
        
        sl = trade['stop_loss']
        tp = trade['take_profit']
        
        # Calculate position on scale from SL (0%) to TP (100%)
        if side == 'BUY':
            total_range = tp - sl
            current_position = current_price - sl
        else:
            total_range = sl - tp
            current_position = sl - current_price
        
        progress = min(max(current_position / total_range, 0), 1) if total_range > 0 else 0.5
        
        st.progress(progress)
        
        # Labels
        col_sl, col_mid, col_tp = st.columns(3)
        with col_sl:
            st.caption(f"🛡️ SL: ${sl:,.2f}")
        with col_mid:
            st.caption(f"📍 Current: ${current_price:,.2f}")
        with col_tp:
            st.caption(f"🎯 TP: ${tp:,.2f}")
    
    # Additional info
    entry_time = datetime.fromisoformat(trade['entry_time'])
    duration = datetime.now() - entry_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"⏱️ **Duration:** {hours}h {minutes}m")
    with col2:
        st.info(f"🌊 **Regime:** {trade.get('regime', 'N/A')}")

def display_pending_orders(pending_orders):
    """Display pending limit orders"""
    
    if not pending_orders:
        return
    
    st.markdown("### ⏳ Pending Limit Orders")
    
    for order in pending_orders:
        with st.expander(f"{order['side']} @ ${order['limit_price']:,.2f}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Limit Price", f"${order['limit_price']:,.2f}")
            with col2:
                st.metric("Stop Loss", f"${order['stop_loss']:,.2f}")
            with col3:
                st.metric("Take Profit", f"${order['take_profit']:,.2f}")
            
            candles_waiting = order.get('candles_waiting', 0)
            st.caption(f"⏰ Waiting: {candles_waiting} candles (Max: 2)")

def display_trade_history(history):
    """Display trade history with color-coded PnL"""
    
    st.markdown("### 📜 Trade History")
    
    if not history:
        st.info("No trade history yet")
        return
    
    # Convert to DataFrame
    df_history = pd.DataFrame(history)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_trades = len(df_history)
    
    # Calculate wins/losses
    completed = df_history[df_history['type'].str.contains('EXIT')]
    if len(completed) > 0:
        wins = len(completed[completed['net_pnl'] > 0])
        losses = len(completed[completed['net_pnl'] <= 0])
        win_rate = (wins / len(completed)) * 100
        
        avg_win = completed[completed['net_pnl'] > 0]['net_pnl'].mean() if wins > 0 else 0
        avg_loss = completed[completed['net_pnl'] <= 0]['net_pnl'].mean() if losses > 0 else 0
    else:
        wins = 0
        losses = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Wins / Losses", f"{wins} / {losses}")
    with col4:
        avg_pnl = completed['net_pnl'].mean() if len(completed) > 0 else 0
        st.metric("Avg PnL", f"{avg_pnl:+.2f}%")
    
    # Prepare display dataframe
    display_df = df_history.copy()
    
    # Format time
    display_df['time'] = pd.to_datetime(display_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select columns to display
    columns_to_show = ['time', 'type', 'entry_price', 'exit_price', 'net_pnl', 'reason', 'regime', 'duration']
    
    # Only include columns that exist
    columns_to_show = [col for col in columns_to_show if col in display_df.columns]
    
    display_df = display_df[columns_to_show]
    
    # Rename for better display
    display_df = display_df.rename(columns={
        'time': 'Time',
        'type': 'Type',
        'entry_price': 'Entry $',
        'exit_price': 'Exit $',
        'net_pnl': 'Net PnL %',
        'reason': 'Exit Reason',
        'regime': 'Regime',
        'duration': 'Duration'
    })
    
    # Format prices
    if 'Entry $' in display_df.columns:
        display_df['Entry $'] = display_df['Entry $'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    if 'Exit $' in display_df.columns:
        display_df['Exit $'] = display_df['Exit $'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    
    # Format PnL
    if 'Net PnL %' in display_df.columns:
        display_df['Net PnL %'] = display_df['Net PnL %'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
    
    # Display table with color coding
    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True
    )

# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🤖 MONSTER BOT v14.4 - TITAN INTERACTIVE")
    st.markdown("---")

    # --- Đọc dữ liệu mới nhất ---
    state = load_bot_state()

    # Nếu chưa có file (Engine chưa chạy)
    if not state:
        st.warning("⚠️ Đang chờ Monster Engine khởi động... (Chưa thấy file data)")
        time.sleep(2)
        st.rerun()
        return

    # Lấy dữ liệu ra biến
    last_update = state.get('last_update', 'N/A')
    current_price = state.get('current_price', 0)
    regime = state.get('regime', 'Scanning...')
    balance = state.get('balance', 0)
    open_trades = state.get('open_trades', [])
    trade_history = state.get('trade_history', [])
    config = state.get('config', {})

    # --- SIDEBAR (GIỐNG V13) ---
    with st.sidebar:
        st.header("⚙️ LIVE CONFIGURATION")
        st.success("✅ Engine is Running 24/7")
        st.info(f"🕒 Last Update: {last_update}")
        
        # Hiển thị thông số (Read-only)
        if config:
            st.code(json.dumps(config, indent=2), language='json')
        else:
            st.text("Loading Config...")
            
        st.markdown("---")
        st.metric("REAL-TIME EQUITY", f"${balance:,.2f}")

    # --- KHU VỰC METRICS (GIỐNG V13) ---
    # Tính toán chỉ số Winrate từ lịch sử
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    if trade_history:
        df_hist = pd.DataFrame(trade_history)
        # Giả sử trong history có cột 'pnl_percent' hoặc 'net_pnl'
        # Logic đếm win/loss
        wins = len([t for t in trade_history if t.get('pnl', 0) > 0])
        losses = len([t for t in trade_history if t.get('pnl', 0) <= 0])
        total_pnl = sum([t.get('pnl', 0) for t in trade_history])
        
        win_rate = (wins / len(trade_history)) * 100 if len(trade_history) > 0 else 0
    else:
        win_rate = 0.0

    # Hiển thị 4 cột chỉ số
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CURRENT PRICE", f"${current_price:,.2f}", f"{regime}")
    col2.metric("WIN RATE", f"{win_rate:.1f}%")
    col3.metric("WINS / LOSSES", f"{wins} / {losses}")
    col4.metric("TOTAL NET PNL", f"${total_pnl:.2f}")

    # --- KHU VỰC LỆNH ĐANG MỞ (ACTIVE TRADE) ---
    if open_trades:
        st.markdown("### ⚡ ACTIVE POSITIONS")
        for trade in open_trades:
            # Tính toán PnL tạm tính
            entry_price = trade.get('entry_price', current_price)
            side = trade.get('side', 'BUY')
            
            if side == 'BUY':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Màu sắc
            color = "green" if pnl_pct >= 0 else "red"
            
            with st.container():
                c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
                c1.markdown(f"**{trade.get('symbol')} ({side})**")
                c2.markdown(f"Entry: `${entry_price:,.2f}`")
                c3.progress(0.5) # Để tạm 50%, nếu muốn tính chính xác cần TP/SL
                c4.markdown(f":{color}[**{pnl_pct:+.2f}%**]")
                st.caption(f"Reason: {trade.get('entry_reason', 'AI Signal')}")
        st.markdown("---")

    # --- KHU VỰC NHẬT KÝ LỆNH (LOG TABLE) ---
    st.markdown("### 📜 TRADING LOG (LIVE)")
    
    if trade_history:
        # Tạo DataFrame hiển thị
        df_log = pd.DataFrame(trade_history)
        
        # Sắp xếp mới nhất lên đầu
        df_log = df_log.iloc[::-1]
        
        # Format lại bảng cho đẹp
        st.dataframe(
            df_log,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": "Time",
                "symbol": "Symbol",
                "side": "Side",
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
                "pnl_percent": st.column_config.NumberColumn("Net PnL %", format="%.2f%%"),
                "exit_reason": "Reason"
            }
        )
    else:
        st.info("Chưa có lệnh nào được đóng.")

    # --- AUTO REFRESH (Cơ chế tự động làm mới trang) ---
    # Tự động refresh sau mỗi 5 giây để cập nhật giá
    time.sleep(5) 
    st.rerun()

if __name__ == "__main__":
    main()
