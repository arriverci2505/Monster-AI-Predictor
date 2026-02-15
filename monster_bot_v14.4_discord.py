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
    page_title="Monster Bot v14.4",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Fix Sidebar Label Overlapping */
    [data-testid="stSidebar"] {
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 1rem;
    }
    
    /* Custom Card Styling */
    .status-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #4a90e2;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .trade-card {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00ff41;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        background: rgba(255,255,255,0.05);
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.1rem;
        color: #ffffff;
        font-weight: bold;
    }
    
    .profit-positive {
        color: #00ff41 !important;
        font-weight: bold;
    }
    
    .profit-negative {
        color: #ff0000 !important;
        font-weight: bold;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .status-running {
        background: #00ff41;
        color: #000;
    }
    
    .status-stopped {
        background: #ff0000;
        color: #fff;
    }
    
    .status-error {
        background: #ffa500;
        color: #000;
    }
    
    /* Progress Bar Custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff0000 0%, #ffa500 50%, #00ff41 100%);
    }
    
    /* Table Styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00ff41;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

STATE_FILE = "bot_state.json"

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
    """Main Streamlit app"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Settings")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        if auto_refresh:
            refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 15)
        
        st.markdown("---")
        
        st.markdown("### 📊 System Info")
        st.info("""
        **Monster Bot v14.4**
        
        ✅ Cloud-optimized architecture
        ✅ Read-only dashboard
        ✅ Real-time state monitoring
        
        **Features:**
        - Smart Exit Logic
        - Profit Lock System
        - Trailing Stop
        - Discord Alerts
        """)
        
        st.markdown("---")
        
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    
    # Load state
    state = load_bot_state()
    
    if state is None:
        st.error("Failed to load bot state. Please check if monster_engine.py is running.")
        return
    
    # Display header
    display_header(state)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Active trades
        if state.get('open_trades'):
            for trade in state['open_trades']:
                # Try to get current price from last known state
                # In production, you might fetch this from exchange
                display_active_trade(trade)
        else:
            st.info("🔍 No active positions")
        
        # Pending orders
        if state.get('pending_orders'):
            display_pending_orders(state['pending_orders'])
    
    with col2:
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        
        balance = state.get('balance', 10000.0)
        initial_balance = 10000.0
        total_pnl = ((balance - initial_balance) / initial_balance) * 100
        
        st.metric(
            "Portfolio PnL",
            f"{total_pnl:+.2f}%",
            delta=f"${balance - initial_balance:+,.2f}"
        )
        
        st.metric("Open Positions", len(state.get('open_trades', [])))
        st.metric("Pending Orders", len(state.get('pending_orders', [])))
        
        history = state.get('trade_history', [])
        st.metric("Total Trades", len(history))
    
    # Trade history (full width)
    st.markdown("---")
    display_trade_history(state.get('trade_history', []))
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
