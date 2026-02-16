"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER MATRIX UI v15.0 - ADVANCED COMMAND CENTER                       ║
║  🎯 Features: Kill Switch | Analytics Pro | System Monitor | Auto-Backup ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# 🔧 FIX: st.set_page_config MUST be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="MONSTER MATRIX v15.0", layout="wide", page_icon="👾")

import pandas as pd
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
import streamlit.components.v1 as components
import subprocess
import psutil
import plotly.graph_objects as go
import plotly.express as px

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

STATE_FILE = os.path.abspath("bot_state.json")
BACKUP_DIR = "backups"

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def is_bot_running():
    """Check if monster_engine.py is running"""
    try:
        for proc in psutil.process_iter(['cmdline']):
            if proc.info['cmdline'] and 'monster_engine.py' in ' '.join(proc.info['cmdline']):
                return True, proc.pid
    except:
        pass
    return False, None

def kill_bot(pid):
    """Stop the bot process"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        return True
    except:
        try:
            process.kill()
            return True
        except:
            return False

def load_data():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            content = f.read()
            if not content: return None
            return json.loads(content)
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

def backup_state():
    """Create backup of current state"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    if os.path.exists(STATE_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"bot_state_{timestamp}.json")
        try:
            import shutil
            shutil.copy(STATE_FILE, backup_file)
            return True
        except:
            return False
    return False

def calculate_total_pnl(trade_history):
    """Calculate cumulative PnL from trade history"""
    total_pnl = 0
    for trade in trade_history:
        try:
            pnl_str = trade.get('dollar_pnl', '$0.00')
            pnl_value = float(pnl_str.replace('$', '').replace(',', ''))
            total_pnl += pnl_value
        except:
            pass
    return total_pnl

def get_system_stats():
    """Get CPU and RAM usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return cpu_percent, memory.percent

# ════════════════════════════════════════════════════════════════════════════
# MATRIX STYLE CSS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Dark Matrix Theme */
    .stApp { 
        background: linear-gradient(180deg, #0a0e17 0%, #0d1117 100%);
        color: #00ff41;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00ff41 !important;
        text-shadow: 0 0 15px #00ff41;
        font-family: 'Courier New', monospace;
        font-size: 2rem !important;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0a0e17 0%, #1a1f2e 100%);
        border: 2px solid #00ff4133;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 255, 65, 0.15);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #00ff41;
        box-shadow: 0 6px 30px rgba(0, 255, 65, 0.3);
        transform: translateY(-2px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ff41 !important;
        text-shadow: 0 0 20px #00ff41;
        font-family: 'Courier New', monospace;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%);
        color: #0a0e17;
        border: none;
        font-weight: bold;
        padding: 10px 30px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 255, 65, 0.4);
        transition: all 0.3s ease;
        font-family: 'Courier New', monospace;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 255, 65, 0.6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e17 0%, #1a1f2e 100%);
        border-right: 2px solid #00ff4133;
    }
    
    /* DataFrames */
    .dataframe {
        background-color: #0a0e17 !important;
        color: #00ff41 !important;
        border: 1px solid #00ff4133 !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #00ff41;
        box-shadow: 0 0 10px #00ff41;
    }
    
    /* Terminal box */
    .terminal-box {
        background: #000000;
        border: 2px solid #00ff41;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        color: #00ff41;
        max-height: 300px;
        overflow-y: auto;
        box-shadow: 0 4px 20px rgba(0, 255, 65, 0.2);
    }
    
    /* Status badges */
    .status-online {
        display: inline-block;
        padding: 5px 15px;
        background: #00ff41;
        color: #000;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 0 15px #00ff41;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        display: inline-block;
        padding: 5px 15px;
        background: #ff4444;
        color: #fff;
        border-radius: 20px;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Card style */
    .info-card {
        background: linear-gradient(135deg, #0a0e17 0%, #1a1f2e 100%);
        border: 1px solid #00ff4133;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 255, 65, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# BOT STARTUP LOGIC
# ════════════════════════════════════════════════════════════════════════════

# Check if bot is running
bot_is_running, bot_pid = is_bot_running()

# Start bot if not running (only once)
if 'engine_started' not in st.session_state:
    if not bot_is_running:
        with st.spinner("🚀 Đang khởi động Monster Engine..."):
            subprocess.Popen([sys.executable, "monster_engine.py"])
            time.sleep(3)
    st.session_state['engine_started'] = True
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR - CONTROL PANEL
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🎮 CONTROL PANEL")
    st.markdown("---")
    
    # System Status
    st.markdown("### ⚡ SYSTEM STATUS")
    cpu_usage, ram_usage = get_system_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU", f"{cpu_usage:.1f}%", delta=None)
    with col2:
        st.metric("RAM", f"{ram_usage:.1f}%", delta=None)
    
    st.markdown("---")
    
    # Bot Status
    st.markdown("### 🤖 BOT STATUS")
    bot_is_running, bot_pid = is_bot_running()
    
    if bot_is_running:
        st.markdown('<span class="status-online">● ONLINE</span>', unsafe_allow_html=True)
        st.caption(f"PID: {bot_pid}")
    else:
        st.markdown('<span class="status-offline">● OFFLINE</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Kill Switch
    st.markdown("### 🛑 KILL SWITCH")
    
    if bot_is_running:
        if st.button("⚠️ STOP BOT", use_container_width=True):
            with st.spinner("Stopping bot..."):
                if kill_bot(bot_pid):
                    # Update JSON status
                    try:
                        data = load_data()
                        if data:
                            data['bot_status'] = 'Stopped'
                            with open(STATE_FILE, 'w') as f:
                                json.dump(data, f, indent=2)
                    except:
                        pass
                    st.success("✅ Bot stopped successfully!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Failed to stop bot")
    else:
        if st.button("▶️ START BOT", use_container_width=True):
            subprocess.Popen([sys.executable, "monster_engine.py"])
            st.success("✅ Bot starting...")
            time.sleep(2)
            st.rerun()
    
    st.markdown("---")
    
    # Backup System
    st.markdown("### 💾 BACKUP SYSTEM")
    if st.button("📦 CREATE BACKUP", use_container_width=True):
        if backup_state():
            st.success("✅ Backup created!")
        else:
            st.error("❌ Backup failed")
    
    # Show backup count
    if os.path.exists(BACKUP_DIR):
        backup_count = len([f for f in os.listdir(BACKUP_DIR) if f.endswith('.json')])
        st.caption(f"📂 {backup_count} backups available")
    
    st.markdown("---")
    
    # Auto-refresh control
    st.markdown("### 🔄 AUTO-REFRESH")
    refresh_interval = st.slider("Interval (seconds)", 3, 30, 5)
    if st.checkbox("Enable Auto-Refresh", value=True):
        st.session_state['auto_refresh'] = True
    else:
        st.session_state['auto_refresh'] = False

# ════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<h1 style='text-align: center; font-size: 3rem;'>
    👾 MONSTER NEXUS COMMAND CENTER v15.0
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# Load data
data = load_data()

if data:
    # ═══════════════════════════════════════════════════════════════════
    # TOP METRICS ROW
    # ═══════════════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data.get('current_price', 0)
        st.metric("💰 CURRENT PRICE", f"${current_price:,.2f}")
    
    with col2:
        # Calculate win rate
        wr = data.get('win_rate')
        if wr is None:
            history = data.get('trade_history', [])
            if history:
                wins = len([t for t in history if float(str(t.get('net_pnl', '0%')).replace('%', '')) > 0])
                wr = (wins / len(history) * 100)
            else:
                wr = 0
        st.metric("🎯 WIN RATE", f"{wr:.1f}%")
    
    with col3:
        balance = data.get('balance', 0)
        st.metric("💵 BALANCE", f"${balance:,.2f}")
    
    with col4:
        # Calculate total PnL
        total_pnl = calculate_total_pnl(data.get('trade_history', []))
        delta_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("📈 TOTAL PnL", f"${total_pnl:,.2f}", delta=f"{(total_pnl/10000)*100:.2f}%")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════
    # ANALYTICS PRO SECTION
    # ═══════════════════════════════════════════════════════════════════
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("## 📊 ANALYTICS PRO")
        
        # Win/Loss Statistics
        history = data.get('trade_history', [])
        if history:
            wins = len([t for t in history if float(str(t.get('net_pnl', '0%')).replace('%', '')) > 0])
            losses = len(history) - wins
            
            st.markdown(f"**Total Trades:** {len(history)}")
            st.markdown(f"**Wins:** {wins} 🟢")
            st.markdown(f"**Losses:** {losses} 🔴")
            
            # Win/Loss Progress Bar
            if len(history) > 0:
                win_percentage = (wins / len(history)) * 100
                st.progress(win_percentage / 100)
                st.caption(f"Win Rate: {win_percentage:.1f}%")
            
            st.markdown("---")
            
            # PnL Distribution
            pnl_values = []
            for trade in history[:20]:  # Last 20 trades
                try:
                    pnl = float(str(trade.get('net_pnl', '0%')).replace('%', ''))
                    pnl_values.append(pnl)
                except:
                    pass
            
            if pnl_values:
                # Create bar chart
                fig_pnl = go.Figure()
                colors = ['#00ff41' if x > 0 else '#ff4444' for x in pnl_values]
                
                fig_pnl.add_trace(go.Bar(
                    y=pnl_values,
                    marker_color=colors,
                    name='PnL %'
                ))
                
                fig_pnl.update_layout(
                    title="Last 20 Trades PnL Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#00ff41'),
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No trade history yet")
        
        st.markdown("---")
        
        # System Info
        st.markdown("### ⚙️ SYSTEM INFO")
        st.markdown(f"**Status:** {data.get('bot_status', 'Unknown')}")
        st.markdown(f"**Last Update:** {data.get('last_update_time', 'N/A')}")
        st.markdown(f"**Open Trades:** {len(data.get('open_trades', []))}")
        st.markdown(f"**Pending Orders:** {len(data.get('pending_orders', []))}")
    
    with col_right:
        # ═══════════════════════════════════════════════════════════════
        # TRADINGVIEW CHART
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("## 📈 LIVE MARKET FEED")
        
        tv_html = """
        <div style="height:500px; border: 2px solid #00ff41; border-radius: 10px; overflow: hidden;">
            <div id="tv_chart" style="height:100%;"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({
                "autosize": true,
                "symbol": "BINANCE:BTCUSDT",
                "interval": "15",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#0a0e17",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tv_chart",
                "studies": [
                    "RSI@tv-basicstudies",
                    "MACD@tv-basicstudies"
                ]
            });
            </script>
        </div>
        """
        components.html(tv_html, height=520)
        
        # ═══════════════════════════════════════════════════════════════
        # 3D PRICE VISUALIZATION (Optional Enhancement)
        # ═══════════════════════════════════════════════════════════════
        
        if history and len(history) > 5:
            st.markdown("### 🎨 PERFORMANCE HEATMAP")
            
            # Create heatmap data
            recent_trades = history[:20]
            
            sides = []
            pnls = []
            entry_prices = []
            
            for trade in recent_trades:
                sides.append(trade.get('side', 'N/A'))
                try:
                    pnl = float(str(trade.get('net_pnl', '0%')).replace('%', ''))
                    pnls.append(pnl)
                except:
                    pnls.append(0)
                
                try:
                    price_str = trade.get('entry_price', '$0')
                    price = float(price_str.replace('$', '').replace(',', ''))
                    entry_prices.append(price)
                except:
                    entry_prices.append(0)
            
            # Create DataFrame
            df_heatmap = pd.DataFrame({
                'Trade': [f"#{i+1}" for i in range(len(recent_trades))],
                'Side': sides,
                'PnL %': pnls,
                'Entry Price': entry_prices
            })
            
            # Create heatmap
            fig_heat = px.scatter(
                df_heatmap,
                x='Trade',
                y='PnL %',
                size=[abs(x) + 1 for x in pnls],
                color='PnL %',
                color_continuous_scale=['red', 'yellow', 'green'],
                hover_data=['Side', 'Entry Price']
            )
            
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00ff41'),
                height=300
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════
    # TRADE HISTORY TABLE
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("## 📜 RECENT TRADES")
    
    if history:
        # Create DataFrame with enhanced display
        df_history = pd.DataFrame(history[:15])  # Show last 15 trades
        
        # Color-code PnL
        def color_pnl(val):
            try:
                pnl = float(str(val).replace('%', ''))
                if pnl > 0:
                    return f'color: #00ff41; font-weight: bold'
                else:
                    return f'color: #ff4444; font-weight: bold'
            except:
                return ''
        
        # Apply styling
        styled_df = df_history.style.applymap(color_pnl, subset=['net_pnl'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download button
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History (CSV)",
            data=csv,
            file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trade history available yet")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════
    # TERMINAL OUTPUT (Real-time logs simulation)
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("## 💻 SYSTEM TERMINAL")
    
    terminal_output = f"""
    [SYSTEM] Monster Engine v15.0 Active
    [STATUS] Bot Status: {data.get('bot_status', 'Unknown')}
    [PRICE]  Current BTC/USDT: ${data.get('current_price', 0):,.2f}
    [TRADES] Open Positions: {len(data.get('open_trades', []))}
    [ORDERS] Pending Limit Orders: {len(data.get('pending_orders', []))}
    [STATS]  Total Trades Executed: {len(history)}
    [STATS]  Win Rate: {wr:.1f}%
    [STATS]  Total PnL: ${total_pnl:,.2f}
    [MEMORY] CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}%
    [TIME]   Last Update: {data.get('last_update_time', 'N/A')}
    [SYNC]   State File: {STATE_FILE}
    """
    
    st.markdown(f"""
    <div class="terminal-box">
        <pre>{terminal_output}</pre>
    </div>
    """, unsafe_allow_html=True)

else:
    # ═══════════════════════════════════════════════════════════════════
    # WAITING FOR DATA
    # ═══════════════════════════════════════════════════════════════════
    
    st.warning("📡 Waiting for data from Monster Engine...")
    st.info("The bot is fetching market data. This may take 10-15 seconds on first startup.")
    
    with st.spinner("Initializing system..."):
        time.sleep(2)

# ════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH MECHANISM
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.get('auto_refresh', True):
    time.sleep(refresh_interval)
    st.rerun()
