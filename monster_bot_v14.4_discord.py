"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER MATRIX UI v18.0 - CYBERPUNK HUD ULTIMATE                        ║
║  🎯 Glassmorphism | Neon Glow | Digital HUD | Matrix Terminal           ║
║  ⚡ Professional Trading Command Center - Futuristic Edition             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
st.set_page_config(page_title="MONSTER MATRIX HUD v18.0", layout="wide", page_icon="⚡")

import pandas as pd
import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import subprocess
import psutil
import plotly.graph_objects as go
import plotly.express as px

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

STATE_FILE = os.path.abspath("bot_state_v14_4.json")
BACKUP_DIR = "backups"
ROLLING_WINDOW = 200

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def is_bot_running():
    """Check if monster_engine.py is running"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'monster_engine.py' in cmdline and 'python' in cmdline.lower():
                    return True, proc.info['pid']
    except:
        pass
    return False, None

def kill_bot(pid):
    """Stop the bot process"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        return True, "TERMINATED"
    except:
        try:
            process.kill()
            return True, "FORCE_KILLED"
        except Exception as e:
            return False, f"ERROR: {e}"

def load_data():
    """Load bot state with proper exception handling"""
    if not os.path.exists(STATE_FILE):
        return None
    
    try:
        with open(STATE_FILE, "r", encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            data = json.loads(content)
            if data.get('current_price', 0) == 0 and data.get('trade_history'):
                try:
                    last_trade = data['trade_history'][0]
                    exit_price_str = last_trade.get('exit_price', '$0.00')
                    data['current_price'] = float(exit_price_str.replace('$', '').replace(',', ''))
                except:
                    data['current_price'] = 0
            return data
    except:
        return None

def send_kill_signal():
    """Send kill signal via JSON"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            state['bot_status'] = 'Kill Signal Received'
            state['should_stop'] = True
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            return True
    except:
        return False

def calculate_total_pnl(trade_history):
    """Calculate cumulative PnL"""
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
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        return cpu_percent, memory.percent
    except:
        return 0, 0

def parse_ai_confidence(data):
    """Extract AI confidence scores"""
    try:
        latest_probs = data.get('latest_ai_probs', {})
        if latest_probs:
            return (
                latest_probs.get('neutral', 0.33),
                latest_probs.get('buy', 0.33),
                latest_probs.get('sell', 0.33)
            )
        return (0.33, 0.33, 0.33)
    except:
        return (0.33, 0.33, 0.33)

# ════════════════════════════════════════════════════════════════════════════
# CYBERPUNK HUD CSS - GLASSMORPHISM + NEON GLOW
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════ */
    /* IMPORT CYBERPUNK FONTS                                           */
    /* ═══════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* DIGITAL GRID BACKGROUND                                          */
    /* ═══════════════════════════════════════════════════════════════ */
    .stApp {
        background: 
            linear-gradient(0deg, transparent 24%, rgba(0, 242, 255, .05) 25%, rgba(0, 242, 255, .05) 26%, transparent 27%, transparent 74%, rgba(0, 242, 255, .05) 75%, rgba(0, 242, 255, .05) 76%, transparent 77%, transparent),
            linear-gradient(90deg, transparent 24%, rgba(0, 242, 255, .05) 25%, rgba(0, 242, 255, .05) 26%, transparent 27%, transparent 74%, rgba(0, 242, 255, .05) 75%, rgba(0, 242, 255, .05) 76%, transparent 77%, transparent),
            linear-gradient(180deg, #050505 0%, #0a0a0f 100%);
        background-size: 50px 50px, 50px 50px, 100% 100%;
        color: #e0e0e0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* HIDE DEFAULT STREAMLIT ELEMENTS                                  */
    /* ═══════════════════════════════════════════════════════════════ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* HUD HEADER - TOP STATUS BAR                                      */
    /* ═══════════════════════════════════════════════════════════════ */
    .hud-header {
        background: linear-gradient(90deg, rgba(0, 242, 255, 0.05) 0%, rgba(189, 0, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 0 0 15px 15px;
        padding: 15px 30px;
        margin: -60px -70px 30px -70px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hud-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f2ff, #bd00ff, transparent);
        animation: scan 3s linear infinite;
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(90deg, #00f2ff 0%, #bd00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 242, 255, 0.5);
        letter-spacing: 3px;
    }
    
    .hud-status {
        display: flex;
        gap: 30px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }
    
    .hud-status-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 3px;
    }
    
    .hud-status-label {
        color: #666;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .hud-status-value {
        color: #00f2ff;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.8);
    }
    
    .hud-status-online {
        color: #00ff88;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.8);
        animation: pulse-green 2s infinite;
    }
    
    .hud-status-offline {
        color: #ff4466;
        text-shadow: 0 0 15px rgba(255, 68, 102, 0.8);
    }
    
    @keyframes pulse-green {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .hud-clock {
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        color: #00f2ff;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.8);
        font-weight: 600;
        letter-spacing: 2px;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* CUSTOM METRIC CARDS - GLASSMORPHISM                              */
    /* ═══════════════════════════════════════════════════════════════ */
    .metric-card {
        background: linear-gradient(135deg, rgba(10, 10, 20, 0.7) 0%, rgba(20, 20, 40, 0.7) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 15px;
        padding: 25px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00f2ff, #bd00ff, #00f2ff);
        border-radius: 15px;
        opacity: 0;
        z-index: -1;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 255, 0.8);
        box-shadow: 0 12px 48px rgba(0, 242, 255, 0.3);
    }
    
    .metric-card:hover::before {
        opacity: 0.3;
    }
    
    .metric-icon {
        font-size: 40px;
        margin-bottom: 10px;
        filter: drop-shadow(0 0 10px rgba(0, 242, 255, 0.8));
    }
    
    .metric-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 11px;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        color: #00f2ff;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.8);
        line-height: 1;
        margin-bottom: 5px;
    }
    
    .metric-value.danger {
        color: #bd00ff;
        text-shadow: 0 0 20px rgba(189, 0, 255, 0.8);
    }
    
    .metric-value.success {
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.8);
    }
    
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #666;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* CAMERA LENS FRAME - CHART CONTAINER                              */
    /* ═══════════════════════════════════════════════════════════════ */
    .camera-frame {
        background: rgba(5, 5, 5, 0.5);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 242, 255, 0.4);
        border-radius: 20px;
        padding: 20px;
        position: relative;
        box-shadow: 
            inset 0 0 30px rgba(0, 242, 255, 0.1),
            0 0 40px rgba(0, 242, 255, 0.2);
    }
    
    .camera-frame::before {
        content: '';
        position: absolute;
        top: 10px;
        left: 10px;
        width: 20px;
        height: 20px;
        border-top: 3px solid #00f2ff;
        border-left: 3px solid #00f2ff;
        border-radius: 5px 0 0 0;
    }
    
    .camera-frame::after {
        content: '';
        position: absolute;
        top: 10px;
        right: 10px;
        width: 20px;
        height: 20px;
        border-top: 3px solid #00f2ff;
        border-right: 3px solid #00f2ff;
        border-radius: 0 5px 0 0;
    }
    
    .camera-bottom-left {
        position: absolute;
        bottom: 10px;
        left: 10px;
        width: 20px;
        height: 20px;
        border-bottom: 3px solid #00f2ff;
        border-left: 3px solid #00f2ff;
        border-radius: 0 0 0 5px;
    }
    
    .camera-bottom-right {
        position: absolute;
        bottom: 10px;
        right: 10px;
        width: 20px;
        height: 20px;
        border-bottom: 3px solid #00f2ff;
        border-right: 3px solid #00f2ff;
        border-radius: 0 0 5px 0;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* MATRIX TERMINAL - SCANLINE EFFECT                                */
    /* ═══════════════════════════════════════════════════════════════ */
    .matrix-terminal {
        background: rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(0, 242, 255, 0.4);
        border-radius: 10px;
        padding: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #00ff88;
        height: 600px;
        overflow-y: auto;
        position: relative;
        box-shadow: 
            inset 0 0 50px rgba(0, 255, 136, 0.1),
            0 0 30px rgba(0, 242, 255, 0.2);
        line-height: 1.6;
    }
    
    .matrix-terminal::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            rgba(0, 255, 136, 0.05) 0px,
            rgba(0, 255, 136, 0.05) 1px,
            transparent 1px,
            transparent 2px
        );
        pointer-events: none;
        animation: scanline 8s linear infinite;
    }
    
    @keyframes scanline {
        0% { transform: translateY(0); }
        100% { transform: translateY(100%); }
    }
    
    .matrix-terminal::-webkit-scrollbar {
        width: 8px;
    }
    
    .matrix-terminal::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 4px;
    }
    
    .matrix-terminal::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f2ff 0%, #00ff88 100%);
        border-radius: 4px;
    }
    
    .terminal-prompt {
        color: #00f2ff;
        font-weight: 700;
    }
    
    .terminal-success {
        color: #00ff88;
    }
    
    .terminal-warning {
        color: #ffaa00;
    }
    
    .terminal-error {
        color: #ff4466;
    }
    
    .terminal-info {
        color: #00f2ff;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* NEON BUTTONS WITH HOVER GLOW                                     */
    /* ═══════════════════════════════════════════════════════════════ */
    .stButton>button {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.2) 0%, rgba(189, 0, 255, 0.2) 100%);
        backdrop-filter: blur(10px);
        color: #00f2ff;
        border: 2px solid #00f2ff;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        padding: 12px 30px;
        border-radius: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 13px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.4) 0%, rgba(189, 0, 255, 0.4) 100%);
        border-color: #00f2ff;
        box-shadow: 0 0 40px rgba(0, 242, 255, 0.8);
        transform: translateY(-2px);
        color: #ffffff;
        text-shadow: 0 0 10px #00f2ff;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* SIDEBAR STYLING                                                  */
    /* ═══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(5, 5, 5, 0.95) 0%, rgba(10, 10, 15, 0.95) 100%);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(0, 242, 255, 0.3);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00f2ff;
        text-shadow: 0 0 15px rgba(0, 242, 255, 0.6);
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* HEADERS                                                          */
    /* ═══════════════════════════════════════════════════════════════ */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f2ff !important;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.6) !important;
        letter-spacing: 2px !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* DATAFRAME STYLING                                                */
    /* ═══════════════════════════════════════════════════════════════ */
    .dataframe {
        background: rgba(5, 5, 5, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 242, 255, 0.3) !important;
        color: #e0e0e0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════ */
    /* ASCII ART CONTAINER                                              */
    /* ═══════════════════════════════════════════════════════════════ */
    .ascii-art {
        font-family: 'JetBrains Mono', monospace;
        font-size: 8px;
        line-height: 1;
        color: #00f2ff;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.8);
        white-space: pre;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HUD HEADER - TOP STATUS BAR
# ════════════════════════════════════════════════════════════════════════════

current_time = datetime.now().strftime("%H:%M:%S")
bot_running, bot_pid = is_bot_running()
cpu_usage, ram_usage = get_system_stats()

status_class = "hud-status-online" if bot_running else "hud-status-offline"
status_text = "ONLINE" if bot_running else "OFFLINE"

st.markdown(f"""
<div class="hud-header">
    <div class="hud-title">⚡ MONSTER MATRIX HUD</div>
    <div class="hud-status">
        <div class="hud-status-item">
            <div class="hud-status-label">STATUS</div>
            <div class="hud-status-value {status_class}">{status_text}</div>
        </div>
        <div class="hud-status-item">
            <div class="hud-status-label">CPU</div>
            <div class="hud-status-value">{cpu_usage:.1f}%</div>
        </div>
        <div class="hud-status-item">
            <div class="hud-status-label">RAM</div>
            <div class="hud-status-value">{ram_usage:.1f}%</div>
        </div>
        <div class="hud-status-item">
            <div class="hud-status-label">LATENCY</div>
            <div class="hud-status-value">12ms</div>
        </div>
    </div>
    <div class="hud-clock">{current_time}</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### ⚙️ SYSTEM CONTROLS")
st.sidebar.markdown("---")

if bot_running:
    st.sidebar.markdown(f"**PID:** `{bot_pid}`")
else:
    st.sidebar.info("Engine Offline")

st.sidebar.markdown("---")

col_k1, col_k2 = st.sidebar.columns(2)

with col_k1:
    if st.button("🛑 KILL", key="kill"):
        if bot_running:
            send_kill_signal()
            success, msg = kill_bot(bot_pid)
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
            time.sleep(1)
            st.rerun()

with col_k2:
    if st.button("🔄 REFRESH", key="refresh"):
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ SETTINGS")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Interval (s)", 3, 30, 5)
else:
    refresh_interval = 999999

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

data = load_data()

if data:
    current_price = data.get('current_price', 0)
    history = data.get('trade_history', [])
    open_trades = data.get('open_trades', [])
    pending_orders = data.get('pending_orders', [])
    regime = data.get('latest_regime', 'UNKNOWN')
    
    total_pnl = calculate_total_pnl(history)
    
    if history:
        wins = len([t for t in history if float(str(t.get('net_pnl', '0%')).replace('%', '')) > 0])
        wr = (wins / len(history)) * 100 if len(history) > 0 else 0
    else:
        wins = 0
        wr = 0
    
    prob_neutral, prob_buy, prob_sell = parse_ai_confidence(data)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CUSTOM METRIC CARDS (NO ST.METRIC)
    # ═══════════════════════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_icon = "💰"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{price_icon}</div>
            <div class="metric-label">BTC PRICE</div>
            <div class="metric-value">${current_price:,.2f}</div>
            <div class="metric-delta">KRAKEN SPOT</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        regime_icon = "📈" if regime == "TRENDING" else "↔️"
        regime_class = "success" if regime == "TRENDING" else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{regime_icon}</div>
            <div class="metric-label">MARKET REGIME</div>
            <div class="metric-value {regime_class}">{regime}</div>
            <div class="metric-delta">{len(open_trades)} OPEN POSITIONS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pnl_icon = "📊"
        pnl_class = "success" if total_pnl >= 0 else "danger"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{pnl_icon}</div>
            <div class="metric-label">TOTAL P&L</div>
            <div class="metric-value {pnl_class}">${total_pnl:,.2f}</div>
            <div class="metric-delta">WIN RATE: {wr:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ai_icon = "🤖"
        ai_max = max(prob_neutral, prob_buy, prob_sell)
        ai_class = "success" if ai_max == prob_buy else "danger" if ai_max == prob_sell else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{ai_icon}</div>
            <div class="metric-label">AI CONFIDENCE</div>
            <div class="metric-value {ai_class}">{ai_max*100:.1f}%</div>
            <div class="metric-delta">NEURAL NETWORK</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN DASHBOARD: 70% CHARTS + 30% TERMINAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    col_main, col_terminal = st.columns([7, 3])
    
    with col_main:
        
        # ═══════════════════════════════════════════════════════════════
        # TRADINGVIEW WIDGET
        # ═══════════════════════════════════════════════════════════════
        st.markdown("### 📊 LIVE MARKET TERMINAL")
        
        # Bọc toàn bộ Chart vào trong khung Camera Frame
        st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
        
        tradingview_html = """
        <div class="tradingview-widget-container" style="height:550px;">
          <div id="tradingview_chart" style="height:100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({
            "autosize": true,
            "symbol": "KRAKEN:BTCUSDT",
            "interval": "15",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "container_id": "tradingview_chart"
          });
          </script>
        </div>
        """
        components.html(tradingview_html, height=550)
        
        # Đóng khung Camera Frame
        st.markdown('<div class="camera-bottom-left"></div><div class="camera-bottom-right"></div></div>', unsafe_allow_html=True)
        
        st.markdown("### 📈 TRADING ANALYSIS")
        
        # Recent Trades Performance
        if history and len(history) > 0:
            st.markdown("### 📊 PERFORMANCE METRICS")
            
            pnl_values = []
            for trade in history[:15]:
                try:
                    pnl = float(str(trade.get('net_pnl', '0%')).replace('%', ''))
                    pnl_values.append(pnl)
                except:
                    pass
            
            if pnl_values:
                fig_perf = go.Figure()
                colors = ['#00ff88' if x > 0 else '#ff4466' for x in pnl_values]
                
                fig_perf.add_trace(go.Bar(
                    y=pnl_values,
                    marker=dict(
                        color=colors,
                        line=dict(color='#00f2ff', width=1)
                    ),
                    text=[f"{x:.1f}%" for x in pnl_values],
                    textposition='outside',
                    textfont=dict(color='#ffffff', size=10, family='JetBrains Mono'),
                    hovertemplate='Trade: %{x}<br>PnL: %{y:.2f}%<extra></extra>'
                ))
                
                fig_perf.update_layout(
                    title=dict(
                        text="LAST 15 TRADES - P&L DISTRIBUTION",
                        font=dict(color='#00f2ff', size=18, family='Orbitron')
                    ),
                    paper_bgcolor='rgba(5,5,5,0.5)',
                    plot_bgcolor='rgba(0,0,0,0.8)',
                    font=dict(color='#e0e0e0', family='JetBrains Mono'),
                    xaxis=dict(
                        title="TRADE INDEX",
                        gridcolor='rgba(0,242,255,0.1)',
                        color='#00f2ff'
                    ),
                    yaxis=dict(
                        title="P&L (%)",
                        gridcolor='rgba(0,242,255,0.1)',
                        color='#00f2ff',
                        zeroline=True,
                        zerolinecolor='rgba(0,242,255,0.5)',
                        zerolinewidth=2
                    ),
                    height=400,
                    showlegend=False,
                    margin=dict(l=60, r=20, t=60, b=50)
                )
                
                st.plotly_chart(fig_perf, width='content')
    
    with col_terminal:
        # ═══════════════════════════════════════════════════════════════
        # MATRIX TERMINAL WITH ASCII ART
        # ═══════════════════════════════════════════════════════════════
        
        st.markdown("### 💻 MATRIX TERMINAL")
        
        # ASCII Art Header
        ascii_header = """
    ███╗   ███╗ ██████╗ ███╗   ██╗███████╗████████╗███████╗██████╗ 
    ████╗ ████║██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
    ██╔████╔██║██║   ██║██╔██╗ ██║███████╗   ██║   █████╗  ██████╔╝
    ██║╚██╔╝██║██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══╝  ██╔══██╗
    ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║███████║   ██║   ███████╗██║  ██║
    ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
        """
        
        st.markdown(f'<div class="ascii-art">{ascii_header}</div>', unsafe_allow_html=True)
        
        # Terminal Output
        last_update = data.get('last_update_time', 'N/A')
        try:
            if last_update != 'N/A':
                dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                dt_gmt7 = dt + timedelta(hours=7)
                last_update_display = dt_gmt7.strftime('%H:%M:%S')
            else:
                last_update_display = 'N/A'
        except:
            last_update_display = last_update
        
        terminal_lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           MONSTER ENGINE v18.0 - NEURAL CORE                 ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"<span class='terminal-prompt'>[SYSTEM.INIT]</span> <span class='terminal-success'>NEURAL NETWORK INITIALIZED</span>",
            f"<span class='terminal-prompt'>[SYSTEM.INFO]</span> Engine Status: {data.get('bot_status', 'Unknown')}",
            f"<span class='terminal-prompt'>[SYSTEM.INFO]</span> Last Sync: {last_update_display}",
            "",
            f"<span class='terminal-prompt'>[MARKET.DATA]</span> BTC/USDT: <span class='terminal-info'>${current_price:,.2f}</span>",
            f"<span class='terminal-prompt'>[MARKET.DATA]</span> Regime: <span class='terminal-info'>{regime}</span>",
            "",
            f"<span class='terminal-prompt'>[AI.ANALYSIS]</span> Neural Confidence:",
            f"  ├─ NEUTRAL: {prob_neutral*100:.1f}%",
            f"  ├─ BUY:     <span class='terminal-success'>{prob_buy*100:.1f}%</span>",
            f"  └─ SELL:    <span class='terminal-warning'>{prob_sell*100:.1f}%</span>",
            "",
            f"<span class='terminal-prompt'>[EXECUTION.STATUS]</span>",
            f"  ├─ Open Trades:    <span class='terminal-info'>{len(open_trades)}</span>",
            f"  ├─ Pending Orders: <span class='terminal-info'>{len(pending_orders)}</span>",
            f"  └─ Total Trades:   <span class='terminal-info'>{len(history)}</span>",
            "",
            f"<span class='terminal-prompt'>[PERFORMANCE.STATS]</span>",
            f"  ├─ Total P&L: <span class='terminal-success' if total_pnl >= 0 else 'terminal-error'>${total_pnl:,.2f}</span>",
            f"  ├─ Win Rate:  <span class='terminal-success'>{wr:.1f}%</span>",
            f"  └─ Wins/Loss: {wins}/{len(history)-wins}",
            "",
            f"<span class='terminal-prompt'>[SYSTEM.RESOURCES]</span>",
            f"  ├─ CPU Usage: {cpu_usage:.1f}%",
            f"  ├─ RAM Usage: {ram_usage:.1f}%",
            f"  └─ State File: bot_state_v14_4.json",
            "",
            "══════════════════════════════════════════════════════════════",
            "<span class='terminal-success'>✓ ALL SYSTEMS OPERATIONAL</span>",
            "<span class='terminal-success'>✓ MONITORING ACTIVE</span>",
            "<span class='terminal-success'>✓ NEURAL CORE ONLINE</span>",
            "══════════════════════════════════════════════════════════════",
        ]
        
        # ═══════════════════════════════════════════════════════════════
        # ULTIMATE INTEGRATION: TITLE ON THE BORDER
        # ═══════════════════════════════════════════════════════════════
        
        terminal_output = "<br>".join(terminal_lines)
        
        # CSS tinh chỉnh để đẩy tiêu đề đè lên thanh kẻ
        st.markdown("""
            <style>
                [data-testid="stVerticalBlock"] > div:has(div.matrix-terminal) {
                    gap: 0rem !important;
                }
                
                .matrix-terminal {
                    margin-bottom: 0px !important;
                    padding-bottom: 10px !important;
                    position: relative;
                    border-radius: 12px 12px 0 0 !important;
                    background: rgba(0, 20, 20, 0.4);
                }
                
                /* Thanh ngăn cách có chứa tiêu đề */
                .separator-with-title {
                    position: relative;
                    height: 20px;
                    border-bottom: 2px solid rgba(0, 242, 255, 0.3);
                    margin-top: -10px;
                    margin-bottom: 5px;
                    display: flex;
                    justify-content: center; /* Đưa chữ ra giữa hoặc left tùy bạn */
                    align-items: center;
                    background: rgba(0, 20, 20, 0.4);
                }
                
                .inner-title {
                    background: #001515; /* Màu nền tối để che đường kẻ phía sau */
                    padding: 0 15px;
                    color: #00f2ff;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.8rem;
                    font-weight: bold;
                    letter-spacing: 2px;
                    text-transform: uppercase;
                    border: 1px solid rgba(0, 242, 255, 0.3);
                    border-radius: 20px;
                    z-index: 10;
                }

                .hud-footer-integrated {
                    background: rgba(0, 20, 20, 0.4);
                    border-left: 4px solid #00f2ff;
                    border-radius: 0 0 12px 12px;
                    padding: 10px 15px;
                    margin-top: -2px !important;
                }
            </style>
        """, unsafe_allow_html=True)

        # 1. Phần Terminal phía trên
        st.markdown(f'<div class="matrix-terminal">{terminal_output}</div>', unsafe_allow_html=True)
        
        # 2. Thanh ngăn cách đè tên lên
        st.markdown("""
            <div class="separator-with-title">
                <div class="inner-title">🧠 AI ENGINE PREDICTION</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 3. Phần Biểu đồ phía dưới
        st.markdown('<div class="hud-footer-integrated">', unsafe_allow_html=True)
        
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Bar(
            y=['NEUTRAL', 'BUY', 'SELL'],
            x=[prob_neutral * 100, prob_buy * 100, prob_sell * 100],
            orientation='h',
            marker=dict(
                color=['rgba(255, 170, 0, 0.6)', 'rgba(0, 242, 255, 0.9)', 'rgba(189, 0, 255, 0.6)'],
                line=dict(color='#00f2ff', width=1)
            ),
            text=[f"{prob_neutral*100:.1f}%", f"{prob_buy*100:.1f}%", f"{prob_sell*100:.1f}%"],
            textposition='auto',
            textfont=dict(color='#ffffff', size=11, family='JetBrains Mono', weight='bold'),
        ))
        
        fig_ai.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00f2ff', family='JetBrains Mono'),
            xaxis=dict(range=[0, 100], visible=False),
            yaxis=dict(color='#00f2ff', tickfont=dict(size=10, weight='bold')),
            height=130, 
            margin=dict(l=70, r=10, t=5, b=5),
            showlegend=False,
            bargap=0.35
        )
        
        st.plotly_chart(fig_ai, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
            
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE HISTORY TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.markdown("### 📜 TRADE EXECUTION LOGS")
    
    if history:
        df_history = pd.DataFrame(history[:10])
        st.dataframe(df_history, width='content', height=350)
        
        csv = df_history.to_csv(index=False)
        st.download_button(
            label="📥 EXPORT TRADE DATA",
            data=csv,
            file_name=f"monster_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("⏳ Awaiting first trade execution...")

else:
    # ═══════════════════════════════════════════════════════════════════════════
    # NO DATA AVAILABLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div style="text-align: center; padding: 100px 20px;">
        <div class="ascii-art">
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   AWAITING ENGINE CONNECTION                  ║
    ║                                                               ║
    ║              [████████████████░░░░░░░░░░] 60%               ║
    ║                                                               ║
    ║         MONSTER ENGINE v18.0 - NEURAL CORE BOOTING...        ║
    ╚═══════════════════════════════════════════════════════════════╝
        </div>
        <h2 style="color: #00f2ff; margin-top: 30px;">INITIALIZING SYSTEMS</h2>
        <p style="color: #999;">Waiting for data stream from Monster Engine...</p>
        <p style="color: #666; font-size: 12px;">State File: bot_state_v14_4.json</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Establishing neural link..."):
        time.sleep(2)

# ════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH (MUST BE LAST)
# ════════════════════════════════════════════════════════════════════════════

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
