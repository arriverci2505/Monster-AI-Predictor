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
from datetime import datetime, timedelta, timezone
import streamlit.components.v1 as components
import subprocess
import psutil
import plotly.graph_objects as go
import plotly.express as px

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

STATE_FILE = os.path.abspath("bot_state_v14_5_1.json")
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

def start_engine():
    """Khởi động Engine độc lập hoàn toàn với giao diện UI"""
    try:
        engine_path = "monster_engine.py"
        if not os.path.exists(engine_path):
            st.sidebar.error(f"❌ Không tìm thấy file: {engine_path}")
            return

        # Sử dụng nohup để tiến trình không bị kill khi người dùng đóng trình duyệt
        # 'python -u' giúp log được đẩy ra file ngay lập tức (unbuffered)
        cmd = f"nohup {sys.executable} -u {engine_path} > engine_debug.log 2>&1 &"
        
        # Thực thi lệnh hệ thống
        subprocess.Popen(cmd, shell=True)
        
        st.sidebar.success("🚀 Engine đã được kích hoạt chạy ngầm!")
        time.sleep(2) # Đợi hệ thống phản hồi
        st.rerun()    # Làm mới lại UI
    except Exception as e:
        st.sidebar.error(f"Lỗi khởi động: {e}")
        
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

def restart_bot():
    """Kill existing engine and start a new one"""
    try:
        # Tìm và tắt các tiến trình liên quan đến monster_engine.py
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['cmdline'] and 'monster_engine.py' in ' '.join(proc.info['cmdline']):
                os.kill(proc.info['pid'], 9)
        
        # Khởi chạy lại engine (đảm bảo file monster_engine.py nằm cùng thư mục)
        subprocess.Popen([sys.executable, "monster_engine.py"])
        return True
    except Exception as e:
        st.error(f"Error restarting bot: {e}")
        return False
        
# [Tìm hàm load_data cũ và thay thế bằng đoạn này]

def load_data():
    """Load bot state - Lấy dữ liệu để hàm calculate_total_pnl chạy được"""
    if not os.path.exists(STATE_FILE):
        return None
    
    try:
        with open(STATE_FILE, "r", encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            data = json.loads(content)
            
        # Lấy danh sách lịch sử từ Bot
        history = data.get('trade_history', [])
        
        # Đảm bảo gán lại vào data để hàm tính toán bên ngoài có thể truy cập
        data['trade_history'] = history

        # Lấy xác suất AI liên tục (mỗi phút) từ state['ai_probs']
        # Điều này giúp UI nhảy số 43% ngay cả khi chưa vào lệnh
        if 'ai_probs' not in data:
            data['ai_probs'] = {'neutral': 0.33, 'buy': 0.0, 'sell': 0.0}
            
        return data
    except Exception:
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
            pnl_value = trade.get('pnl_pct', 0)
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
    """Trích xuất xác suất AI liên tục từ trạng thái mới nhất của Bot"""
    try:
        live_probs = data.get('ai_probs', {})
        if live_probs:
            return (
                live_probs.get('neutral', 0.33),
                live_probs.get('buy', 0.0),
                live_probs.get('sell', 0.0)
            )

        return (0.33, 0.33, 0.33)
    except Exception:
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
    [data-testid="stHeader"] {
        visibility: visible !important;
        background: rgba(5, 5, 5, 0.8) !important;
        display: flex !important;
    }
    [data-testid="stHeader"] button {
        color: #00f2ff !important;
    }
    
    /* Hiện lại Sidebar bên trái */
    [data-testid="stSidebar"] {
        visibility: visible !important;
        transform: translateX(0) !important;
    }
    
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
        overflow: visible !important;
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
        height: auto;
        min-height: 550px;
        overflow-y: hidden;
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
    /* ═══════════════════════════════════════════════════════════════ */
    /* CHỈ KÍCH HOẠT KHI XEM TRÊN ĐIỆN THOẠI (KHÔNG LỖI PC)             */
    /* ═══════════════════════════════════════════════════════════════ */
    @media (max-width: 768px) {
        /* 1. Ép các cột Streamlit xếp chồng lên nhau */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 15px !important;
        }

        /* Fix lỗi thanh Sidebar bên trái trên Mobile */
        [data-testid="stSidebar"] {
            width: 80% !important; /* Không cho nó chiếm hết 100% màn hình mobile */
        }
    
        /* Đưa nút đóng/mở Sidebar lên lớp trên cùng để không bị HUD đè */
        [data-testid="stSidebarCollapsedControl"] {
            z-index: 999999 !important;
            background-color: rgba(10, 10, 15, 0.8) !important;
            border-radius: 0 10px 10px 0 !important;
            top: 10px !important;
        }
    
        /* Quan trọng: Reset lại Margin âm của HUD trên Mobile để không đè Sidebar */
        .hud-header {
            margin: 0 !important; /* Bỏ margin -60px -70px trên điện thoại */
            width: 100% !important;
            border-radius: 0 !important;
        }
    
        /* Đẩy toàn bộ nội dung xuống dưới một chút để không bị nút Sidebar đè */
        .stApp {
            padding-top: 50px !important;
        }
    
        /* 2. Fix Header HUD bị tràn mép màn hình mobile */
        .hud-header {
            margin: -60px -10px 30px -10px !important; /* Giảm âm lề trái/phải */
            padding: 10px 15px !important;
            flex-direction: column !important; /* Xếp dọc các item trong HUD */
            gap: 10px;
        }

        .hud-title {
            font-size: 18px !important; /* Nhỏ lại để không xuống dòng */
            letter-spacing: 1px !important;
        }

        .hud-status {
            gap: 15px !important;
            flex-wrap: wrap;
            justify-content: center;
        }

        /* 3. Metric Card - Thu nhỏ để vừa màn hình dọc */
        .metric-card {
            padding: 15px !important;
            min-height: 140px !important;
            margin-bottom: 20px !important;
            width: 100% !important;
        }

        .metric-value {
            font-size: 24px !important; /* 32px trên PC là quá to cho mobile */
        }

        .metric-icon {
            font-size: 28px !important;
        }

        /* 4. Terminal và Camera Frame */
        .camera-frame {
            padding: 10px !important;
        }
        
        .matrix-terminal {
            min-height: 300px !important;
            font-size: 10px !important;
        }

        /* 5. Ẩn bớt các khoảng trống thừa của Streamlit trên mobile */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HUD HEADER - TOP STATUS BAR
# ════════════════════════════════════════════════════════════════════════════

vn_time = datetime.now(timezone.utc) + timedelta(hours=7)
current_time = vn_time.strftime("%H:%M:%S")
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

with st.sidebar:
    
    # Nút Restart to nhất, nổi bật
    if st.button("🚀 RESTART MONSTER ENGINE"):
        with st.spinner("Re-linking neural core..."):
            if restart_bot():
                st.success("Engine Restarted!")
                time.sleep(1)
                st.rerun()

    # Chia 2 cột cho các nút chức năng phụ
    col_k1, col_k2 = st.columns(2)
    
    with col_k1:
        if st.button("🛑 KILL BOT", key="kill"):
            if bot_running:
                send_kill_signal()
                success, msg = kill_bot(bot_pid)
                if success: st.success("Killed")
                else: st.error("Failed")
                time.sleep(1)
                st.rerun()

    with col_k2:
        if st.button("🔄 REFRESH", key="refresh"):
            st.rerun()

    # Nút dọn dẹp để ở dưới cùng của nhóm control
    if st.button("🧹 CLEAR STATE DATA"):
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            st.warning("Data cleared!")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ SETTINGS")
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Interval (s)", 3, 30, 5)

    if bot_running:
        st.sidebar.markdown(f"**PID:** `{bot_pid}`")
    else:
        st.sidebar.info("Engine Offline")
    
# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

data = load_data()

if data:
    current_price = data.get('current_price', 0)
    history = data.get('trade_history', [])
    open_trades = data.get('open_trades', [])
    pending_orders = data.get('pending_orders', [])
    ai_probs = data.get('ai_probs', {})
    regime = ai_probs.get('regime', 'UNKNOWN')
    
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
        if regime == "TRENDING":
            regime_icon = "📈"
            regime_class = "success"
            regime_display = "TRENDING"
        elif regime == "SIDEWAY":
            regime_icon = "↔️"
            regime_class = "info"
            regime_display = "SIDEWAY"
        else:
            regime_icon = "🔍"
            regime_class = "warning"
            regime_display = "SCANNING"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{regime_icon}</div>
            <div class="metric-label">MARKET REGIME</div>
            <div class="metric-value {regime_class}">{regime_display}</div>
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
        <div id="tv_chart_container" style="height: 600px; width: 100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({
          "autosize": true,
          "symbol": "KRAKEN:BTCUSDT",
          "interval": "15",
          "timezone": "Asia/Ho_Chi_Minh",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#050505",
          "enable_publishing": false,
          "hide_top_toolbar": false, /* Hiện lại toolbar để không bị thụt */
          "hide_legend": false,
          "save_image": false,
          "container_id": "tv_chart_container",
          "studies": [
            {
                "id": "MASimple@tv-basicstudies",
                "inputs": { "length": 200 }
            },
            "RSI@tv-basicstudies"
          ],
          "overrides": {
            "paneProperties.background": "#050505",
            "paneProperties.vertGridProperties.color": "rgba(0, 242, 255, 0.05)",
            "paneProperties.horzGridProperties.color": "rgba(0, 242, 255, 0.05)",
            "scalesProperties.textColor": "#00f2ff",
            "mainSeriesProperties.candleStyle.upColor": "#00f2ff",
            "mainSeriesProperties.candleStyle.downColor": "#ff0055",
            "mainSeriesProperties.candleStyle.borderColor": "#00f2ff",
            "mainSeriesProperties.candleStyle.borderUpColor": "#00f2ff",
            "mainSeriesProperties.candleStyle.borderDownColor": "#ff0055",
            "mainSeriesProperties.candleStyle.wickUpColor": "#00f2ff",
            "mainSeriesProperties.candleStyle.wickDownColor": "#ff0055",
            "studiesOverrides.moving average.line.color": "#FFD700",
            "studiesOverrides.moving average.line.linewidth": 3
          }
        });
        </script>
        """
        
        # Hiển thị vào khung
        components.html(tradingview_html, height=600)
        
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
                
                st.plotly_chart(fig_perf)
    
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
        last_update_iso = data.get('last_update_time')
        last_update_display = "N/A"
        is_stalled = False
        lag_seconds = 0
        
        if last_update_iso:
            try:
                # Parse thời gian UTC từ Engine
                last_update_utc = datetime.fromisoformat(last_update_iso)
                if last_update_utc.tzinfo is None:
                    last_update_utc = last_update_utc.replace(tzinfo=timezone.utc)
                
                # Tính độ trễ
                lag_seconds = (datetime.now(timezone.utc) - last_update_utc).total_seconds()
                
                # Chuyển sang GMT+7 để hiển thị terminal
                vn_time = last_update_utc + timedelta(hours=7)
                last_update_display = vn_time.strftime("%H:%M:%S")
                
                # Nếu quá 120 giây không có dữ liệu mới -> Coi như treo
                if lag_seconds > 120:
                    is_stalled = True
            except:
                last_update_display = "SYNC ERROR"
        
        # Định dạng màu cho Last Sync
        sync_color = "terminal-error" if is_stalled else "terminal-success"
        sync_status_text = f"<span class='{sync_color}'>{last_update_display} ({int(lag_seconds)}s ago)</span>"
        if is_stalled:
            sync_status_text += " <span class='terminal-error'>[STALLED]</span>"

        
        terminal_lines = [
            "MONSTER ENGINE v18.0 - NEURAL CORE",
            "",
            f"<span class='terminal-prompt'>[SYSTEM.INIT]</span> <span class='terminal-success'>NEURAL NETWORK INITIALIZED</span>",
            f"<span class='terminal-prompt'>[SYSTEM.INFO]</span> Engine Status: {data.get('bot_status', 'Unknown')}",
            
            # --- DÒNG THAY ĐỔI: Hiển thị Last Sync có màu và cảnh báo ---
            f"<span class='terminal-prompt'>[SYSTEM.INFO]</span> Last Sync: {sync_status_text}",
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
            f"  ├─ Total P&L: <span class='{'terminal-success' if total_pnl >= 0 else 'terminal-error'}'>${total_pnl:,.2f}</span>",
            f"  ├─ Win Rate:  <span class='terminal-success'>{wr:.1f}%</span>",
            f"  └─ Wins/Loss: {wins}/{len(history)-wins}",
            "",
            f"<span class='terminal-prompt'>[SYSTEM.RESOURCES]</span>",
            f"  ├─ CPU Usage: {cpu_usage:.1f}%",
            f"  ├─ RAM Usage: {ram_usage:.1f}%",
            f"  └─ State File: bot_state_v14_5_1.json",
            "",
        ]
        
        # ═══════════════════════════════════════════════════════════════
        # ULTIMATE INTEGRATION: TITLE ON THE BORDER
        # ═══════════════════════════════════════════════════════════════
        
        terminal_output = "<br>".join(terminal_lines)
        
        st.markdown(f"""
        <div class="matrix-terminal">
            {terminal_output}
        </div>
        """, unsafe_allow_html=True)
            
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE HISTORY TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.markdown("### 📜 TRADE EXECUTION LOGS")
    
    if history:
        df_history = pd.DataFrame(history[:10])
        st.dataframe(df_history, height=350)
        
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
        <p style="color: #666; font-size: 12px;">State File: bot_state_v14_5_1.json</p>
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
