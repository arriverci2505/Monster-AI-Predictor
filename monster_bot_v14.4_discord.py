"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT v14.4 - TITAN INTERACTIVE (PAPER TRADING)                   ║
║  Enhanced Execution + Risk Management + Smart Exit                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import streamlit.components.v1 as components
import time
from datetime import datetime, timedelta
from scipy import signal as scipy_signal
import warnings
import logging
import requests
import os

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thực thi lệnh thực tế
SLIPPAGE = 0.0005  # 0.05% slippage
COMMISSION = 0.00075  # 0.075% commission per trade

# Live Config
LIVE_CONFIG = {
    'exchange': 'kraken', 
    'symbol': 'BTC/USDT', 
    'timeframe': '15m',
    'sequence_length': 30,
    'atr_multiplier_sl': 4.0,
    'atr_multiplier_tp': 20.0,
    'adx_threshold_trending': 25,
    'temperature': 0.7,
    'refresh_interval': 60,
    'config': {
        'input_dim': 29, 
        'hidden_dim': 256, 
        'num_lstm_layers': 2, 
        'num_transformer_layers': 2, 
        'num_heads': 4, 
        'num_classes': 3
    }
}

# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def backup_trade_log(new_entry):
    """Backup trade log to CSV file"""
    file_name = "titan_audit_trail.csv"
    df_new = pd.DataFrame([new_entry])
    if not os.path.isfile(file_name):
        df_new.to_csv(file_name, index=False)
    else:
        df_new.to_csv(file_name, mode='a', header=False, index=False)

def send_discord_alert(webhook_url, msg_type, data):
    """
    Professional Discord Webhook alerts with Embed format
    
    Args:
        webhook_url: Discord webhook URL
        msg_type: 'entry', 'exit', 'profit_lock', 'trailing_update'
        data: dict with relevant trade information
    
    Color codes:
        - Green (BUY/Win): 0x00FF41
        - Red (SELL/Loss): 0xFF0000
        - Yellow (Profit Lock): 0xFFFF00
        - Blue (Info): 0x3498DB
    """
    if not webhook_url:
        return
    
    try:
        embed = {}
        timestamp = datetime.utcnow().isoformat()
        
        if msg_type == 'entry':
            # Calculate R:R Ratio
            if data['action'] == 'BUY':
                rr_ratio = (data['tp'] - data['entry']) / (data['entry'] - data['sl'])
            else:
                rr_ratio = (data['entry'] - data['tp']) / (data['sl'] - data['entry'])
            
            # Color based on action
            color = 0x00FF41 if data['action'] == 'BUY' else 0xFF0000
            
            embed = {
                "title": f"🚀 NEW SIGNAL: {data['symbol']} {data['action']}",
                "description": f"**Market Regime:** {data['regime']}\n**Confidence Level:** {data['confidence']:.1%}",
                "color": color,
                "fields": [
                    {
                        "name": "💰 Entry Price",
                        "value": f"${data['entry']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "🎯 Target TP",
                        "value": f"${data['tp']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "🛡️ Stop Loss",
                        "value": f"${data['sl']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "📈 Risk/Reward",
                        "value": f"{rr_ratio:.2f}:1",
                        "inline": True
                    },
                    {
                        "name": "🌊 Market State",
                        "value": data['regime'],
                        "inline": True
                    },
                    {
                        "name": "⏰ Execution Time",
                        "value": data['time'],
                        "inline": True
                    }
                ],
                "footer": {
                    "text": f"Monster Bot v14.4 | Paper Trading"
                },
                "timestamp": timestamp
            }
        
        elif msg_type == 'exit':
            # Determine color based on PnL
            if data['pnl'] > 0:
                color = 0x00FF41  # Green for profit
                pnl_emoji = "💰"
            else:
                color = 0xFF0000  # Red for loss
                pnl_emoji = "💀"
            
            # Reason emoji mapping
            reason_emoji = {
                'TAKE_PROFIT': '🎯',
                'STOP_LOSS': '🛑',
                'TRAILING_STOP': '🔄',
                'PROFIT_LOCK': '🔒',
                'PROFIT_LOCK_1.5%': '🔒',
                'PROFIT_LOCK_3%': '🔒',
                'PROFIT_LOCK_5%': '🔒',
                'TIMEOUT': '⏱️'
            }.get(data['reason'], '🏁')
            
            embed = {
                "title": f"{pnl_emoji} POSITION CLOSED: {data['symbol']} {data['action']}",
                "description": f"{reason_emoji} **Exit Reason:** {data['reason']}",
                "color": color,
                "fields": [
                    {
                        "name": "💵 Entry Price",
                        "value": f"${data['entry']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "💵 Exit Price",
                        "value": f"${data['exit']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "⏱️ Duration",
                        "value": data['duration'],
                        "inline": True
                    },
                    {
                        "name": "📊 Net PnL",
                        "value": f"**{data['pnl']:+.2%}**",
                        "inline": True
                    },
                    {
                        "name": "🔄 Gross PnL",
                        "value": f"{data['gross_pnl']:+.2%}",
                        "inline": True
                    },
                    {
                        "name": "💸 Trading Fees",
                        "value": f"{data['fees']:.3%}",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": f"Monster Bot v14.4 | Closed at {data['time']}"
                },
                "timestamp": timestamp
            }
        
        elif msg_type == 'profit_lock':
            # Yellow color for profit lock updates
            embed = {
                "title": f"🔒 PROFIT LOCK ACTIVATED: {data['symbol']}",
                "description": f"**Stop Loss Updated to Secure Profits**",
                "color": 0xFFFF00,  # Yellow
                "fields": [
                    {
                        "name": "📈 Current Profit",
                        "value": f"{data['current_pnl']:+.2%}",
                        "inline": True
                    },
                    {
                        "name": "🛡️ Old Stop Loss",
                        "value": f"${data['old_sl']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "🛡️ New Stop Loss",
                        "value": f"${data['new_sl']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "💰 Current Price",
                        "value": f"${data['current_price']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "🔒 Lock Level",
                        "value": data['lock_level'],
                        "inline": True
                    },
                    {
                        "name": "⏰ Update Time",
                        "value": data['time'],
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Monster Bot v14.4 | Dynamic Risk Management"
                },
                "timestamp": timestamp
            }
        
        elif msg_type == 'trailing_update':
            # Blue color for trailing stop updates
            embed = {
                "title": f"🔄 TRAILING STOP UPDATED: {data['symbol']}",
                "description": f"**Stop Loss Following Price Movement**",
                "color": 0x3498DB,  # Blue
                "fields": [
                    {
                        "name": "📈 Highest Price",
                        "value": f"${data['highest_price']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "🛡️ New Stop Loss",
                        "value": f"${data['new_sl']:,.2f}",
                        "inline": True
                    },
                    {
                        "name": "💰 Current Price",
                        "value": f"${data['current_price']:,.2f}",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Monster Bot v14.4 | Trailing Stop Active"
                },
                "timestamp": timestamp
            }
        
        else:
            return
        
        # Send webhook
        payload = {"embeds": [embed]}
        response = requests.post(webhook_url, json=payload, timeout=5)
        
        if response.status_code == 204:
            logger.info(f"Discord alert sent successfully: {msg_type}")
        else:
            logger.warning(f"Discord alert failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        logger.error(f"Discord Webhook Error: {e}")

def display_logic_gate(check_results, metrics):
    """Hiển thị thông số với viền phát sáng (Glow) chuẩn style Titan"""
    cols = st.columns(len(check_results))
    for i, (label, passed) in enumerate(check_results.items()):
        color = "#00FF41" if passed else "#FF0000"
        with cols[i]:
            st.markdown(f"""
            <div style="padding: 15px; border: 2px solid {color}; background: rgba(0, 30, 0, 0.4);
                box-shadow: 0 0 15px {color}44;text-align: center;border-radius: 8px;
                margin-bottom: 10px;font-family: 'Fira Code', monospace;">
                <div style="color: {color}; font-size: 11px; text-shadow: 0 0 5px {color}; opacity: 0.8;">{label}</div>
                <div style="color: white; font-size: 18px; font-weight: bold; margin: 8px 0;">{metrics.get(label, 'N/A')}</div>
                <div style="color: {color}; font-size: 9px; letter-spacing: 1px;">{">> PASS" if passed else ">> FAIL"}</div>
            </div>
            """, unsafe_allow_html=True)

def display_position_progress(position, current_price):
    """Display visual progress bar showing distance to TP/SL"""
    if position is None:
        return
    
    entry = position['Entry']
    sl = position['SL']
    tp = position['TP']
    
    # Calculate position in range
    if position['Action'] == 'BUY':
        total_range = tp - sl
        price_from_sl = current_price - sl
        progress = (price_from_sl / total_range) * 100 if total_range > 0 else 50
    else:  # SELL
        total_range = sl - tp
        price_from_tp = current_price - tp
        progress = (price_from_tp / total_range) * 100 if total_range > 0 else 50
    
    progress = max(0, min(100, progress))
    
    # Calculate current PnL
    if position['Action'] == 'BUY':
        gross_pnl = ((current_price - entry) / entry) * 100
    else:
        gross_pnl = ((entry - current_price) / entry) * 100
    
    net_pnl = gross_pnl - (2 * COMMISSION * 100)
    
    # Color coding
    if net_pnl > 0:
        pnl_color = "#00FF41"
    elif net_pnl < -1:
        pnl_color = "#FF0000"
    else:
        pnl_color = "#FFFF00"
    
    st.markdown(f"""
    <div style="background: rgba(0, 20, 0, 0.6); padding: 15px; border: 1px solid #004400; border-radius: 8px; margin: 10px 0;">
        <div style="color: #00FF41; font-size: 14px; font-family: 'Fira Code', monospace; margin-bottom: 10px;">
            ACTIVE POSITION: {position['Action']} @ ${entry:,.2f}
        </div>
        <div style="background: #001100; height: 30px; border-radius: 5px; position: relative; overflow: hidden;">
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {progress}%; 
                background: linear-gradient(90deg, #FF0000 0%, #FFFF00 50%, #00FF41 100%); 
                transition: width 0.3s;"></div>
            <div style="position: absolute; width: 100%; height: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 10px; font-size: 11px; color: white; font-family: 'Fira Code', monospace;">
                <span>SL: ${sl:,.1f}</span>
                <span style="color: {pnl_color}; font-weight: bold; font-size: 14px;">{net_pnl:+.2f}%</span>
                <span>TP: ${tp:,.1f}</span>
            </div>
        </div>
        <div style="color: #888; font-size: 10px; margin-top: 5px; text-align: right; font-family: 'Fira Code', monospace;">
            Current: ${current_price:,.2f} | Gross: {gross_pnl:+.2f}% | Fees: {(2*COMMISSION*100):.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): 
        return x + self.pe[:, :x.size(1), :]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        s = x.mean(dim=1)
        e = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * e.unsqueeze(1)

class HybridTransformerLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.input_proj = nn.Linear(config['input_dim'], self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=config['num_heads'], 
            dim_feedforward=self.hidden_dim * 4, 
            dropout=config.get('dropout', 0.3), 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_transformer_layers'])
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, 
            num_layers=config['num_lstm_layers'], 
            batch_first=True, bidirectional=True
        )
        self.se_block = SEBlock(self.hidden_dim * 2, config.get('se_reduction_ratio', 16))
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, config['num_classes'])
        )
    
    def forward(self, x):
        x = self.pos_encoding(self.input_proj(x))
        x = self.transformer(x)
        x, _ = self.lstm(x)
        return self.classifier(self.se_block(x)[:, -1, :])

# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def enrich_features_v13(df):
    """Feature engineering với các chỉ báo kỹ thuật"""
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    tr = pd.concat([
        df['High']-df['Low'], 
        abs(df['High']-df['Close'].shift(1)), 
        abs(df['Low']-df['Close'].shift(1))
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # ADX
    p = 14
    plus_dm = np.where(
        (df['High'].diff() > df['Low'].shift(1)-df['Low']), 
        np.maximum(df['High'].diff(), 0), 0
    )
    minus_dm = np.where(
        (df['Low'].shift(1)-df['Low'] > df['High'].diff()), 
        np.maximum(df['Low'].shift(1)-df['Low'], 0), 0
    )
    pdi = 100 * (pd.Series(plus_dm).rolling(p).mean() / df['ATR'])
    mdi = 100 * (pd.Series(minus_dm).rolling(p).mean() / df['ATR'])
    df['ADX'] = (100 * abs(pdi-mdi)/(pdi+mdi)).rolling(p).mean()
    
    df['SMA_distance'] = (df['Close'].rolling(20).mean() - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    
    # Fourier features
    for i in range(1, 6):
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * df.index / 100)
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * df.index / 100)
    
    # Bollinger Bands
    df['BB_width'] = (
        (df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()) - 
        (df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std())
    )
    df['BB_position'] = (
        (df['Close'] - (df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std())) / 
        df['BB_width']
    )
    
    # Additional features
    df['frac_diff_close'] = df['Close'].diff()
    df['volume_imbalance'] = df['Volume'].diff()
    df['entropy'] = df['Close'].rolling(10).std()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Regime indicators
    df['regime_trending'] = (df['ADX'] > 25).astype(int)
    df['regime_uptrend'] = ((df['SMA_distance'] > 0) & (df['regime_trending'] == 1)).astype(int)
    df['regime_downtrend'] = ((df['SMA_distance'] < 0) & (df['regime_trending'] == 1)).astype(int)
    
    # RSI & MACD
    delta = df['Close'].diff()
    u = (delta.where(delta > 0, 0)).rolling(14).mean()
    d = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + u/d))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # Volatility adjusted
    vol = df['Close'].pct_change().rolling(20).std()
    df['volatility_zscore'] = (vol - vol.rolling(100).mean()) / vol.rolling(100).std()
    df['RSI_vol_adj'] = df['RSI'] / (vol * 100)
    df['ROC_vol_adj'] = (df['Close'].pct_change(10) * 100) / (vol * 100)

    return df.ffill().fillna(0)

def apply_rolling_normalization(df, cols):
    """Apply rolling normalization to features"""
    df_norm = df.copy()
    for col in cols:
        if col in df_norm.columns:
            mean = df_norm[col].rolling(window=100, min_periods=1).mean()
            std = df_norm[col].rolling(window=100, min_periods=1).std()
            df_norm[col] = (df_norm[col] - mean) / (std + 1e-8)
    return df_norm.fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT & SMART EXIT
# ════════════════════════════════════════════════════════════════════════════

def update_dynamic_exit(position, current_price, enable_profit_lock, enable_trailing):
    """
    Cập nhật SL động dựa trên Profit Lock và Trailing Stop
    
    Returns:
        tuple: (updated_position, update_info)
        update_info: dict with update details for alerts, or None if no update
    
    Profit Lock Levels:
    - >= 1.5%: Move SL to Entry + 0.3%
    - >= 3.0%: Move SL to Entry + 1.2%
    - >= 5.0%: Move SL to Entry + 2.5%
    
    Trailing Stop:
    - Activation: 1.0% profit
    - Distance: 0.4% from highest price
    """
    if position is None:
        return position, None
    
    entry = position['Entry']
    action = position['Action']
    current_sl = position['SL']
    
    # Calculate current PnL
    if action == 'BUY':
        pnl_pct = ((current_price - entry) / entry) * 100
    else:  # SELL
        pnl_pct = ((entry - current_price) / entry) * 100
    
    new_sl = current_sl
    update_reason = None
    update_info = None
    
    # ─── PROFIT LOCK LOGIC ───
    if enable_profit_lock and pnl_pct > 0:
        if pnl_pct >= 5.0:
            target_sl = entry * (1.025 if action == 'BUY' else 0.975)
            if (action == 'BUY' and target_sl > current_sl) or (action == 'SELL' and target_sl < current_sl):
                new_sl = target_sl
                update_reason = "PROFIT_LOCK_5%"
        elif pnl_pct >= 3.0:
            target_sl = entry * (1.012 if action == 'BUY' else 0.988)
            if (action == 'BUY' and target_sl > current_sl) or (action == 'SELL' and target_sl < current_sl):
                new_sl = target_sl
                update_reason = "PROFIT_LOCK_3%"
        elif pnl_pct >= 1.5:
            target_sl = entry * (1.003 if action == 'BUY' else 0.997)
            if (action == 'BUY' and target_sl > current_sl) or (action == 'SELL' and target_sl < current_sl):
                new_sl = target_sl
                update_reason = "PROFIT_LOCK_1.5%"
    
    # ─── TRAILING STOP LOGIC ───
    if enable_trailing and pnl_pct >= 1.0:
        if 'highest_price' not in position:
            position['highest_price'] = current_price if action == 'BUY' else current_price
        
        if action == 'BUY':
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            trailing_sl = position['highest_price'] * 0.996  # 0.4% below highest
            if trailing_sl > new_sl:
                new_sl = trailing_sl
                update_reason = "TRAILING_STOP"
        else:  # SELL
            if current_price < position['highest_price']:
                position['highest_price'] = current_price
            trailing_sl = position['highest_price'] * 1.004  # 0.4% above lowest
            if trailing_sl < new_sl:
                new_sl = trailing_sl
                update_reason = "TRAILING_STOP"
    
    # Update position if SL changed
    if new_sl != current_sl:
        position['SL'] = new_sl
        position['last_sl_update'] = datetime.now()
        position['sl_update_reason'] = update_reason
        logger.info(f"SL Updated: {current_sl:.2f} → {new_sl:.2f} ({update_reason})")
        
        # Prepare update info for alert
        update_info = {
            'old_sl': current_sl,
            'new_sl': new_sl,
            'current_price': current_price,
            'current_pnl': pnl_pct / 100,  # Convert to decimal
            'update_type': update_reason,
            'highest_price': position.get('highest_price', current_price)
        }
    
    return position, update_info

def calculate_actual_entry_price(market_price, action):
    """
    Calculate actual entry price with slippage
    Long: market_price * (1 + SLIPPAGE)
    Short: market_price * (1 - SLIPPAGE)
    """
    if action == 'BUY':
        return market_price * (1 + SLIPPAGE)
    else:  # SELL
        return market_price * (1 - SLIPPAGE)

def calculate_net_pnl(entry_price, exit_price, action):
    """
    Calculate net PnL with commission
    Net PnL = Gross PnL - (2 * COMMISSION)
    """
    if action == 'BUY':
        gross_pnl_pct = ((exit_price - entry_price) / entry_price)
    else:  # SELL
        gross_pnl_pct = ((entry_price - exit_price) / entry_price)
    
    net_pnl_pct = gross_pnl_pct - (2 * COMMISSION)
    
    return {
        'gross_pnl': gross_pnl_pct,
        'net_pnl': net_pnl_pct,
        'fees': 2 * COMMISSION
    }

# ════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION LOGIC
# ════════════════════════════════════════════════════════════════════════════

def check_limit_order_fill(pending_order, current_candle):
    """
    Check if limit order should be filled
    Returns: (filled: bool, fill_price: float or None)
    """
    if pending_order is None:
        return False, None
    
    action = pending_order['Action']
    limit_price = pending_order['LimitPrice']
    high = current_candle['High']
    low = current_candle['Low']
    
    if action == 'BUY':
        # For BUY limit, need price to dip to or below limit
        if low <= limit_price:
            return True, limit_price
    else:  # SELL
        # For SELL limit, need price to rise to or above limit
        if high >= limit_price:
            return True, limit_price
    
    return False, None

@st.cache_resource
def load_monster_model():
    """Load the AI model"""
    model = HybridTransformerLSTM(LIVE_CONFIG['config'])
    # In production: model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()
    return model

# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="TITAN INTEL TERMINAL v14.4", layout="wide")

    # ─── CSS STYLING ───
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap');
        
        .stApp { 
            background-color: #030a03; 
        }

        .crt-glow {
            color: #00FF41 !important;
            font-family: 'Fira Code', monospace !important;
            text-shadow: 
                0 0 5px rgba(0, 255, 65, 1),
                0 0 10px rgba(0, 255, 65, 0.6);
            letter-spacing: 1px;
        }

        .signal-card { 
            padding: 25px; 
            border: 2px solid #00FF41; 
            background: rgba(0, 30, 0, 0.4);
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.3);
            text-align: center;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .trade-setup { 
            background: #000; 
            border-left: 4px solid #00FF41; 
            padding: 12px; 
            margin-top: 10px;
            font-family: 'Fira Code', monospace;
            color: #00FF41;
        }

        /* Sidebar Fix - Prevent Label Overlap */
        [data-testid="stSidebar"] {
            background-color: #010801 !important;
            border-right: 1px solid #004400;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            padding: 0.5rem 0;
        }
        
        .stSlider label, .stToggle label, .stSelectbox label, 
        [data-testid="stWidgetLabel"] p {
            color: #00AA00 !important; 
            font-family: 'Fira Code', monospace !important;
            font-size: 12px !important;
            text-shadow: none !important;
            margin-bottom: 0.3rem !important;
            padding-bottom: 0 !important;
        }

        /* Improved spacing for sidebar widgets */
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stToggle,
        [data-testid="stSidebar"] .stTextInput {
            margin-bottom: 1rem !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #003300;
            filter: sepia(80%) hue-rotate(80deg) brightness(90%);
        }
        
        hr { 
            border: 0.5px solid #002200; 
            margin: 1.5rem 0;
        }

        /* Enhanced styling for expanders */
        [data-testid="stExpander"] {
            background-color: rgba(0, 20, 0, 0.3) !important;
            border: 1px solid #003300 !important;
            border-radius: 5px;
            margin-bottom: 0.5rem !important;
        }

        [data-testid="stExpander"] summary {
            color: #00FF41 !important;
            font-family: 'Fira Code', monospace !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ─── SIDEBAR CONFIGURATION ───
    st.sidebar.markdown("<h2 class='crt-glow' style='font-size:20px; margin-bottom: 1rem;'>⚙️ TERMINAL_CONFIG</h2>", unsafe_allow_html=True)
    
    # AI Core Settings
    with st.sidebar.expander("🤖 OPERATIONAL_PARAMS", expanded=True):
        ui_temp = st.slider("Signal_Temperature", 0.1, 1.5, 0.5, step=0.01, 
                           help="Lower = more conservative, Higher = more aggressive")
        ui_buy_threshold = st.slider("Buy_Threshold", 0.3, 0.8, 0.40, step=0.01)
        ui_sell_threshold = st.slider("Sell_Threshold", 0.3, 0.8, 0.40, step=0.01)

    # Market Filters
    with st.sidebar.expander("📡 RADAR_FILTERS", expanded=True):
        ui_adx_min = st.slider("Min_ADX_Level", 10, 50, 20, step=1)
        ui_adx_max = st.slider("Max_ADX_Level", 50, 100, 100, step=1)
        ui_use_dynamic = st.toggle("Activate_SMA_Filter", value=True)

    # Risk Management
    with st.sidebar.expander("⚖️ RISK_MANAGEMENT", expanded=True):
        ui_atr_sl = st.slider("Stop_Loss (ATR)", 1.0, 10.0, 3.5, step=0.1)
        ui_atr_tp = st.slider("Take_Profit (ATR)", 5.0, 50.0, 20.0, step=0.1)

    # Advanced Exit
    with st.sidebar.expander("🛡️ ADVANCED_EXIT", expanded=True):
        ui_use_profit_lock = st.toggle("Enable_Profit_Lock", value=True,
                                       help="Auto-adjust SL at 1.5%, 3%, 5% profit")
        ui_use_trailing = st.toggle("Enable_Trailing_Stop", value=True,
                                    help="Activate at 1% profit, 0.4% distance")
        st.caption("📊 Profit Lock: 1.5%→0.3% | 3%→1.2% | 5%→2.5%")
        st.caption("🔄 Trailing: Activate@1% | Distance: 0.4%")
    
    # Discord Integration
    with st.sidebar.expander("📡 DISCORD_WEBHOOK", expanded=False):
        discord_webhook = st.text_input("Webhook_URL", type="password", 
                                       placeholder="https://discord.com/api/webhooks/...")
        st.caption("🔔 Professional alerts with Embed format")
        st.caption("📘 [Get Webhook URL](https://support.discord.com/hc/en-us/articles/228383668)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div class='crt-glow' style='font-size:10px; text-align:center;'>v14.4 | PAPER TRADING MODE</div>", 
                       unsafe_allow_html=True)

    # ─── MAIN LAYOUT ───
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown("<div class='crt-glow' style='font-size:16px;'>[SYSTEM_STATUS: ACTIVE]</div>", 
                   unsafe_allow_html=True)
        signal_placeholder = st.empty()
        progress_placeholder = st.empty()
        setup_placeholder = st.empty()
        
        st.markdown("<div class='crt-glow' style='font-size:16px; margin-top:20px;'>[EXECUTION_LOG]</div>", 
                   unsafe_allow_html=True)
        log_placeholder = st.empty()

    with col_right:
        st.markdown("<div class='crt-glow' style='font-size:16px;'>[LIVE_MARKET_FEED]</div>", 
                   unsafe_allow_html=True)
        tv_html = f"""
        <div style="height:750px; border: 2px solid #004400; border-radius:5px; overflow:hidden; 
                    filter: brightness(0.7) contrast(1.2) sepia(100%) hue-rotate(70deg);">
            <div id="tv_chart_v14" style="height:100%;"></div>
            <script src="https://s3.tradingview.com/tv.js"></script>
            <script>
                new TradingView.widget({{
                    "autosize": true,
                    "symbol": "KRAKEN:BTCUSDT",
                    "interval": "15",
                    "theme": "dark",
                    "container_id": "tv_chart_v14",
                    "style": "1",
                    "enable_publishing": false,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true
                }});
            </script>
        </div>
        """
        components.html(tv_html, height=760)
        
    # ─── SESSION STATE INITIALIZATION ───
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'active_position' not in st.session_state:
        st.session_state.active_position = None
    if 'pending_order' not in st.session_state:
        st.session_state.pending_order = None
    if 'pending_candle_count' not in st.session_state:
        st.session_state.pending_candle_count = 0
        
    # ─── INITIALIZATION ───
    try:
        model = load_monster_model()
        exchange = ccxt.kraken({'enableRateLimit': True})
        
        feature_cols = [
            'log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close',
            'fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5',
            'fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5',
            'volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance',
            'regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD',
            'MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj'
        ]
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return

    # ─── MAIN TRADING LOOP ───
    while True:
        try:
            # Fetch market data
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            df_enriched = enrich_features_v13(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            # Current market state
            current_candle = df.iloc[-1]
            price = current_candle['Close']
            atr = df_enriched['ATR'].iloc[-1]
            adx = df_enriched['ADX'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]
            
            # ═══ CHECK PENDING LIMIT ORDER ═══
            if st.session_state.pending_order is not None:
                filled, fill_price = check_limit_order_fill(st.session_state.pending_order, current_candle)
                
                if filled:
                    # Convert pending to active position
                    pending = st.session_state.pending_order
                    st.session_state.active_position = {
                        'Action': pending['Action'],
                        'Entry': fill_price,
                        'TP': pending['TP'],
                        'SL': pending['SL'],
                        'EntryTime': datetime.now(),
                        'Regime': pending['Regime']
                    }
                    
                    # Log the fill
                    new_entry = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Action": f"✅ {pending['Action']}",
                        "Price": f"${fill_price:,.2f}",
                        "TP": f"${pending['TP']:,.2f}",
                        "SL": f"${pending['SL']:,.2f}",
                        "Type": "LIMIT_FILLED",
                        "Net PnL %": "---",
                        "Duration": "---"
                    }
                    st.session_state.trade_log.insert(0, new_entry)
                    
                    # Discord notification
                    if discord_webhook:
                        send_discord_alert(discord_webhook, 'entry', {
                            'action': pending['Action'],
                            'symbol': 'BTC/USDT',
                            'entry': fill_price,
                            'tp': pending['TP'],
                            'sl': pending['SL'],
                            'confidence': pending.get('Confidence', 0),
                            'regime': pending['Regime'],
                            'time': datetime.now().strftime("%H:%M:%S")
                        })
                    
                    st.session_state.pending_order = None
                    st.session_state.pending_candle_count = 0
                else:
                    # Check timeout (2 candles)
                    st.session_state.pending_candle_count += 1
                    if st.session_state.pending_candle_count >= 2:
                        # Cancel pending order
                        cancel_entry = {
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Action": f"❌ CANCELLED",
                            "Price": f"${price:,.2f}",
                            "TP": "---",
                            "SL": "---",
                            "Type": "TIMEOUT",
                            "Net PnL %": "---",
                            "Duration": "---"
                        }
                        st.session_state.trade_log.insert(0, cancel_entry)
                        st.session_state.pending_order = None
                        st.session_state.pending_candle_count = 0

            # ═══ MANAGE ACTIVE POSITION ═══
            if st.session_state.active_position is not None:
                pos = st.session_state.active_position
                
                # Update dynamic exit if enabled
                if ui_use_profit_lock or ui_use_trailing:
                    pos, update_info = update_dynamic_exit(pos, price, ui_use_profit_lock, ui_use_trailing)
                    st.session_state.active_position = pos
                    
                    # Send Discord alert for SL updates
                    if update_info and discord_webhook:
                        if 'PROFIT_LOCK' in update_info['update_type']:
                            send_discord_alert(discord_webhook, 'profit_lock', {
                                'symbol': 'BTC/USDT',
                                'old_sl': update_info['old_sl'],
                                'new_sl': update_info['new_sl'],
                                'current_price': update_info['current_price'],
                                'current_pnl': update_info['current_pnl'],
                                'lock_level': update_info['update_type'],
                                'time': datetime.now().strftime("%H:%M:%S")
                            })
                        elif update_info['update_type'] == 'TRAILING_STOP':
                            send_discord_alert(discord_webhook, 'trailing_update', {
                                'symbol': 'BTC/USDT',
                                'highest_price': update_info['highest_price'],
                                'new_sl': update_info['new_sl'],
                                'current_price': update_info['current_price'],
                                'time': datetime.now().strftime("%H:%M:%S")
                            })
                
                # Check exit conditions
                is_closed = False
                exit_reason = ""
                exit_price = price
                
                if pos['Action'] == 'BUY':
                    if price >= pos['TP']:
                        is_closed, exit_reason = True, "TAKE_PROFIT"
                    elif price <= pos['SL']:
                        is_closed, exit_reason = True, "STOP_LOSS"
                        if pos.get('sl_update_reason'):
                            exit_reason = pos['sl_update_reason']
                elif pos['Action'] == 'SELL':
                    if price <= pos['TP']:
                        is_closed, exit_reason = True, "TAKE_PROFIT"
                    elif price >= pos['SL']:
                        is_closed, exit_reason = True, "STOP_LOSS"
                        if pos.get('sl_update_reason'):
                            exit_reason = pos['sl_update_reason']
                
                if is_closed:
                    # Calculate PnL
                    pnl_data = calculate_net_pnl(pos['Entry'], exit_price, pos['Action'])
                    duration = datetime.now() - pos['EntryTime']
                    duration_str = f"{int(duration.total_seconds()//60)}m"
                    
                    # Log the exit
                    close_entry = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Action": f"🏁 CLOSE_{pos['Action']}",
                        "Price": f"${exit_price:,.2f}",
                        "TP": "---",
                        "SL": "---",
                        "Type": exit_reason,
                        "Net PnL %": f"{pnl_data['net_pnl']*100:+.2f}%",
                        "Duration": duration_str
                    }
                    st.session_state.trade_log.insert(0, close_entry)
                    
                    # Discord notification
                    if discord_webhook:
                        send_discord_alert(discord_webhook, 'exit', {
                            'action': pos['Action'],
                            'symbol': 'BTC/USDT',
                            'entry': pos['Entry'],
                            'exit': exit_price,
                            'pnl': pnl_data['net_pnl'],
                            'gross_pnl': pnl_data['gross_pnl'],
                            'fees': pnl_data['fees'],
                            'reason': exit_reason,
                            'duration': duration_str,
                            'time': datetime.now().strftime("%H:%M:%S")
                        })
                    
                    # Backup to CSV
                    try:
                        backup_trade_log(close_entry)
                    except:
                        pass
                    
                    # Reset position
                    st.session_state.active_position = None
            
            # ═══ AI PREDICTION (Only if no active position) ═══
            if st.session_state.active_position is None and st.session_state.pending_order is None:
                # Prepare features
                X_last = df_norm[feature_cols].tail(30).values
                X_tensor = torch.FloatTensor(X_last).unsqueeze(0)
                
                # Get AI prediction
                with torch.no_grad():
                    logits = model(X_tensor)
                    probs = torch.softmax(logits / ui_temp, dim=-1).numpy()[0]
                
                p_neutral, p_buy, p_sell = probs[0], probs[1], probs[2]
                
                # Determine raw signal
                raw_sig = "NEUTRAL"
                if p_buy > ui_buy_threshold:
                    raw_sig = "BUY"
                elif p_sell > ui_sell_threshold:
                    raw_sig = "SELL"
                
                # ═══ GATE CHECKS ═══
                current_conf = max(p_buy, p_sell)
                
                # 1. AI Probability
                ai_pass = False
                if raw_sig == "BUY":
                    ai_pass = p_buy > ui_buy_threshold
                elif raw_sig == "SELL":
                    ai_pass = p_sell > ui_sell_threshold
                
                # 2. ADX Range
                adx_pass = ui_adx_min <= adx <= ui_adx_max
                
                # 3. SMA Filter
                sma_pass = True
                if ui_use_dynamic:
                    if raw_sig == "BUY":
                        sma_pass = price > sma200
                    elif raw_sig == "SELL":
                        sma_pass = price < sma200
                
                gate_status = {
                    "AI_PROB": ai_pass,
                    "ADX_LEVEL": adx_pass,
                    "SMA_BIAS": sma_pass
                }
                
                gate_metrics = {
                    "AI_PROB": f"{max(p_buy, p_sell):.1%}",
                    "ADX_LEVEL": f"{adx:.1f}",
                    "SMA_BIAS": f"{(price - sma200):+.1f}"
                }
                
                # Determine final signal
                if all(gate_status.values()) and raw_sig != "NEUTRAL":
                    final_sig = raw_sig
                    reason = "READY"
                else:
                    final_sig = "NEUTRAL"
                    if not ai_pass:
                        reason = "Prob < Threshold"
                    elif not adx_pass:
                        reason = "ADX Out of Range"
                    elif not sma_pass:
                        reason = "Wrong Side of SMA"
                    else:
                        reason = "No Signal"
                
                # ═══ ORDER EXECUTION LOGIC ═══
                if final_sig != "NEUTRAL":
                    # Determine regime
                    is_trending = adx > LIVE_CONFIG['adx_threshold_trending']
                    regime_name = "TRENDING" if is_trending else "SIDEWAY"
                    
                    if is_trending:
                        # Market Order - Instant execution with slippage
                        entry_price_actual = calculate_actual_entry_price(price, final_sig)
                        sl_val = entry_price_actual - (atr * ui_atr_sl) if final_sig == "BUY" else entry_price_actual + (atr * ui_atr_sl)
                        tp_val = entry_price_actual + (atr * ui_atr_tp) if final_sig == "BUY" else entry_price_actual - (atr * ui_atr_tp)
                        
                        # Create active position immediately
                        st.session_state.active_position = {
                            'Action': final_sig,
                            'Entry': entry_price_actual,
                            'TP': tp_val,
                            'SL': sl_val,
                            'EntryTime': datetime.now(),
                            'Regime': regime_name
                        }
                        
                        # Log entry
                        new_entry = {
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Action": f"⚡ {final_sig}",
                            "Price": f"${entry_price_actual:,.2f}",
                            "TP": f"${tp_val:,.2f}",
                            "SL": f"${sl_val:,.2f}",
                            "Type": "MARKET",
                            "Net PnL %": "---",
                            "Duration": "---"
                        }
                        st.session_state.trade_log.insert(0, new_entry)
                        
                        # Discord alert
                        if discord_webhook:
                            send_discord_alert(discord_webhook, 'entry', {
                                'action': final_sig,
                                'symbol': 'BTC/USDT',
                                'entry': entry_price_actual,
                                'tp': tp_val,
                                'sl': sl_val,
                                'confidence': current_conf,
                                'regime': regime_name,
                                'time': datetime.now().strftime("%H:%M:%S")
                            })
                        
                    else:
                        # Limit Order - Wait for better price
                        if final_sig == "BUY":
                            limit_price = price * 0.999  # 0.1% below current
                        else:
                            limit_price = price * 1.001  # 0.1% above current
                        
                        sl_val = limit_price - (atr * ui_atr_sl) if final_sig == "BUY" else limit_price + (atr * ui_atr_sl)
                        tp_val = limit_price + (atr * ui_atr_tp) if final_sig == "BUY" else limit_price - (atr * ui_atr_tp)
                        
                        # Create pending order
                        st.session_state.pending_order = {
                            'Action': final_sig,
                            'LimitPrice': limit_price,
                            'TP': tp_val,
                            'SL': sl_val,
                            'Regime': regime_name,
                            'Confidence': current_conf
                        }
                        st.session_state.pending_candle_count = 0
                        
                        # Log pending
                        pending_entry = {
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Action": f"⏳ {final_sig}",
                            "Price": f"${limit_price:,.2f}",
                            "TP": f"${tp_val:,.2f}",
                            "SL": f"${sl_val:,.2f}",
                            "Type": "LIMIT_PENDING",
                            "Net PnL %": "---",
                            "Duration": "---"
                        }
                        st.session_state.trade_log.insert(0, pending_entry)
            else:
                # Use last known values for display
                final_sig = "NEUTRAL"
                reason = "Position Active"
                gate_status = {"AI_PROB": False, "ADX_LEVEL": False, "SMA_BIAS": False}
                gate_metrics = {"AI_PROB": "---", "ADX_LEVEL": "---", "SMA_BIAS": "---"}
                current_conf = 0
            
            # ═══ UI UPDATE ═══
            # Signal Card
            sig_color = "#00FF41" if final_sig == "BUY" else "#FF0000" if final_sig == "SELL" else "#FFFF00"
            glow_style = f"text-shadow: 0 0 20px {sig_color}, 0 0 30px {sig_color}; color: {sig_color} !important;"
            conf = current_conf if final_sig != "NEUTRAL" else 0
            bar_len = int(conf * 20)
            signal_bar = "█" * bar_len + "░" * (20 - bar_len)

            with signal_placeholder.container():
                st.markdown(f"""
                <div class='signal-card' style='border-color: {sig_color};'>
                    <div style='{glow_style} font-size:60px; font-weight:bold; font-family:Fira Code;'>{final_sig}</div>
                    <div class='crt-glow' style='font-size:20px; color:white !important;'>PRICE: ${price:,.1f}</div>
                    <div class='crt-glow' style='font-size:12px; font-family: Courier New;'>
                        STRENGTH: [{signal_bar}] {conf:.1%}
                    </div>
                    <div class='crt-glow' style='font-size:12px; opacity:0.6; margin-bottom:10px;'>STATUS: {reason}</div>
                </div>
                """, unsafe_allow_html=True)
                
                display_logic_gate(gate_status, gate_metrics)
            
            # Position Progress
            if st.session_state.active_position is not None:
                with progress_placeholder.container():
                    display_position_progress(st.session_state.active_position, price)
            else:
                progress_placeholder.empty()
            
            # Trade Log
            with log_placeholder.container():
                if st.session_state.trade_log:
                    df_log = pd.DataFrame(st.session_state.trade_log)
                    
                    # Summary metrics
                    cols = st.columns(4)
                    cols[0].metric("TOTAL_TRADES", len(df_log))
                    
                    # Count wins/losses
                    closed_trades = df_log[df_log['Net PnL %'] != '---']
                    if len(closed_trades) > 0:
                        wins = len(closed_trades[closed_trades['Net PnL %'].str.contains(r'^\+', regex=True)])
                        losses = len(closed_trades) - wins
                        win_rate = (wins / len(closed_trades)) * 100 if len(closed_trades) > 0 else 0
                        cols[1].metric("WIN_RATE", f"{win_rate:.1f}%")
                        cols[2].metric("WINS/LOSSES", f"{wins}/{losses}")
                    else:
                        cols[1].metric("WIN_RATE", "N/A")
                        cols[2].metric("WINS/LOSSES", "0/0")
                    
                    cols[3].metric("LAST_ACTION", final_sig)
                    
                    # Display log table
                    st.dataframe(
                        df_log.head(20), 
                        use_container_width=True, 
                        hide_index=True
                    )
            
            # ═══ SLEEP UNTIL NEXT CANDLE ═══
            now = datetime.now()
            seconds_until_next_minute = 60 - now.second
            time.sleep(max(1, seconds_until_next_minute))
            st.rerun()
            
        except Exception as e:
            st.error(f"⚠️ SYSTEM ERROR: {e}")
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    main()
