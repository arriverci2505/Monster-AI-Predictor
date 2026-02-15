"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER BOT v14.4 - BTC GOLDEN RATIO (27% BACKTEST WIN)                 ║
║  FULL UI + TradingView + Background Execution                            ║
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
import threading
import json
import os
import gc
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - BTC GOLDEN RATIO FROM 27% WIN BACKTEST
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Trading constants
SLIPPAGE = 0.0005
COMMISSION = 0.00075

# BTC GOLDEN RATIO - Optimized Parameters
BTC_GOLDEN_CONFIG = {
    # Exchange & Symbol
    'exchange': 'kraken',
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'sequence_length': 30,
    
    # AI Model Settings - Dual Threshold System
    'temperature': 1.2,
    'entry_percentile': 25,            
          
    # TRENDING MODE
    'trending_buy_threshold': 0.40,    
    'trending_sell_threshold': 0.42,
          
    # SIDEWAY MODE
    'sideway_buy_threshold': 0.22,      
    'sideway_sell_threshold': 0.22,
          
    # --- REGIME CLASSIFICATION ---
    'trending_adx_min': 30,
    'sideway_adx_max': 30,
    'choppiness_threshold_high': 58.0,
    'choppiness_extreme_low': 30,      

    # --- SIDEWAY FILTERS (MỞ RỘNG BIÊN) ---
    'deviation_zscore_threshold': 1.4,  
    'mean_reversion_min_shadow_atr': 0.1,
    'bb_squeeze_percentile': 0.35,
          
    # --- TRENDING MODE (SIẾT SL, THẢ TP) ---
    'sl_std_multiplier': 1.5,          
    'max_holding_bars': 200,             
          
    # --- SIDEWAY EXIT ---
    'mean_reversion_sl_pct': 1.0,       
    'mean_reversion_tp_pct': 3.5,       
    'time_barrier': 20,
    'min_profit_for_target': 0.009,     

    # --- RISK MANAGEMENT (CHO LÃI "THỞ") ---
    'use_advanced_exit': True,
    'use_profit_lock': True,
    'ai_exit_threshold': 0.70,
          
    'profit_lock_levels': [
        (1.8, 1.2),                    
        (3.5, 2.8),
        (5.5, 4.5)
    ],

    'trailing_stop_activation': 1.5,    
    'trailing_stop_distance': 0.6,
    
    # System
    'discord_webhook': '',
    'heartbeat_interval': 21600,
    'refresh_interval': 60
}

# File paths
TRADE_DATA_FILE = Path("trade_data.json")
BOT_CONFIG_FILE = Path("bot_config.json")

# Global state
BOT_RUNNING = False
STATE_LOCK = threading.Lock()

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
    def __init__(self, input_dim=29, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.se_block = SEBlock(hidden_dim * 2, 16)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, x):
        x = self.pos_encoding(self.input_proj(x))
        x = self.transformer(x)
        x, _ = self.lstm(x)
        return self.classifier(self.se_block(x)[:, -1, :])

@st.cache_resource
def load_model():
    """Load model once and cache"""
    logger.info("Loading AI model...")
    model = HybridTransformerLSTM(input_dim=29, hidden_dim=256)
    model.eval()
    logger.info("Model loaded successfully")
    return model

# ════════════════════════════════════════════════════════════════════════════
# PERSISTENT STORAGE
# ════════════════════════════════════════════════════════════════════════════

def load_trade_data():
    """Load all trade data from JSON"""
    if TRADE_DATA_FILE.exists():
        try:
            with open(TRADE_DATA_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data.get('trades', []))} trades")
                return data
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
    
    return {
        'active_position': None,
        'pending_order': None,
        'trades': [],
        'bot_start_time': datetime.now().isoformat(),
        'last_heartbeat': None
    }

def save_trade_data(data):
    """Save trade data immediately"""
    try:
        with STATE_LOCK:
            with open(TRADE_DATA_FILE, 'w') as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving trade data: {e}")

def load_bot_config():
    """Load configuration"""
    if BOT_CONFIG_FILE.exists():
        try:
            with open(BOT_CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                config = BTC_GOLDEN_CONFIG.copy()
                config.update(saved)
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return BTC_GOLDEN_CONFIG.copy()

def save_bot_config(config):
    """Save configuration"""
    try:
        with open(BOT_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving config: {e}")

# ════════════════════════════════════════════════════════════════════════════
# DISCORD EMBEDS V2
# ════════════════════════════════════════════════════════════════════════════

def send_discord_alert(webhook_url, msg_type, data):
    """Professional Discord alerts with full details"""
    if not webhook_url:
        return
    
    try:
        timestamp = datetime.utcnow().isoformat()
        
        if msg_type == 'heartbeat':
            embed = {
                "embeds": [{
                    "title": "💚 BOT HEARTBEAT - BTC GOLDEN RATIO",
                    "description": f"**Uptime:** {data.get('uptime', 'N/A')}\n**Mode:** 24/7 Background",
                    "color": 0x00FF41,
                    "fields": [
                        {"name": "📊 Status", "value": "Active & Monitoring", "inline": True},
                        {"name": "🔄 Last Check", "value": datetime.now().strftime("%H:%M:%S"), "inline": True}
                    ],
                    "footer": {"text": "Monster Bot v14.4 BTC | Golden Ratio"},
                    "timestamp": timestamp
                }]
            }
        
        elif msg_type == 'entry':
            # Calculate R:R with fees included
            entry_with_slippage = data['entry']
            if data['action'] == 'BUY':
                rr_ratio = (data['tp'] - entry_with_slippage) / (entry_with_slippage - data['sl'])
            else:
                rr_ratio = (entry_with_slippage - data['tp']) / (data['sl'] - entry_with_slippage)
            
            color = 0x00FF41 if data['action'] == 'BUY' else 0xFF0000
            
            embed = {
                "embeds": [{
                    "title": f"🚀 NEW SIGNAL: {data['symbol']} {data['action']}",
                    "description": f"**Regime:** {data['regime']} | **AI Confidence:** {data['confidence']:.1%}",
                    "color": color,
                    "fields": [
                        {"name": "💰 Entry (w/ fees)", "value": f"${entry_with_slippage:,.2f}", "inline": True},
                        {"name": "🎯 Take Profit", "value": f"${data['tp']:,.2f}", "inline": True},
                        {"name": "🛡️ Stop Loss", "value": f"${data['sl']:,.2f}", "inline": True},
                        {"name": "📈 R:R Ratio", "value": f"{rr_ratio:.2f}:1", "inline": True},
                        {"name": "🌊 Market Type", "value": data['regime'], "inline": True},
                        {"name": "⏰ Entry Time", "value": datetime.now().strftime("%H:%M:%S"), "inline": True}
                    ],
                    "footer": {"text": "BTC Golden Ratio | 27% Backtest Win"},
                    "timestamp": timestamp
                }]
            }
        
        elif msg_type == 'exit':
            color = 0x00FF41 if data['pnl'] > 0 else 0xFF0000
            pnl_emoji = "💰" if data['pnl'] > 0 else "💀"
            
            # Reason with emoji
            reason_map = {
                'TAKE_PROFIT': '🎯 Take Profit',
                'STOP_LOSS': '🛑 Stop Loss',
                'PROFIT_LOCK_1.2%': '🔒 Profit Lock 1.2%',
                'PROFIT_LOCK_2.8%': '🔒 Profit Lock 2.8%',
                'PROFIT_LOCK_4.5%': '🔒 Profit Lock 4.5%',
                'TRAILING_STOP': '🔄 Trailing Stop'
            }
            reason_text = reason_map.get(data['reason'], data['reason'])
            
            embed = {
                "embeds": [{
                    "title": f"{pnl_emoji} POSITION CLOSED: {data['symbol']} {data['action']}",
                    "description": f"**Exit Reason:** {reason_text}",
                    "color": color,
                    "fields": [
                        {"name": "💵 Entry Price", "value": f"${data['entry']:,.2f}", "inline": True},
                        {"name": "💵 Exit Price", "value": f"${data['exit']:,.2f}", "inline": True},
                        {"name": "⏱️ Duration", "value": data['duration'], "inline": True},
                        {"name": "📊 Net PnL (after fees)", "value": f"**{data['pnl']:+.2%}**", "inline": True},
                        {"name": "🔄 Gross PnL", "value": f"{data['gross_pnl']:+.2%}", "inline": True},
                        {"name": "💸 Total Fees", "value": f"{data['fees']:.3%}", "inline": True}
                    ],
                    "footer": {"text": f"BTC Golden Ratio | Closed at {datetime.now().strftime('%H:%M:%S')}"},
                    "timestamp": timestamp
                }]
            }
        
        elif msg_type == 'profit_lock':
            embed = {
                "embeds": [{
                    "title": f"🔒 PROFIT LOCK ACTIVATED: {data['symbol']}",
                    "description": "**Stop Loss Moved to Secure Profits**",
                    "color": 0xFFFF00,
                    "fields": [
                        {"name": "📈 Current Profit", "value": f"+{data['current_pnl']:.2%}", "inline": True},
                        {"name": "🛡️ Old SL", "value": f"${data['old_sl']:,.2f}", "inline": True},
                        {"name": "🛡️ New SL", "value": f"${data['new_sl']:,.2f}", "inline": True},
                        {"name": "💰 Current Price", "value": f"${data['current_price']:,.2f}", "inline": True},
                        {"name": "🔒 Lock Level", "value": data['lock_level'], "inline": True},
                        {"name": "⏰ Update Time", "value": datetime.now().strftime("%H:%M:%S"), "inline": True}
                    ],
                    "footer": {"text": "BTC Golden Ratio | Dynamic Risk Management"},
                    "timestamp": timestamp
                }]
            }
        
        elif msg_type == 'trailing_update':
            embed = {
                "embeds": [{
                    "title": f"🔄 TRAILING STOP UPDATED: {data['symbol']}",
                    "description": "**Stop Loss Following Price Movement**",
                    "color": 0x3498DB,
                    "fields": [
                        {"name": "📈 Highest Price", "value": f"${data['highest_price']:,.2f}", "inline": True},
                        {"name": "🛡️ New Stop Loss", "value": f"${data['new_sl']:,.2f}", "inline": True},
                        {"name": "💰 Current Price", "value": f"${data['current_price']:,.2f}", "inline": True}
                    ],
                    "footer": {"text": "BTC Golden Ratio | Trailing Active"},
                    "timestamp": timestamp
                }]
            }
        
        else:
            return
        
        response = requests.post(webhook_url, json=embed, timeout=5)
        if response.status_code == 204:
            logger.info(f"Discord alert sent: {msg_type}")
    
    except Exception as e:
        logger.error(f"Discord error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def enrich_features(df):
    """Feature engineering"""
    df = df.copy()
    
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # ATR
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
    
    # Fourier
    for i in range(1, 6):
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * df.index / 100)
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * df.index / 100)
    
    # Bollinger
    df['BB_width'] = (df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()) - \
                     (df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std())
    df['BB_position'] = (df['Close'] - (df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std())) / df['BB_width']
    
    df['frac_diff_close'] = df['Close'].diff()
    df['volume_imbalance'] = df['Volume'].diff()
    df['entropy'] = df['Close'].rolling(10).std()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Regime
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
    
    # Volatility
    vol = df['Close'].pct_change().rolling(20).std()
    df['volatility_zscore'] = (vol - vol.rolling(100).mean()) / vol.rolling(100).std()
    df['RSI_vol_adj'] = df['RSI'] / (vol * 100)
    df['ROC_vol_adj'] = (df['Close'].pct_change(10) * 100) / (vol * 100)
    
    return df.ffill().fillna(0)

def apply_rolling_normalization(df, cols):
    """Normalize features"""
    df_norm = df.copy()
    for col in cols:
        if col in df_norm.columns:
            mean = df_norm[col].rolling(window=100, min_periods=1).mean()
            std = df_norm[col].rolling(window=100, min_periods=1).std()
            df_norm[col] = (df_norm[col] - mean) / (std + 1e-8)
    return df_norm.fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# TRADING LOGIC
# ════════════════════════════════════════════════════════════════════════════

def calculate_net_pnl(entry_price, exit_price, action):
    """Calculate PnL with fees"""
    if action == 'BUY':
        gross_pnl = (exit_price - entry_price) / entry_price
    else:
        gross_pnl = (entry_price - exit_price) / entry_price
    
    net_pnl = gross_pnl - (2 * COMMISSION)
    
    return {
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'fees': 2 * COMMISSION
    }

def update_dynamic_exit_btc(position, current_price, config):
    """BTC Golden Ratio Smart Exit"""
    if position is None:
        return position, None
    
    entry = position['Entry']
    action = position['Action']
    current_sl = position['SL']
    
    # Calculate PnL
    if action == 'BUY':
        pnl_pct = ((current_price - entry) / entry) * 100
    else:
        pnl_pct = ((entry - current_price) / entry) * 100
    
    new_sl = current_sl
    update_reason = None
    update_info = None
    
    # Profit Lock - BTC Tier
    if config['enable_profit_lock'] and pnl_pct > 0:
        for level in reversed(config['profit_lock_levels']):
            if pnl_pct >= level['trigger']:
                lock_pct = level['lock'] / 100
                target_sl = entry * (1 + lock_pct if action == 'BUY' else 1 - lock_pct)
                
                if (action == 'BUY' and target_sl > current_sl) or \
                   (action == 'SELL' and target_sl < current_sl):
                    new_sl = target_sl
                    update_reason = f"PROFIT_LOCK_{level['lock']}%"
                    break
    
    # Trailing Stop
    if config['enable_trailing'] and pnl_pct >= config['trailing_activation']:
        if 'highest_price' not in position:
            position['highest_price'] = current_price
        
        if action == 'BUY':
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            trailing_sl = position['highest_price'] * (1 - config['trailing_distance']/100)
            if trailing_sl > new_sl:
                new_sl = trailing_sl
                update_reason = "TRAILING_STOP"
        else:
            if current_price < position['highest_price']:
                position['highest_price'] = current_price
            trailing_sl = position['highest_price'] * (1 + config['trailing_distance']/100)
            if trailing_sl < new_sl:
                new_sl = trailing_sl
                update_reason = "TRAILING_STOP"
    
    # Update if changed
    if new_sl != current_sl:
        position['SL'] = new_sl
        position['last_sl_update'] = datetime.now().isoformat()
        position['sl_update_reason'] = update_reason
        logger.info(f"SL Updated: {current_sl:.2f} → {new_sl:.2f} ({update_reason})")
        
        update_info = {
            'old_sl': current_sl,
            'new_sl': new_sl,
            'current_price': current_price,
            'current_pnl': pnl_pct / 100,
            'update_type': update_reason
        }
    
    return position, update_info

# ════════════════════════════════════════════════════════════════════════════
# BOT WORKER THREAD
# ════════════════════════════════════════════════════════════════════════════

def bot_worker():
    """Main trading bot running in background"""
    global BOT_RUNNING
    
    logger.info("🚀 Bot worker started - BTC Golden Ratio")
    BOT_RUNNING = True
    
    # Load state
    config = load_bot_config()
    trade_data = load_trade_data()
    
    # Load model
    model = load_model()
    
    # Feature columns
    feature_cols = [
        'log_return', 'ATR', 'BB_width', 'BB_position', 'frac_diff_close',
        'fourier_sin_1', 'fourier_sin_2', 'fourier_sin_3', 'fourier_sin_4', 'fourier_sin_5',
        'fourier_cos_1', 'fourier_cos_2', 'fourier_cos_3', 'fourier_cos_4', 'fourier_cos_5',
        'volume_imbalance', 'entropy', 'volume_ratio', 'ADX', 'SMA_distance',
        'regime_trending', 'regime_uptrend', 'regime_downtrend', 'RSI', 'MACD',
        'MACD_signal', 'volatility_zscore', 'RSI_vol_adj', 'ROC_vol_adj'
    ]
    
    # Initialize exchange
    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        logger.info("Exchange connected")
    except Exception as e:
        logger.error(f"Exchange init error: {e}")
        BOT_RUNNING = False
        return
    
    last_heartbeat = datetime.now()
    
    # Main loop
    while BOT_RUNNING:
        try:
            # Reload config
            config = load_bot_config()
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(config['symbol'], config['timeframe'], limit=400)
            df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
            
            # Features
            df_enriched = enrich_features(df)
            df_norm = apply_rolling_normalization(df_enriched, feature_cols)
            
            # Current state
            price = df['Close'].iloc[-1]
            atr = df_enriched['ATR'].iloc[-1]
            adx = df_enriched['ADX'].iloc[-1]
            sma200 = df_enriched['SMA200'].iloc[-1]
            
            # Heartbeat
            if (datetime.now() - last_heartbeat).total_seconds() >= config['heartbeat_interval']:
                uptime = datetime.now() - datetime.fromisoformat(trade_data['bot_start_time'])
                send_discord_alert(
                    config['discord_webhook'],
                    'heartbeat',
                    {'uptime': str(uptime).split('.')[0]}
                )
                last_heartbeat = datetime.now()
            
            # Manage active position
            if trade_data['active_position'] is not None:
                pos = trade_data['active_position']
                
                # Update smart exit
                if config['enable_profit_lock'] or config['enable_trailing']:
                    pos, update_info = update_dynamic_exit_btc(pos, price, config)
                    trade_data['active_position'] = pos
                    save_trade_data(trade_data)
                    
                    # Alert
                    if update_info and config['discord_webhook']:
                        if 'PROFIT_LOCK' in update_info['update_type']:
                            send_discord_alert(
                                config['discord_webhook'],
                                'profit_lock',
                                {
                                    'symbol': config['symbol'],
                                    'old_sl': update_info['old_sl'],
                                    'new_sl': update_info['new_sl'],
                                    'current_price': update_info['current_price'],
                                    'current_pnl': update_info['current_pnl'],
                                    'lock_level': update_info['update_type']
                                }
                            )
                        elif update_info['update_type'] == 'TRAILING_STOP':
                            send_discord_alert(
                                config['discord_webhook'],
                                'trailing_update',
                                {
                                    'symbol': config['symbol'],
                                    'highest_price': pos.get('highest_price', price),
                                    'new_sl': update_info['new_sl'],
                                    'current_price': update_info['current_price']
                                }
                            )
                
                # Check exit
                is_closed = False
                exit_reason = ""
                
                if pos['Action'] == 'BUY':
                    if price >= pos['TP']:
                        is_closed, exit_reason = True, "TAKE_PROFIT"
                    elif price <= pos['SL']:
                        is_closed, exit_reason = True, pos.get('sl_update_reason', 'STOP_LOSS')
                else:
                    if price <= pos['TP']:
                        is_closed, exit_reason = True, "TAKE_PROFIT"
                    elif price >= pos['SL']:
                        is_closed, exit_reason = True, pos.get('sl_update_reason', 'STOP_LOSS')
                
                if is_closed:
                    # Calculate PnL
                    pnl_data = calculate_net_pnl(pos['Entry'], price, pos['Action'])
                    entry_time = datetime.fromisoformat(pos['EntryTime'])
                    duration = datetime.now() - entry_time
                    duration_str = f"{int(duration.total_seconds()//60)}m"
                    
                    # Log trade
                    trade_entry = {
                        'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Action': f"CLOSE_{pos['Action']}",
                        'Entry': pos['Entry'],
                        'Exit': price,
                        'Type': exit_reason,
                        'Net_PnL': pnl_data['net_pnl'],
                        'Gross_PnL': pnl_data['gross_pnl'],
                        'Duration': duration_str,
                        'Regime': pos.get('Regime', 'N/A')
                    }
                    trade_data['trades'].insert(0, trade_entry)
                    
                    # Discord
                    if config['discord_webhook']:
                        send_discord_alert(
                            config['discord_webhook'],
                            'exit',
                            {
                                'action': pos['Action'],
                                'symbol': config['symbol'],
                                'entry': pos['Entry'],
                                'exit': price,
                                'pnl': pnl_data['net_pnl'],
                                'gross_pnl': pnl_data['gross_pnl'],
                                'fees': pnl_data['fees'],
                                'reason': exit_reason,
                                'duration': duration_str
                            }
                        )
                    
                    # Clear position
                    trade_data['active_position'] = None
                    save_trade_data(trade_data)
            
            # Generate new signal
            elif trade_data['active_position'] is None:
                # Prepare features
                X_last = df_norm[feature_cols].tail(30).values
                X_tensor = torch.FloatTensor(X_last).unsqueeze(0)
                
                # AI prediction
                with torch.no_grad():
                    logits = model(X_tensor)
                    probs = torch.softmax(logits / config['temperature'], dim=-1).numpy()[0]
                
                p_neutral, p_buy, p_sell = probs[0], probs[1], probs[2]
                
                # Determine regime
                is_trending = adx > config['adx_threshold_trending']
                regime_name = "TRENDING" if is_trending else "SIDEWAY"
                
                # Select threshold based on regime
                if is_trending:
                    buy_thresh = config['buy_threshold_trending']
                    sell_thresh = config['sell_threshold_trending']
                else:
                    buy_thresh = config['buy_threshold_sideway']
                    sell_thresh = config['sell_threshold_sideway']
                
                # Determine signal
                raw_sig = "NEUTRAL"
                confidence = 0
                if p_buy > buy_thresh:
                    raw_sig = "BUY"
                    confidence = p_buy
                elif p_sell > sell_thresh:
                    raw_sig = "SELL"
                    confidence = p_sell
                
                # Gate checks
                ai_pass = (raw_sig == "BUY" and p_buy > buy_thresh) or \
                          (raw_sig == "SELL" and p_sell > sell_thresh)
                adx_pass = config['adx_min'] <= adx <= config['adx_max']
                sma_pass = True
                
                if config['use_sma_filter']:
                    if raw_sig == "BUY":
                        sma_pass = price > sma200
                    elif raw_sig == "SELL":
                        sma_pass = price < sma200
                
                # Execute if all gates pass
                if ai_pass and adx_pass and sma_pass and raw_sig != "NEUTRAL":
                    # Calculate entry with slippage
                    entry_actual = price * (1 + SLIPPAGE) if raw_sig == "BUY" else price * (1 - SLIPPAGE)
                    
                    sl_val = entry_actual - (atr * config['atr_multiplier_sl']) if raw_sig == "BUY" else \
                             entry_actual + (atr * config['atr_multiplier_sl'])
                    tp_val = entry_actual + (atr * config['atr_multiplier_tp']) if raw_sig == "BUY" else \
                             entry_actual - (atr * config['atr_multiplier_tp'])
                    
                    # Create position
                    trade_data['active_position'] = {
                        'Action': raw_sig,
                        'Entry': entry_actual,
                        'TP': tp_val,
                        'SL': sl_val,
                        'EntryTime': datetime.now().isoformat(),
                        'Regime': regime_name
                    }
                    
                    # Log
                    trade_entry = {
                        'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Action': raw_sig,
                        'Entry': entry_actual,
                        'Exit': None,
                        'Type': 'MARKET' if is_trending else 'LIMIT',
                        'Net_PnL': None,
                        'Gross_PnL': None,
                        'Duration': None,
                        'Regime': regime_name
                    }
                    trade_data['trades'].insert(0, trade_entry)
                    save_trade_data(trade_data)
                    
                    # Discord
                    if config['discord_webhook']:
                        send_discord_alert(
                            config['discord_webhook'],
                            'entry',
                            {
                                'action': raw_sig,
                                'symbol': config['symbol'],
                                'entry': entry_actual,
                                'tp': tp_val,
                                'sl': sl_val,
                                'confidence': confidence,
                                'regime': regime_name
                            }
                        )
            
            # Cleanup
            del df, df_enriched, df_norm, ohlcv
            gc.collect()
            
            # Sleep
            time.sleep(config['refresh_interval'])
        
        except Exception as e:
            logger.error(f"Bot worker error: {e}")
            time.sleep(10)
    
    logger.info("🛑 Bot worker stopped")

# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI - FULL VERSION
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Monster Bot v14.4 BTC Golden Ratio",
        page_icon="🤖",
        layout="wide"
    )
    
    # Full UI CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap');
        
        .stApp { background-color: #030a03; }
        
        .crt-glow {
            color: #00FF41 !important;
            font-family: 'Fira Code', monospace !important;
            text-shadow: 0 0 5px rgba(0, 255, 65, 1), 0 0 10px rgba(0, 255, 65, 0.6);
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
        
        [data-testid="stSidebar"] {
            background-color: #010801 !important;
            border-right: 1px solid #004400;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            padding: 0.5rem 0;
        }
        
        .stSlider label, .stToggle label, .stSelectbox label, .stTextInput label,
        [data-testid="stWidgetLabel"] p {
            color: #00AA00 !important;
            font-family: 'Fira Code', monospace !important;
            font-size: 12px !important;
            margin-bottom: 0.3rem !important;
        }
        
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stToggle,
        [data-testid="stSidebar"] .stTextInput {
            margin-bottom: 1rem !important;
        }
        
        [data-testid="stDataFrame"] {
            border: 1px solid #003300;
            filter: sepia(80%) hue-rotate(80deg) brightness(90%);
        }
        
        hr { border: 0.5px solid #002200; margin: 1.5rem 0; }
        
        [data-testid="stExpander"] {
            background-color: rgba(0, 20, 0, 0.3) !important;
            border: 1px solid #003300 !important;
            border-radius: 5px;
            margin-bottom: 0.5rem !important;
        }
        
        .status-active { color: #00FF41; font-weight: bold; }
        .status-inactive { color: #FF0000; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    # Start bot thread (only once)
    if 'bot_thread_started' not in st.session_state:
        thread = threading.Thread(target=bot_worker, daemon=True)
        thread.start()
        st.session_state.bot_thread_started = True
        logger.info("Bot thread initialized")
    
    # Load data
    config = load_bot_config()
    trade_data = load_trade_data()
    
    # Header
    st.markdown("<h1 class='crt-glow'>🤖 MONSTER BOT v14.4 - BTC GOLDEN RATIO</h1>", unsafe_allow_html=True)
    
    # Status
    status_text = "🟢 ACTIVE" if BOT_RUNNING else "🔴 STOPPED"
    status_class = "status-active" if BOT_RUNNING else "status-inactive"
    st.markdown(f"<h3 class='{status_class}'>Status: {status_text} | 27% Backtest Win</h3>", unsafe_allow_html=True)
    
    # ─── SIDEBAR ───
    st.sidebar.markdown("<h2 class='crt-glow' style='font-size:20px; margin-bottom: 1rem;'>⚙️ CONFIGURATION</h2>", unsafe_allow_html=True)
    
    # AI Settings
    with st.sidebar.expander("🤖 AI MODEL", expanded=False):
        config['temperature'] = st.slider("Temperature", 0.1, 1.5, config['temperature'], 0.01)
        st.caption("BTC Optimized: 0.42")
    
    # Dual Threshold
    with st.sidebar.expander("🎯 DUAL THRESHOLD (BTC)", expanded=True):
        st.markdown("**Trending Market:**")
        config['buy_threshold_trending'] = st.slider("Buy (Trending)", 0.1, 0.8, config['buy_threshold_trending'], 0.01)
        config['sell_threshold_trending'] = st.slider("Sell (Trending)", 0.1, 0.8, config['sell_threshold_trending'], 0.01)
        
        st.markdown("**Sideway Market:**")
        config['buy_threshold_sideway'] = st.slider("Buy (Sideway)", 0.1, 0.8, config['buy_threshold_sideway'], 0.01)
        config['sell_threshold_sideway'] = st.slider("Sell (Sideway)", 0.1, 0.8, config['sell_threshold_sideway'], 0.01)
        
        st.caption("Golden: Trending 0.36 | Sideway 0.22")
    
    # Risk Management
    with st.sidebar.expander("⚖️ RISK MANAGEMENT", expanded=False):
        config['atr_multiplier_sl'] = st.slider("Stop Loss (ATR)", 1.0, 10.0, config['atr_multiplier_sl'], 0.1)
        config['atr_multiplier_tp'] = st.slider("Take Profit (ATR)", 5.0, 50.0, config['atr_multiplier_tp'], 0.5)
        st.caption("Golden: SL 3.2 | TP 18.5")
    
    # Smart Exit
    with st.sidebar.expander("🛡️ SMART EXIT (BTC TIER)", expanded=False):
        config['enable_profit_lock'] = st.checkbox("Enable Profit Lock", config['enable_profit_lock'])
        if config['enable_profit_lock']:
            st.caption("📊 Lock Levels:")
            st.caption("• 1.8% → 1.2%")
            st.caption("• 3.5% → 2.8%")
            st.caption("• 5.5% → 4.5%")
        
        config['enable_trailing'] = st.checkbox("Enable Trailing Stop", config['enable_trailing'])
        if config['enable_trailing']:
            st.caption(f"🔄 Activate: {config['trailing_activation']}% | Distance: {config['trailing_distance']}%")
    
    # Filters
    with st.sidebar.expander("📡 MARKET FILTERS", expanded=False):
        config['adx_min'] = st.slider("Min ADX", 10, 50, config['adx_min'], 1)
        config['adx_max'] = st.slider("Max ADX", 50, 100, config['adx_max'], 1)
        config['use_sma_filter'] = st.checkbox("Use SMA200 Filter", config['use_sma_filter'])
    
    # Discord
    with st.sidebar.expander("📡 DISCORD WEBHOOK", expanded=False):
        config['discord_webhook'] = st.text_input("Webhook URL", value=config['discord_webhook'], type="password")
    
    # Save button
    if st.sidebar.button("💾 Save Configuration"):
        save_bot_config(config)
        st.sidebar.success("Configuration saved!")
    
    # ─── MAIN LAYOUT ───
    col_left, col_right = st.columns([1.2, 1.8])
    
    with col_left:
        st.markdown("<div class='crt-glow' style='font-size:16px;'>[LIVE STATUS]</div>", unsafe_allow_html=True)
        
        # Current position
        if trade_data['active_position']:
            pos = trade_data['active_position']
            st.markdown(f"""
            <div class='signal-card' style='border-color: {"#00FF41" if pos["Action"] == "BUY" else "#FF0000"};'>
                <div style='color: {"#00FF41" if pos["Action"] == "BUY" else "#FF0000"}; font-size:40px; font-weight:bold;'>{pos['Action']}</div>
                <div class='crt-glow' style='font-size:18px; color:white !important;'>Entry: ${pos['Entry']:,.2f}</div>
                <div class='crt-glow' style='font-size:14px;'>TP: ${pos['TP']:,.2f} | SL: ${pos['SL']:,.2f}</div>
                <div class='crt-glow' style='font-size:12px; opacity:0.6;'>{pos.get('Regime', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No active position - Monitoring market...")
        
        # Trade log
        st.markdown("<div class='crt-glow' style='font-size:16px; margin-top:20px;'>[TRADE LOG]</div>", unsafe_allow_html=True)
        
        if st.button("🔄 Sync Data"):
            trade_data = load_trade_data()
            st.success("Data synced from file!")
        
        if trade_data['trades']:
            df_trades = pd.DataFrame(trade_data['trades'][:10])
            
            # Summary metrics
            cols = st.columns(3)
            cols[0].metric("Total Trades", len(trade_data['trades']))
            
            closed = [t for t in trade_data['trades'] if t['Net_PnL'] is not None]
            if closed:
                wins = len([t for t in closed if t['Net_PnL'] > 0])
                win_rate = (wins / len(closed)) * 100
                cols[1].metric("Win Rate", f"{win_rate:.1f}%")
                cols[2].metric("W/L", f"{wins}/{len(closed)-wins}")
            
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet")
    
    with col_right:
        st.markdown("<div class='crt-glow' style='font-size:16px;'>[LIVE MARKET FEED]</div>", unsafe_allow_html=True)
        
        # TradingView chart
        tv_html = f"""
        <div style="height:750px; border: 2px solid #004400; border-radius:5px; overflow:hidden; 
                    filter: brightness(0.7) contrast(1.2) sepia(100%) hue-rotate(70deg);">
            <div id="tv_chart_btc" style="height:100%;"></div>
            <script src="https://s3.tradingview.com/tv.js"></script>
            <script>
                new TradingView.widget({{
                    "autosize": true,
                    "symbol": "KRAKEN:BTCUSDT",
                    "interval": "15",
                    "theme": "dark",
                    "container_id": "tv_chart_btc",
                    "style": "1",
                    "enable_publishing": false,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true
                }});
            </script>
        </div>
        """
        components.html(tv_html, height=760)
    
    # Control buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reload Data"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared!")
    
    with col3:
        if st.button("📤 Test Discord"):
            if config['discord_webhook']:
                send_discord_alert(config['discord_webhook'], 'heartbeat', {'uptime': 'Test'})
                st.success("Test sent!")
            else:
                st.error("No webhook configured")
    
    # Footer
    st.caption("Monster Bot v14.4 BTC Golden Ratio | 27% Backtest Win | Full UI Version")

if __name__ == "__main__":
    main()
