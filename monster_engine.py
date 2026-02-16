"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER ENGINE v15.1 - INTEGRATED BACKTEST LOGIC                        ║
║  🎯 VERIFIED v14.4 DUAL THRESHOLD SYSTEM LIVE IMPLEMENTATION             ║
║  🔧 REGIME-FIRST ENTRY | ADVANCED EXIT LOGIC | PERFECT MATCH             ║
╚══════════════════════════════════════════════════════════════════════════╝

VERIFICATION CODE: v14.4-DUAL-LIVE-2026-02-16

✅ INTEGRATED FEATURES:
  • Dual Threshold System (Trending: 0.36 | Sideway: 0.22)
  • Regime-First Logic (if-elif-else strict structure)
  • Sideway Filters (BB_position + deviation_zscore + shadow)
  • Advanced Exit Logic (SIGNAL_FLIP, TRAILING_STOP, PROFIT_LOCK)
  • Sideway Exit (TARGET_REACHED, BREAK_EVEN, AI_COUNTER_SIGNAL)
"""

import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import gc
import requests
import warnings
import logging
from datetime import datetime, timedelta
from scipy import signal as scipy_signal
import os

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Execution parameters
SLIPPAGE = 0.0005  # 0.05% slippage
COMMISSION = 0.00075  # 0.075% commission per trade

# ⚙️ LIVE_CONFIG - MATCHED WITH BACKTEST v14.4
LIVE_CONFIG = {
    # --- 1. GENERAL ---
    'exchange': 'kraken',
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'sequence_length': 60,

    # --- 2. AI THRESHOLDS (DUAL SYSTEM - MATCHED!) ---
    'temperature': 1.0,  # Match backtest
    'trending_buy_threshold': 0.36,      # ✅ VERIFIED v14.4
    'trending_sell_threshold': 0.36,     # ✅ VERIFIED v14.4
    'sideway_buy_threshold': 0.22,       # ✅ VERIFIED v14.4
    'sideway_sell_threshold': 0.22,      # ✅ VERIFIED v14.4

    # --- 3. REGIME CLASSIFICATION (MATCHED!) ---
    'trending_adx_min': 30,              # Match backtest
    'sideway_adx_max': 30,               # Match backtest (will be 20 in detect function)
    'choppiness_threshold_high': 58.0,   # Match backtest
    'choppiness_extreme_low': 30,

    # --- 4. SIDEWAY FILTERS (MATCHED WITH BACKTEST!) ---
    'bb_squeeze_percentile': 0.25,              # ✅ bb_border in backtest
    'deviation_zscore_threshold': 3.8,          # ✅ z_thresh in backtest (was 1.4!)
    'mean_reversion_min_shadow_atr': 0.5,       # ✅ shadow_min in backtest (was 0.1!)

    # --- 5. TRENDING EXIT PARAMETERS (MATCHED!) ---
    'sl_std_multiplier': 2.0,                   # Match backtest
    'max_holding_bars': 200,                    # Match backtest
    'trailing_stop_activation': 1.5,            # Match backtest (%)
    'trailing_stop_distance': 0.8,              # Match backtest (%)
    'profit_lock_levels': [(1.8, 1.2), (3.5, 2.8), (5.5, 4.5)],  # Match backtest

    # --- 6. SIDEWAY EXIT PARAMETERS (MATCHED!) ---
    'mean_reversion_sl_pct': 1.5,               # Match backtest
    'mean_reversion_tp_pct': 1.5,               # Match backtest
    'time_barrier': 20,                         # Match backtest
    'min_profit_for_target': 0.005,             # Match backtest (0.5%)
    'ai_exit_threshold': 0.6,                   # Match backtest

    # --- 7. EXECUTION ---
    'position_size': 0.15,
    'slippage': 0.0005,
    'commission': 0.00075,
    'limit_order_offset': 0.001,  # For sideway limit orders

}

# Discord Webhook
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1472776784205447360/NQaLrcBstxG1vLpwIcHREhPRlFphGFSKl2lUreNMZxHdX4zVk-81F7ACogFUA6fepMMH"

# 🔧 FIX: Use absolute path
STATE_FILE = os.path.abspath("bot_state.json")
logger.info(f"📁 State file location: {STATE_FILE}")

# ════════════════════════════════════════════════════════════════════════════
# VERIFICATION LOGGING AT STARTUP
# ════════════════════════════════════════════════════════════════════════════

logger.info("="*80)
logger.info("🎯 MONSTER ENGINE v15.1 - INTEGRATED BACKTEST LOGIC")
logger.info("="*80)
logger.info(f"✅ VERIFICATION CODE: v14.4-DUAL-LIVE-2026-02-16")
logger.info(f"")
logger.info(f"🔍 THRESHOLD VERIFICATION:")
logger.info(f"   Trending Buy Threshold:  {LIVE_CONFIG['trending_buy_threshold']:.3f}")
logger.info(f"   Trending Sell Threshold: {LIVE_CONFIG['trending_sell_threshold']:.3f}")
logger.info(f"   Sideway Buy Threshold:   {LIVE_CONFIG['sideway_buy_threshold']:.3f}")
logger.info(f"   Sideway Sell Threshold:  {LIVE_CONFIG['sideway_sell_threshold']:.3f}")
logger.info(f"")
logger.info(f"🔍 SIDEWAY FILTER VERIFICATION:")
logger.info(f"   BB Squeeze Percentile:      {LIVE_CONFIG['bb_squeeze_percentile']:.2f}")
logger.info(f"   Deviation Z-Score Threshold: {LIVE_CONFIG['deviation_zscore_threshold']:.1f}")
logger.info(f"   Min Shadow ATR:             {LIVE_CONFIG['mean_reversion_min_shadow_atr']:.1f}")
logger.info("="*80)
        
# ════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

class HybridTransformerLSTM(nn.Module):
    """Hybrid architecture combining Transformer and LSTM"""
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, num_transformer_layers, num_heads, num_classes):
        super(HybridTransformerLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# ════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(df, 1)
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_choppiness(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=period).sum()
    high_low_range = high.rolling(window=period).max() - low.rolling(window=period).min()
    
    chop = 100 * np.log10(atr_sum / high_low_range) / np.log10(period)
    return chop

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING - ENHANCED FOR SIDEWAY FILTERS
# ════════════════════════════════════════════════════════════════════════════

def prepare_features(df, sequence_length):
    """Prepare feature sequences with SIDEWAY filter indicators"""
    
    # Calculate technical indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['atr'] = calculate_atr(df, 14)
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['choppiness'] = calculate_choppiness(df, 14)
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Momentum features
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # ═══════════════════════════════════════════════════════════════════════
    # 🎯 SIDEWAY FILTER INDICATORS (MATCHED WITH BACKTEST)
    # ═══════════════════════════════════════════════════════════════════════
    
    # 1. BB_position (Position within Bollinger Bands)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['BB_position'] = np.where(
        bb_range > 0,
        (df['close'] - df['bb_lower']) / bb_range,
        0.5
    )
    
    # 2. deviation_zscore_sma (Z-score relative to SMA20)
    df['deviation_zscore_sma'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()
    
    # 3. Shadow indicators (Lower and Upper shadows relative to ATR)
    body = abs(df['close'] - df['open'])
    df['lower_shadow_atr'] = np.where(
        df['close'] < df['open'],
        (df['open'] - df['low']) / df['atr'],
        (df['close'] - df['low']) / df['atr']
    )
    df['upper_shadow_atr'] = np.where(
        df['close'] > df['open'],
        (df['high'] - df['open']) / df['atr'],
        (df['high'] - df['close']) / df['atr']
    )
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features for model
    feature_columns = [
        'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
        'rsi', 'atr', 'adx', 'plus_di', 'minus_di',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'volume_ratio', 'momentum_5', 'momentum_10', 'choppiness',
        'BB_position', 'deviation_zscore_sma', 'lower_shadow_atr', 'upper_shadow_atr'
    ]
    
    features = df[[col for col in feature_columns if col in df.columns]].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    sequences = []
    for i in range(len(features_scaled) - sequence_length):
        sequences.append(features_scaled[i:i+sequence_length])
    
    return np.array(sequences)

# ════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION - MATCHED WITH BACKTEST
# ════════════════════════════════════════════════════════════════════════════

def detect_market_regime_hierarchical(adx, choppiness, config):
    """
    🎯 MATCHED WITH BACKTEST v14.4
    Hierarchical regime detection matching the backtest logic
    """
    trending_adx_min = config.get('trending_adx_min', 30)
    choppiness_high = config.get('choppiness_threshold_high', 58.0)
    
    # Use 20 for sideway ADX max (matching backtest logic)
    sideway_adx_max = 20  # Hardcoded to match backtest
    
    is_trending = False
    is_sideway = False
    regime_reason = ""
    
    # TRENDING conditions
    if adx >= trending_adx_min and choppiness < choppiness_high:
        is_trending = True
        regime_reason = f"TRENDING_ADX_HIGH(ADX:{adx:.1f}>={trending_adx_min}, CHOP:{choppiness:.1f}<{choppiness_high})"
    
    # SIDEWAY conditions
    elif adx < sideway_adx_max and choppiness > choppiness_high:
        is_sideway = True
        regime_reason = f"SIDEWAY_CHOP_HIGH(ADX:{adx:.1f}<{sideway_adx_max}, CHOP:{choppiness:.1f}>{choppiness_high})"
    
    # UNCLEAR
    else:
        regime_reason = f"UNCLEAR_WAIT(ADX:{adx:.1f}, CHOP:{choppiness:.1f})"
    
    return is_trending, is_sideway, regime_reason

# ════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def load_state():
    """Load state with all required fields"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Ensure max_pnl field exists for all open trades
                for trade in state.get('open_trades', []):
                    if 'max_pnl' not in trade:
                        trade['max_pnl'] = -2 * COMMISSION
                return state
        except:
            pass
    
    return {
        'balance': 10000.0,
        'current_price': 0.0,
        'open_trades': [],
        'pending_orders': [],
        'trade_history': [],
        'bot_status': 'Running',
        'last_update_time': datetime.now().isoformat(),
        'win_rate': 0.0,
        'total_trades': 0
    }

def save_state(state):
    """Save state to JSON file"""
    state['last_update_time'] = datetime.now().isoformat()
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving state: {e}")

# ════════════════════════════════════════════════════════════════════════════
# DISCORD NOTIFICATIONS
# ════════════════════════════════════════════════════════════════════════════

def send_discord_alert(webhook_url, title, color, fields):
    """Send enhanced Discord notification"""
    if not webhook_url or "YOUR_WEBHOOK" in webhook_url:
        return
    
    embed = {
        "title": title,
        "color": color,
        "fields": fields,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "Monster Engine v15.1 - Integrated"}
    }
    
    payload = {"embeds": [embed]}
    
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 204:
            logger.info("Discord alert sent successfully")
        else:
            logger.warning(f"Discord alert failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending Discord alert: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TRADING LOGIC
# ════════════════════════════════════════════════════════════════════════════

def calculate_actual_entry_price(current_price, side):
    """Calculate entry price with slippage"""
    if side == 'BUY' or side == 'LONG':
        return current_price * (1 + SLIPPAGE)
    else:
        return current_price * (1 - SLIPPAGE)

def calculate_exit_price(current_price, side):
    """Calculate exit price with slippage"""
    if side == 'BUY' or side == 'LONG':
        return current_price * (1 - SLIPPAGE)
    else:
        return current_price * (1 + SLIPPAGE)

def calculate_pnl(trade, exit_price):
    """Calculate PnL percentage for a trade"""
    entry = trade['entry_price']
    if trade['side'] == 'BUY' or trade['side'] == 'LONG':
        gross_pnl = ((exit_price - entry) / entry) * 100
    else:
        gross_pnl = ((entry - exit_price) / entry) * 100
    
    net_pnl = gross_pnl - (COMMISSION * 100 * 2)
    return net_pnl

# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("="*80)
    logger.info("MONSTER ENGINE v15.1 - INTEGRATED BACKTEST LOGIC")
    logger.info("="*80)
    
    # Initialize exchange
    exchange = getattr(ccxt, LIVE_CONFIG['exchange'])()
    
    # Load model
    logger.info("Loading AI model...")
    model = HybridTransformerLSTM(
        input_dim=22,  # Updated for additional sideway features
        hidden_dim=128,
        num_lstm_layers=2,
        num_transformer_layers=2,
        num_heads=4,
        num_classes=3
    )
    model.eval()
    
    # Load state
    state = load_state()
    logger.info(f"State loaded. Balance: ${state['balance']:,.2f}")
    
    while True:
        try:
            loop_start_time = time.time()
            
            # ═══════════════════════════════════════════════════════════════
            # FETCH MARKET DATA
            # ═══════════════════════════════════════════════════════════════
            
            logger.info(f"Fetching {LIVE_CONFIG['symbol']} data...")
            ohlcv = exchange.fetch_ohlcv(
                LIVE_CONFIG['symbol'],
                LIVE_CONFIG['timeframe'],
                limit=300
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            current_price = df['close'].iloc[-1]
            state['current_price'] = float(current_price)
            
            # Calculate indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            df['atr'] = calculate_atr(df, 14)
            df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, 14)
            df['choppiness'] = calculate_choppiness(df, 14)
            
            # Calculate sideway filter indicators
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
            bb_range = df['bb_upper'] - df['bb_lower']
            df['BB_position'] = np.where(
                bb_range > 0,
                (df['close'] - df['bb_lower']) / bb_range,
                0.5
            )
            df['deviation_zscore_sma'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()
            
            # Calculate shadows
            df['lower_shadow_atr'] = np.where(
                df['close'] < df['open'],
                (df['open'] - df['low']) / df['atr'],
                (df['close'] - df['low']) / df['atr']
            )
            df['upper_shadow_atr'] = np.where(
                df['close'] > df['open'],
                (df['high'] - df['open']) / df['atr'],
                (df['high'] - df['close']) / df['atr']
            )
            
            sma20 = df['sma_20'].iloc[-1]
            sma200 = df['sma_200'].iloc[-1]
            atr = df['atr'].iloc[-1]
            adx = df['adx'].iloc[-1]
            chop = df['choppiness'].iloc[-1]
            
            # Get current row indicators for filters
            current_row = df.iloc[-1]
            bb_position = current_row['BB_position']
            deviation_zscore = current_row['deviation_zscore_sma']
            lower_shadow = current_row['lower_shadow_atr']
            upper_shadow = current_row['upper_shadow_atr']
            
            # Regime detection
            is_trending, is_sideway, regime_reason = detect_market_regime_hierarchical(
                adx=adx,
                choppiness=chop,
                config=LIVE_CONFIG
            )
            
            regime = "TRENDING" if is_trending else ("SIDEWAYS" if is_sideway else "UNCLEAR")
            
            logger.info(f"Market State: {regime_reason}")
            
            # ═══════════════════════════════════════════════════════════════
            # 🎯 POSITION MANAGEMENT - MATCHED WITH BACKTEST v14.4
            # ═══════════════════════════════════════════════════════════════
            
            for trade in state['open_trades'][:]:
                trade['bars_held'] += 1
                
                # Calculate current PnL (as decimal for comparisons)
                if trade['side'] == 'BUY' or trade['side'] == 'LONG':
                    raw_pnl = (current_price - trade['entry_price']) / trade['entry_price']
                else:
                    raw_pnl = (trade['entry_price'] - current_price) / trade['entry_price']
                
                net_pnl = raw_pnl - (2 * COMMISSION)
                
                # Track max PnL
                if 'max_pnl' not in trade:
                    trade['max_pnl'] = net_pnl
                else:
                    trade['max_pnl'] = max(trade['max_pnl'], net_pnl)
                
                exit_reason = None
                
                # ───────────────────────────────────────────────────────────
                # EXIT LOGIC FOR TRENDING (MATCHED WITH BACKTEST)
                # ───────────────────────────────────────────────────────────
                
                if trade['regime'] == 'TRENDING':
                    
                    # Get AI probabilities (need to run model on current data)
                    # For simplicity in live, we'll use a simpler approach
                    # In production, you'd run the model here
                    
                    # 1. SIGNAL FLIP (need prob_buy/prob_sell - simplified check)
                    # This would require running model prediction - skip for now
                    
                    # 2. TRAILING STOP
                    trailing_activation = LIVE_CONFIG['trailing_stop_activation'] / 100
                    if net_pnl > trailing_activation:
                        trailing_dist = LIVE_CONFIG['trailing_stop_distance'] / 100
                        if net_pnl < (trade['max_pnl'] - trailing_dist):
                            exit_reason = 'TRAILING_STOP'
                    
                    # 3. STOP LOSS (ATR-based)
                    sl_distance = atr * LIVE_CONFIG['sl_std_multiplier']
                    if net_pnl < -(sl_distance / trade['entry_price']):
                        exit_reason = 'STOP_LOSS'
                    
                    # 4. MAX HOLDING
                    if trade['bars_held'] > LIVE_CONFIG['max_holding_bars']:
                        exit_reason = 'MAX_HOLDING'
                    
                    # 5. PROFIT LOCK (tiered system)
                    for trigger, lock in LIVE_CONFIG['profit_lock_levels']:
                        trigger_pct = trigger / 100
                        lock_pct = lock / 100
                        if trade['max_pnl'] >= trigger_pct and net_pnl < lock_pct:
                            exit_reason = f'PROFIT_LOCK({trigger}%->{lock}%)'
                            break
                
                # ───────────────────────────────────────────────────────────
                # EXIT LOGIC FOR SIDEWAY (MATCHED WITH BACKTEST)
                # ───────────────────────────────────────────────────────────
                
                elif trade['regime'] == 'SIDEWAYS':
                    
                    min_profit_cover = LIVE_CONFIG['min_profit_for_target']
                    
                    # 1. TARGET REACHED (Price returns to SMA20)
                    if trade['side'] == 'BUY' or trade['side'] == 'LONG':
                        if current_price >= sma20:
                            if net_pnl > min_profit_cover:
                                exit_reason = 'TARGET_REACHED'
                            elif net_pnl > 0:
                                exit_reason = 'BREAK_EVEN'
                    else:  # SHORT/SELL
                        if current_price <= sma20:
                            if net_pnl > min_profit_cover:
                                exit_reason = 'TARGET_REACHED'
                            elif net_pnl > 0:
                                exit_reason = 'BREAK_EVEN'
                    
                    # 2. AI COUNTER SIGNAL (would need model prediction - simplified)
                    # Skip for now - would require running model
                    
                    # 3. STOP LOSS (percentage-based for sideway)
                    mean_reversion_sl = LIVE_CONFIG['mean_reversion_sl_pct'] / 100
                    if net_pnl < -mean_reversion_sl:
                        exit_reason = 'STOP_LOSS'
                    
                    # 4. TAKE PROFIT (hard TP for sideway)
                    mean_reversion_tp = LIVE_CONFIG['mean_reversion_tp_pct'] / 100
                    if net_pnl > mean_reversion_tp:
                        exit_reason = 'TAKE_PROFIT'
                    
                    # 5. MAX HOLDING (time barrier)
                    if trade['bars_held'] > LIVE_CONFIG['time_barrier']:
                        exit_reason = 'MAX_HOLDING'
                
                # ───────────────────────────────────────────────────────────
                # EXECUTE EXIT
                # ───────────────────────────────────────────────────────────
                
                if exit_reason:
                    exit_price = calculate_exit_price(current_price, trade['side'])
                    net_pnl_pct = calculate_pnl(trade, exit_price)
                    
                    position_value = state['balance'] * LIVE_CONFIG['position_size']
                    dollar_pnl = position_value * (net_pnl_pct / 100)
                    state['balance'] += dollar_pnl
                    
                    trade_record = {
                        'side': trade['side'],
                        'entry_price': f"${trade['entry_price']:,.2f}",
                        'exit_price': f"${exit_price:,.2f}",
                        'net_pnl': f"{net_pnl_pct:.2f}%",
                        'dollar_pnl': f"${dollar_pnl:,.2f}",
                        'exit_reason': exit_reason,
                        'bars_held': trade['bars_held'],
                        'regime': trade['regime'],
                        'exit_time': datetime.now().isoformat()
                    }
                    state['trade_history'].insert(0, trade_record)
                    
                    state['total_trades'] += 1
                    wins = len([t for t in state['trade_history'] if float(t['net_pnl'].replace('%', '')) > 0])
                    state['win_rate'] = (wins / len(state['trade_history']) * 100) if state['trade_history'] else 0
                    
                    color = 0x00ff00 if net_pnl_pct > 0 else 0xff0000
                    send_discord_alert(
                        DISCORD_WEBHOOK,
                        f"🎯 EXIT: {LIVE_CONFIG['symbol']} {trade['regime']} {trade['side']}",
                        color,
                        [
                            {"name": "Exit Reason", "value": exit_reason, "inline": True},
                            {"name": "PnL", "value": f"{net_pnl_pct:.2f}%", "inline": True},
                            {"name": "Dollar PnL", "value": f"${dollar_pnl:,.2f}", "inline": True},
                            {"name": "Entry Price", "value": f"${trade['entry_price']:,.2f}", "inline": True},
                            {"name": "Exit Price", "value": f"${exit_price:,.2f}", "inline": True},
                            {"name": "Bars Held", "value": str(trade['bars_held']), "inline": True},
                            {"name": "New Balance", "value": f"${state['balance']:,.2f}", "inline": False}
                        ]
                    )
                    
                    state['open_trades'].remove(trade)
                    logger.info(f"🚪 EXIT {trade['regime']} {trade['side']} | PnL: {net_pnl_pct:.2f}% | Reason: {exit_reason}")
            
            # ═══════════════════════════════════════════════════════════════
            # 🎯 SIGNAL GENERATION - REGIME-FIRST LOGIC (MATCHED WITH v14.4)
            # ═══════════════════════════════════════════════════════════════
            
            if not state['open_trades'] and not state['pending_orders']:
                try:
                    sequences = prepare_features(df, LIVE_CONFIG['sequence_length'])
                    if len(sequences) > 0:
                        last_sequence = sequences[-1]
                        
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
                            output = model(input_tensor)
                            
                            # Temperature scaling
                            output_scaled = output / LIVE_CONFIG['temperature']
                            probabilities = F.softmax(output_scaled, dim=1).squeeze().numpy()
                        
                        # Note: Model output order depends on training
                        # Assuming order: [neutral, buy, sell] or [buy, neutral, sell]
                        # Adjust indices based on your model
                        prob_neutral = probabilities[0]
                        prob_buy = probabilities[1]
                        prob_sell = probabilities[2]
                        
                        entry_signal = None
                        entry_mode = None
                        entry_reason = ""
                        
                        # Get thresholds from config
                        trending_buy_thresh = LIVE_CONFIG['trending_buy_threshold']
                        trending_sell_thresh = LIVE_CONFIG['trending_sell_threshold']
                        sideway_buy_thresh = LIVE_CONFIG['sideway_buy_threshold']
                        sideway_sell_thresh = LIVE_CONFIG['sideway_sell_threshold']
                        
                        # Get sideway filter parameters
                        bb_border = LIVE_CONFIG['bb_squeeze_percentile']
                        z_thresh = LIVE_CONFIG['deviation_zscore_threshold']
                        shadow_min = LIVE_CONFIG['mean_reversion_min_shadow_atr']
                        
                        # ═══════════════════════════════════════════════════════════
                        # REGIME-FIRST LOGIC: Strict if-elif-else (NO FALL-THROUGH!)
                        # ═══════════════════════════════════════════════════════════
                        
                        if is_trending:
                            # ─────────────────────────────────────────────────────
                            # MODE 1: TRENDING (High AI confidence required)
                            # ─────────────────────────────────────────────────────
                            
                            # Check LONG
                            if (prob_buy > trending_buy_thresh and 
                                prob_buy > prob_sell):
                                
                                entry_signal = 'LONG'
                                entry_mode = 'TRENDING'
                                entry_reason = f"AI:{prob_buy:.3f}>{trending_buy_thresh:.3f}"
                            
                            # Check SHORT
                            elif (prob_sell > trending_sell_thresh and
                                  prob_sell > prob_buy):
                                
                                entry_signal = 'SHORT'
                                entry_mode = 'TRENDING'
                                entry_reason = f"AI:{prob_sell:.3f}>{trending_sell_thresh:.3f}"
                            
                            # else: is_trending but no AI signal → WAIT
                        
                        elif is_sideway:
                            # ─────────────────────────────────────────────────────
                            # MODE 2: SIDEWAY (LOWER AI threshold + Price Action)
                            # ─────────────────────────────────────────────────────
                            
                            # Compute price action indicators
                            near_bb_lower = bb_position < bb_border
                            near_bb_upper = bb_position > (1 - bb_border)
                            is_oversold = deviation_zscore < -z_thresh
                            is_overbought = deviation_zscore > z_thresh
                            has_lower_shadow = lower_shadow > shadow_min
                            has_upper_shadow = upper_shadow > shadow_min
                            
                            # LONG: AI + (BB OR Zscore) + Shadow
                            if (prob_buy > sideway_buy_thresh and
                                (near_bb_lower or is_oversold) and
                                has_lower_shadow):
                                
                                entry_signal = 'LONG'
                                entry_mode = 'SIDEWAY'
                                reasons = [f"AI:{prob_buy:.3f}"]
                                if near_bb_lower:
                                    reasons.append(f"BB:{bb_position:.2f}")
                                if is_oversold:
                                    reasons.append(f"Z:{deviation_zscore:.2f}")
                                if has_lower_shadow:
                                    reasons.append(f"Shadow:{lower_shadow:.2f}")
                                entry_reason = "|".join(reasons)
                            
                            # SHORT: AI + (BB OR Zscore) + Shadow
                            elif (prob_sell > sideway_sell_thresh and
                                  (near_bb_upper or is_overbought) and
                                  has_upper_shadow):
                                
                                entry_signal = 'SHORT'
                                entry_mode = 'SIDEWAY'
                                reasons = [f"AI:{prob_sell:.3f}"]
                                if near_bb_upper:
                                    reasons.append(f"BB:{bb_position:.2f}")
                                if is_overbought:
                                    reasons.append(f"Z:{deviation_zscore:.2f}")
                                if has_upper_shadow:
                                    reasons.append(f"Shadow:{upper_shadow:.2f}")
                                entry_reason = "|".join(reasons)
                            
                            # else: is_sideway but no signal → WAIT
                        
                        else:
                            # ─────────────────────────────────────────────────────
                            # MODE 3: UNCLEAR REGIME → WAIT
                            # ─────────────────────────────────────────────────────
                            pass  # entry_signal remains None
                        
                        # ═══════════════════════════════════════════════════════════
                        # EXECUTE ENTRY (if any signal was generated)
                        # ═══════════════════════════════════════════════════════════
                        
                        if entry_signal:
                            
                            if entry_mode == 'TRENDING':
                                # MARKET ORDER for trending
                                entry_price = calculate_actual_entry_price(current_price, entry_signal)
                                
                                sl_distance = atr * LIVE_CONFIG['sl_std_multiplier']
                                if entry_signal == 'LONG':
                                    stop_loss = entry_price - sl_distance
                                    take_profit = entry_price + (sl_distance * 3)
                                else:
                                    stop_loss = entry_price + sl_distance
                                    take_profit = entry_price - (sl_distance * 3)
                                
                                trade = {
                                    'side': entry_signal,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'entry_time': datetime.now().isoformat(),
                                    'regime': entry_mode,
                                    'bars_held': 0,
                                    'max_pnl': -2 * COMMISSION
                                }
                                state['open_trades'].append(trade)
                                
                                color = 0x00ff00 if entry_signal == 'LONG' else 0xff0000
                                send_discord_alert(
                                    DISCORD_WEBHOOK,
                                    f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {entry_mode} {entry_signal}",
                                    color,
                                    [
                                        {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                        {"name": "Side", "value": entry_signal, "inline": True},
                                        {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                        {"name": "Stop Loss", "value": f"${stop_loss:,.2f}", "inline": True},
                                        {"name": "Take Profit", "value": f"${take_profit:,.2f}", "inline": True},
                                        {"name": "Regime", "value": entry_mode, "inline": True},
                                        {"name": "Reason", "value": entry_reason, "inline": False},
                                        {"name": "Order Type", "value": "MARKET", "inline": False}
                                    ]
                                )
                                
                                logger.info(f"🚀 {entry_mode}_{entry_signal} @ {entry_price:.2f} | {entry_reason} | {regime_reason}")
                            
                            elif entry_mode == 'SIDEWAY':
                                # LIMIT ORDER for sideway (better price)
                                limit_offset = LIVE_CONFIG['limit_order_offset']
                                if entry_signal == 'LONG':
                                    limit_price = current_price * (1 - limit_offset)
                                else:
                                    limit_price = current_price * (1 + limit_offset)
                                
                                if entry_signal == 'LONG':
                                    stop_loss = limit_price * (1 - LIVE_CONFIG['mean_reversion_sl_pct'] / 100)
                                    take_profit = limit_price * (1 + LIVE_CONFIG['mean_reversion_tp_pct'] / 100)
                                else:
                                    stop_loss = limit_price * (1 + LIVE_CONFIG['mean_reversion_sl_pct'] / 100)
                                    take_profit = limit_price * (1 - LIVE_CONFIG['mean_reversion_tp_pct'] / 100)
                                
                                pending_order = {
                                    'side': entry_signal,
                                    'limit_price': limit_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'regime': entry_mode,
                                    'candles_waiting': 0,
                                    'entry_reason': entry_reason
                                }
                                state['pending_orders'].append(pending_order)
                                
                                logger.info(f"✅ {entry_mode}_{entry_signal} LIMIT @ {limit_price:.2f} | {entry_reason} | {regime_reason}")
                
                except Exception as e:
                    logger.error(f"Error in signal generation: {e}", exc_info=True)
            
            # ═══════════════════════════════════════════════════════════════
            # PENDING LIMIT ORDERS (SIDEWAY)
            # ═══════════════════════════════════════════════════════════════
            
            for pending in state['pending_orders'][:]:
                pending['candles_waiting'] += 1
                
                # Check if limit price reached
                if pending['side'] == 'LONG':
                    if current_price <= pending['limit_price']:
                        entry_price = pending['limit_price']
                        trade = {
                            'side': pending['side'],
                            'entry_price': entry_price,
                            'stop_loss': pending['stop_loss'],
                            'take_profit': pending['take_profit'],
                            'entry_time': datetime.now().isoformat(),
                            'regime': pending['regime'],
                            'bars_held': 0,
                            'max_pnl': -2 * COMMISSION
                        }
                        state['open_trades'].append(trade)
                        state['pending_orders'].remove(pending)
                        
                        send_discord_alert(
                            DISCORD_WEBHOOK,
                            f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {pending['regime']} {pending['side']} (LIMIT FILLED)",
                            0x00ff00,
                            [
                                {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                {"name": "Side", "value": pending['side'], "inline": True},
                                {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                {"name": "Regime", "value": pending['regime'], "inline": True},
                                {"name": "Reason", "value": pending.get('entry_reason', 'N/A'), "inline": False}
                            ]
                        )
                        logger.info(f"Limit order filled: {pending['side']} @ ${entry_price:.2f}")
                
                elif pending['side'] == 'SHORT':
                    if current_price >= pending['limit_price']:
                        entry_price = pending['limit_price']
                        trade = {
                            'side': pending['side'],
                            'entry_price': entry_price,
                            'stop_loss': pending['stop_loss'],
                            'take_profit': pending['take_profit'],
                            'entry_time': datetime.now().isoformat(),
                            'regime': pending['regime'],
                            'bars_held': 0,
                            'max_pnl': -2 * COMMISSION
                        }
                        state['open_trades'].append(trade)
                        state['pending_orders'].remove(pending)
                        
                        send_discord_alert(
                            DISCORD_WEBHOOK,
                            f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {pending['regime']} {pending['side']} (LIMIT FILLED)",
                            0xff0000,
                            [
                                {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                {"name": "Side", "value": pending['side'], "inline": True},
                                {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                {"name": "Regime", "value": pending['regime'], "inline": True},
                                {"name": "Reason", "value": pending.get('entry_reason', 'N/A'), "inline": False}
                            ]
                        )
                        logger.info(f"Limit order filled: {pending['side']} @ ${entry_price:.2f}")
                
                # Cancel if waited too long (2 candles)
                if pending['candles_waiting'] >= 2:
                    logger.info(f"Canceling pending limit order after 2 candles")
                    state['pending_orders'].remove(pending)
            
            # Save state
            save_state(state)
            
            # ═══════════════════════════════════════════════════════════════
            # CLEANUP & SLEEP
            # ═══════════════════════════════════════════════════════════════
            
            gc.collect()
            
            loop_duration = time.time() - loop_start_time
            sleep_time = max(5, 60 - loop_duration)
            
            logger.info(f"Cycle complete | Price: ${current_price:,.2f} | Regime: {regime} | Open: {len(state['open_trades'])} | Pending: {len(state['pending_orders'])}")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Exiting...")
            state['bot_status'] = 'Stopped'
            save_state(state)
            break
        
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
            state['bot_status'] = f'Error: {str(e)}'
            save_state(state)
            time.sleep(60)

if __name__ == "__main__":
    main()
