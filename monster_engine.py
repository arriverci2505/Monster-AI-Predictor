"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER ENGINE v15.0 - HEADLESS BACKEND WORKER                          ║
║  Cloud-Optimized Trading Engine with Smart Exit & Discord Alerts         ║
║  🔧 FIXED: Current Price Tracking + Absolute Path Support                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

# ⚙️ LIVE_CONFIG - 27 MANDATORY PARAMETERS
LIVE_CONFIG = {
    # --- 1. GENERAL ---
    'exchange': 'kraken',
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'sequence_length': 60,

    # --- 2. AI THRESHOLDS ---
    'temperature': 1.2,
    'entry_percentile': 25,
    'trending_buy_threshold': 0.40,
    'trending_sell_threshold': 0.42,
    'sideway_buy_threshold': 0.22,
    'sideway_sell_threshold': 0.22,

    # --- 3. REGIME CLASSIFICATION ---
    'trending_adx_min': 30,             # BTC Golden Ratio
    'sideway_adx_max': 30,              # BTC Golden Ratio
    'choppiness_threshold_high': 58.0,
    'choppiness_extreme_low': 30,

    # --- 4. SIDEWAY FILTERS ---
    'deviation_zscore_threshold': 1.4,       # Sensitive Entry
    'mean_reversion_min_shadow_atr': 0.1,    # Low shadow requirement
    'bb_squeeze_percentile': 0.35,

    # --- 5. TRENDING PARAMETERS ---
    'sl_std_multiplier': 1.5,
    'max_holding_bars': 200,

    # --- 6. SIDEWAY EXIT PARAMETERS ---
    'mean_reversion_sl_pct': 1.0,
    'mean_reversion_tp_pct': 3.5,            # High reward target
    'time_barrier': 20,
    'min_profit_for_target': 0.009,
    'limit_order_offset': 0.001,             # 0.1% better price

    # --- 7. RISK MANAGEMENT (SMART EXIT) ---
    'use_advanced_exit': True,
    'use_profit_lock': True,
    'ai_exit_threshold': 0.70,
    'profit_lock_levels': [(1.8, 1.2), (3.5, 2.8), (5.5, 4.5)], # Tier 1, 2, 3
    'trailing_stop_activation': 1.5,
    'trailing_stop_distance': 0.6,

    # --- 8. EXECUTION ---
    'position_size': 0.15,
    'slippage': 0.0005,
    'commission': 0.00075,

}

# Discord Webhook (set via environment variable or edit here)
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1472776784205447360/NQaLrcBstxG1vLpwIcHREhPRlFphGFSKl2lUreNMZxHdX4zVk-81F7ACogFUA6fepMMH"

# 🔧 FIX: Use absolute path for STATE_FILE to avoid file loss on cloud
STATE_FILE = os.path.abspath("bot_state.json")
logger.info(f"📁 State file location: {STATE_FILE}")
        
# ════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

class HybridTransformerLSTM(nn.Module):
    """
    Hybrid architecture combining Transformer and LSTM for time series prediction
    """
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, num_transformer_layers, num_heads, num_classes):
        super(HybridTransformerLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for Transformer
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
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
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def prepare_features(df, sequence_length):
    """
    Prepare feature sequences for model input
    """
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
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features for model
    feature_columns = [
        'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
        'rsi', 'atr', 'adx', 'plus_di', 'minus_di',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'volume_ratio', 'momentum_5', 'momentum_10', 'choppiness'
    ]
    
    features = df[feature_columns].values
    
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
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def load_state():
    """🔧 FIX: Added 'current_price': 0.0 to default state"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Default state with current_price initialized
    return {
        'balance': 10000.0,
        'current_price': 0.0,  # 🔧 FIX: Added default current_price
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
    """Send enhanced Discord notification with embedded fields"""
    if not webhook_url or webhook_url == "YOUR_WEBHOOK_URL_HERE":
        return
    
    embed = {
        "title": title,
        "color": color,
        "fields": fields,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "Monster Engine v15.0"}
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
    if side == 'BUY':
        return current_price * (1 + SLIPPAGE)
    else:
        return current_price * (1 - SLIPPAGE)

def calculate_exit_price(current_price, side):
    """Calculate exit price with slippage"""
    if side == 'BUY':
        return current_price * (1 - SLIPPAGE)
    else:
        return current_price * (1 + SLIPPAGE)

def calculate_pnl(trade, exit_price):
    """Calculate PnL percentage for a trade"""
    entry = trade['entry_price']
    if trade['side'] == 'BUY':
        gross_pnl = ((exit_price - entry) / entry) * 100
    else:
        gross_pnl = ((entry - exit_price) / entry) * 100
    
    # Subtract commission
    net_pnl = gross_pnl - (COMMISSION * 100 * 2)  # Entry + Exit
    return net_pnl

# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 80)
    logger.info("MONSTER ENGINE v15.0 - STARTING")
    logger.info("=" * 80)
    
    # Initialize exchange
    exchange = getattr(ccxt, LIVE_CONFIG['exchange'])()
    
    # Load model
    logger.info("Loading AI model...")
    model = HybridTransformerLSTM(
        input_dim=19,
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
            
            # 🔧 FIX: Save current_price to state immediately after fetching
            state['current_price'] = float(current_price)
            
            # Calculate indicators
            df['sma_200'] = df['close'].rolling(window=200).mean()
            df['atr'] = calculate_atr(df, 14)
            df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, 14)
            df['choppiness'] = calculate_choppiness(df, 14)
            
            sma200 = df['sma_200'].iloc[-1]
            atr = df['atr'].iloc[-1]
            adx = df['adx'].iloc[-1]
            chop = df['choppiness'].iloc[-1]
            
            # Regime detection
            is_trending = (adx >= LIVE_CONFIG['trending_adx_min']) and \
                         (chop < LIVE_CONFIG['choppiness_threshold_high'])
            regime = "TRENDING" if is_trending else "SIDEWAYS"
            
            logger.info(f"Market State: {regime} | ADX: {adx:.1f} | CHOP: {chop:.1f}")
            
            # ═══════════════════════════════════════════════════════════════
            # POSITION MANAGEMENT
            # ═══════════════════════════════════════════════════════════════
            
            for trade in state['open_trades'][:]:
                trade['bars_held'] += 1
                
                if trade['side'] == 'BUY':
                    pnl_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
                else:
                    pnl_pct = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # Stop Loss
                if trade['side'] == 'BUY' and current_price <= trade['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif trade['side'] == 'SELL' and current_price >= trade['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop Loss"
                
                # Take Profit
                elif trade['side'] == 'BUY' and current_price >= trade['take_profit']:
                    should_exit = True
                    exit_reason = "Take Profit"
                elif trade['side'] == 'SELL' and current_price <= trade['take_profit']:
                    should_exit = True
                    exit_reason = "Take Profit"
                
                # Time barrier for sideways
                elif trade['regime'] == 'SIDEWAYS' and trade['bars_held'] >= LIVE_CONFIG['time_barrier']:
                    should_exit = True
                    exit_reason = "Time Barrier"
                
                # Max holding for trending
                elif trade['regime'] == 'TRENDING' and trade['bars_held'] >= LIVE_CONFIG['max_holding_bars']:
                    should_exit = True
                    exit_reason = "Max Holding"
                
                if should_exit:
                    exit_price = calculate_exit_price(current_price, trade['side'])
                    net_pnl = calculate_pnl(trade, exit_price)
                    
                    # Calculate dollar PnL
                    position_value = state['balance'] * LIVE_CONFIG['position_size']
                    dollar_pnl = position_value * (net_pnl / 100)
                    state['balance'] += dollar_pnl
                    
                    # Record trade
                    trade_record = {
                        'side': trade['side'],
                        'entry_price': f"${trade['entry_price']:,.2f}",
                        'exit_price': f"${exit_price:,.2f}",
                        'net_pnl': f"{net_pnl:.2f}%",
                        'dollar_pnl': f"${dollar_pnl:,.2f}",
                        'exit_reason': exit_reason,
                        'bars_held': trade['bars_held'],
                        'exit_time': datetime.now().isoformat()
                    }
                    state['trade_history'].insert(0, trade_record)
                    
                    # Update stats
                    state['total_trades'] += 1
                    wins = len([t for t in state['trade_history'] if float(t['net_pnl'].replace('%', '')) > 0])
                    state['win_rate'] = (wins / len(state['trade_history']) * 100) if state['trade_history'] else 0
                    
                    # Discord notification
                    color = 0x00ff00 if net_pnl > 0 else 0xff0000
                    send_discord_alert(
                        DISCORD_WEBHOOK,
                        f"🎯 EXIT: {LIVE_CONFIG['symbol']} {trade['side']}",
                        color,
                        [
                            {"name": "Exit Reason", "value": exit_reason, "inline": True},
                            {"name": "PnL", "value": f"{net_pnl:.2f}%", "inline": True},
                            {"name": "Dollar PnL", "value": f"${dollar_pnl:,.2f}", "inline": True},
                            {"name": "Entry Price", "value": f"${trade['entry_price']:,.2f}", "inline": True},
                            {"name": "Exit Price", "value": f"${exit_price:,.2f}", "inline": True},
                            {"name": "Bars Held", "value": str(trade['bars_held']), "inline": True},
                            {"name": "New Balance", "value": f"${state['balance']:,.2f}", "inline": False}
                        ]
                    )
                    
                    state['open_trades'].remove(trade)
                    logger.info(f"Trade closed: {exit_reason} | PnL: {net_pnl:.2f}%")
            
            # ═══════════════════════════════════════════════════════════════
            # PENDING LIMIT ORDERS
            # ═══════════════════════════════════════════════════════════════
            
            for pending in state['pending_orders'][:]:
                pending['candles_waiting'] += 1
                
                # Check if limit price reached
                if pending['side'] == 'BUY':
                    if current_price <= pending['limit_price']:
                        # Execute limit order
                        entry_price = pending['limit_price']
                        trade = {
                            'side': pending['side'],
                            'entry_price': entry_price,
                            'stop_loss': pending['stop_loss'],
                            'take_profit': pending['take_profit'],
                            'entry_time': datetime.now().isoformat(),
                            'regime': regime,
                            'bars_held': 0
                        }
                        state['open_trades'].append(trade)
                        state['pending_orders'].remove(pending)
                        
                        # Discord notification
                        color = 0x00ff00
                        send_discord_alert(
                            DISCORD_WEBHOOK,
                            f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {pending['side']}",
                            color,
                            [
                                {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                {"name": "Side", "value": pending['side'], "inline": True},
                                {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                {"name": "Stop Loss", "value": f"${pending['stop_loss']:,.2f}", "inline": True},
                                {"name": "Take Profit", "value": f"${pending['take_profit']:,.2f}", "inline": True},
                                {"name": "Regime", "value": regime, "inline": True},
                                {"name": "Order Type", "value": "LIMIT (Filled)", "inline": False}
                            ]
                        )
                        
                        logger.info(f"Limit order filled: {pending['side']} @ ${entry_price:.2f}")
                
                elif pending['side'] == 'SELL':
                    if current_price >= pending['limit_price']:
                        entry_price = pending['limit_price']
                        trade = {
                            'side': pending['side'],
                            'entry_price': entry_price,
                            'stop_loss': pending['stop_loss'],
                            'take_profit': pending['take_profit'],
                            'entry_time': datetime.now().isoformat(),
                            'regime': regime,
                            'bars_held': 0
                        }
                        state['open_trades'].append(trade)
                        state['pending_orders'].remove(pending)
                        
                        color = 0xff0000
                        send_discord_alert(
                            DISCORD_WEBHOOK,
                            f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {pending['side']}",
                            color,
                            [
                                {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                {"name": "Side", "value": pending['side'], "inline": True},
                                {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                {"name": "Stop Loss", "value": f"${pending['stop_loss']:,.2f}", "inline": True},
                                {"name": "Take Profit", "value": f"${pending['take_profit']:,.2f}", "inline": True},
                                {"name": "Regime", "value": regime, "inline": True},
                                {"name": "Order Type", "value": "LIMIT (Filled)", "inline": False}
                            ]
                        )
                        
                        logger.info(f"Limit order filled: {pending['side']} @ ${entry_price:.2f}")
                
                # Cancel if waited too long (2 candles)
                if pending['candles_waiting'] >= 2:
                    logger.info(f"Canceling pending limit order after 2 candles")
                    state['pending_orders'].remove(pending)
            
            # ═══════════════════════════════════════════════════════════════
            # SIGNAL GENERATION
            # ═══════════════════════════════════════════════════════════════
            
            if not state['open_trades'] and not state['pending_orders']:
                try:
                    sequences = prepare_features(df, LIVE_CONFIG['sequence_length'])
                    if len(sequences) > 0:
                        last_sequence = sequences[-1]
                        
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
                            output = model(input_tensor)
                            
                            output_scaled = output / LIVE_CONFIG['temperature']
                            probabilities = torch.softmax(output_scaled, dim=1).squeeze().numpy()
                        
                        p_buy = probabilities[0]
                        p_neutral = probabilities[1]
                        p_sell = probabilities[2]
                        
                        # Determine signal based on regime
                        if is_trending:
                            buy_threshold = LIVE_CONFIG['trending_buy_threshold']
                            sell_threshold = LIVE_CONFIG['trending_sell_threshold']
                        else:
                            buy_threshold = LIVE_CONFIG['sideway_buy_threshold']
                            sell_threshold = LIVE_CONFIG['sideway_sell_threshold']
                        
                        signal = None
                        if p_buy > buy_threshold and p_buy > p_sell:
                            signal = 'BUY'
                        elif p_sell > sell_threshold and p_sell > p_buy:
                            signal = 'SELL'
                        
                        # Additional filters
                        if signal:
                            if signal == 'BUY' and current_price < sma200:
                                logger.info(f"BUY signal rejected: Price below SMA200")
                                signal = None
                            elif signal == 'SELL' and current_price > sma200:
                                logger.info(f"SELL signal rejected: Price above SMA200")
                                signal = None
                        
                        # Execute signal
                        if signal:
                            logger.info(f"Signal detected: {signal} | Regime: {regime} | ADX: {adx:.1f}")
                            
                            if is_trending:
                                # MARKET ORDER
                                entry_price = calculate_actual_entry_price(current_price, signal)
                                
                                sl_distance = atr * LIVE_CONFIG['sl_std_multiplier']
                                if signal == 'BUY':
                                    stop_loss = entry_price - sl_distance
                                    take_profit = entry_price + (sl_distance * 3)
                                else:
                                    stop_loss = entry_price + sl_distance
                                    take_profit = entry_price - (sl_distance * 3)
                                
                                trade = {
                                    'side': signal,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'entry_time': datetime.now().isoformat(),
                                    'regime': regime,
                                    'bars_held': 0
                                }
                                state['open_trades'].append(trade)
                                
                                color = 0x00ff00 if signal == 'BUY' else 0xff0000
                                send_discord_alert(
                                    DISCORD_WEBHOOK,
                                    f"🚀 ENTRY: {LIVE_CONFIG['symbol']} {signal}",
                                    color,
                                    [
                                        {"name": "Symbol", "value": LIVE_CONFIG['symbol'], "inline": True},
                                        {"name": "Side", "value": signal, "inline": True},
                                        {"name": "Entry Price", "value": f"${entry_price:,.2f}", "inline": True},
                                        {"name": "Stop Loss", "value": f"${stop_loss:,.2f}", "inline": True},
                                        {"name": "Take Profit", "value": f"${take_profit:,.2f}", "inline": True},
                                        {"name": "Regime", "value": regime, "inline": True},
                                        {"name": "Order Type", "value": "MARKET", "inline": False}
                                    ]
                                )
                                
                                logger.info(f"Market order executed: {signal} @ ${entry_price:.2f}")
                                
                            else:
                                # LIMIT ORDER
                                limit_offset = LIVE_CONFIG['limit_order_offset']
                                if signal == 'BUY':
                                    limit_price = current_price * (1 - limit_offset)
                                else:
                                    limit_price = current_price * (1 + limit_offset)
                                
                                if signal == 'BUY':
                                    stop_loss = limit_price * (1 - LIVE_CONFIG['mean_reversion_sl_pct'] / 100)
                                    take_profit = limit_price * (1 + LIVE_CONFIG['mean_reversion_tp_pct'] / 100)
                                else:
                                    stop_loss = limit_price * (1 + LIVE_CONFIG['mean_reversion_sl_pct'] / 100)
                                    take_profit = limit_price * (1 - LIVE_CONFIG['mean_reversion_tp_pct'] / 100)
                                
                                pending_order = {
                                    'side': signal,
                                    'limit_price': limit_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'candles_waiting': 0
                                }
                                state['pending_orders'].append(pending_order)
                                
                                logger.info(f"Limit order placed: {signal} @ ${limit_price:.2f} (Market: ${current_price:.2f})")
                
                except Exception as e:
                    logger.error(f"Error in signal generation: {e}")
            
            # 🔧 FIX: Save state with updated current_price
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
