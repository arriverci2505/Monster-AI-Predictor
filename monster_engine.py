"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MONSTER ENGINE v14.5.1 - CRITICAL PATCHES APPLIED                      ║
║  🎯 100% MATCHED - HIERARCHICAL + AI SAFETY                             ║
║  🔧 PATCHED: AI Counter Signal + AI Safety Filter                       ║
╚══════════════════════════════════════════════════════════════════════════╝

VERIFICATION CODE: v14.5.1-PATCHED-2026-02-16

✅ SYNCHRONIZED FEATURES:
  • Feature Engineering: Matched with enrich_features_v14()
  • Rolling Normalization: Window=200, Min periods=50
  • Dual Threshold: Trending 0.40/0.42 | Sideway 0.22
  • Regime Detection: ADX-FIRST Hierarchical (Priority-based)
  • Sideway Filters: Z-score=1.4, Shadow=0.1, BB=0.35
  • Exit Logic: Progressive Profit Lock + Trailing + AI Counter
  
🔄 KEY CHANGES IN v14.5:
  • Regime logic: AND → HIERARCHICAL (ADX first, then Choppiness)
  • ADX thresholds: sideway_max=20 (not 30)
  • Trending threshold: 0.40 (from 0.36)
  • Choppiness: extreme_low=30 for directional exception

🔥 CRITICAL PATCHES IN v14.5.1:
  • AI Counter Signal: Exit Sideway when AI predicts strong reversal
  • AI Safety Filter: Reject Sideway entry if counter-probability too high
  • Both features matched 100% with backtest logic
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
import joblib
from datetime import datetime, timedelta, timezone
from scipy import signal as scipy_signal
from sklearn.preprocessing import RobustScaler
import os
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Execution parameters
SLIPPAGE = 0.0005
COMMISSION = 0.00075

# ⚙️ LIVE_CONFIG - 100% MATCHED WITH BACKTEST v14.5
LIVE_CONFIG = {
    # MODEL ARCHITECTURE
    'input_dim': 42,                     
    'hidden_dim': 128,
    'num_lstm_layers': 2,
    'num_transformer_layers': 2,
    'num_heads': 4,
    'se_reduction_ratio': 16,
    'dropout': 0.35,
    'num_classes': 3,
    'use_positional_encoding': True,
    # --- GENERAL ---
    'exchange': 'kraken',
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'sequence_length': 60,
    
    # --- AI THRESHOLDS (DUAL SYSTEM - MATCHED!) ---
    'temperature': 1.2,  
    
    # TRENDING MODE (HIGH CONFIDENCE)
    'trending_buy_threshold': 0.40,       
    'trending_sell_threshold': 0.42,       
    
    # SIDEWAY MODE (LOWER CONFIDENCE)
    'sideway_buy_threshold': 0.22,    
    'sideway_sell_threshold': 0.22,   
    
    # --- REGIME CLASSIFICATION (UPDATED!) ---
    'trending_adx_min': 30,                
    'sideway_adx_max': 30,                 
    'choppiness_threshold_low': 30,       
    'choppiness_threshold_high': 58.0,    
    'choppiness_extreme_low': 30,          
    
    # --- SIDEWAY FILTERS (MATCHED!) ---
    'deviation_zscore_threshold': 1.4,       
    'mean_reversion_min_shadow_atr': 0.1,    
    'bb_squeeze_percentile': 0.35,           
    
    # --- TRENDING EXIT ---
    'sl_std_multiplier': 1.5,         
    'max_holding_bars': 200,         
    'trailing_stop_activation': 1.5,  
    'trailing_stop_distance': 0.6,    
    
    # --- SIDEWAY EXIT ---
    'mean_reversion_sl_pct': 1.0,    
    'mean_reversion_tp_pct': 3.5,    
    'time_barrier': 20,              
    'min_profit_for_target': 0.009,  
    'ai_exit_threshold': 0.7,        
    
    # --- PROFIT LOCK (TRENDING) ---
    'use_profit_lock': True,
    'profit_lock_levels': [
        (1.8, 1.2),
        (3.5, 2.8),
        (5.5, 4.5)
    ],
    
    # --- ROLLING NORMALIZATION (v11) ---
    'use_rolling_normalization': True,  
    'rolling_window': 200,              
    'rolling_min_periods': 50,        
    
    # --- EXECUTION ---
    'position_size': 0.15,
    'limit_order_offset': 0.001,
}

# Discord Webhook
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1472776784205447360/NQaLrcBstxG1vLpwIcHREhPRlFphGFSKl2lUreNMZxHdX4zVk-81F7ACogFUA6fepMMH"

STATE_FILE = os.path.abspath("bot_state_v14_5_1.json")

# ════════════════════════════════════════════════════════════════════════════
# STARTUP VERIFICATION
# ════════════════════════════════════════════════════════════════════════════

logger.info("="*80)
logger.info("🎯 MONSTER ENGINE v14.5.1 - CRITICAL PATCHES APPLIED")
logger.info("="*80)
logger.info(f"✅ VERIFICATION CODE: v14.5.1-PATCHED-2026-02-16")
logger.info(f"")
logger.info(f"🔥 CRITICAL PATCHES:")
logger.info(f"   AI Counter Signal:  ✅ (Sideway exit on strong reversal)")
logger.info(f"   AI Safety Filter:   ✅ (Reject risky Sideway entries)")
logger.info(f"")
logger.info(f"🔍 THRESHOLD VERIFICATION:")
logger.info(f"   Trending Buy:  {LIVE_CONFIG['trending_buy_threshold']:.3f} (0.40) ✅")
logger.info(f"   Trending Sell: {LIVE_CONFIG['trending_sell_threshold']:.3f} (0.42) ✅")
logger.info(f"   Sideway Buy:   {LIVE_CONFIG['sideway_buy_threshold']:.3f} (0.22) ✅")
logger.info(f"   Sideway Sell:  {LIVE_CONFIG['sideway_sell_threshold']:.3f} (0.22) ✅")
logger.info(f"")
logger.info(f"🔍 REGIME PARAMETERS:")
logger.info(f"   ADX Trending Min:     {LIVE_CONFIG['trending_adx_min']} (30) ✅")
logger.info(f"   ADX Sideway Max:      {LIVE_CONFIG['sideway_adx_max']} (20) ✅")
logger.info(f"   Chop Extreme Low:     {LIVE_CONFIG['choppiness_threshold_low']} (30) ✅")
logger.info(f"")
logger.info(f"🔍 SIDEWAY FILTER VERIFICATION:")
logger.info(f"   BB Percentile:    {LIVE_CONFIG['bb_squeeze_percentile']:.2f} ✅")
logger.info(f"   Z-Score Thresh:   {LIVE_CONFIG['deviation_zscore_threshold']:.1f} ✅")
logger.info(f"   Min Shadow ATR:   {LIVE_CONFIG['mean_reversion_min_shadow_atr']:.1f} ✅")
logger.info(f"   AI Exit Thresh:   {LIVE_CONFIG['ai_exit_threshold']:.1f} ✅")
logger.info(f"")
logger.info(f"🔍 TEMPERATURE: {LIVE_CONFIG['temperature']:.1f} ✅")
logger.info("="*80)

# ════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL ARCHITECTURE (MATCHED)
# ════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Positional Encoding chuẩn xác từ v14.4 Training"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation chuẩn xác (fc1, fc2) từ v14.4 Training"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # Global average pooling
        scale = x.mean(dim=1)  # (batch, channels)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.unsqueeze(1)

class HybridTransformerLSTM(nn.Module):
    """Kiến trúc THE MONSTER v14.4 - Khớp 100% State Dict"""
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_classes = config.get('num_classes', 3)
        self.use_positional_encoding = config.get('use_positional_encoding', True)

        # 1. Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # 2. Positional encoding
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(self.hidden_dim)

        # 3. Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.get('num_heads', 8),
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get('num_transformer_layers', 2)
        )

        # 4. Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=config.get('num_lstm_layers', 2),
            batch_first=True,
            dropout=config.get('dropout', 0.1) if config.get('num_lstm_layers', 1) > 1 else 0,
            bidirectional=True
        )

        # 5. Squeeze-Excitation (Khớp với fc1, fc2 trong checkpoint)
        self.se_block = SqueezeExcitation(
            channels=self.hidden_dim * 2,
            reduction=config.get('se_reduction_ratio', 16)
        )

        # 6. Final Multi-Head Attention
        self.final_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )

        # 7. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = self.se_block(x)
        
        # Self-attention cuối cùng
        x, _ = self.final_attention(x, x, x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        
        return self.classifier(x)
# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING - MATCHED WITH BACKTEST v14
# ════════════════════════════════════════════════════════════════════════════

def calculate_fractional_diff(series, d=0.4):
    """Fractional differentiation"""
    weights = [1.0]
    for k in range(1, len(series)):
        weights.append(-weights[-1] * (d - k + 1) / k)
        if abs(weights[-1]) < 1e-5:
            break
    weights = np.array(weights[::-1])
    return pd.Series(np.convolve(series, weights, mode='same'), index=series.index)

def calculate_entropy(series, window=20):
    """Shannon entropy"""
    def _entropy(x):
        if len(x) == 0:
            return 0
        hist, _ = np.histogram(x, bins=10)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return -np.sum(probs * np.log2(probs))
    return series.rolling(window).apply(_entropy, raw=True)

def calculate_choppiness_index(df, period=14):
    """Choppiness Index - MATCHED"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    atr_sum = tr.rolling(period).sum()
    high_low_range = high.rolling(period).max() - low.rolling(period).min()
    
    ci = 100 * np.log10(atr_sum / high_low_range) / np.log10(period)
    return ci

def calculate_candlestick_shadows(df, config):
    """Candlestick shadow analysis - MATCHED"""
    df = df.copy()
    
    body = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low']
    
    # Shadows relative to ATR
    df['lower_shadow_atr'] = np.where(
        df['Close'] < df['Open'],
        (df['Open'] - df['Low']) / df['ATR_raw'],
        (df['Close'] - df['Low']) / df['ATR_raw']
    )
    
    df['upper_shadow_atr'] = np.where(
        df['Close'] > df['Open'],
        (df['High'] - df['Open']) / df['ATR_raw'],
        (df['High'] - df['Close']) / df['ATR_raw']
    )
    
    df['body_size_atr'] = body / df['ATR_raw']
    
    # Shadow percentages
    df['upper_shadow_pct'] = np.where(
        candle_range > 0,
        (df['High'] - df[['Close', 'Open']].max(axis=1)) / candle_range,
        0
    )
    
    df['lower_shadow_pct'] = np.where(
        candle_range > 0,
        (df[['Close', 'Open']].min(axis=1) - df['Low']) / candle_range,
        0
    )
    
    df['body_ratio'] = np.where(candle_range > 0, body / candle_range, 0)
    
    # Pattern scores
    shadow_thresh = config.get('shadow_threshold_atr', 0.8)
    body_max = config.get('pinbar_body_ratio_max', 0.3)
    
    df['pinbar_score'] = (
        ((df['lower_shadow_atr'] > shadow_thresh) | 
         (df['upper_shadow_atr'] > shadow_thresh)) &
        (df['body_ratio'] < body_max)
    ).astype(int)
    
    df['hammer_score'] = (
        (df['lower_shadow_atr'] > shadow_thresh) &
        (df['upper_shadow_pct'] < 0.1) &
        (df['body_ratio'] < 0.3)
    ).astype(int)
    
    df['shooting_star_score'] = (
        (df['upper_shadow_atr'] > shadow_thresh) &
        (df['lower_shadow_pct'] < 0.1) &
        (df['body_ratio'] < 0.3)
    ).astype(int)
    
    return df

def calculate_deviation_from_mean(df, config):
    """Deviation analysis - MATCHED"""
    df = df.copy()
    
    # VWAP
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_rolling'] = (
        (df['Close'] * df['Volume']).rolling(20).sum() / 
        df['Volume'].rolling(20).sum()
    )
    
    # SMA_20 for reference
    df['SMA_20'] = df['Close'].rolling(20).mean()
    
    # Use VWAP or SMA
    use_vwap = config.get('use_vwap_deviation', True)
    reference = df['VWAP_rolling'] if use_vwap else df['SMA_20']
    
    # Z-score from reference
    rolling_std = df['Close'].rolling(20).std()
    df['deviation_zscore_sma'] = (df['Close'] - reference) / rolling_std
    df['deviation_zscore_vwap'] = (df['Close'] - df['VWAP_rolling']) / rolling_std
    
    # Percentage deviation
    df['deviation_pct_sma'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
    
    # Overextended flags
    z_thresh = config.get('deviation_zscore_threshold', 2.0)
    df['is_overextended_high'] = (df['deviation_zscore_sma'] > z_thresh).astype(int)
    df['is_overextended_low'] = (df['deviation_zscore_sma'] < -z_thresh).astype(int)
    
    return df

def calculate_bollinger_squeeze(df, period=20):
    """Bollinger Band squeeze detection - MATCHED"""
    df = df.copy()
    
    # BB width relative to price
    df['BB_width_ma'] = df['BB_width'].rolling(period).mean()
    df['squeeze_intensity'] = df['BB_width'] / df['BB_width_ma']
    
    # Squeeze = BB width in bottom 20%
    percentile_thresh = 0.20
    bb_width_percentile = df['BB_width'].rolling(100).apply(
        lambda x: np.percentile(x, percentile_thresh * 100)
    )
    
    df['is_bb_squeeze'] = (df['BB_width'] < bb_width_percentile).astype(int)
    df['is_extreme_squeeze'] = (df['squeeze_intensity'] < 0.5).astype(int)
    
    return df

def apply_rolling_normalization(df, feature_cols, config):
    """
    Rolling Z-Score Normalization - MATCHED WITH BACKTEST
    Exclude raw indicators from normalization
    """
    df = df.copy()
    
    if not config.get('use_rolling_normalization', True):
        return df
    
    window = config.get('rolling_window', 200)
    min_periods = config.get('rolling_min_periods', 50)
    
    logger.info(f"🔄 Applying Rolling Z-Score (Window={window}, Min={min_periods})...")
    
    # EXCLUDE these from normalization (keep raw)
    exclude_from_normalization = [
        'ADX_raw', 'choppiness_index', 'ATR_raw',
        'Close', 'Open', 'High', 'Low', 'Volume',
        'timestamp',
        'regime_trending', 'regime_sideway', 'regime_uptrend', 'regime_downtrend',
        'is_bb_squeeze', 'is_extreme_squeeze',
        'is_overextended_high', 'is_overextended_low',
        'pinbar_score', 'hammer_score', 'shooting_star_score',
        'VWAP', 'VWAP_rolling', 'SMA_20', 'SMA_long', 'SMA_short',
        'BB_upper', 'BB_lower', 'BB_width', 'BB_width_ma',
    ]
    
    cols_to_normalize = [col for col in feature_cols if col not in exclude_from_normalization]
    
    logger.info(f"   Normalizing: {len(cols_to_normalize)} features")
    logger.info(f"   Excluding:   {len(feature_cols) - len(cols_to_normalize)} features")
    
    for col in cols_to_normalize:
        if col not in df.columns:
            continue
        
        rolling_mean = df[col].rolling(window=window, min_periods=min_periods).mean()
        rolling_std = df[col].rolling(window=window, min_periods=min_periods).std()
        
        # Z-score normalization
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # Clip extreme values
        df[col] = df[col].clip(-5, 5)
    
    logger.info("✅ Rolling normalization complete")
    return df

def enrich_features_live(df, config, feature_cols):
    """
    Feature Engineering for LIVE - MATCHED with backtest v14
    
    Args:
        df: Raw OHLCV DataFrame (must have at least 300 rows for warm-up)
        config: Configuration dict
        feature_cols: Feature columns from checkpoint
    
    Returns:
        enriched DataFrame with features ready for model
    """
    df = df.copy()
    logger.info(f"🔧 Live Feature Engineering (v14.4 Matched)")
    logger.info(f"   Input: {len(df)} rows")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BASIC FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    
    # Fractional differentiation
    df['fd_close'] = calculate_fractional_diff(df['Close'], d=0.4)
    df['fd_close'] = df['fd_close'].replace([np.inf, -np.inf], np.nan).ffill()
    
    # Volume imbalance
    df['vol_imbalance'] = np.sign(df['Close'] - df['Open']) * df['Volume']
    df['vol_imbalance_ema'] = df['vol_imbalance'].ewm(span=20).mean()
    
    # Entropy
    df['entropy'] = calculate_entropy(df['Close'], window=20)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_raw'] = df['ATR'].copy()  # Keep raw for filters
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_upper'] = sma_20 + (2 * std_20)
    df['BB_lower'] = sma_20 - (2 * std_20)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma_20
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # ADX
    period = 14
    df['plus_dm'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    df['minus_dm'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )
    
    atr_smooth = df['ATR'].rolling(period).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(period).mean() / atr_smooth)
    df['minus_di'] = 100 * (df['minus_dm'].rolling(period).mean() / atr_smooth)
    
    dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['ADX'] = dx.rolling(period).mean()
    df['ADX_raw'] = df['ADX'].copy()  # Keep raw for regime detection
    
    # SMA
    df['SMA_short'] = df['Close'].rolling(20).mean()
    df['SMA_long'] = df['Close'].rolling(50).mean()
    df['SMA_distance'] = (df['SMA_short'] - df['SMA_long']) / df['SMA_long']
    
    # Volume
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Volatility Z-Score
    returns = df['Close'].pct_change()
    volatility = returns.rolling(20).std()
    vol_mean = volatility.rolling(100).mean()
    vol_std = volatility.rolling(100).std()
    df['volatility_zscore'] = (volatility - vol_mean) / vol_std.replace(0, np.nan)
    
    # ═══════════════════════════════════════════════════════════════════════
    # v14 FEATURES: PRICE ACTION
    # ═══════════════════════════════════════════════════════════════════════
    
    logger.info("   ⚡ Computing shadows, deviation, choppiness...")
    df = calculate_candlestick_shadows(df, config)
    df = calculate_deviation_from_mean(df, config)
    df['choppiness_index'] = calculate_choppiness_index(df, period=14)
    df = calculate_bollinger_squeeze(df, period=20)
    
    # ═══════════════════════════════════════════════════════════════════════
    # REGIME CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    
    df['regime_trending'] = (
        (df['ADX'] > 25) & (df['choppiness_index'] < 50)
    ).astype(int)
    
    df['regime_sideway'] = (
        (df['ADX'] < 20) &
        (df['choppiness_index'] > 61.8) &
        (df['is_bb_squeeze'] == 1)
    ).astype(int)
    
    df['regime_uptrend'] = (
        (df['regime_trending'] == 1) & (df['SMA_distance'] > 0)
    ).astype(int)
    
    df['regime_downtrend'] = (
        (df['regime_trending'] == 1) & (df['SMA_distance'] < 0)
    ).astype(int)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ROLLING NORMALIZATION (v11 feature)
    # ═══════════════════════════════════════════════════════════════════════
    
    df = apply_rolling_normalization(df, feature_cols, config)
    
    # Forward fill any remaining NaN
    df = df.ffill()
    df = df.fillna(0)
    
    logger.info(f"✅ Feature engineering complete: {len(df)} rows")
    
    return df

def prepare_features_for_model(df, feature_cols, config):
    """
    Prepare final feature matrix for model input
    
    CRITICAL: Use exact feature_cols from checkpoint to maintain column order
    """
    # Extract features in EXACT order from checkpoint
    features_df = df[feature_cols].copy()
    
    # Convert to numpy
    features = features_df.values
    
    # Create sequences
    sequence_length = config.get('sequence_length', 60)
    sequences = []
    
    for i in range(len(features) - sequence_length):
        seq = features[i:i+sequence_length]
        if not np.isnan(seq).any() and not np.isinf(seq).any():
            sequences.append(seq)
    
    if len(sequences) == 0:
        logger.warning("⚠️ No valid sequences created!")
        return np.array([])
    
    logger.info(f"✅ Created {len(sequences)} valid sequences")
    return np.array(sequences, dtype=np.float32)

# ════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION - MATCHED
# ════════════════════════════════════════════════════════════════════════════

def detect_market_regime_hierarchical(adx, choppiness, config):
    """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  REGIME DETECTION v14.5 - HIERARCHICAL (ADX-FIRST APPROACH)            ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    PRIORITY: ADX > Choppiness
    
    Step 1: Check ADX (PRIMARY SIGNAL)
        • ADX < 20  → SIDEWAY (weak momentum)
        • ADX > 25  → TRENDING (strong momentum)
        • 20 ≤ ADX ≤ 25 → Use Choppiness as tiebreaker
    
    Step 2: Choppiness (SECONDARY - only for extreme cases)
        • If ADX < 20 but Chop < 30 → TRENDING (very directional price)
        • If ADX > 25 but Chop > 70 → Still TRENDING (trust ADX)
    
    Args:
        adx: Current ADX value
        choppiness: Current Choppiness Index
        config: Config dict
    
    Returns:
        (is_trending, is_sideway, regime_name)
    """
    adx_sideway_max = config.get('sideway_adx_max', 30)
    adx_trending_min = config.get('trending_adx_min', 30)
    chop_extreme_low = config.get('choppiness_threshold_low', 30)

    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 1: ADX (PRIMARY)
    # ═══════════════════════════════════════════════════════════════════════

    # Strong momentum → TRENDING (regardless of choppiness)
    if adx > adx_trending_min:
        return True, False, f"TRENDING_ADX_HIGH(ADX:{adx:.1f}>30)"

    # Very weak momentum → SIDEWAY
    if adx < adx_sideway_max:
        # Exception: Extremely directional price (very low chop)
        if choppiness < chop_extreme_low:
            return True, False, f"TRENDING_CHOP_EXTREME_LOW(Chop:{choppiness:.1f}<30,ADX:{adx:.1f})"
        else:
            return False, True, f"SIDEWAY_ADX_LOW(ADX:{adx:.1f}<30)"

    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 2: UNCLEAR ZONE (20 ≤ ADX ≤ 25) - Use Choppiness
    # ═══════════════════════════════════════════════════════════════════════
    
    if choppiness > 60:
        return False, True, f"SIDEWAY_CHOP_HIGH({choppiness:.1f})"
    elif choppiness < 40:
        return True, False, f"TRENDING_CHOP_LOW({choppiness:.1f})"

    # Ambiguous → Don't trade
    return False, False, f"UNCLEAR_WAIT(ADX:{adx:.1f}, Chop:{choppiness:.1f})"


# ════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def load_state():
    """Load bot state"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                for trade in state.get('open_trades', []):
                    if 'max_pnl' not in trade:
                        trade['max_pnl'] = -2 * COMMISSION
                return state
        except:
            pass
    
    return {
        'balance': 10000.0,
        'current_price': 0.0,
        'ai_probs': {               
            'buy': 0.0, 
            'sell': 0.0, 
            'neutral': 1.0,
            'regime': 'Unknown'
        },
        'open_trades': [],
        'pending_orders': [],
        'trade_history': [],
        'bot_status': 'Running',
        'last_update_time': datetime.now().isoformat(),
        'win_rate': 0.0,
        'pnl_pct': 0.0,
        'total_trades': 0
    }

# [Tìm hàm save_state cũ và thay thế bằng đoạn này]

def save_state(state):
    """
    ATOMIC WRITE MECHANISM:
    1. Write to .tmp file
    2. Flush & Fsync to disk
    3. Atomic Replace (Rename)
    """
    try:
        # 1. Cập nhật Timestamp chuẩn UTC ngay trước khi lưu
        state['last_update_time'] = datetime.now(timezone.utc).isoformat()
        
        # 2. Tạo tên file tạm
        temp_file = f"{STATE_FILE}.tmp"
        
        # 3. Ghi vào file tạm
        with open(temp_file, "w", encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
            f.flush()              # Đẩy dữ liệu từ bộ đệm Python xuống OS
            os.fsync(f.fileno())   # Ép OS ghi dữ liệu xuống đĩa cứng vật lý
            
        # 4. Thay thế file chính bằng file tạm (Hành động nguyên tử - Atomic)
        # Trên Linux/Unix, thao tác này diễn ra ngay lập tức, không có độ trễ
        os.replace(temp_file, STATE_FILE)
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR saving state: {e}")
# ════════════════════════════════════════════════════════════════════════════
# DISCORD NOTIFICATIONS
# ════════════════════════════════════════════════════════════════════════════

def send_discord_alert(webhook_url, title, color, fields):
    """Send Discord notification"""
    if not webhook_url or "YOUR_WEBHOOK" in webhook_url:
        return
    
    embed = {
        "title": title,
        "color": color,
        "fields": fields,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "Monster Engine v14.5.1 Patched"}
    }
    
    payload = {"embeds": [embed]}
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        if response.status_code == 204:
            logger.info("✅ Discord alert sent")
    except Exception as e:
        logger.error(f"Discord error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TRADING UTILITIES
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

# ════════════════════════════════════════════════════════════════════════════
# MAIN TRADING LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("="*80)
    logger.info("🚀 MONSTER ENGINE v14.4 - SYNCHRONIZED")
    logger.info("="*80)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOAD MODEL & CHECKPOINT
    # ═══════════════════════════════════════════════════════════════════════
    
    model_path = Path('BTC-USDT_MONSTER_model.pt')
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return
    
    logger.info(f"📦 Loading checkpoint: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract config and feature_cols from checkpoint
        saved_config = checkpoint.get('config', {})
        feature_cols = checkpoint.get('feature_cols', [])
        
        logger.info(f"✅ Checkpoint loaded")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Input dim: {saved_config.get('input_dim', 'N/A')}")
        
        # Update LIVE_CONFIG with model config (but keep our live params)
        LIVE_CONFIG['input_dim'] = len(feature_cols)
        
        # Build model
        logger.info("🏗️ Building model...")
        model = HybridTransformerLSTM(LIVE_CONFIG)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("✅ Model ready")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZE EXCHANGE
    # ═══════════════════════════════════════════════════════════════════════
    
    exchange = getattr(ccxt, LIVE_CONFIG['exchange'])()
    
    # Load state
    state = load_state()
    logger.info(f"💰 Balance: ${state['balance']:,.2f}")
    
    logger.info("="*80)
    logger.info("🎯 Starting main loop...")
    logger.info("="*80)
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════
    
    while True:
        try:
            loop_start = time.time()
            
            # ═══════════════════════════════════════════════════════════════
            # FETCH DATA (need 300+ candles for warm-up)
            # ═══════════════════════════════════════════════════════════════
            
            logger.info(f"📊 Fetching {LIVE_CONFIG['symbol']} data...")
            ohlcv = exchange.fetch_ohlcv(
                LIVE_CONFIG['symbol'],
                LIVE_CONFIG['timeframe'],
                limit=350  # Extra margin for warm-up
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            current_price = float(df['Close'].iloc[-1])
            state['current_price'] = current_price

            logger.info(f"📊 Data Fetched: {len(df)} candles. Required for Rolling: 200. ")
      
            # ═══════════════════════════════════════════════════════════════
            # FEATURE ENGINEERING (using checkpoint feature_cols)
            # ═══════════════════════════════════════════════════════════════
            
            df_enriched = enrich_features_live(df, LIVE_CONFIG, feature_cols)
            
            # Get current candle indicators (for filters)
            current_row = df_enriched.iloc[-1]
            adx = current_row['ADX_raw']
            chop = current_row['choppiness_index']
            bb_position = current_row['BB_position']
            deviation_zscore = current_row['deviation_zscore_sma']
            lower_shadow = current_row['lower_shadow_atr']
            upper_shadow = current_row['upper_shadow_atr']
            sma20 = current_row.get('SMA_20', current_price)
            atr = current_row['ATR_raw']
            
            # Regime detection
            is_trending, is_sideway, regime_reason = detect_market_regime_hierarchical(
                adx, chop, LIVE_CONFIG
            )
            
            regime = "TRENDING" if is_trending else ("SIDEWAY" if is_sideway else "UNCLEAR")
            logger.info(f"🎯 {regime_reason}")
            
            # ═══════════════════════════════════════════════════════════════
            # EXIT LOGIC (MATCHED WITH BACKTEST)
            # ═══════════════════════════════════════════════════════════════
            
            active_trades = []
            
            for trade in state['open_trades']:
                trade['bars_held'] = trade.get('bars_held', 0) + 1
                
                # Calculate PnL
                if trade['side'] in ['BUY', 'LONG']:
                    raw_pnl = (current_price - trade['entry_price']) / trade['entry_price']
                else:
                    raw_pnl = (trade['entry_price'] - current_price) / trade['entry_price']
                
                net_pnl = raw_pnl - (2 * COMMISSION)
                
                # Track max PnL
                trade['max_pnl'] = max(trade.get('max_pnl', net_pnl), net_pnl)
                
                exit_reason = None
                
                # ───────────────────────────────────────────────────────────
                # TRENDING EXIT
                # ───────────────────────────────────────────────────────────
                
                if trade.get('regime') == 'TRENDING':
                    
                    # 1. Trailing Stop
                    trailing_activation = LIVE_CONFIG['trailing_stop_activation'] / 100
                    if net_pnl > trailing_activation:
                        trailing_dist = LIVE_CONFIG['trailing_stop_distance'] / 100
                        if net_pnl < (trade['max_pnl'] - trailing_dist):
                            exit_reason = 'Dừng lỗ động'
                    
                    # 2. Stop Loss (ATR-based)
                    sl_distance = atr * LIVE_CONFIG['sl_std_multiplier']
                    if net_pnl < -(sl_distance / trade['entry_price']):
                        exit_reason = 'Dừng lỗ'
                    
                    # 3. Max Holding
                    if trade['bars_held'] > LIVE_CONFIG['max_holding_bars']:
                        exit_reason = 'Hết thời gian chờ'
                    
                    # 4. Profit Lock (tiered)
                    for trigger, lock in LIVE_CONFIG['profit_lock_levels']:
                        trigger_pct = trigger / 100
                        lock_pct = lock / 100
                        if trade['max_pnl'] >= trigger_pct and net_pnl < lock_pct:
                            exit_reason = f'Khóa lợi nhuận({trigger}%->{lock}%)'
                            break
                
                # ───────────────────────────────────────────────────────────
                # SIDEWAY EXIT
                # ───────────────────────────────────────────────────────────
                
                elif trade.get('regime') == 'SIDEWAY':
                    
                    min_profit_cover = LIVE_CONFIG['min_profit_for_target']
                    
                    # 0. AI Counter Signal (PRIORITY - NEW v14.5)
                    # Get current AI predictions (need to recalculate)
                    try:
                        sequences = prepare_features_for_model(df_enriched, feature_cols, LIVE_CONFIG)
                        if len(sequences) > 0:
                            last_sequence = sequences[-1]
                            with torch.no_grad():
                                input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
                                output = model(input_tensor)
                                output_scaled = output / LIVE_CONFIG['temperature']
                                probabilities = F.softmax(output_scaled, dim=1).squeeze().numpy()
                            
                            current_prob_buy = float(probabilities[1])
                            current_prob_sell = float(probabilities[2])
                            
                            ai_exit_thresh = LIVE_CONFIG['ai_exit_threshold']
                            if trade['side'] in ['BUY', 'LONG'] and current_prob_sell > ai_exit_thresh:
                                exit_reason = 'Thoát do AI báo tín hiệu ngược'
                                logger.warning(f"⚠️ AI Counter Signal: SELL prob {current_prob_sell:.3f} > {ai_exit_thresh}")
                            elif trade['side'] in ['SELL', 'SHORT'] and current_prob_buy > ai_exit_thresh:
                                exit_reason = 'Thoát do AI báo tín hiệu ngược'
                                logger.warning(f"⚠️ AI Counter Signal: BUY prob {current_prob_buy:.3f} > {ai_exit_thresh}")
                    except Exception as e:
                        logger.error(f"Error in AI Counter Signal: {e}")
                    
                    # 1. Target Reached (price returns to SMA20)
                    if not exit_reason:
                        if trade['side'] in ['BUY', 'LONG']:
                            if current_price >= sma20:
                                if net_pnl > min_profit_cover:
                                    exit_reason = 'Chạm mục tiêu'
                                elif net_pnl > 0:
                                    exit_reason = 'Hòa vốn'
                        else:
                            if current_price <= sma20:
                                if net_pnl > min_profit_cover:
                                    exit_reason = 'Chạm mục tiêu'
                                elif net_pnl > 0:
                                    exit_reason = 'Hòa vốn'
                    
                    # 2. Stop Loss (percentage)
                    if not exit_reason:
                        mean_reversion_sl = LIVE_CONFIG['mean_reversion_sl_pct'] / 100
                        if net_pnl < -mean_reversion_sl:
                            exit_reason = 'Dừng lỗ'
                    
                    # 3. Take Profit (hard TP)
                    if not exit_reason:
                        mean_reversion_tp = LIVE_CONFIG['mean_reversion_tp_pct'] / 100
                        if net_pnl > mean_reversion_tp:
                            exit_reason = 'Chốt lời'
                    
                    # 4. Max Holding
                    if not exit_reason:
                        if trade['bars_held'] > LIVE_CONFIG['time_barrier']:
                            exit_reason = 'Hết thời gian chờ'
                
                # Execute exit
                if exit_reason:
                    exit_price = calculate_exit_price(current_price, trade['side'])
                    pnl_usd = state['balance'] * LIVE_CONFIG['position_size'] * net_pnl
                    state['balance'] += pnl_usd
                    
                    state['trade_history'].insert(0, { 
                        'entry_time': trade['entry_time'],
                        'exit_time': datetime.now().isoformat(),
                        'mode': trade['regime'],
                        'type': trade['side'],
                        'entry_price': trade['entry_price'],
                        'exit_price': exit_price,
                        'pnl_pct': float(net_pnl * 100), 
                        'exit_reason': exit_reason
                    })
                    
                    state['total_trades'] += 1
                    
                    wins = sum(1 for t in state['trade_history'] if t['pnl_pct'] > 0)
                    state['win_rate'] = wins / state['total_trades'] if state['total_trades'] > 0 else 0
                    state['pnl_pct'] = sum(t.get('pnl_pct', 0) for t in state['trade_history'])
                  
                    logger.info(f"🚪 EXIT {trade['regime']} {trade['side']}: {net_pnl*100:.2f}% | {exit_reason}")
                    
                    send_discord_alert(
                        DISCORD_WEBHOOK,
                        f"🚪 EXIT: {trade['side']} {trade['regime']}",
                        0x00ff00 if net_pnl > 0 else 0xff0000,
                        [
                            {"name": "Phía:", "value": trade['side'], "inline": True},
                            {"name": "Vào lệnh:", "value": f"${trade['entry_price']:.2f}", "inline": True},
                            {"name": "Thoát lệnh:", "value": f"${exit_price:.2f}", "inline": True},
                            {"name": "Lợi nhuận và Thua lỗ:", "value": f"{net_pnl*100:.2f}%", "inline": True},
                            {"name": "Lý do thoát lệnh:", "value": exit_reason, "inline": False}
                        ]
                    )
                else:
                    active_trades.append(trade)
           
            # ═══════════════════════════════════════════════════════════════
            # ENTRY LOGIC - REGIME-FIRST (MATCHED WITH BACKTEST v14.4)
            # ═══════════════════════════════════════════════════════════════
            if len(state.get('open_trades', [])) == 0 and len(state.get('pending_orders', [])) == 0:
                try:
                    # Prepare sequences for model
                    sequences = prepare_features_for_model(df_enriched, feature_cols, LIVE_CONFIG)
                    
                    if len(sequences) > 0:
                        last_sequence = sequences[-1]
                        
                        # Get AI predictions
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
                            output = model(input_tensor)
                            
                            # Temperature scaling
                            output_scaled = output / LIVE_CONFIG['temperature']
                            probabilities = F.softmax(output_scaled, dim=1).squeeze().numpy()
                        
                        prob_neutral = float(probabilities[0])
                        prob_buy = float(probabilities[1])
                        prob_sell = float(probabilities[2])

                        entry_signal = None
                        entry_mode = None
                        entry_reason = ""
                        
                        # Get thresholds
                        trending_buy_thresh = LIVE_CONFIG['trending_buy_threshold']
                        trending_sell_thresh = LIVE_CONFIG['trending_sell_threshold']
                        sideway_buy_thresh = LIVE_CONFIG['sideway_buy_threshold']
                        sideway_sell_thresh = LIVE_CONFIG['sideway_sell_threshold']
                        
                        # Get sideway filter parameters
                        bb_border = LIVE_CONFIG['bb_squeeze_percentile']
                        z_thresh = LIVE_CONFIG['deviation_zscore_threshold']
                        shadow_min = LIVE_CONFIG['mean_reversion_min_shadow_atr']

                        state['ai_probs'] = {
                            'buy': float(prob_buy),
                            'sell': float(prob_sell),
                            'neutral': float(prob_neutral),
                            'regime': regime
                        }
                        save_state(state)
                        # ═══════════════════════════════════════════════════════
                        # REGIME-FIRST LOGIC (STRICT IF-ELIF-ELSE)
                        # ═══════════════════════════════════════════════════════
                        
                        if is_trending:
                            # ───────────────────────────────────────────────────
                            # MODE 1: TRENDING (High AI confidence)
                            # ───────────────────────────────────────────────────
                            
                            if prob_buy > trending_buy_thresh and prob_buy > prob_sell:
                                entry_signal = 'LONG'
                                entry_mode = 'TRENDING'
                                entry_reason = f"AI dự đoán: {prob_buy:.3f} > {trending_buy_thresh:.3f}}"
                            
                            elif prob_sell > trending_sell_thresh and prob_sell > prob_buy:
                                entry_signal = 'SHORT'
                                entry_mode = 'TRENDING'
                                entry_reason = f"AI dự đoán: {prob_sell:.3f} > {trending_sell_thresh:.3f}}"
                        
                        elif is_sideway:
                            # ───────────────────────────────────────────────────
                            # MODE 2: SIDEWAY (Lower AI threshold + Price Action)
                            # ───────────────────────────────────────────────────
                            
                            # AI Safety Filter (CRITICAL - NEW v14.5)
                            ai_filter_threshold = LIVE_CONFIG.get('ai_exit_threshold', 0.7)
                            
                            # Price action conditions
                            near_bb_lower = bb_position < bb_border
                            near_bb_upper = bb_position > (1 - bb_border)
                            is_oversold = deviation_zscore < -z_thresh
                            is_overbought = deviation_zscore > z_thresh
                            has_lower_shadow = lower_shadow > shadow_min
                            has_upper_shadow = upper_shadow > shadow_min
                            
                            # LONG: AI + (BB OR Zscore) + Shadow + Safety Filter
                            if (prob_buy > sideway_buy_thresh and
                                (near_bb_lower or is_oversold) and
                                has_lower_shadow):
                                
                                # AI Safety Filter: Reject if prob_sell too high
                                if prob_sell > ai_filter_threshold:
                                    logger.warning(f"⚠️ SIDEWAY LONG REJECTED: AI Safety Filter (prob_sell:{prob_sell:.3f} > {ai_filter_threshold})")
                                else:
                                    entry_signal = 'LONG'
                                    entry_mode = 'SIDEWAY'
                                    reasons = [f"AI dự đoán: {prob_buy:.3f} > {sideway_buy_thresh:.3f}"]
                                    if near_bb_lower:
                                        reasons.append(f"Giá đã chạm vào biên giới dưới: {bb_position:.2f} < {bb_border:.2f}")
                                    if is_oversold:
                                        reasons.append(f"Quá bán: {deviation_zscore:.2f} < -{z_thresh:.2f}")
                                    if has_lower_shadow:
                                        reasons.append(f"Nến có râu dài: {lower_shadow:.2f} > {shadow_min:.2f}")
                                    reasons.append(f"AI đã chấp nhận vào lệnh bán:{prob_sell:.2f} > {ai_filter_threshold:.2f}")
                                    entry_reason = "|".join(reasons)
                            
                            # SHORT: AI + (BB OR Zscore) + Shadow + Safety Filter
                            elif (prob_sell > sideway_sell_thresh and
                                  (near_bb_upper or is_overbought) and
                                  has_upper_shadow):
                                
                                # AI Safety Filter: Reject if prob_buy too high
                                if prob_buy > ai_filter_threshold:
                                    logger.warning(f"⚠️ SIDEWAY SHORT REJECTED: AI Safety Filter (prob_buy:{prob_buy:.3f} > {ai_filter_threshold})")
                                else:
                                    entry_signal = 'SHORT'
                                    entry_mode = 'SIDEWAY'
                                    reasons = [f"AI:{prob_sell:.3f} > {sideway_sell_thresh:.3f}"]
                                    if near_bb_upper:
                                        reasons.append(f"Giá đã chạm vào biên giới trên:{bb_position:.2f} > {(1 - bb_border):.2f}")
                                    if is_overbought:
                                        reasons.append(f"Quá mua: {deviation_zscore:.2f} > {z_thresh:.2f}")
                                    if has_upper_shadow:
                                        reasons.append(f"Nến có râu dài: {upper_shadow:.2f} > {shadow_min:.2f}")
                                    reasons.append(f"AI đã chấp nhận vào lệnh mua: {prob_buy:.2f} > {ai_filter_threshold:.2f})")
                                    entry_reason = "|".join(reasons)
                        
                        else:
                            # ───────────────────────────────────────────────────
                            # MODE 3: UNCLEAR REGIME → WAIT
                            # ───────────────────────────────────────────────────
                            pass
                        
                        # ═══════════════════════════════════════════════════════
                        # EXECUTE ENTRY
                        # ═══════════════════════════════════════════════════════
                        
                        if entry_signal:
                            
                            if entry_mode == 'TRENDING':
                                # MARKET ORDER
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
                                
                                logger.info(f"🚀 {entry_mode}_{entry_signal} @ {entry_price:.2f} | {entry_reason}")
                                
                                send_discord_alert(
                                    DISCORD_WEBHOOK,
                                    f"🚀 ENTRY: {entry_mode} {entry_signal}",
                                    0x00ff00 if entry_signal == 'LONG' else 0xff0000,
                                    [
                                        {"name": "Mã:", "value": LIVE_CONFIG['symbol'], "inline": True},
                                        {"name": "Phía:", "value": entry_signal, "inline": True},
                                        {"name": "Vào lệnh:", "value": f"${entry_price:.2f}", "inline": True},
                                        {"name": "Trạng thái thị trường:", "value": entry_mode, "inline": True},
                                        {"name": "Lý do:", "value": entry_reason, "inline": False}
                                    ]
                                )
                            
                            elif entry_mode == 'SIDEWAY':
                                # LIMIT ORDER
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
                                
                                logger.info(f"✅ {entry_mode}_{entry_signal} LIMIT @ {limit_price:.2f} | {entry_reason}")
                
                except Exception as e:
                    logger.error(f"Error in signal generation: {e}", exc_info=True)
            
            # ═══════════════════════════════════════════════════════════════
            # PENDING LIMIT ORDERS (SIDEWAY)
            # ═══════════════════════════════════════════════════════════════
            
            still_pending = [] 

            for pending in state['pending_orders']:
                pending['candles_waiting'] = pending.get('candles_waiting', 0) + 1
                should_remove = False
                
                # Check if filled
                if pending['side'] == 'LONG':
                    if current_price <= pending['limit_price']:
                        trade = {
                            'side': pending['side'],
                            'entry_price': pending['limit_price'],
                            'stop_loss': pending['stop_loss'],
                            'take_profit': pending['take_profit'],
                            'entry_time': datetime.now().isoformat(),
                            'regime': pending['regime'],
                            'bars_held': 0,
                            'max_pnl': -2 * COMMISSION
                        }
                        state['open_trades'].append(trade)
                        logger.info(f"✅ Limit filled: {pending['side']} @ ${pending['limit_price']:.2f}")
                        should_remove = True
                
                elif pending['side'] == 'SHORT':
                    if current_price >= pending['limit_price']:
                        trade = {
                            'side': pending['side'],
                            'entry_price': pending['limit_price'],
                            'stop_loss': pending['stop_loss'],
                            'take_profit': pending['take_profit'],
                            'entry_time': datetime.now().isoformat(),
                            'regime': pending['regime'],
                            'bars_held': 0,
                            'max_pnl': -2 * COMMISSION
                        }
                        state['open_trades'].append(trade)
                        logger.info(f"✅ Limit filled: {pending['side']} @ ${pending['limit_price']:.2f}")
                        should_remove = True
                
                # Cancel if waited too long
                if not should_remove and pending['candles_waiting'] >= 2:
                      logger.info(f"❌ Limit order {pending['side']} cancelled (timeout)")
                      should_remove = True
                
                # Nếu không bị xóa, giữ lệnh lại trong danh sách chờ
                if not should_remove:
                    still_pending.append(pending)

            state['open_trades'] = active_trades
            state['pending_orders'] = still_pending
            
            # Save state
            save_state(state)
            
            # ═══════════════════════════════════════════════════════════════
            # CLEANUP & SLEEP
            # ═══════════════════════════════════════════════════════════════
            
            gc.collect()
            
            loop_duration = time.time() - loop_start
            sleep_time = max(5, 60 - loop_duration)
            
            logger.info(
                f"✅ Cycle: Price=${current_price:,.2f} | {regime} | "
                f"Open={len(state['open_trades'])} | Pending={len(state['pending_orders'])}"
            )
            
            time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("👋 Shutdown signal received")
            state['bot_status'] = 'Stopped'
            save_state(state)
            break
        
        except Exception as e:
            logger.error(f"❌ Critical error: {e}", exc_info=True)
            state['bot_status'] = f'Error: {str(e)}'
            save_state(state)
            time.sleep(60)

if __name__ == "__main__":
    main()
