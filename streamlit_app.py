"""
SCSP神器 - Streamlit Application
Streamlit web interface for the mean-reversion trading strategy
"""

import streamlit as st
import pandas as pd
import ta
import yfinance as yf
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Version information
VERSION = "2.0.1"

# Try to read version from version.txt if it exists
try:
    version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            VERSION = f.read().strip()
except:
    pass

# Page config
st.set_page_config(
    page_title="SCSP神器 - 交易策略分析器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Yahoo Finance-style professional theme
st.markdown("""
<style>
    /* Main background - Yahoo Finance style clean white */
    .main {
        background-color: #ffffff;
        padding-top: 1rem;
    }
    .stApp {
        background-color: #ffffff;
    }
    
    /* Remove default Streamlit padding - Yahoo Finance style */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Clean typography - Yahoo Finance style */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 1.75rem;
        margin-bottom: 0.25rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }
    
    h2 {
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }
    
    h3 {
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    }
    
    /* Professional containers */
    .stContainer {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        color: #1a1a1a;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0066CC;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
    }
    
    /* Button styling - Yahoo Finance style */
    .stButton > button {
        background-color: #0066CC;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        font-size: 0.95rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
    }
    
    /* Metric cards - Bloomberg style */
    [data-testid="stMetricValue"] {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 2rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #e0f2fe;
        border-left: 4px solid #0066CC;
        border-radius: 4px;
    }
    
    .stSuccess {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
    }
    
    .stWarning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    
    .stError {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #374151;
        line-height: 1.6;
    }
    
    .stMarkdown strong {
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    /* Code blocks */
    code {
        background-color: #f3f4f6;
        color: #0066CC;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-size: 0.875rem;
    }
    
    /* Remove Streamlit default styling */
    .stApp > header {
        background-color: #ffffff;
        border-bottom: 2px solid #0066CC;
    }
    
    /* Professional spacing */
    .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def detect_bullish_pin_bar(row):
    """Detect if a candle is a Bullish Pin Bar."""
    body_size = abs(row['close'] - row['open'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    
    if body_size == 0:
        return lower_shadow > 0
    
    return lower_shadow >= 2 * body_size


def calculate_indicators(df):
    """Calculate all required technical indicators."""
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Calculate RSI (period 14)
    rsi_indicator = ta.momentum.RSIIndicator(df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()
    
    # Calculate Bollinger Bands (period 20, std dev 2)
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    
    # Calculate ATR (Average True Range) with window=14
    atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()
    
    # Calculate ADX (period 14) using Futu's formula with DMI+ and DMI-
    # Both moving average periods are 14 (N=14, M=14)
    from adx_futu import calculate_adx_futu_ewm
    adx_result = calculate_adx_futu_ewm(df, n=14, m=14)
    df['adx'] = adx_result['adx']
    df['dmi_plus'] = adx_result['pdi']  # DMI+ (PDI)
    df['dmi_minus'] = adx_result['mdi']  # DMI- (MDI)
    
    # Calculate ADX slope
    df['adx_slope'] = df['adx'].diff()
    
    # Detect Bullish Pin Bar
    df['is_pin_bar'] = df.apply(detect_bullish_pin_bar, axis=1)
    
    return df


def generate_analysis(df):
    """
    Generate detailed market analysis in Traditional Chinese.
    
    Returns a formatted analysis string covering:
    1. Trend Analysis (ADX & DI)
    2. Momentum Analysis (RSI)
    3. Position Analysis (Bollinger Bands)
    """
    if len(df) < 1:
        return "❌ 數據不足，無法進行分析"
    
    latest = df.iloc[-1]
    
    # Get values with safe defaults
    current_adx = latest.get('adx', pd.NA)
    pdi = latest.get('dmi_plus', pd.NA)
    mdi = latest.get('dmi_minus', pd.NA)
    rsi = latest.get('rsi', pd.NA)
    close_price = latest.get('close', pd.NA)
    bb_upper = latest.get('bb_upper', pd.NA)
    bb_lower = latest.get('bb_lower', pd.NA)
    
    analysis_parts = []
    
    # 1. Trend Analysis (ADX & DI)
    trend_desc = ""
    adx_value_str = "N/A"
    if pd.notna(current_adx):
        adx_value = float(current_adx)
        adx_value_str = f"{adx_value:.2f}"
        if adx_value > ADX_THRESHOLD:
            trend_desc = "強勢趨勢"
        elif adx_value < 25:
            trend_desc = "弱勢趨勢 / 橫盤整理"
        else:
            trend_desc = "中等趨勢 / 轉換期"
    else:
        trend_desc = "無法判斷"
    
    direction_desc = ""
    if pd.notna(pdi) and pd.notna(mdi):
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        if pdi_val > mdi_val:
            direction_desc = "多頭主導（上升趨勢）"
        else:
            direction_desc = "空頭主導（下降趨勢）"
    else:
        direction_desc = "無法判斷方向"
    
    analysis_parts.append(f"📊 **趨勢分析：** {trend_desc}（ADX: {adx_value_str}，{direction_desc}）")
    
    # 2. Momentum Analysis (RSI)
    momentum_desc = ""
    rsi_value_str = "N/A"
    if pd.notna(rsi):
        rsi_value = float(rsi)
        rsi_value_str = f"{rsi_value:.2f}"
        if rsi_value > 70:
            momentum_desc = "🔥 超買（過熱）"
        elif rsi_value < 30:
            momentum_desc = "❄️ 超賣（過冷）"
        elif 45 <= rsi_value <= 55:
            momentum_desc = "⚖️ 中性（無明確方向）"
        else:
            momentum_desc = "適中"
    else:
        momentum_desc = "無法判斷"
    
    analysis_parts.append(f"💪 **動量分析：** {momentum_desc}（RSI: {rsi_value_str}）")
    
    # 3. Position Analysis (Bollinger Bands)
    position_desc = ""
    if pd.notna(close_price) and pd.notna(bb_upper) and pd.notna(bb_lower):
        close_val = float(close_price)
        upper_val = float(bb_upper)
        lower_val = float(bb_lower)
        
        if upper_val > lower_val:
            # Calculate distance to bands
            distance_to_upper = abs(close_val - upper_val) / upper_val * 100
            distance_to_lower = abs(close_val - lower_val) / lower_val * 100
            
            if distance_to_upper < 1:
                position_desc = "測試阻力位（接近上軌）"
            elif distance_to_lower < 1:
                position_desc = "測試支撐位（接近下軌）"
            else:
                position_desc = "位於中間通道（無明顯優勢）"
        else:
            position_desc = "無法判斷（布林通道數據異常）"
    else:
        position_desc = "無法判斷"
    
    analysis_parts.append(f"📍 **位置分析：** {position_desc}")
    
    return "\n\n".join(analysis_parts)


def get_detailed_wait_analysis(df, signal_type='wait'):
    """
    Generate detailed analysis for WAIT signals explaining WHY there's no trade signal.
    Returns specific explanations for different WAIT scenarios.
    """
    if len(df) < 1:
        return ""
    
    latest = df.iloc[-1]
    
    current_adx = latest.get('adx', pd.NA)
    pdi = latest.get('dmi_plus', pd.NA)
    mdi = latest.get('dmi_minus', pd.NA)
    rsi = latest.get('rsi', pd.NA)
    close_price = latest.get('close', pd.NA)
    bb_upper = latest.get('bb_upper', pd.NA)
    bb_lower = latest.get('bb_lower', pd.NA)
    is_pin_bar = latest.get('is_pin_bar', False)
    
    wait_analysis_parts = []
    
    # Check if we have valid data
    if pd.isna(close_price) or pd.isna(bb_upper) or pd.isna(bb_lower):
        return ""
    
    close_val = float(close_price)
    upper_val = float(bb_upper)
    lower_val = float(bb_lower)
    rsi_val = float(rsi) if pd.notna(rsi) else None
    
    # NEW: Scenario - Choppy Trend (ADX > ADX_THRESHOLD but PDI/MDI gap < PDI_MDI_GAP)
    if pd.notna(current_adx) and pd.notna(pdi) and pd.notna(mdi):
        adx_val = float(current_adx)
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        
        if adx_val > ADX_THRESHOLD:
            pdi_mdi_gap = abs(pdi_val - mdi_val)
            if pdi_mdi_gap < PDI_MDI_GAP:
                wait_analysis_parts.append("🌪️ **趨勢混亂：多空力量接近**")
                wait_analysis_parts.append(f"雖然 ADX 顯示強勢趨勢（{adx_val:.2f} > {ADX_THRESHOLD}），但多空雙方力量接近（PDI: {pdi_val:.2f}, MDI: {mdi_val:.2f}，差距僅 {pdi_mdi_gap:.2f} < {PDI_MDI_GAP}）。")
                wait_analysis_parts.append("多空雙方正在激烈爭奪，趨勢方向不明確。這是市場噪音，而非明確趨勢。此時交易風險較高，建議等待更明確的方向。")
                return "\n\n".join(wait_analysis_parts)
    
    # NEW: Scenario - Band Squeeze (Bandwidth < BB_BANDWIDTH_MIN%)
    bb_middle = latest.get('bb_middle', pd.NA)
    if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_middle):
        bandwidth_pct = ((float(bb_upper) - float(bb_lower)) / float(bb_middle)) * 100
        if bandwidth_pct < BB_BANDWIDTH_MIN:
            wait_analysis_parts.append("🤏 **波動率收窄：布林通道過緊**")
            wait_analysis_parts.append(f"布林通道寬度僅 {bandwidth_pct:.2f}% < {BB_BANDWIDTH_MIN}%，波動率過低，通道過於緊窄。")
            wait_analysis_parts.append("這通常預示著即將出現大幅波動（突破或崩跌）。在通道收窄時進行均值回歸交易風險極高，建議等待方向明確後再進場，避免在波動爆發前被套。")
            return "\n\n".join(wait_analysis_parts)
    
    # Scenario 4: Trend Confusion (check first as it can apply regardless of price position)
    # But only if we have the necessary data
    trend_confusion_detected = False
    if pd.notna(current_adx) and pd.notna(rsi) and rsi_val is not None:
        adx_val = float(current_adx)
        adx_slope = latest.get('adx_slope', pd.NA)
        
        # Check if ADX is rising (trend strengthening)
        adx_rising = pd.notna(adx_slope) and float(adx_slope) > 0
        
        # Check for conflicting signals
        if adx_val > 25 and adx_rising and pd.notna(pdi) and pd.notna(mdi):
            pdi_val = float(pdi)
            mdi_val = float(mdi)
            # Uptrend but RSI not confirming
            if pdi_val > mdi_val and rsi_val < 50:
                wait_analysis_parts.append("🌪️ **訊號衝突：趨勢指標與動量指標不一致**")
                wait_analysis_parts.append("ADX 顯示上升趨勢正在加強，但 RSI 顯示動量不足。")
                wait_analysis_parts.append("趨勢指標和動量指標出現分歧，最好暫時觀望，等待更明確的信號。")
                trend_confusion_detected = True
            # Downtrend but RSI not confirming
            elif mdi_val > pdi_val and rsi_val > 50:
                wait_analysis_parts.append("🌪️ **訊號衝突：趨勢指標與動量指標不一致**")
                wait_analysis_parts.append("ADX 顯示下降趨勢正在加強，但 RSI 顯示動量仍然強勁。")
                wait_analysis_parts.append("趨勢指標和動量指標出現分歧，最好暫時觀望，等待更明確的信號。")
                trend_confusion_detected = True
    
    # If trend confusion detected, return it (it's more important than price position)
    if trend_confusion_detected:
        return "\n\n".join(wait_analysis_parts)
    
    # Scenario 1: Price broke/touched LOWER Band, but NO Signal
    if close_val <= lower_val:
        # Check why there's no signal
        rsi_not_oversold = rsi_val is None or rsi_val >= 30
        no_pin_bar = not is_pin_bar
        
        if rsi_not_oversold and no_pin_bar:
            wait_analysis_parts.append("⚠️ **危險：價格已跌破下軌，但無交易訊號**")
            wait_analysis_parts.append("價格已經跌破布林下軌，但 RSI 未達到超賣水平（<30），且沒有出現看漲針形（拒絕信號）。")
            wait_analysis_parts.append("這看起來像是「接飛刀」的情況，等待價格穩定後再考慮進場。")
            return "\n\n".join(wait_analysis_parts)
    
    # Scenario 2: Price broke/touched UPPER Band, but NO Signal
    if close_val >= upper_val:
        # Check why there's no signal
        rsi_not_overbought = rsi_val is None or rsi_val <= 70
        
        if rsi_not_overbought:
            wait_analysis_parts.append("⚠️ **謹慎：價格測試上軌，但無交易訊號**")
            wait_analysis_parts.append("價格正在測試布林上軌，但 RSI 未達到超買水平（>70），不足以支持賣出認購期權。")
            wait_analysis_parts.append("動量可能推動價格繼續上漲，等待動能耗盡的信號。")
            return "\n\n".join(wait_analysis_parts)
    
    # Scenario 3: Price is in the Middle
    if lower_val < close_val < upper_val:
        wait_analysis_parts.append("⚖️ **中性：價格位於布林通道中間**")
        wait_analysis_parts.append("價格目前浮動在布林通道的中間區域，風險回報比不佳。")
        wait_analysis_parts.append("需要耐心等待價格接近上軌或下軌時再考慮交易機會。")
        return "\n\n".join(wait_analysis_parts)
    
    return ""


def get_analysis_text(df, signal_type=None):
    """
    Smart Analyst Commentary - Explains the "Why" behind the market status and signals.
    Returns detailed commentary in Traditional Chinese.
    Includes stability filter explanations.
    """
    if len(df) < 1:
        return "❌ 數據不足，無法進行分析"
    
    latest = df.iloc[-1]
    
    current_adx = latest.get('adx', pd.NA)
    pdi = latest.get('dmi_plus', pd.NA)
    mdi = latest.get('dmi_minus', pd.NA)
    rsi = latest.get('rsi', pd.NA)
    close_price = latest.get('close', pd.NA)
    bb_upper = latest.get('bb_upper', pd.NA)
    bb_lower = latest.get('bb_lower', pd.NA)
    bb_middle = latest.get('bb_middle', pd.NA)
    
    commentary_parts = []
    
    # 1. Trend Analysis with Emoji (with stability filter awareness)
    if pd.notna(current_adx) and pd.notna(pdi) and pd.notna(mdi):
        adx_val = float(current_adx)
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        pdi_mdi_gap = abs(pdi_val - mdi_val)
        
        if adx_val > ADX_THRESHOLD:
            if pdi_val > (mdi_val + PDI_MDI_GAP):
                commentary_parts.append("🚀 **趨勢：強勢上升趨勢**")
                commentary_parts.append(f"ADX 非常強勁（{adx_val:.2f} > {ADX_THRESHOLD}），市場呈現強勁的多頭動能，上升趨勢明確且持續（PDI {pdi_val:.2f} 領先 MDI {mdi_val:.2f} 超過 {PDI_MDI_GAP} 點）。")
            elif mdi_val > (pdi_val + PDI_MDI_GAP):
                commentary_parts.append("📉 **趨勢：強勢下降趨勢**")
                commentary_parts.append(f"ADX 非常強勁（{adx_val:.2f} > {ADX_THRESHOLD}），市場呈現強勁的空頭動能，下降趨勢明確且持續（MDI {mdi_val:.2f} 領先 PDI {pdi_val:.2f} 超過 {PDI_MDI_GAP} 點）。")
            else:
                # Choppy trend - gap is too small
                commentary_parts.append("🌪️ **趨勢：混亂趨勢**")
                commentary_parts.append(f"雖然 ADX 顯示強勢趨勢（{adx_val:.2f} > {ADX_THRESHOLD}），但 PDI 和 MDI 線交織在一起（差距僅 {pdi_mdi_gap:.2f} < {PDI_MDI_GAP}）。")
                commentary_parts.append("這是市場噪音，而非明確趨勢。多空雙方正在激烈爭奪，趨勢方向不明確。")
        elif adx_val <= ADX_THRESHOLD:
            commentary_parts.append("📊 **趨勢：橫盤整理 / 弱勢趨勢**")
            # Check bandwidth for squeeze warning
            if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_middle):
                bandwidth_pct = ((float(bb_upper) - float(bb_lower)) / float(bb_middle)) * 100
                if bandwidth_pct < BB_BANDWIDTH_MIN:
                    commentary_parts.append(f"⚠️ **注意：** 布林通道過於緊窄（寬度 {bandwidth_pct:.2f}% < {BB_BANDWIDTH_MIN}%），波動率收窄，預期即將出現大幅波動。")
                else:
                    commentary_parts.append("市場缺乏明確方向，價格在區間內震盪，適合均值回歸策略。")
            else:
                commentary_parts.append("市場缺乏明確方向，價格在區間內震盪，適合均值回歸策略。")
        else:
            commentary_parts.append("⚡ **趨勢：過渡期 / 中等趨勢**")
            commentary_parts.append("市場處於趨勢轉換階段，建議謹慎觀察，等待更明確的信號。")
    
    # 2. Momentum Analysis
    if pd.notna(rsi):
        rsi_val = float(rsi)
        if rsi_val > 70:
            commentary_parts.append("🔥 **動量：超買狀態**")
            commentary_parts.append("RSI 顯示市場過熱，價格可能面臨回調壓力。")
        elif rsi_val < 30:
            commentary_parts.append("❄️ **動量：超賣狀態**")
            commentary_parts.append("RSI 顯示市場過冷，價格可能出現反彈機會。")
        elif 45 <= rsi_val <= 55:
            commentary_parts.append("⚖️ **動量：中性狀態**")
            commentary_parts.append("RSI 處於中性區域，動量指標無明顯偏向。")
        else:
            commentary_parts.append("💪 **動量：適中**")
            commentary_parts.append("RSI 顯示動量適中，市場情緒平衡。")
    
    # 3. Position Analysis (Bollinger Bands) - Detailed
    if pd.notna(close_price) and pd.notna(bb_upper) and pd.notna(bb_lower):
        close_val = float(close_price)
        upper_val = float(bb_upper)
        lower_val = float(bb_lower)
        
        if upper_val > lower_val:
            # Calculate percentage distance to bands
            distance_to_upper_pct = abs(close_val - upper_val) / upper_val * 100
            distance_to_lower_pct = abs(close_val - lower_val) / lower_val * 100
            
            # Determine position status
            if close_val > upper_val:
                position_desc = f"突破上軌（價格 ${close_val:.2f} 高於上軌 ${upper_val:.2f}）"
                position_status = "Breakout (Above Upper Band)"
            elif close_val < lower_val:
                position_desc = f"跌破下軌（價格 ${close_val:.2f} 低於下軌 ${lower_val:.2f}）"
                position_status = "Breakdown (Below Lower Band)"
            elif distance_to_upper_pct < 1:
                position_desc = f"測試阻力位（價格 ${close_val:.2f} 接近上軌 ${upper_val:.2f}，距離 {distance_to_upper_pct:.2f}%）"
                position_status = "Testing Resistance (Upper Band)"
            elif distance_to_lower_pct < 1:
                position_desc = f"測試支撐位（價格 ${close_val:.2f} 接近下軌 ${lower_val:.2f}，距離 {distance_to_lower_pct:.2f}%）"
                position_status = "Testing Support (Lower Band)"
            else:
                position_desc = f"浮動在中間通道（價格 ${close_val:.2f}，上軌 ${upper_val:.2f}，下軌 ${lower_val:.2f}）"
                position_status = "Floating in Middle Channel (Neutral)"
            
            commentary_parts.append("")
            commentary_parts.append(f"📍 **位置分析：** {position_desc}")
        else:
            commentary_parts.append("")
            commentary_parts.append("📍 **位置分析：** 無法判斷（布林通道數據異常）")
    else:
        commentary_parts.append("")
        commentary_parts.append("📍 **位置分析：** 無法判斷（缺少數據）")
    
    # 4. Action Explanation (will be enhanced by signal generation)
    commentary_parts.append("")
    commentary_parts.append("💡 **策略建議：**")
    
    # 5. Add detailed WAIT analysis if signal is WAIT
    if signal_type == 'wait':
        detailed_wait = get_detailed_wait_analysis(df, signal_type)
        if detailed_wait:
            commentary_parts.append("")
            commentary_parts.append("---")
            commentary_parts.append("**詳細等待分析：**")
            commentary_parts.append(detailed_wait)
    
    return "\n\n".join(commentary_parts)


# ============================================================================
# STABILITY FILTER CONSTANTS
# ============================================================================
# These constants prevent whipsaw signals and false positives by requiring
# clear market conditions before generating trade signals.
# Stricter thresholds to reduce false signals and increase signal quality.
# ============================================================================
ADX_THRESHOLD = 35  # ADX value above which trend-following strategy is used (raised from 30 to filter weak trends)
PDI_MDI_GAP = 5.0  # Minimum spread required between PDI and MDI for trend signals (prevents whipsaws)
BB_BANDWIDTH_MIN = 3.0  # Minimum Bollinger Bandwidth % to avoid squeeze detection (prevents false range signals)
# ============================================================================


def generate_trading_signal(df):
    """
    Generate trading signal with Trend-Following and Mean-Reversion strategies.
    Includes strict stability filters to reduce whipsaws and false signals.
    
    Scenarios:
    A: RANGE MARKET (ADX <= 35) -> Mean Reversion (with Bandwidth filter)
    B: STRONG UPTREND (ADX > 35 & PDI > MDI + 5) -> Trend Following (Short Put)
    C: STRONG DOWNTREND (ADX > 35 & MDI > PDI + 5) -> Trend Following (Short Call)
    D: TRANSITION (ADX 25-35) -> Wait/Caution
    E: CHOPPY TREND (ADX > 35 but PDI/MDI gap < 5) -> Wait
    F: BAND SQUEEZE (Bandwidth < 3%) -> Wait
    
    Note: ADX threshold raised to 35 to filter out weak trends and reduce false signals.
    """
    if len(df) < 2:
        return {
            'advice': '❌ 錯誤：數據不足，無法進行分析',
            'signal_type': 'error',
            'details': {},
            'strategy_type': None,
            'commentary': None
        }
    
    latest = df.iloc[-1]
    
    current_adx = latest['adx']
    adx_slope = latest['adx_slope']
    close_price = latest['close']
    rsi = latest['rsi']
    bb_lower = latest['bb_lower']
    bb_upper = latest['bb_upper']
    is_pin_bar = latest['is_pin_bar']
    pdi = latest.get('dmi_plus', 0)
    mdi = latest.get('dmi_minus', 0)
    
    if pd.isna(current_adx) or pd.isna(adx_slope) or pd.isna(close_price):
        return {
            'advice': '❌ 錯誤：缺少技術指標數據',
            'signal_type': 'error',
            'details': {},
            'strategy_type': None,
            'commentary': None
        }
    
    # Initialize strike prices
    suggested_put_strike = None
    suggested_call_strike = None
    atr = latest.get('atr', 0)
    has_valid_data = pd.notna(atr) and pd.notna(close_price) and pd.notna(bb_lower) and pd.notna(bb_upper)
    
    details = {
        'close_price': float(close_price),
        'rsi': float(rsi),
        'adx': float(current_adx),
        'adx_slope': float(adx_slope),
        'dmi_plus': float(pdi) if pd.notna(pdi) else 0,
        'dmi_minus': float(mdi) if pd.notna(mdi) else 0,
        'atr': float(atr) if pd.notna(atr) else 0,
        'bb_upper': float(bb_upper),
        'bb_lower': float(bb_lower),
        'is_pin_bar': bool(is_pin_bar),
        'suggested_put_strike': None,
        'suggested_call_strike': None
    }
    
    # Get base commentary (will be enhanced with signal-specific details)
    base_commentary = get_analysis_text(df)
    commentary = base_commentary
    
    # SCENARIO B: STRONG UPTREND (ADX > ADX_THRESHOLD & PDI > MDI + PDI_MDI_GAP) -> Trend Following
    # STABILITY FIX: Require clear gap between PDI and MDI to prevent whipsaws
    if current_adx > ADX_THRESHOLD and pd.notna(pdi) and pd.notna(mdi):
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        pdi_mdi_gap = pdi_val - mdi_val
        
        # Only trigger if gap is significant (>= PDI_MDI_GAP)
        if pdi_val > (mdi_val + PDI_MDI_GAP):
            # Suggest SHORT PUT (Bullish) - Trading with the trend
            # AGGRESSIVE: Use 1.5x ATR (ignore Lower Band as it's too far away)
            if has_valid_data:
                suggested_put_strike = close_price - (1.5 * atr)
                details['suggested_put_strike'] = float(suggested_put_strike)
            
            commentary += "\n\n✅ **策略：順勢交易（趨勢跟隨）**"
            commentary += "\n趨勢強勁且向上，適合賣出認沽期權。"
            commentary += "\n**理由：** 趨勢明確向上，支撐位持續上升，賣出認沽期權相對安全。"
            commentary += "\n**目標行使價：** 收盤價減 1.5 倍 ATR（積極策略，獲取更好溢價）。"
            
            return {
                'advice': '🟢 訊號：賣出認沽期權（趨勢跟隨策略）',
                'signal_type': 'buy',
                'details': details,
                'strategy_type': 'trend_following',
                'commentary': commentary
            }
        elif mdi_val > (pdi_val + PDI_MDI_GAP):
            # SCENARIO C: STRONG DOWNTREND (ADX > ADX_THRESHOLD & MDI > PDI + PDI_MDI_GAP) -> Trend Following
            # Suggest SHORT CALL (Bearish) - Trading with the trend
            # AGGRESSIVE: Use 1.5x ATR (ignore Upper Band as it's too far away)
            if has_valid_data:
                suggested_call_strike = close_price + (1.5 * atr)
                details['suggested_call_strike'] = float(suggested_call_strike)
            
            commentary += "\n\n✅ **策略：順勢交易（趨勢跟隨）**"
            commentary += "\n趨勢強勁且向下，適合賣出認購期權。"
            commentary += "\n**理由：** 趨勢明確向下，阻力位持續下降，賣出認購期權相對安全。"
            commentary += "\n**目標行使價：** 收盤價加 1.5 倍 ATR（積極策略，獲取更好溢價）。"
            
            return {
                'advice': '🔴 訊號：賣出認購期權（趨勢跟隨策略）',
                'signal_type': 'sell',
                'details': details,
                'strategy_type': 'trend_following',
                'commentary': commentary
            }
        else:
            # SCENARIO E: CHOPPY TREND - Gap is less than PDI_MDI_GAP
            # Market is undecided despite high ADX
            # Get detailed WAIT analysis
            detailed_wait = get_detailed_wait_analysis(df, 'wait')
            
            commentary += "\n\n🌪️ **策略：等待（趨勢混亂）**"
            commentary += f"\n雖然 ADX 顯示強勢趨勢（{current_adx:.2f}），但多空雙方力量接近（PDI: {pdi_val:.2f}, MDI: {mdi_val:.2f}，差距僅 {abs(pdi_mdi_gap):.2f} < {PDI_MDI_GAP}）。"
            commentary += "\n**理由：** 市場方向不明確，多空雙方正在激烈爭奪，此時交易風險較高。這是市場噪音，而非明確趨勢。"
            
            # Add detailed WAIT analysis if available
            if detailed_wait:
                commentary += "\n\n---"
                commentary += "\n**詳細等待分析：**"
                commentary += "\n" + detailed_wait
            
            return {
                'advice': f'☕ 等待：趨勢混亂（ADX={current_adx:.1f}，但PDI/MDI差距僅{abs(pdi_mdi_gap):.1f}）',
                'signal_type': 'wait',
                'details': details,
                'strategy_type': 'transition',
                'commentary': commentary
            }
    
    # SCENARIO A: RANGE MARKET (ADX <= 35) -> Mean Reversion
    # Note: ADX <= 35 is treated as Range Market (or weak trend)
    # STABILITY FIX: Check Bandwidth before generating signals to avoid squeeze
    if current_adx <= ADX_THRESHOLD:
        # Special case: ADX 25-35 is transition period - always wait
        if 25 <= current_adx <= ADX_THRESHOLD:
            # SCENARIO D: TRANSITION (ADX between 25-35) -> Wait/Caution
            detailed_wait = get_detailed_wait_analysis(df, 'wait')
            
            commentary += "\n\n⚠️ **策略：等待 / 謹慎觀察**"
            commentary += f"\n市場處於趨勢轉換期，ADX 在 25-{ADX_THRESHOLD} 之間（當前 {current_adx:.2f}），建議等待更明確的信號。"
            commentary += f"\n**理由：** 趨勢強度不足（ADX < {ADX_THRESHOLD}），不足以支持趨勢跟隨策略，但也不夠弱到明確的橫盤整理。此時交易風險較高。"
            
            # Add detailed WAIT analysis if available
            if detailed_wait:
                commentary += "\n\n---"
                commentary += "\n**詳細等待分析：**"
                commentary += "\n" + detailed_wait
            
            return {
                'advice': f'☕ 等待：趨勢轉換期（ADX {current_adx:.1f} 在 25-{ADX_THRESHOLD} 之間），建議謹慎觀察',
                'signal_type': 'wait',
                'details': details,
                'strategy_type': 'transition',
                'commentary': commentary
            }
        
        # ADX < 25: Clear Range Market - proceed with Mean Reversion logic
        # Calculate Bollinger Bandwidth to detect squeeze
        bb_middle = latest.get('bb_middle', pd.NA)
        if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_middle) and pd.notna(close_price):
            bandwidth_pct = ((float(bb_upper) - float(bb_lower)) / float(bb_middle)) * 100
            
            # SCENARIO F: BAND SQUEEZE - If bandwidth is too narrow, return WAIT
            if bandwidth_pct < BB_BANDWIDTH_MIN:
                # Get detailed WAIT analysis
                detailed_wait = get_detailed_wait_analysis(df, 'wait')
                
                commentary += "\n\n🤏 **策略：等待（波動率收窄）**"
                commentary += f"\n布林通道過於緊窄（寬度 {bandwidth_pct:.2f}% < {BB_BANDWIDTH_MIN}%），波動率過低。"
                commentary += "\n**理由：** 這通常預示著即將出現大幅波動（突破或崩跌）。在通道收窄時進行均值回歸交易風險極高，建議等待方向明確後再進場。"
                
                # Add detailed WAIT analysis if available
                if detailed_wait:
                    commentary += "\n\n---"
                    commentary += "\n**詳細等待分析：**"
                    commentary += "\n" + detailed_wait
                
                return {
                    'advice': f'☕ 等待：波動率收窄（通道寬度{bandwidth_pct:.1f}% < {BB_BANDWIDTH_MIN}%），預期大幅波動',
                    'signal_type': 'wait',
                    'details': details,
                    'strategy_type': 'none',
                    'commentary': commentary
                }
        
        # Bandwidth is OK (>= BB_BANDWIDTH_MIN%), proceed with Mean Reversion logic
        # Logic B: SHORT PUT SIGNAL (Mean Reversion)
        if close_price <= bb_lower and (rsi < 30 or is_pin_bar):
            reason_parts = []
            if close_price <= bb_lower:
                reason_parts.append("超賣")
            if rsi < 30:
                reason_parts.append("RSI < 30")
            if is_pin_bar:
                reason_parts.append("看漲針形")
            reason = " + ".join(reason_parts)
            
            if has_valid_data:
                put_strike_1 = close_price - (2 * atr)
                put_strike_2 = bb_lower
                suggested_put_strike = min(put_strike_1, put_strike_2)
                details['suggested_put_strike'] = float(suggested_put_strike)
            
            commentary += "\n\n✅ **策略：均值回歸**"
            commentary += "\n市場處於橫盤整理，價格接近下軌，適合賣出認沽期權。"
            commentary += f"\n**理由：** {reason}，預期價格回歸均值。"
            commentary += "\n**目標行使價：** 使用布林下軌或收盤價減 2 倍 ATR。"
            
            return {
                'advice': f'🟢 訊號：賣出認沽期權（均值回歸策略，原因：{reason}）',
                'signal_type': 'buy',
                'details': details,
                'strategy_type': 'mean_reversion',
                'commentary': commentary
            }
        
        # Logic C: SHORT CALL SIGNAL (Mean Reversion)
        if close_price >= bb_upper or rsi > 70:
            reason_parts = []
            if close_price >= bb_upper:
                reason_parts.append("超買")
            if rsi > 70:
                reason_parts.append("RSI > 70")
            reason = " + ".join(reason_parts)
            
            if has_valid_data:
                call_strike_1 = close_price + (2 * atr)
                call_strike_2 = bb_upper
                suggested_call_strike = max(call_strike_1, call_strike_2)
                details['suggested_call_strike'] = float(suggested_call_strike)
            
            commentary += "\n\n✅ **策略：均值回歸**"
            commentary += "\n市場處於橫盤整理，價格接近上軌，適合賣出認購期權。"
            commentary += f"\n**理由：** {reason}，預期價格回歸均值。"
            commentary += "\n**目標行使價：** 使用布林上軌或收盤價加 2 倍 ATR。"
            
            return {
                'advice': f'🔴 訊號：賣出認購期權（均值回歸策略，原因：{reason}）',
                'signal_type': 'sell',
                'details': details,
                'strategy_type': 'mean_reversion',
                'commentary': commentary
            }
    
    # Default: NO ACTION - This is where detailed WAIT analysis is most important
    # Get detailed WAIT analysis for the "no signal" case
    detailed_wait = get_detailed_wait_analysis(df, 'wait')
    
    commentary += "\n\n☕ **策略：等待**"
    commentary += "\n目前無明確的交易訊號，建議繼續觀察市場變化。"
    
    # Add detailed WAIT analysis explaining WHY there's no signal
    if detailed_wait:
        commentary += "\n\n---"
        commentary += "\n**詳細等待分析：**"
        commentary += "\n" + detailed_wait
    
    return {
        'advice': '☕ 等待：無明確訊號',
        'signal_type': 'wait',
        'details': details,
        'strategy_type': 'none',
        'commentary': commentary
    }


def normalize_stock_code(input_code):
    """
    Normalize stock code input to Yahoo Finance format.
    
    Examples:
        "700" -> "0700.HK"
        "00700" -> "0700.HK"
        "HK.00700" -> "0700.HK"
        "AAPL" -> "AAPL"
        "US.AAPL" -> "AAPL"
    
    Args:
        input_code: User input (e.g., "700", "00700", "HK.00700", "AAPL", "US.AAPL")
    
    Returns:
        Normalized stock code in Yahoo Finance format
    """
    input_code = input_code.strip().upper()
    
    # Handle HK stocks (Futu format: HK.00700 or just 00700)
    if input_code.startswith('HK.'):
        # Extract the number part (e.g., "00700" from "HK.00700")
        # Yahoo Finance requires 4-digit format with leading zeros (e.g., "0700.HK")
        number_part = input_code[3:].zfill(4)  # Pad to 4 digits with leading zeros
        return f"{number_part}.HK"
    
    # Handle US stocks (Futu format: US.AAPL)
    if input_code.startswith('US.'):
        # Extract the ticker (e.g., "AAPL" from "US.AAPL")
        return input_code[3:]
    
    # Check if it's a number (HK stock like "700" or "00700")
    if input_code.isdigit():
        # Yahoo Finance requires 4-digit format with leading zeros (e.g., "0700.HK")
        number_part = input_code.zfill(4)  # Pad to 4 digits with leading zeros
        return f"{number_part}.HK"
    
    # Check if it's all letters (likely US stock)
    if input_code.isalpha():
        return input_code
    
    # If mixed or unclear, try to extract numbers for HK stock
    digits = ''.join(filter(str.isdigit, input_code))
    if digits:
        # Yahoo Finance requires 4-digit format with leading zeros
        number_part = digits.zfill(4)  # Pad to 4 digits with leading zeros
        return f"{number_part}.HK"
    
    # Default: assume it's a US stock code (return as is)
    return input_code


def analyze_stock(stock_code, original_input=None):
    """Analyze a stock and return trading signal using Yahoo Finance."""
    if original_input is None:
        original_input = stock_code
    
    try:
        # Fetch 5 years of daily data using Yahoo Finance
        data = yf.download(stock_code, period="5y", interval="1d", progress=False)
        
        if data.empty:
            return {
                'success': False,
                'error': f'No data returned for {stock_code}'
            }
        
        # Handle MultiIndex columns from Yahoo Finance
        # Yahoo Finance returns MultiIndex columns like ('Open', 'Close', etc.) when downloading multiple tickers
        # For single ticker, it's usually a simple Index, but we handle both cases
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex: take the first level (usually the column name)
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        # Rename columns to lowercase to match expected format
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'  # Keep for reference but we'll use 'close'
        }
        
        # Rename columns
        df = data.copy()
        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                'success': False,
                'error': f'Missing required columns: {missing_cols}'
            }
        
        # Reset index to make Date a column (Yahoo Finance uses Date as index)
        # This ensures we have a 'time' column for consistency with the rest of the code
        df = df.reset_index()
        
        # Rename the date column to 'time' if it exists
        if 'Date' in df.columns:
            df['time'] = df['Date']
        elif len(df.columns) > 0 and df.columns[0] == 'Date':
            # Sometimes the reset_index creates a column with the index name
            date_col = df.columns[0] if len(df.columns) > 0 else None
            if date_col:
                df['time'] = df[date_col]
        else:
            # Create a time column from the index if it was datetime
            df['time'] = df.index if hasattr(df.index, '__iter__') else range(len(df))
        
        # Sort by time to ensure chronological order
        df = df.sort_values('time').reset_index(drop=True)
        
        # Get stock basic info (name, current price) from yfinance
        stock_name = stock_code  # Default to stock code if name not available
        current_price = None
        try:
            ticker = yf.Ticker(stock_code)
            info = ticker.info
            if 'longName' in info:
                stock_name = info['longName']
            elif 'shortName' in info:
                stock_name = info['shortName']
            elif 'symbol' in info:
                stock_name = info['symbol']
            
            # Get current price from info or latest close
            if 'currentPrice' in info:
                current_price = float(info['currentPrice'])
            elif 'regularMarketPrice' in info:
                current_price = float(info['regularMarketPrice'])
        except Exception as e:
            print(f"Warning: Could not fetch stock info: {e}")
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Get latest price if not available from snapshot
        if current_price is None:
            current_price = float(df.iloc[-1]['close'])
        
        # Calculate price change from yesterday's close
        price_change = None
        price_change_percent = None
        if len(df) >= 2:
            yesterday_close = float(df.iloc[-2]['close'])
            price_change = current_price - yesterday_close
            if yesterday_close > 0:
                price_change_percent = (price_change / yesterday_close) * 100
        
        # Prepare price history for Candlestick chart with Bollinger Bands (last 50 days)
        price_history = df.tail(50).copy()
        
        # Format dates for chart (extract date part if datetime)
        dates = []
        if 'time' in price_history.columns:
            for dt in price_history['time']:
                if pd.notna(dt):
                    # Keep as datetime for Plotly
                    dates.append(dt)
                else:
                    dates.append(None)
        else:
            dates = price_history.index.tolist()
        
        chart_data = {
            'dates': dates,
            'open': [float(x) for x in price_history['open'].tolist() if pd.notna(x)],
            'high': [float(x) for x in price_history['high'].tolist() if pd.notna(x)],
            'low': [float(x) for x in price_history['low'].tolist() if pd.notna(x)],
            'close': [float(x) for x in price_history['close'].tolist() if pd.notna(x)],
            'volume': [float(x) for x in price_history['volume'].tolist() if pd.notna(x)],
            'bb_upper': [float(x) for x in price_history['bb_upper'].tolist() if pd.notna(x)],
            'bb_middle': [float(x) for x in price_history['bb_middle'].tolist() if pd.notna(x)],
            'bb_lower': [float(x) for x in price_history['bb_lower'].tolist() if pd.notna(x)]
        }
        
        # Generate signal
        signal = generate_trading_signal(df)
        
        # Generate detailed market analysis
        market_analysis = generate_analysis(df)
        
        # Use commentary from signal if available, otherwise use market_analysis
        analyst_commentary = signal.get('commentary', market_analysis) if signal else market_analysis
        
        return {
            'success': True,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'original_input': original_input,
            'data_points': len(df),
            'chart_data': chart_data,
            'signal': signal,
            'market_analysis': market_analysis,
            'analyst_commentary': analyst_commentary,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Main Streamlit App
def main():
    # Compact header
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.markdown("## SCSP神器 - 交易策略分析器")
    with col_header2:
        st.markdown(f"<div style='text-align: right; color: #6b7280; font-size: 0.75rem; padding-top: 0.5rem;'>v{VERSION}</div>", unsafe_allow_html=True)
    
    # Input section - compact
    col1, col2 = st.columns([4, 1])
    with col1:
        stock_input = st.text_input(
            "股票代碼",
            value="700",
            placeholder="輸入股票代碼（例如：700, AAPL, 1）",
            help="支援格式：700, 00700, HK.00700, AAPL, US.AAPL",
            label_visibility="visible"
        )
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("分析", type="primary", use_container_width=True)
    
    # Analyze button clicked or Enter key pressed
    if analyze_button or stock_input:
        if not stock_input.strip():
            st.warning("請輸入股票代碼")
        else:
            with st.spinner("正在分析股票數據..."):
                # Normalize stock code
                stock_code = normalize_stock_code(stock_input)
                
                # Analyze stock
                result = analyze_stock(stock_code, original_input=stock_input)
                
                if result['success']:
                    # Yahoo Finance-style Ticker Tape Header
                    price_change = result.get('price_change')
                    price_change_percent = result.get('price_change_percent')
                    current_price = result['current_price']
                    
                    # Determine color based on price change
                    if price_change is not None:
                        if price_change > 0:
                            price_color = "#16a34a"  # Green for up
                            change_color = "#16a34a"
                            change_prefix = "+"
                        elif price_change < 0:
                            price_color = "#dc2626"  # Red for down
                            change_color = "#dc2626"
                            change_prefix = ""
                        else:
                            price_color = "#1a1a1a"  # Black for no change
                            change_color = "#6b7280"
                            change_prefix = ""
                    else:
                        price_color = "#1a1a1a"
                        change_color = "#6b7280"
                        change_prefix = ""
                    
                    # Unified header layout
                    header_col1, header_col2, header_col3 = st.columns([3, 2, 1])
                    with header_col1:
                        st.markdown(f"<div style='margin-bottom: 0.5rem;'><span style='font-size: 1.75rem; font-weight: 700; color: #1a1a1a;'>{result['stock_name']}</span> <span style='font-size: 1.25rem; font-weight: 600; color: #6b7280; margin-left: 0.5rem;'>{result['stock_code']}</span></div>", unsafe_allow_html=True)
                    
                    with header_col2:
                        if price_change is not None and price_change_percent is not None:
                            delta_display = f"{change_prefix}{price_change:.2f} ({change_prefix}{price_change_percent:.2f}%)"
                        else:
                            delta_display = "N/A"
                        
                        st.markdown(f"""
                        <div style='text-align: right;'>
                            <div style='font-size: 2.25rem; font-weight: 700; color: {price_color}; line-height: 1.2;'>{current_price:.2f}</div>
                            <div style='font-size: 1rem; font-weight: 600; color: {change_color}; margin-top: 0.25rem;'>{delta_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with header_col3:
                        st.markdown(f"<div style='text-align: right; color: #9ca3af; font-size: 0.75rem; padding-top: 1.5rem;'>{result['timestamp']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Display Candlestick Chart with Bollinger Bands
                    if result.get('chart_data'):
                        chart_data = result['chart_data']
                        
                        # Prepare data
                        dates = chart_data['dates']
                        opens = chart_data['open']
                        highs = chart_data['high']
                        lows = chart_data['low']
                        closes = chart_data['close']
                        bb_upper = chart_data['bb_upper']
                        bb_middle = chart_data['bb_middle']
                        bb_lower = chart_data['bb_lower']
                        
                        fig = go.Figure()
                        
                        # Add Bollinger Bands as filled area (semi-transparent)
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=bb_upper,
                            name='布林上軌',
                            line=dict(color='rgba(239, 68, 68, 0.3)', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=bb_lower,
                            name='布林下軌',
                            line=dict(color='rgba(239, 68, 68, 0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(239, 68, 68, 0.1)',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Add Bollinger Middle line
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=bb_middle,
                            name='布林中線',
                            line=dict(color='#9ca3af', width=1, dash='dot'),
                            hovertemplate='<b>布林中線</b><br>日期: %{x}<br>價格: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Add Bollinger Upper line (visible)
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=bb_upper,
                            name='布林上軌',
                            line=dict(color='#ef4444', width=1.5, dash='dash'),
                            hovertemplate='<b>布林上軌</b><br>日期: %{x}<br>價格: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Add Bollinger Lower line (visible)
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=bb_lower,
                            name='布林下軌',
                            line=dict(color='#10b981', width=1.5, dash='dash'),
                            hovertemplate='<b>布林下軌</b><br>日期: %{x}<br>價格: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Add Candlestick chart
                        fig.add_trace(go.Candlestick(
                            x=dates,
                            open=opens,
                            high=highs,
                            low=lows,
                            close=closes,
                            name='價格',
                            increasing_line_color='#16a34a',
                            decreasing_line_color='#dc2626',
                            increasing_fillcolor='#16a34a',
                            decreasing_fillcolor='#dc2626',
                            hovertemplate='<b>%{fullData.name}</b><br>日期: %{x}<br>開盤: %{open:.2f}<br>最高: %{high:.2f}<br>最低: %{low:.2f}<br>收盤: %{close:.2f}<extra></extra>'
                        ))
                        
                        # Yahoo Finance-style layout
                        fig.update_layout(
                            title_text=f"{result['stock_code']} - 價格圖表",
                            title_x=0.5,
                            title_font_size=16,
                            title_font_color='#1a1a1a',
                            xaxis_title="",
                            yaxis_title="價格",
                            hovermode='x unified',
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff',
                            font_family='Arial, sans-serif',
                            height=500,
                            margin=dict(l=50, r=30, t=50, b=30),
                            xaxis_rangeslider_visible=False,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font_size=10
                            )
                        )
                        
                        # Update axis styling
                        fig.update_xaxes(
                            title_font_size=11,
                            tickfont_size=10,
                            tickfont_color='#6b7280',
                            gridcolor='#e5e7eb',
                            showgrid=True,
                            linecolor='#d1d5db',
                            linewidth=1
                        )
                        
                        fig.update_yaxes(
                            title_font_size=11,
                            title_font_color='#6b7280',
                            tickfont_size=10,
                            tickfont_color='#6b7280',
                            gridcolor='#e5e7eb',
                            showgrid=True,
                            linecolor='#d1d5db',
                            linewidth=1
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Key Statistics Dashboard
                    signal = result.get('signal', {})
                    details = signal.get('details', {}) if signal else {}
                    
                    if details:
                        st.markdown("### 關鍵統計指標")
                        st.markdown("---")
                        
                        # Metrics grid - 6 columns
                        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5, stat_col6 = st.columns(6)
                        
                        # RSI with color coding
                        rsi_val = details.get('rsi', 0)
                        rsi_color = "#dc2626" if rsi_val > 70 else "#16a34a" if rsi_val < 30 else "#1a1a1a"
                        with stat_col1:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>RSI</div><div style='color: {rsi_color}; font-size: 1.5rem; font-weight: 700;'>{rsi_val:.2f}</div></div>", unsafe_allow_html=True)
                        
                        with stat_col2:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>ADX</div><div style='color: #1a1a1a; font-size: 1.5rem; font-weight: 700;'>{details.get('adx', 0):.2f}</div></div>", unsafe_allow_html=True)
                        
                        with stat_col3:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>ADX 斜率</div><div style='color: #1a1a1a; font-size: 1.5rem; font-weight: 700;'>{details.get('adx_slope', 0):.2f}</div></div>", unsafe_allow_html=True)
                        
                        with stat_col4:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>PDI</div><div style='color: #1a1a1a; font-size: 1.5rem; font-weight: 700;'>{details.get('dmi_plus', 0):.2f}</div></div>", unsafe_allow_html=True)
                        
                        with stat_col5:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>MDI</div><div style='color: #1a1a1a; font-size: 1.5rem; font-weight: 700;'>{details.get('dmi_minus', 0):.2f}</div></div>", unsafe_allow_html=True)
                        
                        with stat_col6:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>ATR</div><div style='color: #1a1a1a; font-size: 1.5rem; font-weight: 700;'>{details.get('atr', 0):.2f}</div></div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                    
                    # Signal Badge & Analyst Report
                    if signal:
                        signal_type = signal.get('signal_type', 'wait')
                        advice_text = signal.get('advice', '無訊號')
                        
                        # Signal badge
                        if signal_type == 'buy':
                            st.success(f"**交易訊號：** {advice_text}")
                        elif signal_type == 'sell':
                            st.error(f"**交易訊號：** {advice_text}")
                        elif signal_type == 'wait':
                            st.warning(f"**交易訊號：** {advice_text}")
                        elif signal_type == 'error':
                            st.error(f"**錯誤：** {advice_text}")
                        else:
                            st.info(f"**狀態：** {advice_text}")
                        
                        # Analyst Report - Premium Insight Style
                        commentary = result.get('analyst_commentary') or result.get('market_analysis')
                        if commentary:
                            st.markdown("---")
                            st.markdown("### 📊 分析師報告")
                            
                            # Strategy type badge
                            strategy_type = signal.get('strategy_type', 'none')
                            strategy_badge = ""
                            if strategy_type == 'trend_following':
                                strategy_badge = '<span style="background-color: #dbeafe; color: #0066CC; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;">📈 趨勢跟隨</span>'
                            elif strategy_type == 'mean_reversion':
                                strategy_badge = '<span style="background-color: #fef3c7; color: #92400e; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;">📊 均值回歸</span>'
                            elif strategy_type == 'transition':
                                strategy_badge = '<span style="background-color: #f3f4f6; color: #6b7280; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;">⚡ 轉換期</span>'
                            
                            st.markdown(f"**策略類型：** {strategy_badge}", unsafe_allow_html=True)
                            
                            # Display commentary in structured format
                            st.markdown(
                                f'<div style="background-color: #f9fafb; border: 1px solid #e5e7eb; border-left: 4px solid #0066CC; padding: 1.5rem; border-radius: 4px; margin-top: 1rem; line-height: 1.8; font-size: 0.95rem;">{commentary.replace(chr(10), "<br>")}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Actionable Strike Price - Prominent Call-to-Action
                        if details:
                            strike_put = details.get('suggested_put_strike')
                            strike_call = details.get('suggested_call_strike')
                            
                            if strike_put is not None or strike_call is not None:
                                st.markdown("---")
                                st.markdown("### 🎯 建議行使價")
                                
                                # Determine rationale from signal commentary
                                signal_commentary = signal.get('commentary', '')
                                rationale = ""
                                if '1.5 倍 ATR' in signal_commentary:
                                    rationale = "基於 1.5x ATR（積極策略）"
                                elif '2 倍 ATR' in signal_commentary:
                                    rationale = "基於 2x ATR（保守策略）"
                                elif '布林' in signal_commentary:
                                    rationale = "基於布林通道"
                                else:
                                    rationale = "基於技術分析"
                                
                                if strike_put is not None:
                                    st.markdown(
                                        f'<div style="background-color: #d1fae5; border: 2px solid #10b981; padding: 2rem; border-radius: 8px; margin-top: 1rem; text-align: center;">'
                                        f'<div style="font-size: 3.5rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.5rem; line-height: 1;">≤ {strike_put:.1f}</div>'
                                        f'<div style="color: #6b7280; font-size: 1rem; font-weight: 600; margin-bottom: 0.25rem;">賣出認沽期權行使價</div>'
                                        f'<div style="color: #9ca3af; font-size: 0.875rem;">{rationale}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                elif strike_call is not None:
                                    st.markdown(
                                        f'<div style="background-color: #fee2e2; border: 2px solid #ef4444; padding: 2rem; border-radius: 8px; margin-top: 1rem; text-align: center;">'
                                        f'<div style="font-size: 3.5rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.5rem; line-height: 1;">≥ {strike_call:.1f}</div>'
                                        f'<div style="color: #6b7280; font-size: 1rem; font-weight: 600; margin-bottom: 0.25rem;">賣出認購期權行使價</div>'
                                        f'<div style="color: #9ca3af; font-size: 0.875rem;">{rationale}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                else:
                    st.error(f"❌ 錯誤: {result.get('error', '未知錯誤')}")


if __name__ == "__main__":
    main()
