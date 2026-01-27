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
    
    # Calculate MFI (Money Flow Index) - Period 14
    try:
        mfi_indicator = ta.volume.MFIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=14
        )
        # Use the correct method name: money_flow_index()
        df['mfi'] = mfi_indicator.money_flow_index()
    except Exception as e:
        # If MFI calculation fails, set to NaN and continue
        import warnings
        warnings.warn(f"MFI calculation failed: {e}. Setting MFI to NaN.")
        df['mfi'] = pd.NA
    
    # Calculate RVOL (Relative Volume) - Ratio of current volume to 20-day SMA of volume
    # Avoid division by zero
    volume_sma_20 = df['volume'].rolling(window=20).mean()
    df['rvol'] = df['volume'] / volume_sma_20.replace(0, pd.NA)
    # Replace infinite values with NaN
    df['rvol'] = df['rvol'].replace([float('inf'), float('-inf')], pd.NA)
    
    # Calculate SMA 50 and SMA 200 (for trend analysis)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
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
    Senior Trader-Level Analysis - Provides contextual, nuanced, and insightful interpretations.
    Returns detailed commentary in Traditional Chinese with professional trading insights.
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
    mfi = latest.get('mfi', pd.NA)
    rvol = latest.get('rvol', pd.NA)
    
    commentary_parts = []
    
    # 1. Nuanced Trend Analysis (DMI & ADX) - Senior Trader Level
    if pd.notna(current_adx) and pd.notna(pdi) and pd.notna(mdi):
        adx_val = float(current_adx)
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        pdi_mdi_gap = pdi_val - mdi_val
        gap_abs = abs(pdi_mdi_gap)
        
        # Special case: If ADX 30-35 but Gap > 15, treat as Trend (not Range)
        is_dominant_trend = gap_abs > 15
        is_strong_trend = adx_val > ADX_THRESHOLD or (30 <= adx_val <= ADX_THRESHOLD and is_dominant_trend)
        
        if is_strong_trend:
            if pdi_val > mdi_val:
                # Uptrend
                if gap_abs > 15:
                    commentary_parts.append("🚀 **趨勢：主導性多頭行情**")
                    commentary_parts.append(f"多頭正在壓倒空頭（PDI {pdi_val:.2f} 領先 MDI {mdi_val:.2f} 超過 15 點，差距 {gap_abs:.2f}）。這是一個高確信度的走勢，趨勢非常明確。")
                elif gap_abs >= PDI_MDI_GAP:
                    commentary_parts.append("📈 **趨勢：穩健上升趨勢**")
                    commentary_parts.append(f"這是一個明確定義的上升趨勢，買方掌控市場（PDI {pdi_val:.2f} 領先 MDI {mdi_val:.2f}，差距 {gap_abs:.2f}）。趨勢清晰且可持續。")
                else:
                    commentary_parts.append("🌪️ **趨勢：不明確 / 混亂**")
                    commentary_parts.append(f"多空雙方正在激烈爭奪（PDI {pdi_val:.2f} vs MDI {mdi_val:.2f}，差距僅 {gap_abs:.2f} < {PDI_MDI_GAP}）。目前還沒有明確的贏家，這是市場噪音而非明確趨勢。")
            else:
                # Downtrend
                if gap_abs > 15:
                    commentary_parts.append("📉 **趨勢：主導性空頭行情**")
                    commentary_parts.append(f"空頭正在壓倒多頭（MDI {mdi_val:.2f} 領先 PDI {pdi_val:.2f} 超過 15 點，差距 {gap_abs:.2f}）。這是一個高確信度的下跌走勢，趨勢非常明確。")
                elif gap_abs >= PDI_MDI_GAP:
                    commentary_parts.append("📉 **趨勢：穩健下降趨勢**")
                    commentary_parts.append(f"這是一個明確定義的下降趨勢，賣方掌控市場（MDI {mdi_val:.2f} 領先 PDI {pdi_val:.2f}，差距 {gap_abs:.2f}）。趨勢清晰且可持續。")
                else:
                    commentary_parts.append("🌪️ **趨勢：不明確 / 混亂**")
                    commentary_parts.append(f"多空雙方正在激烈爭奪（MDI {mdi_val:.2f} vs PDI {pdi_val:.2f}，差距僅 {gap_abs:.2f} < {PDI_MDI_GAP}）。目前還沒有明確的贏家，這是市場噪音而非明確趨勢。")
        elif adx_val < 25:
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
    
    # 2. Contextual Momentum Analysis (RSI) - Senior Trader Level
    # Interpret "Room to Run" based on RSI and trend context
    if pd.notna(rsi):
        rsi_val = float(rsi)
        is_uptrend = False
        if pd.notna(pdi) and pd.notna(mdi):
            is_uptrend = float(pdi) > float(mdi)
        
        if rsi_val > 75:
            commentary_parts.append("🔥 **動量：過熱危險區**")
            commentary_parts.append(f"RSI {rsi_val:.2f} 顯示市場極度過熱。在此處追高風險極高，預期將出現回調。這是危險區域，不建議在此時進場。")
        elif rsi_val > 70:
            commentary_parts.append("🔥 **動量：超買狀態**")
            commentary_parts.append(f"RSI {rsi_val:.2f} 顯示市場過熱，價格可能面臨回調壓力。需要謹慎觀察。")
        elif is_uptrend and 50 <= rsi_val <= 65:
            commentary_parts.append("⛽ **動量：健康且可持續**")
            commentary_parts.append(f"RSI {rsi_val:.2f} 處於「甜蜜點」區域。動量強勁但未過熱，顯示仍有充足的上漲空間。這是理想的進場時機。")
        elif is_uptrend and 40 <= rsi_val < 50:
            commentary_parts.append("🧘 **動量：蓄勢待發**")
            commentary_parts.append(f"RSI {rsi_val:.2f} 顯示短期整理，讓股票積蓄能量為下一波上漲做準備。這是健康的回調，為後續上漲提供動力。")
        elif rsi_val < 30:
            commentary_parts.append("❄️ **動量：超賣狀態**")
            commentary_parts.append(f"RSI {rsi_val:.2f} 顯示市場極度過冷，價格可能出現反彈機會。這是潛在的買入時機。")
        elif 45 <= rsi_val <= 55:
            commentary_parts.append("⚖️ **動量：中性狀態**")
            commentary_parts.append(f"RSI {rsi_val:.2f} 處於中性區域，動量指標無明顯偏向。市場情緒平衡。")
        else:
            commentary_parts.append("💪 **動量：適中**")
            commentary_parts.append(f"RSI {rsi_val:.2f} 顯示動量適中，市場情緒平衡。")
    
    # 3. Position Analysis (Bollinger Bands) - Senior Trader Level
    # Explain WHERE the price is, not just if it touched a band
    if pd.notna(close_price) and pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(bb_middle):
        close_val = float(close_price)
        upper_val = float(bb_upper)
        lower_val = float(bb_lower)
        middle_val = float(bb_middle)
        
        if upper_val > lower_val:
            # Determine which zone the price is in
            if close_val > upper_val:
                position_desc = f"📍 **位置：** 突破上軌（價格 ${close_val:.2f} 高於上軌 ${upper_val:.2f}）"
                position_status = "Breakout (Above Upper Band)"
            elif close_val < lower_val:
                position_desc = f"📍 **位置：** 跌破下軌（價格 ${close_val:.2f} 低於下軌 ${lower_val:.2f}）"
                position_status = "Breakdown (Below Lower Band)"
            elif middle_val < close_val < upper_val:
                # Upper Channel - Bull Zone
                position_desc = f"📍 **位置：** 股票正在「多頭區域」（上半部）運行。價格 ${close_val:.2f} 位於中線 ${middle_val:.2f} 和上軌 ${upper_val:.2f} 之間。"
                position_desc += f" 在上軌 ${upper_val:.2f} 之前沒有明顯阻力，仍有上漲空間。"
                position_status = "Bull Zone (Upper Half)"
            elif lower_val < close_val < middle_val:
                # Lower Channel - Weak Zone
                position_desc = f"📍 **位置：** 股票被困在「弱勢區域」（下半部）。價格 ${close_val:.2f} 位於下軌 ${lower_val:.2f} 和中線 ${middle_val:.2f} 之間。"
                position_desc += f" 需要重新站上中線 ${middle_val:.2f} 才能轉為正面。"
                position_status = "Weak Zone (Lower Half)"
            else:
                # Very close to middle or exactly at middle
                position_desc = f"📍 **位置：** 價格 ${close_val:.2f} 接近中線 ${middle_val:.2f}，處於關鍵位置。"
                position_status = "Near Middle Band"
            
            commentary_parts.append("")
            commentary_parts.append(position_desc)
        else:
            commentary_parts.append("")
            commentary_parts.append("📍 **位置分析：** 無法判斷（布林通道數據異常）")
    else:
        commentary_parts.append("")
        commentary_parts.append("📍 **位置分析：** 無法判斷（缺少數據）")
    
    # 4. Volume/Money Flow Analysis (MFI & RVOL) - Senior Trader Level
    commentary_parts.append("")
    if pd.notna(mfi) or pd.notna(rvol):
        mfi_val = float(mfi) if pd.notna(mfi) else None
        rvol_val = float(rvol) if pd.notna(rvol) else None
        
        volume_analysis = []
        
        if mfi_val is not None:
            if mfi_val < 20:
                volume_analysis.append(f"💸 **資金流向：** MFI 為 {mfi_val:.2f}，顯示資金正在大量流出，市場處於極度超賣狀態。這是潛在的買入機會。")
            elif mfi_val > 80:
                volume_analysis.append(f"💸 **資金流向：** MFI 為 {mfi_val:.2f}，顯示資金正在大量流入，市場處於極度超買狀態。需要謹慎觀察回調風險。")
            elif 20 <= mfi_val <= 40:
                volume_analysis.append(f"💸 **資金流向：** MFI 為 {mfi_val:.2f}，顯示資金正在流出，但尚未達到極端水平。")
            elif 60 <= mfi_val <= 80:
                volume_analysis.append(f"💸 **資金流向：** MFI 為 {mfi_val:.2f}，顯示資金正在流入，市場情緒積極。")
            else:
                volume_analysis.append(f"💸 **資金流向：** MFI 為 {mfi_val:.2f}，資金流向中性。")
        
        if rvol_val is not None:
            if rvol_val > 2.0:
                volume_analysis.append(f"📊 **相對成交量：** RVOL 為 {rvol_val:.2f}，成交量異常放大（超過平均值的 2 倍）！這可能表示恐慌性拋售或重大消息驅動，需要密切關注。")
            elif rvol_val > 1.5:
                volume_analysis.append(f"📊 **相對成交量：** RVOL 為 {rvol_val:.2f}，成交量明顯放大，市場活躍度提高。")
            elif rvol_val < 0.5:
                volume_analysis.append(f"📊 **相對成交量：** RVOL 為 {rvol_val:.2f}，成交量異常萎縮（低於平均值的一半）。如果價格上漲但成交量低，可能是假突破。")
            elif rvol_val < 1.0:
                volume_analysis.append(f"📊 **相對成交量：** RVOL 為 {rvol_val:.2f}，成交量低於平均值，市場參與度較低。")
            else:
                volume_analysis.append(f"📊 **相對成交量：** RVOL 為 {rvol_val:.2f}，成交量接近平均值，市場參與度正常。")
        
        if volume_analysis:
            commentary_parts.extend(volume_analysis)
        else:
            commentary_parts.append("💸 **資金流向：** 無法判斷（缺少成交量數據）")
    else:
        commentary_parts.append("💸 **資金流向：** 無法判斷（缺少成交量數據）")
    
    # 5. Add detailed WAIT analysis if signal is WAIT (called from signal generation)
    # Note: "The Verdict" section is added in generate_trading_signal, not here
    
    return "\n\n".join(commentary_parts)


# ============================================================================
# STABILITY FILTER CONSTANTS
# ============================================================================
# These constants prevent whipsaw signals and false positives by requiring
# clear market conditions before generating trade signals.
# Stricter thresholds to reduce false signals and increase signal quality.
# ============================================================================
ADX_THRESHOLD = 30.0  # ADX value above which trend-following strategy is used
PDI_MDI_GAP = 5.0  # Minimum spread required between PDI and MDI for trend signals (prevents whipsaws)
BB_BANDWIDTH_MIN = 3.0  # Minimum Bollinger Bandwidth % to avoid squeeze detection (prevents false range signals)
# ============================================================================


def get_fundamental_status(ticker):
    """
    Fetch and analyze fundamental data from yfinance to filter out distressed companies.
    PRIORITY: Solvency & Distress Detection (Zombie Stock Filter)
    
    This function focuses on identifying financially distressed companies that may be
    "zombie stocks" or facing solvency issues, rather than just valuation metrics.
    
    Args:
        ticker: yfinance Ticker object or ticker symbol string
    
    Returns:
        dict: {
            'status': 'healthy' | 'overvalued' | 'unprofitable' | 'toxic' | 'unknown',
            'trailing_pe': float or None,
            'forward_pe': float or None,
            'peg_ratio': float or None,
            'eps': float or None,
            'debt_to_equity': float or None,
            'profit_margins': float or None,
            'current_price': float or None,
            'quick_ratio': float or None,
            'current_ratio': float or None,
            'warnings': list of warning messages,
            'risk_level': 'low' | 'medium' | 'high' | 'toxic',
            'red_flags': list of specific red flag reasons
        }
    """
    try:
        # If ticker is a string, create Ticker object
        if isinstance(ticker, str):
            ticker_obj = yf.Ticker(ticker)
        else:
            ticker_obj = ticker
        
        # Fetch info - this may take a moment
        # Note: yfinance.info is a property that fetches data on access
        # Sometimes yfinance returns an empty dict or None, so we need to handle that
        try:
            info = ticker_obj.info
        except Exception as info_error:
            # Log the error for debugging (don't use st.error here as this function may be called outside Streamlit)
            import traceback
            error_trace = traceback.format_exc()
            print(f"⚠️ yfinance error fetching info: {str(info_error)}")
            print(f"Error trace: {error_trace}")
            raise ValueError(f"Failed to fetch info from yfinance: {str(info_error)}")
        
        # Check if info is empty or None
        if not info:
            print("⚠️ yfinance returned empty info dictionary")
            raise ValueError("Empty or None info dictionary returned from yfinance")
        
        # Debug: Check if info has any keys (for troubleshooting)
        if len(info) == 0:
            print("⚠️ yfinance returned info dictionary with no keys")
            raise ValueError("Info dictionary is empty (no keys found)")
        
        # Additional check: Sometimes yfinance returns a dict with only 'regularMarketPrice' or minimal data
        # Check if we have at least some fundamental data fields
        has_fundamental_data = any(key in info for key in ['trailingPE', 'forwardPE', 'debtToEquity', 'profitMargins', 'trailingEps'])
        
        # DEBUG: Log what we got from yfinance
        print(f"📊 DEBUG: yfinance.info returned {len(info)} keys")
        print(f"📊 DEBUG: Has fundamental data: {has_fundamental_data}")
        
        # Check for fundamental data keys
        fundamental_keys = ['trailingPE', 'forwardPE', 'debtToEquity', 'profitMargins', 'trailingEps', 
                           'quickRatio', 'currentRatio', 'epsTrailingTwelveMonths', 'trailingEps']
        found_keys = [key for key in fundamental_keys if key in info]
        print(f"📊 DEBUG: Found fundamental keys: {found_keys}")
        
        if not has_fundamental_data:
            # Log all available keys for debugging
            print(f"⚠️ yfinance returned data but no fundamental metrics found")
            print(f"📊 DEBUG: Total keys in info: {len(info)}")
            print(f"📊 DEBUG: Sample keys (first 20): {list(info.keys())[:20]}")
            
            # Check if it's a minimal response (common with yfinance issues)
            if len(info) < 10:
                print(f"⚠️ WARNING: Very few keys returned ({len(info)}). This might indicate:")
                print(f"   1. Yahoo Finance API is blocking/rate-limiting requests")
                print(f"   2. The ticker symbol format is incorrect")
                print(f"   3. The stock doesn't have fundamental data available")
                print(f"   4. yfinance library needs an update")
            else:
                # This is the known 2025 yfinance issue - data exists but fundamental fields are missing
                print(f"⚠️ KNOWN ISSUE (2025): yfinance is returning data but fundamental fields are missing.")
                print(f"   This is a known bug where Yahoo Finance changed their API structure.")
                print(f"   The data exists on Yahoo Finance website but yfinance can't parse it.")
            
            # Don't raise an error here - let it continue and extract what we can
            # The extraction code below will handle None values gracefully
        
        # Extract SOLVENCY & DISTRESS metrics (Priority 1)
        debt_to_equity = info.get('debtToEquity', None)
        profit_margins = info.get('profitMargins', None)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
        quick_ratio = info.get('quickRatio', None)
        current_ratio = info.get('currentRatio', None)
        
        # Extract VALUATION metrics (Priority 2)
        trailing_pe = info.get('trailingPE', None)
        forward_pe = info.get('forwardPE', None)
        peg_ratio = info.get('pegRatio', None)
        eps = info.get('trailingEps', info.get('epsTrailingTwelveMonths', None))
        
        # DEBUG: Log what values we extracted
        print(f"📊 DEBUG: Extracted values:")
        print(f"   trailing_pe: {trailing_pe}")
        print(f"   forward_pe: {forward_pe}")
        print(f"   debt_to_equity: {debt_to_equity}")
        print(f"   profit_margins: {profit_margins}")
        print(f"   eps: {eps}")
        print(f"   quick_ratio: {quick_ratio}")
        print(f"   current_ratio: {current_ratio}")
        
        warnings = []
        red_flags = []
        risk_level = 'low'
        status = 'healthy'
        
        # ========================================================================
        # RED FLAG LOGIC: STRICT DISTRESS DETECTION (Priority 1)
        # Only flag truly distressed "zombie stocks" - not healthy companies
        # ========================================================================
        
        # Rule A: The Debt Trap - Extreme Debt Levels
        # NOTE: Yahoo Finance returns debtToEquity as a PERCENTAGE (e.g., 27.2 = 27.2%)
        # Healthy range: < 100% (e.g., 27.2% is very healthy)
        # Extreme: > 200% (e.g., 350% is extremely distressed)
        if debt_to_equity is not None:
            debt_ratio = float(debt_to_equity)
            # Threshold: 200 (meaning 200% debt-to-equity)
            # This catches only truly distressed companies (like 2777.HK with 300+)
            # Healthy companies like 9988.HK (27.2%) will pass this check
            if debt_ratio > 200:
                status = 'toxic'
                risk_level = 'toxic'
                red_flags.append('extreme_debt')
                warnings.append(f"☠️ **極度負債：** 負債權益比 {debt_ratio:.1f}% > 200%，公司面臨嚴重財務壓力")
        
        # Rule B: The Bleeding Cash - Significant Losses
        # Only flag if losing 15%+ (stricter threshold to avoid false positives)
        if profit_margins is not None:
            profit_margin_pct = float(profit_margins)
            if profit_margin_pct < -0.15:  # Negative 15% (stricter than -10%)
                status = 'toxic'
                risk_level = 'toxic'
                red_flags.append('significant_losses')
                warnings.append(f"☠️ **嚴重虧損：** 利潤率 {profit_margin_pct*100:.1f}%，公司正在大量失血")
        
        # Rule C: Penny Stock Risk - Loss-making Penny Stock
        # Stricter: Only flag if price < $2.00 AND losing money (not just < $1.00)
        if current_price is not None and profit_margins is not None:
            price = float(current_price)
            profit_margin_pct = float(profit_margins)
            if price < 2.0 and profit_margin_pct < 0:
                status = 'toxic'
                risk_level = 'toxic'
                red_flags.append('penny_stock_loss')
                warnings.append(f"☠️ **虧損低價股：** 股價 ${price:.2f} < $2.00 且公司虧損，極高風險")
        
        # Rule D: Missing Earnings Data - Only flag if truly suspicious (penny stock)
        # Do NOT flag blue chips (like 9988.HK) that might have temporary N/A due to reporting periods
        if trailing_pe is None and current_price is not None:
            price = float(current_price)
            # Only flag if price < $5 (penny stock territory)
            # Blue chips with complex reporting might have temporary N/A - don't penalize them
            if price < 5.0:
                status = 'toxic'
                risk_level = 'toxic'
                red_flags.append('no_earnings_data')
                warnings.append(f"☠️ **無盈利數據：** 股價 ${price:.2f} < $5.00 且無 PE 數據，很可能虧損")
        
        # ========================================================================
        # VALUATION CHECKS (Priority 2 - Only if not already TOXIC)
        # Safe handling: Skip checks if data is None (don't penalize for missing data)
        # ========================================================================
        
        if status != 'toxic':
            # Check for unprofitable company (negative PE)
            # Only check if PE is available (not None)
            if trailing_pe is not None and trailing_pe < 0:
                status = 'unprofitable'
                risk_level = 'high'
                warnings.append("⚠️ 公司虧損：Trailing PE < 0，公司目前不盈利")
            
            # Check for overvaluation (only if PE is available)
            elif trailing_pe is not None and trailing_pe > 50:
                # Check PEG if available (if None, skip PEG check)
                if peg_ratio is not None and peg_ratio > 2:
                    status = 'overvalued'
                    risk_level = 'high'
                    warnings.append(f"⚠️ 估值過高：Trailing PE ({trailing_pe:.2f}) > 50 且 PEG ({peg_ratio:.2f}) > 2")
                else:
                    status = 'overvalued'
                    risk_level = 'medium'
                    warnings.append(f"⚠️ 估值偏高：Trailing PE ({trailing_pe:.2f}) > 50")
            
            # Check for high PEG (only if PEG is available and PE is reasonable or None)
            # If PE is None, we can still check PEG independently
            elif peg_ratio is not None and peg_ratio > 2:
                status = 'overvalued'
                risk_level = 'medium'
                warnings.append(f"⚠️ 成長估值偏高：PEG ({peg_ratio:.2f}) > 2")
            
            # If PE is None but price is reasonable (> $5), don't flag as unprofitable
            # Blue chips like 9988.HK might have temporary N/A due to reporting periods
            # This is handled by Rule D above (only flags if price < $5)
        
        return {
            'status': status,
            'trailing_pe': trailing_pe,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'eps': eps,
            'debt_to_equity': debt_to_equity,
            'profit_margins': profit_margins,
            'current_price': current_price,
            'quick_ratio': quick_ratio,
            'current_ratio': current_ratio,
            'warnings': warnings,
            'risk_level': risk_level,
            'red_flags': red_flags
        }
    
    except Exception as e:
        # If fundamental data is unavailable, return unknown status
        # Log the error for debugging (but don't expose to user in production)
        error_msg = str(e)
        import traceback
        error_details = traceback.format_exc()
        
        # Check if this is the known 2025 yfinance issue (empty dict or missing fields)
        is_known_issue = ("Empty or None info" in error_msg or 
                          "Info dictionary is empty" in error_msg or
                          "Failed to fetch info" in error_msg)
        
        if is_known_issue:
            warning_msg = "無法獲取基本面數據：這是 yfinance 庫的已知問題（2025年）。Yahoo Finance 更改了 API 結構，導致基本面數據無法通過 yfinance 獲取。"
        else:
            warning_msg = f"無法獲取基本面數據：{error_msg}"
        
        print(f"⚠️ get_fundamental_status error: {error_msg}")
        if is_known_issue:
            print("   This appears to be the known 2025 yfinance issue with fundamental data")
        
        return {
            'status': 'unknown',
            'trailing_pe': None,
            'forward_pe': None,
            'peg_ratio': None,
            'eps': None,
            'debt_to_equity': None,
            'profit_margins': None,
            'current_price': None,
            'quick_ratio': None,
            'current_ratio': None,
            'warnings': [warning_msg],
            'risk_level': 'medium',  # Default to medium risk if data unavailable
            'red_flags': [],
            '_error_details': error_details,  # For debugging only
            '_is_known_issue': is_known_issue  # Flag for UI to show appropriate message
        }


def apply_fundamental_filters(signal, fundamental_status, is_bullish=True):
    """
    Apply fundamental filters to downgrade or override trading signals.
    CRITICAL: "TOXIC" status forces WAIT or SHORT ONLY (never buy).
    
    Args:
        signal: dict with signal information (advice, signal_type, commentary, etc.)
        fundamental_status: dict from get_fundamental_status() or None
        is_bullish: bool, True for buy signals (Short Put), False for sell signals (Short Call)
    
    Returns:
        dict: Modified signal with downgraded status if filters trigger
    """
    warnings = []
    should_downgrade = False
    is_toxic = False
    
    # Check fundamental status (Priority 1: TOXIC status)
    if fundamental_status:
        fund_status = fundamental_status.get('status', 'unknown')
        fund_risk = fundamental_status.get('risk_level', 'low')
        fund_warnings = fundamental_status.get('warnings', [])
        fund_red_flags = fundamental_status.get('red_flags', [])
        
        # CRITICAL: TOXIC status - Force WAIT for buy signals, allow SHORT for sell signals
        if fund_status == 'toxic' or fund_risk == 'toxic':
            is_toxic = True
            should_downgrade = True
            warnings.extend(fund_warnings)
            
            # Build toxic warning message
            toxic_reasons = []
            if 'extreme_debt' in fund_red_flags:
                debt_eq = fundamental_status.get('debt_to_equity', 'N/A')
                toxic_reasons.append(f"極度負債 (負債權益比: {debt_eq})")
            if 'significant_losses' in fund_red_flags:
                profit_margin = fundamental_status.get('profit_margins', 'N/A')
                if isinstance(profit_margin, (int, float)):
                    toxic_reasons.append(f"嚴重虧損 (利潤率: {profit_margin*100:.1f}%)")
            if 'penny_stock_loss' in fund_red_flags:
                toxic_reasons.append("虧損仙股")
            if 'no_earnings_data' in fund_red_flags:
                toxic_reasons.append("無盈利數據")
            
            if toxic_reasons:
                warnings.append(f"☠️ **TOXIC / 高風險資產：** {' & '.join(toxic_reasons)}。強烈建議避免買入。")
            else:
                warnings.append("☠️ **TOXIC / 高風險資產：** 公司財務狀況極度危險。強烈建議避免買入。")
        
        # Check for other high-risk fundamental issues (only if not already TOXIC)
        elif fund_status in ['unprofitable', 'overvalued'] and fund_risk == 'high':
            should_downgrade = True
            warnings.extend(fund_warnings)
            warnings.append("🔴 **基本面風險：** 技術面雖好，但基本面存在高風險。建議等待。")
    
    # Apply filters based on signal type
    signal_type = signal.get('signal_type', 'wait')
    
    # For BUY signals (Short Put): Downgrade to WAIT if any filter triggers
    if signal_type == 'buy' and is_bullish and should_downgrade:
        original_advice = signal.get('advice', '')
        original_commentary = signal.get('commentary', '')
        
        # Create new WAIT signal with appropriate warning level
        if is_toxic:
            filter_header = "**☠️ TOXIC 資產過濾器觸發**"
            advice_prefix = "☠️ TOXIC："
        else:
            filter_header = "**⚠️ 基本面過濾器觸發**"
            advice_prefix = "☕ 等待："
        
        new_commentary = original_commentary + f"\n\n---\n{filter_header}\n"
        new_commentary += "\n".join(warnings)
        
        if is_toxic:
            new_commentary += "\n\n**結論：** 技術面雖顯示買入訊號，但這是高風險/TOXIC 資產（可能面臨財務危機、極度負債或嚴重虧損）。**強烈建議避免買入，等待更好的標的。**"
        else:
            new_commentary += "\n\n**結論：** 技術面雖顯示買入訊號，但基本面存在風險。建議暫時觀望，等待更好的進場時機。"
        
        return {
            'advice': f'{advice_prefix}技術面良好，但基本面存在風險（已過濾原訊號：{original_advice}）',
            'signal_type': 'wait',
            'details': signal.get('details', {}),
            'strategy_type': signal.get('strategy_type', 'none'),
            'commentary': new_commentary,
            'original_signal': original_advice,  # Keep track of what was filtered
            'filter_reasons': warnings,
            'is_toxic': is_toxic
        }
    
    # For SELL signals (Short Call): Allow if TOXIC (can short distressed companies)
    # But still warn about the risks
    if signal_type == 'sell' and not is_bullish and is_toxic:
        original_commentary = signal.get('commentary', '')
        new_commentary = original_commentary + "\n\n---\n**⚠️ TOXIC 資產警告**\n"
        new_commentary += "\n".join(warnings)
        new_commentary += "\n\n**注意：** 雖然技術面支持賣出認購期權，但這是 TOXIC 資產。做空高風險資產需格外謹慎，建議降低倉位。"
        
        return {
            'advice': signal.get('advice', ''),
            'signal_type': 'sell',
            'details': signal.get('details', {}),
            'strategy_type': signal.get('strategy_type', 'none'),
            'commentary': new_commentary,
            'is_toxic': True
        }
    
    # No downgrade needed, return original signal
    return signal


def generate_trading_signal(df, fundamental_status=None):
    """
    Generate trading signal with Trend-Following and Mean-Reversion strategies.
    Includes strict stability filters to reduce whipsaws and false signals.
    Now includes fundamental filters to avoid bad companies.
    
    Scenarios:
    A: RANGE MARKET (ADX <= 35) -> Mean Reversion (with Bandwidth filter)
    B: STRONG UPTREND (ADX > 35 & PDI > MDI + 5) -> Trend Following (Short Put)
    C: STRONG DOWNTREND (ADX > 35 & MDI > PDI + 5) -> Trend Following (Short Call)
    D: TRANSITION (ADX 25-35) -> Wait/Caution
    E: CHOPPY TREND (ADX > 35 but PDI/MDI gap < 5) -> Wait
    F: BAND SQUEEZE (Bandwidth < 3%) -> Wait
    
    Args:
        df: DataFrame with calculated indicators
        fundamental_status: dict from get_fundamental_status() or None
    
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
    mfi = latest.get('mfi', pd.NA)
    rvol = latest.get('rvol', pd.NA)
    
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
    
    # Get SMA values from latest data
    sma_50 = latest.get('sma_50', pd.NA)
    sma_200 = latest.get('sma_200', pd.NA)
    bb_middle = latest.get('bb_middle', pd.NA)
    
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
        'bb_middle': float(bb_middle) if pd.notna(bb_middle) else 0,
        'is_pin_bar': bool(is_pin_bar),
        'mfi': float(mfi) if pd.notna(mfi) else 0,
        'rvol': float(rvol) if pd.notna(rvol) else 0,
        'sma_50': float(sma_50) if pd.notna(sma_50) else None,
        'sma_200': float(sma_200) if pd.notna(sma_200) else None,
        'suggested_put_strike': None,
        'suggested_call_strike': None
    }
    
    # Get base commentary (will be enhanced with signal-specific details)
    base_commentary = get_analysis_text(df)
    commentary = base_commentary
    
    # ========================================================================
    # OVERRIDE LOGIC: SUPER BREAKOUT (Priority 0 - Checks BEFORE normal logic)
    # ========================================================================
    # If PDI/MDI gap is extremely large (>20) AND volume spike (RVOL > 2.0) 
    # AND price breaks above Upper Band, this is an EXPLOSIVE BREAKOUT.
    # Override normal ADX threshold - this is a high-conviction signal.
    # ========================================================================
    if pd.notna(pdi) and pd.notna(mdi) and pd.notna(rvol) and pd.notna(bb_upper):
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        pdi_mdi_gap = pdi_val - mdi_val
        rvol_val = float(rvol)
        
        # Check for SUPER BREAKOUT conditions
        if pdi_mdi_gap > 20 and rvol_val > 2.0 and close_price > bb_upper:
            # EXPLOSIVE BREAKOUT detected - Override normal ADX threshold
            if has_valid_data:
                suggested_put_strike = close_price - (1.5 * atr)
                details['suggested_put_strike'] = float(suggested_put_strike)
            
            commentary += "\n\n🚀 **策略：爆炸性突破（超級突破）**"
            commentary += f"\n成交量爆升 (RVOL {rvol_val:.1f} > 2.0) 確認了突破上軌的訊號。多頭極度主導 (PDI/MDI 差距 {pdi_mdi_gap:.1f} > 20)。"
            commentary += "\n**理由：** 這是罕見的爆炸性突破模式 - 即使 ADX 較低 ({:.1f})，但極大的多空差距和成交量爆升顯示這是高確信度的突破訊號。".format(float(current_adx))
            commentary += "\n**目標行使價：** 收盤價減 1.5 倍 ATR（積極策略，獲取更好溢價）。"
            
            # Add "The Verdict" summary
            strike_price = details.get('suggested_put_strike', close_price - (1.5 * atr) if has_valid_data else None)
            if strike_price:
                verdict_reason = f"爆炸性突破：成交量爆升 (RVOL {rvol_val:.1f}) 且多頭極度主導 (差距 {pdi_mdi_gap:.1f})。這是高確信度的突破訊號，即使 ADX 較低也值得跟進。"
                commentary += f"\n\n💡 **結論：** 賣出認沽期權 @ ${strike_price:.1f}。**為什麼？** {verdict_reason}"
            
            # Create EXPLOSIVE BREAKOUT signal
            original_signal = {
                'advice': '🚀 訊號：爆炸性突破 - 賣出認沽期權（超級突破策略）',
                'signal_type': 'buy',
                'details': details,
                'strategy_type': 'explosive_breakout',
                'commentary': commentary
            }
            
            # Apply fundamental filters (but this is a high-conviction signal)
            filtered_signal = apply_fundamental_filters(
                original_signal, 
                fundamental_status,
                is_bullish=True
            )
            
            return filtered_signal
    
    # SCENARIO B: STRONG TREND (ADX >= ADX_THRESHOLD) -> Trend Following
    # CORRECTED LOGIC: Simple, clear flow to prevent math errors
    if pd.notna(pdi) and pd.notna(mdi) and current_adx >= ADX_THRESHOLD:
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        pdi_mdi_gap = pdi_val - mdi_val
        gap_abs = abs(pdi_mdi_gap)
        
        # Case 1: Clear Uptrend (PDI leads by >= PDI_MDI_GAP) -> SIGNAL: SHORT PUT
        if pdi_val > (mdi_val + PDI_MDI_GAP):
            # Suggest SHORT PUT (Bullish) - Trading with the trend
            # AGGRESSIVE: Use 1.5x ATR (ignore Lower Band as it's too far away)
            if has_valid_data:
                suggested_put_strike = close_price - (1.5 * atr)
                details['suggested_put_strike'] = float(suggested_put_strike)
            
            commentary += "\n\n✅ **策略：順勢交易（趨勢跟隨）**"
            if gap_abs > 15:
                commentary += "\n趨勢非常強勁且向上，多頭主導市場。適合賣出認沽期權。"
                commentary += "\n**理由：** 這是主導性多頭行情（差距 > 15），趨勢明確且高確信度，支撐位持續上升，賣出認沽期權相對安全。"
            else:
                commentary += f"\n強勢上升趨勢（ADX {current_adx:.2f}）。多頭領先 {gap_abs:.2f} 點。適合賣出認沽期權。"
                commentary += "\n**理由：** 趨勢明確向上，支撐位持續上升，賣出認沽期權相對安全。"
            commentary += "\n**目標行使價：** 收盤價減 1.5 倍 ATR（積極策略，獲取更好溢價）。"
            
            # Add "The Verdict" summary
            strike_price = details.get('suggested_put_strike', close_price - (1.5 * atr) if has_valid_data else None)
            if strike_price:
                rsi_val = float(rsi) if pd.notna(rsi) else None
                if rsi_val and 50 <= rsi_val <= 65:
                    verdict_reason = f"趨勢主導（差距 {gap_abs:.1f}）且 RSI 仍有充足上漲空間（{rsi_val:.1f}）。不要害怕緩慢上漲。"
                elif gap_abs > 15:
                    verdict_reason = f"這是主導性多頭行情（差距 {gap_abs:.1f}），趨勢非常明確且高確信度。"
                else:
                    verdict_reason = f"強勢上升趨勢（ADX {current_adx:.1f}）。多頭領先 {gap_abs:.1f} 點。"
                commentary += f"\n\n💡 **結論：** 賣出認沽期權 @ ${strike_price:.1f}。**為什麼？** {verdict_reason}"
            
            # FUNDAMENTAL & NEWS FILTER: Check if we should downgrade this buy signal
            original_signal = {
                'advice': '🟢 訊號：賣出認沽期權（趨勢跟隨策略）',
                'signal_type': 'buy',
                'details': details,
                'strategy_type': 'trend_following',
                'commentary': commentary
            }
            
            # Apply fundamental filters
            filtered_signal = apply_fundamental_filters(
                original_signal,
                fundamental_status,
                is_bullish=True
            )
            
            return filtered_signal
        # Case 2: Clear Downtrend (MDI leads by >= PDI_MDI_GAP) -> SIGNAL: SHORT CALL
        elif mdi_val > (pdi_val + PDI_MDI_GAP):
            # SCENARIO C: STRONG DOWNTREND (ADX > ADX_THRESHOLD & MDI > PDI + PDI_MDI_GAP) -> Trend Following
            # Suggest SHORT CALL (Bearish) - Trading with the trend
            # AGGRESSIVE: Use 1.5x ATR (ignore Upper Band as it's too far away)
            if has_valid_data:
                suggested_call_strike = close_price + (1.5 * atr)
                details['suggested_call_strike'] = float(suggested_call_strike)
            
            commentary += "\n\n✅ **策略：順勢交易（趨勢跟隨）**"
            if gap_abs > 15:
                commentary += "\n趨勢非常強勁且向下，空頭主導市場。適合賣出認購期權。"
                commentary += "\n**理由：** 這是主導性空頭行情（差距 > 15），趨勢明確且高確信度，阻力位持續下降，賣出認購期權相對安全。"
            else:
                commentary += f"\n強勢下降趨勢（ADX {current_adx:.2f}）。空頭領先 {gap_abs:.2f} 點。適合賣出認購期權。"
                commentary += "\n**理由：** 趨勢明確向下，阻力位持續下降，賣出認購期權相對安全。"
            commentary += "\n**目標行使價：** 收盤價加 1.5 倍 ATR（積極策略，獲取更好溢價）。"
            
            # Add "The Verdict" summary
            strike_price = details.get('suggested_call_strike', close_price + (1.5 * atr) if has_valid_data else None)
            if strike_price:
                if gap_abs > 15:
                    verdict_reason = f"這是主導性空頭行情（差距 {gap_abs:.1f}），趨勢非常明確且高確信度。"
                else:
                    verdict_reason = f"強勢下降趨勢（ADX {current_adx:.1f}）。空頭領先 {gap_abs:.1f} 點。"
                commentary += f"\n\n💡 **結論：** 賣出認購期權 @ ${strike_price:.1f}。**為什麼？** {verdict_reason}"
            
            return {
                'advice': '🔴 訊號：賣出認購期權（趨勢跟隨策略）',
                'signal_type': 'sell',
                'details': details,
                'strategy_type': 'trend_following',
                'commentary': commentary
            }
        # Case 3: Gap is too small (< PDI_MDI_GAP) -> WAIT
        else:
            # SCENARIO E: CHOPPY TREND - Gap is less than PDI_MDI_GAP
            # Market is undecided despite high ADX
            # Get detailed WAIT analysis
            detailed_wait = get_detailed_wait_analysis(df, 'wait')
            
            commentary += "\n\n🌪️ **策略：等待（趨勢混亂）**"
            commentary += f"\n雖然 ADX 顯示強勢趨勢（{current_adx:.2f}），但多空雙方力量接近（PDI: {pdi_val:.2f}, MDI: {mdi_val:.2f}，差距僅 {gap_abs:.2f} < {PDI_MDI_GAP}）。"
            commentary += "\n**理由：** 市場方向不明確，多空雙方正在激烈爭奪，此時交易風險較高。這是市場噪音，而非明確趨勢。"
            
            # Add detailed WAIT analysis if available
            if detailed_wait:
                commentary += "\n\n---"
                commentary += "\n**詳細等待分析：**"
                commentary += "\n" + detailed_wait
            
            return {
                'advice': f'☕ 等待：趨勢混亂（ADX={current_adx:.1f}，但PDI/MDI差距僅{gap_abs:.1f} < {PDI_MDI_GAP}）',
                'signal_type': 'wait',
                'details': details,
                'strategy_type': 'transition',
                'commentary': commentary
            }
    
    # SCENARIO A: RANGE MARKET (ADX < 20) -> Mean Reversion
    # STABILITY FIX: Check Bandwidth before generating signals to avoid squeeze
    elif current_adx < 20:
        # ADX < 20: Clear Range Market - proceed with Mean Reversion logic
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
        # Extract volume indicators for mean reversion signals
        mfi_val = float(mfi) if pd.notna(mfi) else None
        rvol_val = float(rvol) if pd.notna(rvol) else None
        
        # Logic B: SHORT PUT SIGNAL (Mean Reversion) - WITH VOLUME FILTER
        # Check for volume confirmation
        
        # Check for MFI divergence (Current MFI > Previous MFI while Price is lower)
        mfi_divergence = False
        if len(df) >= 2 and pd.notna(mfi):
            prev_mfi = df.iloc[-2].get('mfi', pd.NA)
            prev_close = df.iloc[-2].get('close', pd.NA)
            if pd.notna(prev_mfi) and pd.notna(prev_close):
                if float(mfi) > float(prev_mfi) and float(close_price) < float(prev_close):
                    mfi_divergence = True
        
        # Volume filter conditions for SHORT PUT
        volume_confirmed = False
        volume_reason = []
        
        if mfi_val is not None and mfi_val < 20:
            volume_confirmed = True
            volume_reason.append(f"MFI 超賣 ({mfi_val:.2f} < 20)")
        
        if mfi_divergence:
            volume_confirmed = True
            volume_reason.append("MFI 背離（資金流入但價格下跌）")
        
        if rvol_val is not None and rvol_val > 2.0 and rsi < 30:
            volume_confirmed = True
            volume_reason.append(f"恐慌性拋售 (RVOL {rvol_val:.2f} > 2.0)")
        
        # Original condition: Price <= Lower BB AND (RSI < 30 OR Pin Bar)
        base_condition = close_price <= bb_lower and (rsi < 30 or is_pin_bar)
        
        # NEW: Require volume confirmation OR keep original condition if volume data unavailable
        if base_condition and (volume_confirmed or (mfi_val is None and rvol_val is None)):
            reason_parts = []
            if close_price <= bb_lower:
                reason_parts.append("超賣")
            if rsi < 30:
                reason_parts.append("RSI < 30")
            if is_pin_bar:
                reason_parts.append("看漲針形")
            if volume_reason:
                reason_parts.extend(volume_reason)
            reason = " + ".join(reason_parts)
            
            if has_valid_data:
                put_strike_1 = close_price - (2 * atr)
                put_strike_2 = bb_lower
                suggested_put_strike = min(put_strike_1, put_strike_2)
                details['suggested_put_strike'] = float(suggested_put_strike)
            
            commentary += "\n\n✅ **策略：均值回歸（成交量確認）**"
            commentary += "\n市場處於橫盤整理，價格接近下軌，適合賣出認沽期權。"
            if volume_reason:
                commentary += f"\n**成交量確認：** {', '.join(volume_reason)}，顯示資金流向支持反彈。"
            commentary += f"\n**理由：** {reason}，預期價格回歸均值。"
            commentary += "\n**目標行使價：** 使用布林下軌或收盤價減 2 倍 ATR。"
            
            # Add "The Verdict" summary
            strike_price = details.get('suggested_put_strike', None)
            if strike_price:
                rsi_val = float(rsi) if pd.notna(rsi) else None
                if is_pin_bar:
                    verdict_reason = "價格在區間底部且出現看漲反轉信號（Pin Bar），預期反彈。"
                elif rsi_val and rsi_val < 30:
                    verdict_reason = f"價格在區間底部且 RSI 超賣（{rsi_val:.1f}），預期反彈回歸均值。"
                else:
                    verdict_reason = "價格在區間底部，預期反彈回歸均值。"
                commentary += f"\n\n💡 **結論：** 賣出認沽期權 @ ${strike_price:.1f}。**為什麼？** {verdict_reason}"
            
            # FUNDAMENTAL & NEWS FILTER: Check if we should downgrade this buy signal
            original_signal = {
                'advice': f'🟢 訊號：賣出認沽期權（均值回歸策略，原因：{reason}）',
                'signal_type': 'buy',
                'details': details,
                'strategy_type': 'mean_reversion',
                'commentary': commentary
            }
            
            # Apply fundamental filters
            filtered_signal = apply_fundamental_filters(
                original_signal,
                fundamental_status,
                is_bullish=True
            )
            
            return filtered_signal
        
        # Logic C: SHORT CALL SIGNAL (Mean Reversion) - WITH VOLUME FILTER
        # Volume filter conditions for SHORT CALL (mfi_val and rvol_val already extracted above)
        volume_confirmed = False
        volume_reason = []
        fake_breakout = False
        
        if mfi_val is not None and mfi_val > 80:
            volume_confirmed = True
            volume_reason.append(f"MFI 超買 ({mfi_val:.2f} > 80)")
        
        if rvol_val is not None and rvol_val < 1.0 and close_price >= bb_upper:
            volume_confirmed = True
            fake_breakout = True
            volume_reason.append(f"假突破 (RVOL {rvol_val:.2f} < 1.0，價格上漲但成交量萎縮)")
        
        # Original condition: Price >= Upper BB OR RSI > 70
        base_condition = close_price >= bb_upper or rsi > 70
        
        # NEW: Require volume confirmation OR keep original condition if volume data unavailable
        if base_condition and (volume_confirmed or fake_breakout or (mfi_val is None and rvol_val is None)):
            reason_parts = []
            if close_price >= bb_upper:
                reason_parts.append("超買")
            if rsi > 70:
                reason_parts.append("RSI > 70")
            if volume_reason:
                reason_parts.extend(volume_reason)
            reason = " + ".join(reason_parts)
            
            if has_valid_data:
                call_strike_1 = close_price + (2 * atr)
                call_strike_2 = bb_upper
                suggested_call_strike = max(call_strike_1, call_strike_2)
                details['suggested_call_strike'] = float(suggested_call_strike)
            
            commentary += "\n\n✅ **策略：均值回歸（成交量確認）**"
            commentary += "\n市場處於橫盤整理，價格接近上軌，適合賣出認購期權。"
            if volume_reason:
                if fake_breakout:
                    commentary += f"\n**成交量確認：** {', '.join(volume_reason)}，這是假突破信號，預期回調。"
                else:
                    commentary += f"\n**成交量確認：** {', '.join(volume_reason)}，顯示資金流向支持回調。"
            commentary += f"\n**理由：** {reason}，預期價格回歸均值。"
            commentary += "\n**目標行使價：** 使用布林上軌或收盤價加 2 倍 ATR。"
            
            # Add "The Verdict" summary
            strike_price = details.get('suggested_call_strike', None)
            if strike_price:
                rsi_val = float(rsi) if pd.notna(rsi) else None
                if rsi_val and rsi_val > 70:
                    verdict_reason = f"價格在區間頂部且 RSI 超買（{rsi_val:.1f}），預期回調回歸均值。"
                else:
                    verdict_reason = "價格在區間頂部，預期回調回歸均值。"
                commentary += f"\n\n💡 **結論：** 賣出認購期權 @ ${strike_price:.1f}。**為什麼？** {verdict_reason}"
            
            return {
                'advice': f'🔴 訊號：賣出認購期權（均值回歸策略，原因：{reason}）',
                'signal_type': 'sell',
                'details': details,
                'strategy_type': 'mean_reversion',
                'commentary': commentary
            }
    
    # SCENARIO D: TRANSITION (ADX between 20-30) -> Wait/Caution
    # This handles the case where ADX is not high enough for trend following, but not low enough for range trading
    elif 20 <= current_adx < ADX_THRESHOLD:
        detailed_wait = get_detailed_wait_analysis(df, 'wait')
        
        commentary += "\n\n⚠️ **策略：等待 / 謹慎觀察**"
        commentary += f"\n市場處於趨勢轉換期，ADX 在 20-{ADX_THRESHOLD} 之間（當前 {current_adx:.2f}），建議等待更明確的信號。"
        commentary += f"\n**理由：** 趨勢強度不足（ADX < {ADX_THRESHOLD}），不足以支持趨勢跟隨策略，但也不夠弱到明確的橫盤整理。此時交易風險較高。"
        
        # Add detailed WAIT analysis if available
        if detailed_wait:
            commentary += "\n\n---"
            commentary += "\n**詳細等待分析：**"
            commentary += "\n" + detailed_wait
        
        return {
            'advice': f'☕ 等待：趨勢轉換期（ADX {current_adx:.1f} 在 20-{ADX_THRESHOLD} 之間），建議謹慎觀察',
            'signal_type': 'wait',
            'details': details,
            'strategy_type': 'transition',
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
        
        # Fetch fundamental data for filtering and additional data for copy report
        # Always try to get fundamental data, even if it fails
        fundamental_status = None
        extended_fundamental_data = {}  # Store additional data for copy report
        try:
            print(f"📊 DEBUG: Fetching fundamental data for ticker: {stock_code}")
            ticker_obj = yf.Ticker(stock_code)
            print(f"📊 DEBUG: Ticker object created, fetching info...")
            fundamental_status = get_fundamental_status(ticker_obj)
            print(f"📊 DEBUG: Fundamental status retrieved: {fundamental_status.get('status', 'unknown')}")
            
            # Fetch additional data for copy report
            try:
                info = ticker_obj.info
                
                # Market Cap
                market_cap = info.get('marketCap', info.get('enterpriseValue', None))
                extended_fundamental_data['market_cap'] = market_cap
                
                # 52-week high/low
                week_52_high = info.get('fiftyTwoWeekHigh', info.get('52WeekHigh', None))
                week_52_low = info.get('fiftyTwoWeekLow', info.get('52WeekLow', None))
                extended_fundamental_data['week_52_high'] = week_52_high
                extended_fundamental_data['week_52_low'] = week_52_low
                
                # Earnings date
                try:
                    # Try calendar first
                    calendar = ticker_obj.calendar
                    if calendar is not None and not calendar.empty:
                        # calendar is a DataFrame, get the first row's earnings date
                        if 'Earnings Date' in calendar.columns:
                            earnings_date = calendar['Earnings Date'].iloc[0]
                            if pd.notna(earnings_date):
                                if isinstance(earnings_date, pd.Timestamp):
                                    extended_fundamental_data['next_earnings'] = earnings_date.strftime('%Y-%m-%d')
                                else:
                                    extended_fundamental_data['next_earnings'] = str(earnings_date)
                            else:
                                extended_fundamental_data['next_earnings'] = None
                        else:
                            # Try to get from index if it's a datetime index
                            if isinstance(calendar.index, pd.DatetimeIndex) and len(calendar) > 0:
                                next_earnings_date = calendar.index[0]
                                extended_fundamental_data['next_earnings'] = next_earnings_date.strftime('%Y-%m-%d')
                            else:
                                extended_fundamental_data['next_earnings'] = None
                    else:
                        # Try alternative method - earnings_dates
                        try:
                            earnings_dates = ticker_obj.earnings_dates
                            if earnings_dates is not None and not earnings_dates.empty:
                                # Get the first future earnings date
                                now = pd.Timestamp.now()
                                future_dates = earnings_dates[earnings_dates.index > now]
                                if not future_dates.empty:
                                    next_earnings_date = future_dates.index[0]
                                    extended_fundamental_data['next_earnings'] = next_earnings_date.strftime('%Y-%m-%d')
                                else:
                                    extended_fundamental_data['next_earnings'] = None
                            else:
                                extended_fundamental_data['next_earnings'] = None
                        except:
                            extended_fundamental_data['next_earnings'] = None
                except Exception as earnings_error:
                    print(f"⚠️ Could not fetch earnings date: {earnings_error}")
                    extended_fundamental_data['next_earnings'] = None
                    
            except Exception as ext_error:
                print(f"⚠️ Could not fetch extended fundamental data: {ext_error}")
                
        except Exception as fund_error:
            # If fundamental data fetch fails, create a fallback status
            import traceback
            error_details = traceback.format_exc()
            fundamental_status = {
                'status': 'unknown',
                'trailing_pe': None,
                'forward_pe': None,
                'peg_ratio': None,
                'eps': None,
                'debt_to_equity': None,
                'profit_margins': None,
                'current_price': None,
                'quick_ratio': None,
                'current_ratio': None,
                'warnings': [f"無法獲取基本面數據：{str(fund_error)}"],
                'risk_level': 'medium',
                'red_flags': [],
                '_error_details': error_details
            }
            # Log the error but don't fail the entire analysis
            print(f"Warning: Failed to fetch fundamental data: {fund_error}")
        
        # Generate signal (with fundamental filters applied)
        signal = generate_trading_signal(df, fundamental_status)
        
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
            'fundamental_status': fundamental_status,
            'extended_fundamental_data': extended_fundamental_data,  # Additional data for copy report
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        # Even on error, try to include fundamental_status if possible
        fundamental_status = None
        try:
            ticker_obj = yf.Ticker(stock_code)
            fundamental_status = get_fundamental_status(ticker_obj)
        except:
            # If we can't get fundamental data, create a fallback
            fundamental_status = {
                'status': 'unknown',
                'trailing_pe': None,
                'forward_pe': None,
                'peg_ratio': None,
                'eps': None,
                'debt_to_equity': None,
                'profit_margins': None,
                'current_price': None,
                'quick_ratio': None,
                'current_ratio': None,
                'warnings': [f"無法獲取基本面數據：分析過程中發生錯誤"],
                'risk_level': 'medium',
                'red_flags': []
            }
        
        return {
            'success': False,
            'error': str(e),
            'fundamental_status': fundamental_status  # Include even on error
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
                        
                        # Metrics grid - 8 columns (added MFI and RVOL)
                        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5, stat_col6, stat_col7, stat_col8 = st.columns(8)
                        
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
                        
                        # MFI with color coding
                        mfi_val = details.get('mfi', 0)
                        mfi_color = "#dc2626" if mfi_val > 80 else "#16a34a" if mfi_val < 20 else "#1a1a1a"
                        with stat_col7:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>MFI</div><div style='color: {mfi_color}; font-size: 1.5rem; font-weight: 700;'>{mfi_val:.2f}</div></div>", unsafe_allow_html=True)
                        
                        # RVOL with color coding (Red/Bold if > 2.0)
                        rvol_val = details.get('rvol', 0)
                        rvol_color = "#dc2626" if rvol_val > 2.0 else "#1a1a1a"
                        rvol_weight = "700" if rvol_val > 2.0 else "700"
                        with stat_col8:
                            st.markdown(f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>RVOL</div><div style='color: {rvol_color}; font-size: 1.5rem; font-weight: {rvol_weight};'>{rvol_val:.2f}</div></div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Company Health Check Section
                        fundamental_status = result.get('fundamental_status')
                        # Always show the section if we have a successful result
                        # This ensures users can see the data or know when it's missing
                        if result.get('success', False):
                            # DEBUG: Log what we're displaying
                            if fundamental_status:
                                print(f"📊 DEBUG UI: Displaying fundamental_status with status: {fundamental_status.get('status', 'unknown')}")
                            else:
                                print(f"📊 DEBUG UI: fundamental_status is None or missing")
                            st.markdown("### 🏥 公司健康檢查")
                            st.markdown("---")
                            
                            # Create columns for fundamental metrics (expanded to show solvency metrics)
                            health_col1, health_col2, health_col3, health_col4, health_col5, health_col6 = st.columns(6)
                            
                            # Fundamental metrics - handle case when fundamental_status is None or missing
                            if fundamental_status:
                                trailing_pe = fundamental_status.get('trailing_pe')
                                forward_pe = fundamental_status.get('forward_pe')
                                peg_ratio = fundamental_status.get('peg_ratio')
                                eps = fundamental_status.get('eps')
                                debt_to_equity = fundamental_status.get('debt_to_equity')
                                profit_margins = fundamental_status.get('profit_margins')
                                fund_status = fundamental_status.get('status', 'unknown')
                                fund_risk = fundamental_status.get('risk_level', 'low')
                                
                                # Check if this is the known 2025 yfinance issue
                                is_known_issue = fundamental_status.get('_is_known_issue', False)
                                all_values_none = all(v is None for v in [trailing_pe, forward_pe, peg_ratio, eps, debt_to_equity, profit_margins])
                                
                                # Determine status color and icon (TOXIC gets highest priority)
                                if fund_status == 'toxic' or fund_risk == 'toxic':
                                    status_color = "#991b1b"  # Dark Red
                                    status_icon = "☠️"
                                    status_bg = "#fee2e2"  # Light red background
                                elif fund_risk == 'high':
                                    status_color = "#dc2626"  # Red
                                    status_icon = "🔴"
                                    status_bg = "#ffffff"
                                elif fund_risk == 'medium':
                                    status_color = "#f59e0b"  # Orange
                                    status_icon = "🟠"
                                    status_bg = "#ffffff"
                                else:
                                    status_color = "#16a34a"  # Green
                                    status_icon = "🟢"
                                    status_bg = "#ffffff"
                                
                                # Valuation metrics
                                with health_col1:
                                    pe_display = f"{trailing_pe:.2f}" if trailing_pe is not None else "N/A"
                                    pe_color = "#dc2626" if (trailing_pe is not None and trailing_pe > 50) else "#1a1a1a"
                                    st.markdown(
                                        f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>Trailing PE</div><div style='color: {pe_color}; font-size: 1.5rem; font-weight: 700;'>{pe_display}</div></div>",
                                        unsafe_allow_html=True
                                    )
                                
                                with health_col2:
                                    peg_display = f"{peg_ratio:.2f}" if peg_ratio is not None else "N/A"
                                    peg_color = "#dc2626" if (peg_ratio is not None and peg_ratio > 2) else "#1a1a1a"
                                    st.markdown(
                                        f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>PEG Ratio</div><div style='color: {peg_color}; font-size: 1.5rem; font-weight: 700;'>{peg_display}</div></div>",
                                        unsafe_allow_html=True
                                    )
                                
                                with health_col3:
                                    eps_display = f"{eps:.2f}" if eps is not None else "N/A"
                                    st.markdown(
                                        f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>EPS</div><div style='color: #1a1a1a; font-size: 1.5rem; font-weight: 700;'>{eps_display}</div></div>",
                                        unsafe_allow_html=True
                                    )
                                
                                # Solvency metrics (NEW - Priority Display)
                                with health_col4:
                                    debt_display = f"{debt_to_equity:.1f}" if debt_to_equity is not None else "N/A"
                                    # Check if debt is extreme (> 200 or > 2.0)
                                    debt_is_extreme = False
                                    if debt_to_equity is not None:
                                        debt_val = float(debt_to_equity)
                                        debt_is_extreme = debt_val > 200 or (debt_val > 2.0 and debt_val <= 100)
                                    debt_color = "#dc2626" if debt_is_extreme else "#1a1a1a"
                                    st.markdown(
                                        f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>負債權益比</div><div style='color: {debt_color}; font-size: 1.5rem; font-weight: 700;'>{debt_display}</div></div>",
                                        unsafe_allow_html=True
                                    )
                                
                                with health_col5:
                                    profit_display = f"{profit_margins*100:.1f}%" if profit_margins is not None else "N/A"
                                    profit_color = "#dc2626" if (profit_margins is not None and profit_margins < -0.10) else "#16a34a" if (profit_margins is not None and profit_margins > 0) else "#1a1a1a"
                                    st.markdown(
                                        f"<div style='text-align: center;'><div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>利潤率</div><div style='color: {profit_color}; font-size: 1.5rem; font-weight: 700;'>{profit_display}</div></div>",
                                        unsafe_allow_html=True
                                    )
                                
                                with health_col6:
                                    status_text = {
                                        'healthy': '健康',
                                        'overvalued': '估值偏高',
                                        'unprofitable': '虧損',
                                        'toxic': '☠️ TOXIC / 高風險',
                                        'unknown': '未知'
                                    }.get(fund_status, '未知')
                                    
                                    # Special styling for TOXIC status
                                    if fund_status == 'toxic' or fund_risk == 'toxic':
                                        status_html = f"""
                                        <div style='text-align: center; background-color: {status_bg}; border: 2px solid {status_color}; padding: 0.5rem; border-radius: 4px;'>
                                            <div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>基本面狀態</div>
                                            <div style='color: {status_color}; font-size: 1.5rem; font-weight: 900; text-transform: uppercase;'>{status_icon} {status_text}</div>
                                        </div>
                                        """
                                    else:
                                        status_html = f"""
                                        <div style='text-align: center;'>
                                            <div style='color: #6b7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem;'>基本面狀態</div>
                                            <div style='color: {status_color}; font-size: 1.5rem; font-weight: 700;'>{status_icon} {status_text}</div>
                                        </div>
                                        """
                                    st.markdown(status_html, unsafe_allow_html=True)
                                
                                # Display warnings if any (TOXIC warnings get special treatment)
                                warnings = fundamental_status.get('warnings', [])
                                
                                # If all values are None and it's the known issue, show special message
                                if all_values_none and (is_known_issue or len(warnings) > 0):
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.error("""
                                    **⚠️ 所有基本面數據顯示為 N/A**
                                    
                                    這是 yfinance 庫的已知問題（2025年）。Yahoo Finance 更改了其 API 結構，
                                    導致 yfinance 無法正確解析基本面數據字段（P/E、PEG、負債權益比等）。
                                    
                                    **影響範圍：**
                                    - 所有使用 yfinance 獲取基本面數據的應用
                                    - 價格和交易數據仍然正常
                                    - 僅基本面財務比率受影響
                                    
                                    **臨時解決方案：**
                                    - 手動訪問 [Yahoo Finance](https://finance.yahoo.com) 查看基本面數據
                                    - 等待 yfinance 庫更新修復此問題
                                    """)
                                
                                if warnings:
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    for warning in warnings:
                                        if fund_status == 'toxic' or fund_risk == 'toxic':
                                            # Use error styling for TOXIC warnings
                                            st.error(warning)
                                        else:
                                            st.warning(warning)
                            else:
                                # Handle case when fundamental_status is None or missing
                                st.error("⚠️ **無法獲取基本面數據**")
                                st.warning("""
                                **已知問題（2025）：** yfinance 庫目前存在已知問題，無法正常獲取 Yahoo Finance 的基本面數據。
                                
                                受影響的數據包括：
                                - P/E 比率 (Trailing PE, Forward PE)
                                - PEG 比率
                                - 負債權益比 (Debt-to-Equity)
                                - 利潤率 (Profit Margins)
                                - EPS (Earnings Per Share)
                                - 流動比率 (Quick Ratio, Current Ratio)
                                
                                **原因：** Yahoo Finance 更改了其 API 結構，導致 yfinance 無法正確解析這些數據字段。
                                雖然數據在 Yahoo Finance 網站上仍然可用，但 yfinance 庫目前無法檢索它們。
                                """)
                                
                                st.info("""
                                **臨時解決方案：**
                                1. 手動訪問 Yahoo Finance 網站查看基本面數據
                                2. 使用其他數據源（如 Alpha Vantage、Quandl 等）
                                3. 等待 yfinance 庫更新修復此問題
                                4. 檢查終端/控制台輸出以查看詳細調試信息
                                """)
                                
                                # Show debug info if available
                                with st.expander("🔍 調試信息 (Debug Info)", expanded=False):
                                    st.code(f"Stock Code: {result.get('stock_code', 'N/A')}")
                                    st.code(f"Fundamental Status: {fundamental_status}")
                                    st.markdown("**技術細節：**")
                                    st.markdown("""
                                    這是一個已知的 yfinance 庫問題（2025年）。Yahoo Finance 更改了其 API 結構，
                                    導致 yfinance 無法正確解析基本面數據字段。雖然價格和交易數據仍然可以正常獲取，
                                    但財務比率和基本面指標目前無法通過 yfinance 獲取。
                                    
                                    **相關問題：**
                                    - GitHub Issue: yfinance 無法獲取 PEG 比率、P/E 比率等基本面數據
                                    - 影響範圍：所有使用 yfinance 獲取基本面數據的應用
                                    - 狀態：等待 yfinance 庫維護者修復
                                    """)
                            
                            st.markdown("---")
                        
                        # One-Click Copy Section for AI Consultation
                        with st.expander("📋 **複製報告給 AI 分析**", expanded=False):
                            # Format the data summary string
                            ticker = result.get('stock_code', 'N/A')
                            stock_name = result.get('stock_name', 'N/A')
                            current_price_val = result.get('current_price', 0)
                            price_change_val = result.get('price_change')
                            price_change_pct = result.get('price_change_percent')
                            
                            # Format price change
                            if price_change_val is not None and price_change_pct is not None:
                                if price_change_val > 0:
                                    change_str = f"+{price_change_val:.2f} (+{price_change_pct:.2f}%)"
                                elif price_change_val < 0:
                                    change_str = f"{price_change_val:.2f} ({price_change_pct:.2f}%)"
                                else:
                                    change_str = "0.00 (0.00%)"
                            else:
                                change_str = "N/A"
                            
                            # Technical data
                            rsi_val = details.get('rsi', 0)
                            adx_val = details.get('adx', 0)
                            adx_slope_val = details.get('adx_slope', 0)
                            pdi_val = details.get('dmi_plus', 0)
                            mdi_val = details.get('dmi_minus', 0)
                            pdi_mdi_gap = abs(pdi_val - mdi_val)
                            atr_val = details.get('atr', 0)
                            bb_upper_val = details.get('bb_upper', 0)
                            bb_lower_val = details.get('bb_lower', 0)
                            bb_middle_val = details.get('bb_middle', 0)
                            mfi_val = details.get('mfi', 0)
                            rvol_val = details.get('rvol', 0)
                            
                            # Get SMA 50 and SMA 200 from details (calculated in generate_trading_signal)
                            sma_50_val = details.get('sma_50', None)
                            sma_200_val = details.get('sma_200', None)
                            
                            # Fundamental data
                            fundamental_status = result.get('fundamental_status', {})
                            extended_data = result.get('extended_fundamental_data', {})
                            
                            trailing_pe = fundamental_status.get('trailing_pe') if fundamental_status else None
                            forward_pe = fundamental_status.get('forward_pe') if fundamental_status else None
                            peg_ratio = fundamental_status.get('peg_ratio') if fundamental_status else None
                            debt_to_equity = fundamental_status.get('debt_to_equity') if fundamental_status else None
                            profit_margins = fundamental_status.get('profit_margins') if fundamental_status else None
                            
                            market_cap = extended_data.get('market_cap', None)
                            week_52_high = extended_data.get('week_52_high', None)
                            week_52_low = extended_data.get('week_52_low', None)
                            next_earnings = extended_data.get('next_earnings', None)
                            
                            # Format market cap
                            if market_cap is not None:
                                if market_cap >= 1e12:
                                    market_cap_str = f"{market_cap/1e12:.2f}T"
                                elif market_cap >= 1e9:
                                    market_cap_str = f"{market_cap/1e9:.2f}B"
                                elif market_cap >= 1e6:
                                    market_cap_str = f"{market_cap/1e6:.2f}M"
                                else:
                                    market_cap_str = f"{market_cap:.2f}"
                            else:
                                market_cap_str = "N/A"
                            
                            # Format profit margins as percentage
                            if profit_margins is not None:
                                profit_margins_str = f"{profit_margins*100:.2f}%"
                            else:
                                profit_margins_str = "N/A"
                            
                            # Signal and analysis
                            signal_advice = signal.get('advice', '無訊號') if signal else '無訊號'
                            signal_reason = signal.get('commentary', signal.get('reason', '')) if signal else ''
                            
                            # Format values for display (handle None cases)
                            sma_200_str = f"{sma_200_val:.2f}" if sma_200_val is not None else "N/A"
                            sma_50_str = f"{sma_50_val:.2f}" if sma_50_val is not None else "N/A"
                            week_52_low_str = f"{week_52_low:.2f}" if week_52_low is not None else "N/A"
                            week_52_high_str = f"{week_52_high:.2f}" if week_52_high is not None else "N/A"
                            trailing_pe_str = f"{trailing_pe:.2f}" if trailing_pe is not None else "N/A"
                            forward_pe_str = f"{forward_pe:.2f}" if forward_pe is not None else "N/A"
                            peg_ratio_str = f"{peg_ratio:.2f}" if peg_ratio is not None else "N/A"
                            debt_to_equity_str = f"{debt_to_equity:.2f}" if debt_to_equity is not None else "N/A"
                            next_earnings_str = next_earnings if next_earnings else "N/A"
                            
                            # Format the enhanced summary string
                            summary_text = f"""Analyze this stock for me: {ticker} ({stock_name})
Price: {current_price_val:.2f} ({change_str})

[Technical Structure]
RSI: {rsi_val:.2f}
ADX: {adx_val:.2f} (Slope: {adx_slope_val:.2f})
PDI: {pdi_val:.2f} | MDI: {mdi_val:.2f} (Gap: {pdi_mdi_gap:.2f})
ATR: {atr_val:.2f}
Bollinger: Up {bb_upper_val:.2f} | Low {bb_lower_val:.2f} | Mid {bb_middle_val:.2f}
SMA 200: {sma_200_str} | SMA 50: {sma_50_str}
52W Range: {week_52_low_str} - {week_52_high_str}

[Fundamental Health]
Market Cap: {market_cap_str}
PE (Trail/Fwd): {trailing_pe_str} / {forward_pe_str}
PEG: {peg_ratio_str}
Profit Margin: {profit_margins_str}
Debt/Eq: {debt_to_equity_str}

[Risk Check]
Next Earnings: {next_earnings_str}
RVOL: {rvol_val:.2f}
MFI: {mfi_val:.2f}

[Robot Signal]
{signal_advice}
{signal_reason if signal_reason else 'No additional signal details'}"""
                            
                            # Display in code block with copy button
                            st.code(summary_text, language='markdown')
                    
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
                            # Custom styled wait badge with better contrast (dark text on light orange background)
                            st.markdown(
                                f'<div style="background-color: #fef3c7; border: 1px solid #f59e0b; border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">'
                                f'<div style="color: #92400e; font-weight: 600; font-size: 1rem;">⚠️ <strong>交易訊號：</strong> {advice_text}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
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
                            if strategy_type == 'explosive_breakout':
                                strategy_badge = '<span style="background-color: #fef3c7; color: #dc2626; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; margin-left: 0.5rem; border: 2px solid #dc2626;">🚀 爆炸性突破</span>'
                            elif strategy_type == 'trend_following':
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
