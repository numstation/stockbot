"""
SCSP神器 - Web Application
Flask web interface for the mean-reversion trading strategy
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import ta
import yfinance as yf
import sys
import os
from datetime import datetime

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

app = Flask(__name__)


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
        if adx_value > 30:
            trend_desc = "強勢趨勢"
        elif adx_value < 25:
            trend_desc = "弱勢趨勢 / 橫盤整理"
        else:
            trend_desc = "中等趨勢"
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


def get_analysis_text(df):
    """
    Smart Analyst Commentary - Explains the "Why" behind the market status and signals.
    Returns detailed commentary in Traditional Chinese.
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
    
    commentary_parts = []
    
    # 1. Trend Analysis with Emoji
    if pd.notna(current_adx) and pd.notna(pdi) and pd.notna(mdi):
        adx_val = float(current_adx)
        pdi_val = float(pdi)
        mdi_val = float(mdi)
        
        if adx_val > 30:
            if pdi_val > mdi_val:
                commentary_parts.append("🚀 **趨勢：強勢上升趨勢**")
                commentary_parts.append("市場呈現強勁的多頭動能，上升趨勢明確且持續。")
            else:
                commentary_parts.append("📉 **趨勢：強勢下降趨勢**")
                commentary_parts.append("市場呈現強勁的空頭動能，下降趨勢明確且持續。")
        elif adx_val < 25:
            commentary_parts.append("📊 **趨勢：橫盤整理 / 弱勢趨勢**")
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
    
    # 3. Action Explanation (will be enhanced by signal generation)
    commentary_parts.append("")
    commentary_parts.append("💡 **策略建議：**")
    
    return "\n\n".join(commentary_parts)


def generate_trading_signal(df):
    """
    Generate trading signal with Trend-Following and Mean-Reversion strategies.
    
    Scenarios:
    A: RANGE MARKET (ADX < 25) -> Mean Reversion
    B: STRONG UPTREND (ADX > 30 & PDI > MDI) -> Trend Following (Short Put)
    C: STRONG DOWNTREND (ADX > 30 & MDI > PDI) -> Trend Following (Short Call)
    D: TRANSITION (ADX 25-30) -> Wait/Caution
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
    
    # Get base commentary
    base_commentary = get_analysis_text(df)
    commentary = base_commentary
    
    # SCENARIO B: STRONG UPTREND (ADX > 30 & PDI > MDI) -> Trend Following
    if current_adx > 30 and pd.notna(pdi) and pd.notna(mdi) and pdi > mdi:
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
    
    # SCENARIO C: STRONG DOWNTREND (ADX > 30 & MDI > PDI) -> Trend Following
    if current_adx > 30 and pd.notna(pdi) and pd.notna(mdi) and mdi > pdi:
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
    
    # SCENARIO D: TRANSITION (ADX between 25-30) -> Wait/Caution
    if 25 <= current_adx <= 30:
        commentary += "\n\n⚠️ **策略：等待 / 謹慎觀察**"
        commentary += "\n市場處於趨勢轉換期，ADX 在 25-30 之間，建議等待更明確的信號。"
        commentary += "\n**理由：** 趨勢強度中等，方向可能轉換，此時交易風險較高。"
        
        return {
            'advice': '☕ 等待：趨勢轉換期，建議謹慎觀察',
            'signal_type': 'wait',
            'details': details,
            'strategy_type': 'transition',
            'commentary': commentary
        }
    
    # SCENARIO A: RANGE MARKET (ADX < 25) -> Mean Reversion (Original Logic)
    if current_adx < 25:
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
    
    # Default: NO ACTION
    commentary += "\n\n☕ **策略：等待**"
    commentary += "\n目前無明確的交易訊號，建議繼續觀察市場變化。"
    
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
        # Yahoo Finance typically names the index 'Date' after reset_index
        if 'Date' in df.columns:
            df['time'] = df['Date']
        elif len(df.columns) > 0 and isinstance(df.columns[0], str) and 'date' in df.columns[0].lower():
            # Handle case where column might be named differently
            df['time'] = df[df.columns[0]]
        else:
            # Fallback: use index as time
            df['time'] = df.index
        
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
        
        # Prepare price history for Bollinger Bands chart (last 50 days)
        price_history = df.tail(50).copy()
        
        # Format dates for chart (extract date part if datetime)
        dates = []
        if 'time' in price_history.columns:
            for dt in price_history['time']:
                if pd.notna(dt):
                    # Convert to string, extract date part if it's a datetime
                    dt_str = str(dt)
                    if ' ' in dt_str:
                        dt_str = dt_str.split(' ')[0]  # Get date part only
                    dates.append(dt_str)
                else:
                    dates.append('')
        else:
            dates = [f'Day {i+1}' for i in range(len(price_history))]
        
        chart_data = {
            'dates': dates,
            'close_prices': [float(x) for x in price_history['close'].tolist() if pd.notna(x)],
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


@app.route('/')
def index():
    """Main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}", 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': VERSION,
        'message': 'SCSP神器 is running'
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze stock endpoint."""
    data = request.get_json()
    input_code = data.get('stock_code', '').strip()
    
    if not input_code:
        return jsonify({
            'success': False,
            'error': 'Please enter a stock code'
        })
    
    # Normalize the stock code (e.g., "700" -> "HK.00700", "AAPL" -> "US.AAPL")
    stock_code = normalize_stock_code(input_code)
    
    result = analyze_stock(stock_code, original_input=input_code)
    return jsonify(result)


if __name__ == '__main__':
    import socket
    
    # Check if port 5000 is available
    port = 5000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        # Port is in use, try to find and kill the process
        import subprocess
        try:
            # Find process using port 5000
            result = subprocess.run(['lsof', '-ti:5000'], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"⚠️  發現端口 5000 被佔用，正在清理進程: {', '.join(pids)}")
                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], check=False)
                    except:
                        pass
                import time
                time.sleep(2)
        except:
            pass
    
    print("═══════════════════════════════════════════════════════════")
    print("           SCSP神器 - 交易策略分析器")
    print(f"           版本: {VERSION}")
    print("═══════════════════════════════════════════════════════════")
    print("🚀 正在啟動 Web 應用程式...")
    print("📱 請在瀏覽器中打開: http://127.0.0.1:5000")
    print("📊 數據來源: Yahoo Finance")
    print("═══════════════════════════════════════════════════════════")
    print("")
    
    # Try to run on port 5000, if it fails, try 5001
    try:
        print("🌐 啟動 Flask 伺服器...")
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️  端口 5000 被佔用，嘗試使用端口 5001...")
            print("📱 請在瀏覽器中打開: http://127.0.0.1:5001")
            app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False, threaded=True)
        else:
            print(f"❌ 錯誤: {e}")
            raise
