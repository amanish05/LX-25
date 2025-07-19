"""
Mock TA-Lib implementation for testing
Provides basic implementations of commonly used indicators
"""

import numpy as np
import pandas as pd


def RSI(close, timeperiod=14):
    """Calculate RSI"""
    close = np.array(close)
    deltas = np.diff(close)
    seed = deltas[:timeperiod+1]
    up = seed[seed >= 0].sum() / timeperiod
    down = -seed[seed < 0].sum() / timeperiod
    rs = up / down
    rsi = np.zeros_like(close)
    rsi[:timeperiod] = np.nan
    rsi[timeperiod] = 100. - 100. / (1. + rs)
    
    for i in range(timeperiod + 1, len(close)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (timeperiod - 1) + upval) / timeperiod
        down = (down * (timeperiod - 1) + downval) / timeperiod
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi


def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """Calculate MACD"""
    close = pd.Series(close)
    exp1 = close.ewm(span=fastperiod, adjust=False).mean()
    exp2 = close.ewm(span=slowperiod, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values


def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
    """Calculate Stochastic"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=slowd_period).mean()
    
    return k_percent.values, d_percent.values


def CCI(high, low, close, timeperiod=20):
    """Calculate CCI"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    tp = (high + low + close) / 3
    sma = tp.rolling(window=timeperiod).mean()
    mad = tp.rolling(window=timeperiod).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - sma) / (0.015 * mad)
    
    return cci.values


def WILLR(high, low, close, timeperiod=14):
    """Calculate Williams %R"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    highest_high = high.rolling(window=timeperiod).max()
    lowest_low = low.rolling(window=timeperiod).min()
    
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    return wr.values


def MOM(close, timeperiod=10):
    """Calculate Momentum"""
    close = pd.Series(close)
    mom = close.diff(timeperiod)
    return mom.values


def ROC(close, timeperiod=12):
    """Calculate Rate of Change"""
    close = pd.Series(close)
    roc = ((close - close.shift(timeperiod)) / close.shift(timeperiod)) * 100
    return roc.values


def ATR(high, low, close, timeperiod=14):
    """Calculate ATR"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=timeperiod).mean()
    
    return atr.values


def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Calculate Bollinger Bands"""
    close = pd.Series(close)
    
    middle = close.rolling(window=timeperiod).mean()
    std = close.rolling(window=timeperiod).std()
    
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)
    
    return upper.values, middle.values, lower.values


def EMA(close, timeperiod=30):
    """Calculate EMA"""
    close = pd.Series(close)
    return close.ewm(span=timeperiod, adjust=False).mean().values


def SMA(close, timeperiod=30):
    """Calculate SMA"""
    close = pd.Series(close)
    return close.rolling(window=timeperiod).mean().values