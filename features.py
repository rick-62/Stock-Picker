import numpy as np
import pandas as pd
import warnings
import scipy.ndimage
import datetime as dt
np.seterr(invalid='ignore')


def suppress_runtime_warning():
    def decorate(func):
        def call(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = func(*args, **kwargs)
            return result
        return call
    return decorate


def rsi_func(close, n):
    """Relative Strength Index: Compares magnitude of recent gains and losses"""

    # convert close to float
    close = close.astype(float)

    # calc difference between adjacent prices
    deltas = np.diff(close)

    # seed is a list of first n differences
    seed = deltas[: n + 1]

    # separately sum increases & decreases and average over n
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n

    # proportion of increases to decreases
    if down == 0:
        rs = 1
    else:
        rs = up / down

        # create list of length prices filled with zeros
    rsi = np.zeros_like(close)

    # first n values calculated using this formula
    rsi[:n] = 100. - 100. / (1. + rs)

    # calculate remaining rsi values on at a time
    for i in range(n, len(close)):
        delta = deltas[i - 1]  # next delta in delta list

        # determine if increase or decrease
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        # sum with new values and create new average
        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        # proportion of increases to decreases
        if down == 0 or np.isnan(down):
            rs = 1
        else:
            rs = up / down

        # next rsi calculated
        rsi[i] = 100. - 100. / (1. + rs)

    # replaces first n values with nan alternatives
    rsi[:n] = np.nan

    return rsi


def cci_func(high, low, close, win):
    """Commodity Channel Index: compares current price to average"""
    pt = np.divide(high + low + close, 3)
    sma = _simple_moving_average(pt, win)
    tpt = _simple_moving_std(pt, win)
    cci = (pt - sma) / (0.015 * tpt)
    return cci

@suppress_runtime_warning()
def stok_func(high, low, close, n):
    """Stochastic Oscillator"""
    high_win = _simple_moving_max(high, n)
    low_win = _simple_moving_min(low, n)
    k = np.divide(close - low_win, np.subtract(high_win, low_win))
    return 100 * k

def obv_func(close, volume):
    """On-Balance Volume"""
    obv = [0]
    deltas = np.diff(close)
    for i, d in enumerate(deltas):
        if d == 0:
            obv.append(obv[-1])
        elif d > 0:
            obv.append(obv[-1] + volume[i+1])
        elif d < 0:
            obv.append(obv[-1] - volume[i+1])
    return np.array(obv)


def obv_ma_func(obv, n):
    """On-Balance Volume Moving Average"""
    return _simple_moving_average(obv, n)

@suppress_runtime_warning()
def williams_func(close, high, low, n):
    """Returns Williams %R"""
    low_n = _rolling_min(low, n, dir=1)
    high_n = _rolling_max(high, n, dir=1)
    return -100 * (high_n - close) / (high_n - low_n)


def moneyflow_func(high, low, close, volume, n):
    """Returns Money Flow Index based on RSI"""
    pt = np.divide(high + low + close, 3)
    mfr = np.multiply(pt, volume)
    return rsi_func(mfr, n)


def ppo_func(close, n, m):
    """Returns Percentage Price Oscillator using EMA"""
    ppo = np.multiply(np.divide(_ema(close, n) - _ema(close, m),
                                _ema(close, m)), 100)
    return ppo


def relative_ema_func(close, n):
    """Returns EMA relative to price"""
    multiplier = 2 / (n + 1)
    shifted_ema = _shift(_ema(close, n), 1, len(close))
    return np.multiply(close - shifted_ema, multiplier) + shifted_ema


def emacd_func(close, n, m):
    """Calculates Exponential Moving Average Difference"""
    ema1 = _ema(close, n)
    ema2 = _ema(close, m)
    return 100 * (ema1 - ema2) / ema2


def proc_func(close, n):
    """Price Rate of Change"""
    shifted_close = _shift(close, n, len(close))
    return np.multiply(
        np.divide(close - shifted_close, shifted_close), 100)


def weekday_func(dates):
    """Day of the week as an integer"""
    weekdays = pd.DataFrame(dates)
    return np.array(weekdays[0].dt.dayofweek.astype(float))


def create_y(a, close, threshold, polarity, days):
    if polarity == 'neg':
        return _diff(num=_rolling_min(a, days), den=close) < threshold
    elif polarity == 'pos':
        return _diff(num=_rolling_max(a, days), den=close) > threshold
    else:
        raise NameError


def count_true(y):
    """Counts number of instances where value went over the threshold"""
    return np.sum(y)


def gaussian_filter(a, sigma):
    """Smoothing filter applied to raw stock data"""
    return scipy.ndimage.gaussian_filter(a, sigma)


def _rolling_window(a, win):
    """Efficiently creates rolling arrays to perform functions on"""
    shape = a.shape[:-1] + (a.shape[-1] - win + 1, win)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _simple_moving_average(a, win):
    """Uses the rolling window function to return rolling mean"""
    return _shift(np.mean(_rolling_window(a, win), axis=1), win, len(a))


def _ema(a, win):
    """Calculates Exponential Moving Average"""
    weights = np.exp(np.linspace(-1., 0., win))
    weights /= weights.sum()
    ema = np.convolve(a, weights)[:len(a)]
    ema[:win] = ema[win]
    return ema


def _simple_moving_std(a, win):
    """Uses the rolling window function to return rolling std"""
    return _shift(np.std(_rolling_window(a, win), axis=1), win, len(a))


def _simple_moving_max(a, win):
    """Uses the rolling window function to return rolling max"""
    return _shift(np.max(_rolling_window(a, win), axis=1), win, len(a))


def _simple_moving_min(a, win):
    """Uses the rolling window function to return rolling max"""
    return _shift(np.min(_rolling_window(a, win), axis=1), win, len(a))


def _rolling_max(a, days, dir=-1):
    """Uses the rolling window function to return rolling maximum"""
    shift = dir * days
    return _shift(np.amax(_rolling_window(a, days), axis=1), shift, len(a))


def _rolling_min(a, days, dir=-1):
    """Uses the rolling window function to return rolling minimum"""
    shift = dir * days
    return _shift(np.amin(_rolling_window(a, days), axis=1), shift, len(a))


def _shift(arr, num, len_, fill_value=np.nan):
    """specific for padding/shifting days and windows"""
    result = np.empty(len_)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:len_-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-len_-num:]
    else:
        result = arr
    return result


def _diff(num, den):
    """calculates % diff between two series and returns series"""
    return 100 * (num - den) / den
