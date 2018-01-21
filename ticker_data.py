import pandas as pd
import numpy as np
import features
import pickle


def list_of_tickers(location):
    """Reads list of tickers from specified location"""
    with open(location, 'rb') as f:
        return pickle.load(f)


class Ticker:

    imported_tickers_loc = 'imported_tickers'
    price_data_path = 'stock_dfs\{ticker}.csv'
    lst_tickers = list_of_tickers(imported_tickers_loc)
    lst_features = None
    threshold = 3
    series = 'Close'  # or 'High'/'Low'
    polarity = 'pos'
    sigma = 1  # for smoothing of input data
    day_threshold = 0.95

    def __init__(self, name):
        self.error = False  # if any errors occur whilst processing data
        self.name = name
        self.price_data_loc = Ticker.price_data_path.format(ticker=name)
        self.price_data = self._get_price_data()
        self._check_data()
        self.y = None
        self.X = None
        self.rsi_n = 14
        self.cci_win = 50
        self.stok_n = 14
        self.obv_n = 14
        self.william_n = 14
        self.moneyflow_n = 14
        self.ppo_n, self.ppo_m = 9, 26
        self.ema_n = 14
        self.ema1_n, self.ema2_m = 7, 50
        self.proc_n = 14
        self.days = 5
        if not self.error:
            self.X = pd.DataFrame(index=self.Index)
            self.create_features()

    def _check_data(self):
        """Checks some basic metrics on input data to clear rubbish"""
        self.error = \
            self.rows_input < 1000 or \
            np.sum(self.price_data['Volume'][-100:] == 0) > 10
        return

    # change volume to count of 0 on one day
    # remove Close as Volume tells us that something has happened during the day

    def _get_price_data(self):
        """tries to get imported ticker data"""
        try:
            data = pd.read_csv(self.price_data_loc,
                               index_col=0, parse_dates=True)
            self._split_price_data(data)
            self._smooth_price_data(self.sigma)
        except FileNotFoundError:
            self.error = True
            data = None
        return data

    def _split_price_data(self, price_data):
        """splits imported price data into separate arrays"""
        self.High_raw = np.array(price_data['High'])
        self.Low_raw = np.array(price_data['Low'])
        self.Close_raw = np.array(price_data['Close'])
        self.Open_raw = np.array(price_data['Open'])
        self.Volume_raw = np.array(price_data['Volume'])
        self.Index = np.array(price_data.index)

    def _smooth_price_data(self, sigma):
        """Smooths input data according to sigma - only need for model"""
        self.High = features.gaussian_filter(self.High_raw, sigma)
        self.Low = features.gaussian_filter(self.Low_raw, sigma)
        self.Close = features.gaussian_filter(self.Close_raw, sigma)
        self.Open = features.gaussian_filter(self.Open_raw, sigma)
        self.Volume = features.gaussian_filter(self.Volume_raw, sigma)

    @property
    def rows_input(self):
        if not self.error:
            rows_ = len(self.price_data)
        else:
            rows_ = None
        return rows_

    @property
    def rows_X(self):
        if not self.error:
            rows_ = len(self.X)
        else:
            rows_ = None
        return rows_

    @property
    def count_true(self):
        """Counts TRUEs in y array"""
        if not self.error:
            count = features.count_true(self.y)
        else:
            count = None
        return count

    def latest(self, *args):
        """gets the latest feature data"""
        if len(args) == 0:
            return self.X.tail(1)
        else:
            return args[0].tail(1)

    def create_y(self, **kwargs):
        """calls function and saves as attribute separate from features"""
        self.__dict__.update(kwargs)
        y = features.create_y(getattr(self, self.series),
                              self.Close,
                              self.threshold,
                              self.polarity,
                              self.days)
        self.y = pd.DataFrame(index=self.Index, data={'y': y})
        return self.y

    def create_features(self):
        """Initiates creation of features and exports dataframe"""
        try:
            self.rsi_func(self.rsi_n)
            self.cci_func(self.cci_win)
            self.stok_func(self.stok_n)
            self.obv_func(self.obv_n)
            self.williams_func(self.william_n)
            self.moneyflow_func(self.moneyflow_n)
            self.ppo_func(self.ppo_n, self.ppo_m)
            self.relative_ema_func(self.ema_n)
            self.emacd_func(self.ema1_n, self.ema2_m)
            self.proc_func(self.proc_n)
            self.weekday_func()
            self.create_y()
        except (TypeError, RuntimeWarning, ValueError):
            self.X = None
            self.error = True
        return

    def rsi_func(self, n):
        """calls rsi function and appends to feature dataframe"""
        rsi = features.rsi_func(self.Close, n)
        self.X['RSI'] = rsi
        return rsi

    def cci_func(self, win):
        """calls cci function and appends to feature dataframe"""
        cci = features.cci_func(self.High, self.Low, self.Close, win)
        self.X['CCI'] = cci
        return cci

    def stok_func(self, win):
        """calls stok_func and appends to feature dataframe"""
        stok = features.stok_func(self.High, self.Low, self.Close, win)
        self.X['STOK'] = stok
        return stok

    def obv_func(self, n):
        """calls obv_func and ma and appends to feature dataframe"""
        obv = features.obv_func(self.Close, self.Volume)
        self.X['OBV'] = obv
        obv_ma = features.obv_ma_func(obv, self.obv_n)
        self.X['OBV_MA'] = obv_ma
        return obv, obv_ma

    def williams_func(self, n):
        """calls williams_func and appends to feature dataframe"""
        r = features.williams_func(self.Close, self.High, self.Low, n)
        self.X['Williams_R'] = r
        return r

    def moneyflow_func(self, n):
        """calls moneyflow_func and appends to feature dataframe"""
        mfi = features.moneyflow_func(self.High,
                                      self.Low,
                                      self.Close,
                                      self.Volume,
                                      self.moneyflow_n)
        self.X['MFI'] = mfi
        return mfi


    def ppo_func(self, n, m):
        """calls ppo_func and appends to feature dataframe"""
        ppo = features.ppo_func(self.Close, self.ppo_n, self.ppo_m)
        self.X['PPO'] = ppo
        return ppo

    def relative_ema_func(self, n):
        """calls relative_ema_func and appends to feature dataframe"""
        rel_ema = features.relative_ema_func(self.Close, n)
        self.X['Relative_EMA'] = rel_ema
        return rel_ema

    def emacd_func(self, n, m):
        """calls emacd_func and appends to feature dataframe"""
        emacd = features.emacd_func(self.Close, n, m)
        self.X['EMACD'] = emacd
        return emacd

    def proc_func(self, n):
        """calls proc_func and appends to feature dataframe"""
        proc = features.proc_func(self.Close, n)
        self.X['PROC'] = proc
        return proc

    def weekday_func(self):
        """calls weekday function and appends to feature dataframe"""
        weekday = features.weekday_func(self.Index)
        self.X['WeekDay'] = weekday
        return weekday

    def __repr__(self):
        return self.name

