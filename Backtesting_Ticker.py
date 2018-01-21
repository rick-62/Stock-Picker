from ticker_data import Ticker
import datetime
import pandas as pd


class Backtest_Ticker(Ticker):

    # next date can be cycled through each week-date using generator
    start_date = '2017-01-01'
    _date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()

    def __init__(self, name):
        super().__init__(name)
        self.Close_df = self._combine_close_index()
        self.Low_df = self._combine_low_index()
        self.Open_df = self._combine_open_index()

    def _combine_close_index(self):
        """Combines Close np and Index np for later reference"""
        return pd.DataFrame(index=self.Index, data={'Close': self.Close_raw})

    def _combine_low_index(self):
        """Combines Close np and Index np for later reference"""
        return pd.DataFrame(index=self.Index, data={'Low': self.Low_raw})

    def _combine_open_index(self):
        """Combines Close np and Index np for later reference"""
        return pd.DataFrame(index=self.Index, data={'Open': self.Open_raw})

    def update_Xy(self, date, modeling=True, n=None):
        """Updates date to next date and changes X and y"""
        X = self.X.ix[:date]
        y = self.y.ix[:date]
        if modeling:
            n = -1
        return X.ix[:n], y.ix[:n]

    def get_latest_price(self, date):
        """Gets latest price from raw Close"""
        return self.Close_df.ix[:date].tail(1).values[0][0]

    def get_latest_low(self, date):
        """Gets latest price from raw Low"""
        return self.Low_df.ix[:date].tail(1).values[0][0]

    def get_latest_open(self, date):
        """Gets latest price from raw Open"""
        return self.Open_df.ix[:date].tail(1).values[0][0]

    @staticmethod
    def next_date(previous_date):
        """Cycles through to the next day and checks today's date"""
        if previous_date.weekday() > 3:
            days = 7 - previous_date.weekday()
        else:
            days = 1
        _next_date = previous_date + datetime.timedelta(days=days)
        if previous_date == datetime.date.today():
            Backtest_Ticker.complete()
            _next_date = False
        return _next_date

    @staticmethod
    def complete():
        print("All dates complete")



