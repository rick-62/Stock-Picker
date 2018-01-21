import requests
import shutil
import pickle
import time
import pandas as pd
import pandas_datareader.data as web


source = 'http://www.londonstockexchange.com/statistics/companies-and-issuers/' \
         'instruments-defined-by-mifir-identifiers-list-on-lse.xlsx'
file_name = 'lse_company_data.xlsx'
output_name = 'tickers_for_import.csv'
reserved_words = "CON, PRN, AUX, CLOCK$, NUL, COM0, COM1, COM2, COM3, COM4, COM5, " \
                 "COM6, COM7, COM8, COM9, LPT0, LPT1, LPT2, LPT3, LPT4, LPT5, LPT6, " \
                 "LPT7, LPT8, LPT9".split(', ')


def _convert_cap_to_size(x):
    """filter to convert Market Cap into company size"""
    _x = int(x)
    if _x > 4000:
        return 'Large'
    if _x > 500:
        return 'Medium'
    if _x < 150:
        return 'Micro'
    if _x <= 500:
        return 'Small'


def _convert_ticker_for_yahoo(ticker):
    if ticker.endswith('.'):
        return ticker + 'L'
    else:
        return ticker.replace('.', '-') + '.L'


def _convert_ticker_for_google(ticker):
    if ticker.endswith('.'):
        suffix = 'L'
    else:
        suffix = '.L'
    return ticker + suffix


def get_tickers_from_lse(download=True):
    """download FTSE tickers and basic company data from LSE"""

    if download:
        print("Downloading tickers!")
        resp = requests.get(source, stream=True)
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(resp.raw, f)

    df = pd.read_excel(io=file_name,
                       sheetname=1,
                       header=0,
                       skiprows=7,
                       skip_footer=3,
                       index_col=0,
                       names=['Company', 'Industry', 'Sector', 'Start Date', 'Size'],
                       converters={'Security Mkt Cap (in £m)': _convert_cap_to_size},
                       parse_cols=[0, 1, 5, 6, 7, 10],
                       )

    df["yahoo"] = df.index.map(_convert_ticker_for_yahoo)
    df["google"] = df.index.map(_convert_ticker_for_google)
    df.to_csv(output_name)

    print("Import tickers from LSE... Complete!")

    return df


def _cleanse_data(df):
    # just in case Pandas spits some warnings out
    pd.options.mode.chained_assignment = None

    # filter Close, where Close != 0
    df = df[df['Close'] != 0]

    # forward, then back fill Close and Volume
    df['Close'].fillna(method='pad', inplace=True)
    df['Volume'].fillna(method='pad', inplace=True)
    df['Close'].fillna(method='bfill', inplace=True)
    df['Volume'].fillna(method='bfill', inplace=True)

    # Fill missing OHL data with Close column
    for col in ['High', 'Low', 'Open']:
        df[col].fillna(df['Close'], inplace=True)

    # Replace Open with Close as best guess
    df['Open'][df['Open'] == 0] = df['Close'][df['Open'] == 0]

    # Replace High/Low as highest and lowest of Close/Open
    Max = df[['Close', 'Open']].max(axis=1)
    Min = df[['Close', 'Open']].min(axis=1)
    df['High'][df['High'] == 0] = Max[df['High'] == 0]
    df['Low'][df['Low'] == 0] = Min[df['Low'] == 0]

    return df


def _data_sufficient(df, years=5):
    """Checks data to ensure sufficient, using arbitrary values"""
    return len(df) > years*5*52 or \
           len(df['Volume'].unique()) > 100


def _not_reserved_in_dos(ticker_name):
    """Checks ticker name will not clash during fileIO"""
    if ticker_name in reserved_words:
        return False
    else:
        return True


def update_stock_data(source='google'):

    # import list of tickers from CSV
    df = pd.read_csv(output_name, index_col=0, usecols=['TIDM', source])

    # cycle through tickers and import stock price data
    imported = []
    failed = []
    for ticker, source_ticker in zip(df.index, df[source]):
        try:
            source_df = web.DataReader(source_ticker, source)  # get stock data
            cleansed_df = _cleanse_data(source_df)  
            if _data_sufficient(cleansed_df) and _not_reserved_in_dos(ticker):
                cleansed_df.to_csv('stock_dfs/{}.csv'.format(ticker))
                imported.append(ticker)
                print("\r{} ...complete".format(ticker), end="\t\t\t\t")
            else:
                failed.append(ticker)
                print("\r{} ...failed!".format(ticker), end="\t\t\t\t")
        except:
            failed.append(ticker)
            print("\r{} ...failed!".format(ticker), end="\t\t\t\t")

        time.sleep(1)  # wait for 1 second to prevent denial of service

    # save lists of imported and failed tickers
    with open('imported_tickers', 'wb') as f:
        pickle.dump(imported, f)
    with open('failed_tickers', 'wb') as f:
        pickle.dump(failed, f)

    print('\rSummary:\t\t\nImported = {}\nFailed = {}\nTotal = {}'
          .format(len(imported), len(failed), len(df)))



