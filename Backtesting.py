from Backtesting_Ticker import Backtest_Ticker as Ticker
from Investment import Investment as Inv
from Bank import Bank
from model_data import RF_Model as Model
from import_tickers import _not_reserved_in_dos  # temporary
import progressbar
import operator
import math
import datetime
from features import gaussian_filter as smooth


def create_and_model_tickers(n=20):
    """Collect tickers and model each one"""
    tickers = Ticker.lst_tickers
    all_objs = {}
    bar = progressbar.ProgressBar(max_value=len(tickers))
    for i, ticker_name in enumerate(tickers[
                                    :n if n != False else len(tickers)]):
        ### Can be ignored on next refresh
        if not _not_reserved_in_dos(ticker_name):
            continue
        ##################################
        ticker_obj = Ticker(ticker_name)
        if ticker_obj.error:
            continue  # skip
        X, y = ticker_obj.update_Xy(date=Ticker._date)
        ticker_obj.model = Model(X, y)
        try:
            ticker_obj.model.fit_model()
        except ValueError:
            continue
        all_objs[ticker_name] = ticker_obj
        bar.update(i)
    return all_objs


def update_investment_price_and_check_change(investment, lower, upper):
    """Return list of tickers which need selling based on thresholds"""
    new_price = investment.ticker.get_latest_price(date)
    current_low = investment.ticker.get_latest_low(date)
    current_open = investment.ticker.get_latest_open(date)
    low_pct_change = 100 * (current_low - current_open) / current_open
    investment.update_value(new_price)
    change = investment.pct_change
    if low_pct_change < lower:
        return lower
    elif change >= upper:
        return upper
    elif change < lower:
        # redundant
        return lower
    else:
        return False


def sell_investment(investment, pct_change):
    global bank_of_rick
    global investments
    args = investment.sell()
    value = investment.capped_value(pct_change) - args[-2]
    log.sell(value, pct_change, *args)
    del investments[args[-1]]
    return bank_of_rick + value


def make_prediction(ticker_obj):
    """return probability of increase based on threshold"""
    X, _ = ticker_obj.update_Xy(date, modeling=False)
    latest_X = ticker_obj.latest(X)
    prediction = ticker_obj.model.get_prediction_probability(latest_X)
    return prediction


def pick_best_tickers(predictions):
    """yield each of the best tickers"""
    sorted_predictions = sorted(predictions.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    for ticker, prediction in sorted_predictions:
        if prediction < ticker.model.pred_thres:
            continue
        else:
            yield ticker


def attempt_investment(ticker):
    """Attempts to take funds out of bank and Invests"""
    global bank_of_rick
    global investments
    if ticker in investments:
        return False
    ticker_obj = ticker_objs_1[ticker]  # not ideal but will work
    price_per_share = ticker_obj.get_latest_price(date)
    num_shares = math.ceil(2000 / (price_per_share / 100))
    current_value = ((num_shares * price_per_share) / 100)
    if bank_of_rick - (current_value + fee):
        log.buy(date, ticker_obj.name, -(current_value - fee))
        investment = Inv(fee, ticker_obj, price_per_share, num_shares, date)
        investments[ticker] = investment
        return True
    return False


def total_value():
    """Adds cash and investment current values"""
    cash_value = bank_of_rick.current_value
    investment_value = 0
    for ticker, investment in investments.items():
        if investment:
            investment_value += investment.current_value
    return cash_value, investment_value


def pct_change():
    """Calculates the percent change"""
    original_value = bank_of_rick.original_value
    current_total_value = sum(total_value())
    return 100 * (current_total_value - original_value) / original_value


class log:

    all_log = ""
    name = 'log.csv'

    @staticmethod
    def new():
        """Create new log and ensures it is wiped"""
        with open(log.name, 'w+'):
            pass

    @staticmethod
    def write():
        """write log on completion"""
        with open(log.name, 'a+') as f:
            f.write(log.all_log)
        log.all_log = ""
        return

    @staticmethod
    def _update(*data):
        """Adds data to row"""
        string = log._create_string(data)
        log.all_log += string
        return True

    @staticmethod
    def sell(value, pct, *args):
        """Prepares data before updating log"""
        value = value
        ticker = args[-1]
        pct = pct
        log._update("Sell", date, ticker, value, pct)

    @staticmethod
    def buy(*args):
        """Prepares data before updating log"""
        pct = 0
        log._update("Buy", *args, pct)
        return

    @staticmethod
    def _create_string(data):
        """Convert data to string"""
        string = ", ".join([str(i) for i in data])
        return string + "\n"


def roll_next_date():
    """cycles next date"""
    Ticker._date = Ticker.next_date(date)
    if Ticker._date >= end_date:
        return False
    return Ticker._date


"""Create universal parameters"""
Ticker.start_date = '2017-01-01'
Ticker._date = datetime.datetime.strptime(Ticker.start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime('2017-07-20', '%Y-%m-%d').date()
fee = 12.5
log.new()
lower = -10
upper = 3  # was ticker threshold but can possibly do better
bank_of_rick = Bank(10000.0)
investments = {}
n = False
"""Create ticker and model parameters"""

Ticker.threshold = 3  # Careful now!
# Model.pred_thres = 0.5  # Careful now!
pre_ticker_objs_1 = create_and_model_tickers(n=n)
# ticker_objs_1 = pre_ticker_objs_1
# for name, obj in pre_ticker_objs_1.items():
#     score_array = []
#     day_array = range(5,16,1)
#     for days in day_array:
#         print(".", end="", flush=True)
#         obj.days = days
#         obj.create_y()
#         X, y = obj.update_Xy(date=Ticker._date)
#         obj.model = Model(X, y)
#         obj.model.fit_model()
#         score, _ = obj.model.get_cv_score(k=10)
#         score_array.append(score)
#     smooth_score = smooth(score_array, 2)
#     for i, score in enumerate(smooth_score):
#         if score >= obj.day_threshold:
#             obj.days = day_array[i]
#             obj.create_y()
#             X, y = obj.update_Xy(date=Ticker._date)
#             obj.model = Model(X, y)
#             obj.model.fit_model()
#             ticker_objs_1[name] = obj
#             print(name, obj.days, score)
#             break
#         else:
#             continue
# print(len(ticker_objs_1))

# Ticker.threshold = 0
# Ticker.days = 1
# Model.pred_thres = 0.8
# ticker_objs_2 = create_and_model_tickers(n=n)

def _get_prediction_threshold(x, a=0.05):
    x += 0.01
    if x < 0.5:
        return 0.5
    else:
        return math.ceil(x / a) * a

ticker_objs_1 = {}
for name, ticker in pre_ticker_objs_1.items():
    if not _not_reserved_in_dos(name) or ticker.error:
        continue
    model = Model(ticker.X, ticker.y)
    try:
        model.get_cv_score(k=5)
    except:
        continue
    if model.min_threshold == 1:
        continue
    else:
        ticker.pred_thres = _get_prediction_threshold(model.min_threshold)
        ticker_objs_1[name] = ticker


print("\nBegin Loop...")
while True:

    date = Ticker._date
    print(date, end="\t")


    """Update all existing investments and sell if required"""
    for ticker, investment in investments.copy().items():
        sell_signal = update_investment_price_and_check_change(investment, lower, upper)
        if type(sell_signal) != bool:
            sell_investment(investment, sell_signal)


    """Make predictions"""

    predictions_1 = {}
    for ticker, ticker_obj in ticker_objs_1.items():
        prediction = make_prediction(ticker_obj)
        predictions_1[ticker] = prediction

    # predictions_2 = {}
    # for ticker, ticker_obj in ticker_objs_2.items():
    #     prediction = make_prediction(ticker_obj)
    #     predictions_2[ticker] = prediction

    """Invest best predictions"""
    best_1 = list(pick_best_tickers(predictions_1))
    # best_2 = list(pick_best_tickers(predictions_2))
    for ticker in best_1:
        if attempt_investment(ticker) == False:
            break
    # for ticker in best_1:
    #     if attempt_investment(ticker) == False:
    #         break


    """prints percent change and number of investments"""
    print("Change: {:.2f}% \t Investments: {}".format(pct_change(), len(investments)))


    # """prints total value"""
    # cash, investment = total_value()
    # total = cash + investment
    # print("Cash: {:.2f} \t "
    #       "Investment: {:.2f} \t "
    #       "Total: {:.2f}".format(cash, investment, total))


    """Go to next date and repeat"""
    if not roll_next_date():
        log.write()
        print("Complete")
        input()
        quit()

































