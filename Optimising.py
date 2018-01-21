from ticker_data import Ticker
from model_data import RF_Model as Model
from matplotlib import pyplot as plt
import features
import random
import time
from collections import Counter
from import_tickers import _not_reserved_in_dos  # temporary

tickers = Ticker.lst_tickers


"""Prepare test sample of Tickers"""
test_tickers = []
i = 0
while len(test_tickers) < 10:
    next_ticker = Ticker(tickers[i])
    if not next_ticker.error:
        test_tickers.append(next_ticker)
    i += 1


"""Begin looping through various random states, tickers and parameters"""
# true_count = ticker.count_true
for ticker in test_tickers:
    print(ticker)
    y_axis_min = []
    y_axis_mean = []
    p_lst = range(5, 16, 1)
    for r in random.sample(range(200), 1):
        # 5 different random seeds
        # print("")
        Model.random_state = r
        p_min = []
        p_mean = []
        for n in p_lst:
            # each parameter consideration
            # print("{} ".format(n), end="", flush=True)
            ticker.days = n
            ticker.create_y()
            print(".", end="", flush=True)
            model = Model(ticker.X, ticker.y)
            model.fit_model()
            min_score, mean_score = model.get_cv_score(k=5)
            p_min.append(min_score)
            p_mean.append(mean_score)
        y_axis_mean.append(features.gaussian_filter(p_mean, 2))
        y_axis_min.append(features.gaussian_filter(p_min, 2))


    # lst_of_maxi = [p_lst[list(n).index(max(n))] for n in y_axis_mean]
    # c = Counter(lst_of_maxi)
    # print(c.most_common())

    p_min = features.gaussian_filter(
        [sum(n)/float(len(n)) for n in zip(*y_axis_min)], 2)
    p_mean = features.gaussian_filter(
        [sum(n)/float(len(n)) for n in zip(*y_axis_mean)], 2)

    best_p = max(p_min)
    worst_p = min(p_min)
    best_n = p_lst[list(p_min).index(best_p)]
    worst_n = p_lst[list(p_min).index(worst_p)]
    print("\nBest: {:.3f} \t Parameter: {} \t (Low: {:.3f}, Parameter: {})"
          .format(best_p, best_n, worst_p, worst_n))

    # plt.plot(p_lst, p_min, label='minimum score')
    plt.plot(p_lst,p_min, label=ticker.name)

plt.legend()
plt.show()
