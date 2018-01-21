import numpy as np
import pandas as pd
import features
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class RF_Model:

    n_splits = 3
    random_state = None
    shuffle = True
    n_estimators = 100
    n_jobs = -1
    confidence = 0.8

    def __init__(self, X, y):
        self.initiate_RFC(self.n_estimators)
        self.update_Xy(X, y)
        self.n_splits = 3
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.number_predicted = []
        self.scores = None
        self.min_thresholds = None
        self.min_threshold = 1
        self.pred_thres = 0.5

    def update_Xy(self, X, y):
        """Update input data"""
        self.X, self.y = self._prep(X.copy(), y.copy())

    def _prep(self, X, y):
        """Prepare input data X,y"""
        X.dropna(axis=0, how='any', inplace=True)
        y = y.ix[X.index]
        return X.values, y.values.ravel()

    @property
    def rows(self):
        return len(self.X)

    def initiate_RFC(self, n_estimators):
        """Initiates the RFC class object, with parameter n_estimators"""
        self.model = RFC(n_jobs=self.n_jobs,
                         n_estimators=self.n_estimators,
                         random_state=self.random_state)
        return

    def fit_model(self, *args):
        """Fits all data to model"""
        if len(args) == 0:
            X = self.X
            y = self.y
        else:
            X = args[0]
            y = args[1]
        self.model.fit(X, y)
        return True

    def make_prediction(self, X, threshold, prob_flag=True):
        """Make a prediction, with confidence threshold"""
        probs_arr = self.model.predict_proba(X)
        try:
            probability = probs_arr[:,1]
            return probability > threshold
        except IndexError:
            probs_arr[probs_arr==1.] = False
            return probs_arr

    def _get_train_test_indices(self, n_splits):
        """Creates test and training indices"""
        kf = KFold(n_splits, self.shuffle, self.random_state)
        return kf.split(self.X)

    def _get_simple_train_test_indices(self, X, y, ratio):
        np.random.seed(self.random_state)
        msk = np.random.rand(len(X)) < ratio
        self.X_train = X[msk]
        self.X_test = X[~msk]
        self.y_train = y[msk]
        self.y_test = y[~msk]
        return

    def _get_score(self, threshold):
        """Run model, make prediction and return score"""
        self.model.fit(self.X_train, self.y_train)
        y_prediction = self.make_prediction(self.X_test, threshold)
        self.number_predicted.append(features.count_true(y_prediction))
        score = self._measure_precision(y_prediction)
        return score

    def get_basic_score(self, t=0.9):
        """Get basic score running on one train/test mask"""
        self._get_simple_train_test_indices(self.X, self.y, t)
        score = self._get_score(self.pred_thres)
        return score

    def get_cv_score(self, k=10, threshold=0.5):
        """Run model, make prediction and output minimum score"""
        self.scores = []
        self.min_thresholds = []
        self.number_predicted = []
        X, y = self.X, self.y
        for train_index, test_index in self._get_train_test_indices(k):
            self.X_train, self.X_test = X[train_index], X[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]
            self.scores.append(self._get_score(threshold))
            # under construction: min threshold on individual or total???? needs thought.
            self.min_thresholds.append(self._get_min_threshold())
        min_score = np.min(self.scores)
        mean_score = np.mean(self.scores)
        total_score = self._cv_total_score()
        # print(self.min_thresholds)
        self.min_threshold = np.max(self.min_thresholds)
        return min_score, mean_score, total_score

    def _cv_total_score(self):
        """return total score after cross validation"""
        scores = self.scores
        numbers = self.number_predicted
        total = sum(numbers)
        number_correct = sum([s*n for s,n in zip(scores,numbers)])
        total_score = number_correct / total
        return total_score

    def _measure_precision(self, y_prediction):
        """Returns Precision: tp / tp + fp"""
        if np.count_nonzero(y_prediction) == 0:
            return 0.
        else:
            return precision_score(self.y_test, y_prediction)

    def _measure_accuracy(self, y_prediction):
        """Returns accuracy"""
        return accuracy_score(self.y_test, y_prediction)

    def _measure_f1(self, y_prediction):
        """Returns F1 score: combo of precision & recall"""
        return f1_score(self.y_test, y_prediction)

    def get_prediction_probability(self, X):
        """Return % of decision trees tipping prediction"""
        probability_array = self.model.predict_proba(X)
        try:
            return probability_array[:,1]
        except IndexError:
            return 0

    def _get_min_threshold(self, confidence=0.95, pred_thres=0.5):
        """Returns minimum threshold based on a confidence"""
        probability = self.get_prediction_probability(self.X_test)
        prediction = probability > pred_thres
        equality = np.logical_and(
            np.logical_xor(prediction, self.y_test), prediction)
        try:
            mask = probability[equality]
        except TypeError:
            return 1.
        if len(mask) == 0:
            return 1.
        else:
            return np.max(mask)

    #### Under Construction ####
    def _convert_label(self, array):
        le = LabelEncoder()
        return le, le.fit_transform(array)

    # def __repr__(self):
    #     return "RFC"









