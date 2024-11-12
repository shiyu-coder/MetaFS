import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from skrebate import ReliefF


class F_value:

    def __init__(self, n_select):
        self.n_select = n_select

    def fit(self, train_X, train_y, eval_X, eval_y):
        f_scores, p_values = f_regression(train_X, train_y)
        selector = SelectKBest(f_regression, k=self.n_select)
        selector.fit(train_X, train_y)

        feature_selected = list(selector.get_support(indices=True))
        return feature_selected


class Relief:

    def __init__(self, n_select):
        self.n_select = n_select

    def fit(self, train_X, train_y, eval_X, eval_y):
        fs = ReliefF(n_neighbors=3, n_features_to_select=self.n_select, n_jobs=-1, verbose=True)

        fs.fit(train_X, train_y)

        feature_selected = list(fs.top_features_[:self.n_select])
        return feature_selected













