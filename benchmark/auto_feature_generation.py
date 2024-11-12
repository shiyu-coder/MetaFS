import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier, LGBMRegressor
from autofeat import FeatureSelector


class AutoFeat:

    def __init__(self, n_select, metric_name):
        self.n_select = n_select
        self.problem_type = 'classification'
        if metric_name == 'RMSE':
            self.problem_type = 'regression'

    def fit(self, train_X, train_y):
        train_X = np.nan_to_num(train_X)
        train_y = np.nan_to_num(train_y)
        feature_df = pd.DataFrame(train_X, columns=[f"feature_{i}" for i in range(train_X.shape[1])])
        feature_df = feature_df.replace([-np.inf, np.inf], np.nan).fillna(0.).astype(float)

        fs = FeatureSelector(problem_type=self.problem_type, n_jobs=15, verbose=1, featsel_runs=3)
        new_features = fs.fit_transform(feature_df, train_y)
        new_features = [int(feature.split('_')[1]) for feature in new_features.columns]
        new_features = new_features[:self.n_select]
        return new_features


def calculate_iv(feature, target, bins=10):
    df = pd.DataFrame({'feature': feature, 'target': target})
    df['bins'] = pd.qcut(feature, q=bins, duplicates='drop', precision=3)

    grouped = df.groupby('bins')
    metrics = grouped['target'].agg([lambda x: (x == 1).sum(), lambda x: (x == 0).sum()])
    metrics.columns = ['pos', 'neg']

    small_value = 0.0001
    metrics['pos_dist'] = (metrics['pos'] + small_value) / (metrics['pos'].sum() + small_value)
    metrics['neg_dist'] = (metrics['neg'] + small_value) / (metrics['neg'].sum() + small_value)

    metrics['iv'] = (metrics['pos_dist'] - metrics['neg_dist']) * np.log(metrics['pos_dist'] / metrics['neg_dist'])
    return metrics['iv'].sum()


def calculate_iv_for_all_features(X, y, bins=10):
    iv_values = {}
    for column in X.columns:
        iv_values[column] = calculate_iv(X[column], y, bins)
    return iv_values


class SAFE:

    def __init__(self, n_select, metric_name, iv_thres=0.1, bins=10):
        self.lgb_param = {'seed': 100, 'n_jobs': 60, 'verbose': -1, 'max_depth': 8, 'reg_alpha': 0.01, 'reg_lambda': 0.5, 'learning_rate': 0.05,
                          'n_estimators': 300}
        self.n_select = n_select
        self.cla = True
        if metric_name == 'RMSE':
            self.cla = False
        self.iv_thres = iv_thres
        self.bins = bins

    def fit(self, train_X, train_y, eval_X, eval_y):
        factor_df = pd.DataFrame(train_X, columns=[f"feature_{i}" for i in range(train_X.shape[1])])
        iv_values = calculate_iv_for_all_features(factor_df, train_y, self.bins)
        filtered_features = [key for key in iv_values.keys() if iv_values[key] > self.iv_thres]
        print(len(filtered_features))
        if len(filtered_features) < self.n_select:
            filtered_features = sorted(iv_values, key=iv_values.get, reverse=True)[:self.n_select]

        if self.cla:
            model = LGBMClassifier(**self.lgb_param)
        else:
            model = LGBMRegressor(**self.lgb_param)
        model.fit(train_X, train_y)
        gain_importance = model._Booster.feature_importance(importance_type='gain')
        feature_name = model.feature_name_
        importance_df = pd.DataFrame({
            'feature': feature_name,
            'gain_importance': gain_importance,
        })
        feature_selected = importance_df.sort_values(by='gain_importance', ascending=False)[:self.n_select]['feature'].tolist()
        feature_selected = [int(feature.split('_')[1]) for feature in feature_selected]
        return feature_selected










