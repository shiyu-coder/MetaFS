import copy
import os
import sys
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
sys.path.append("..")
from metafs.metafs import MetaFS
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    test_size = int(n * test_ratio)
    train_size = n - test_size
    train_X = X[:train_size]
    train_y = y[:train_size]
    test_X = X[train_size:]
    test_y = y[train_size:]
    return train_X, train_y, test_X, test_y


def train_valid_split(X, y, valid_ratio=0.25):
    n = len(X)
    valid_size = int(n * valid_ratio)
    train_size = n - valid_size
    train_X = X[:train_size]
    train_y = y[:train_size]
    valid_X = X[train_size:]
    valid_y = y[train_size:]
    return train_X, train_y, valid_X, valid_y


def load_dataset(data_id, cat=False):
    df_X, sr_y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=True, parser='auto')
    if cat:
        sr_y = sr_y.cat.codes.astype(int)
    for column in df_X.columns:
        # Assume all object types are categorical variables.
        if df_X[column].dtype == 'object' or 'category':
            labels, unique = pd.factorize(df_X[column])
            df_X[column] = labels
    ndr_X = df_X.astype(float).values
    ndr_y = sr_y.values
    return ndr_X, ndr_y


def model_predict_func(model, X):
    data_X = np.nan_to_num(X)
    y_pred = model.predict_proba(data_X)[:, 1]
    return y_pred


def eval_func(model, X, y):
    y_pred = model_predict_func(model, X)
    score = roc_auc_score(y, y_pred)
    return score


def ensemble_eval_func(y_pred_list, y):
    y_pred = np.zeros(y.shape[0])
    for i in range(len(y_pred_list)):
        y_pred += y_pred_list[i]
    y_pred /= len(y_pred_list)
    score = roc_auc_score(y, y_pred)
    return score


def baseline(X, y, n_select):
    lgb_params = {'max_depth': 7, 'num_leaves': 64, 'reg_alpha': 0.2, 'reg_lambda': 0.2,
                  'learning_rate': 0.075, 'seed': 100, 'n_jobs': 30, 'verbose': -1,
                  'n_estimators': 200}
    model = LGBMClassifier(**lgb_params)
    model.fit(X, y)
    gain_importance = model._Booster.feature_importance(importance_type='gain')
    feature_name = model.feature_name_
    importance_df = pd.DataFrame({
        'feature': feature_name,
        'gain_importance': gain_importance,
    })
    feature_selected = importance_df.sort_values(by='gain_importance', ascending=False)[:n_select]['feature'].tolist()
    feature_selected = [int(feature.split('_')[1]) for feature in feature_selected]
    return feature_selected


if __name__ == '__main__':

    # load higgs dataset
    X, y = load_dataset(data_id=23512, cat=False)
    print(X.shape, y.shape)

    # split dataset
    train_X, train_y, test_X, test_y = train_test_split(X, y)
    train_X, train_y, valid_X, valid_y = train_valid_split(train_X, train_y, valid_ratio=0.25)

    model = LogisticRegression()

    mfs = MetaFS(model, eval_func, model_predict_func=model_predict_func, n_select=5, ensemble_count=1,
                 ensemble_eval_func=ensemble_eval_func, n_meta_samples=200)

    selected_features_list = mfs.fit(train_X, train_y, valid_X, valid_y)

    print(f"Selected feature list: {selected_features_list}")

    # evaluate the model
    y_pred_list = []
    for feature_subsets in selected_features_list:
        model_subsets = copy.deepcopy(model)
        model_subsets.fit(train_X[:, feature_subsets], train_y)
        y_pred = model_predict_func(model_subsets, test_X[:, feature_subsets])
        y_pred_list.append(y_pred)
    score = ensemble_eval_func(y_pred_list, test_y)
    print(f"Test score: {score:.5f}")

    # baseline
    feature_selected = baseline(train_X, train_y, 5)
    print(f"Baseline selected feature list: {feature_selected}")
    model_baseline = copy.deepcopy(model)
    model_baseline.fit(train_X[:, feature_selected], train_y)
    y_pred_baseline = model_predict_func(model_baseline, test_X[:, feature_selected])
    score_baseline = eval_func(model_baseline, test_X[:, feature_selected], test_y)
    print(f"Baseline test score: {score_baseline:.5f}")

