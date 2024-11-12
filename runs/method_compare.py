import copy
import os
import sys
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import mode
from utils import *
sys.path.append("../metafs")
from metafs import MetaFS
from config import *
import argparse
from benchmark.embedded import *
from benchmark.filter import *
from benchmark.wrapper import *
from benchmark.auto_feature_generation import *
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

"""
Since the FT-Transformer does not comply with the standard sklearn interface and
 its multiprocessing implementation is quite complex,
 the corresponding implementation is not provided here.
"""


def get_model_predict_func(model_name):
    if model_name == 'LinearRegression' or 'LGBMRegressor':
        fun = model_predict_func_reg
    elif model_name == 'LogisticRegression' or 'LGBMClassifier':
        fun = model_predict_func_cla
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    return fun


def get_eval_func(metric_name):
    if metric_name == 'AUC':
        fun = eval_func_auc
    elif metric_name == 'Acc':
        fun = eval_func_acc
    elif metric_name == 'RMSE':
        fun = eval_func_rmse
    else:
        raise ValueError(f"Metric name {metric_name} is not supported.")
    return fun


def get_ensemble_eval_func(metric_name):
    if metric_name == 'AUC':
        fun = ensemble_eval_func_auc
    elif metric_name == 'Acc':
        fun = ensemble_eval_func_acc
    elif metric_name == 'RMSE':
        fun = ensemble_eval_func_rmse
    else:
        raise ValueError(f"Metric name {metric_name} is not supported.")
    return fun


def load_dataset(dataset_name, target_feature_num, root_dataset_path):
    train_X = np.load(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/X_train.npy")
    eval_X = np.load(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/X_val.npy")
    test_X = np.load(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/X_test.npy")
    train_y = np.load(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/y_train.npy")
    eval_y = np.load(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/y_val.npy")
    test_y = np.load(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/y_test.npy")

    return train_X, train_y, eval_X, eval_y, test_X, test_y


def get_method(method_name, model, metric_name, n_select):
    model_predict_func = get_model_predict_func(model.__class__.__name__)
    eval_func = get_eval_func(metric_name)
    ensemble_eval_func = get_ensemble_eval_func(metric_name)
    if method_name == 'MetaFS':
        method = MetaFS(model, eval_func, model_predict_func=model_predict_func, n_select=n_select, ensemble_count=1,
                        ensemble_eval_func=ensemble_eval_func, n_meta_samples=200)
    elif method_name == 'MetaFSe':
        method = MetaFS(model, eval_func, model_predict_func=model_predict_func, n_select=n_select, ensemble_count=3,
                        ensemble_eval_func=ensemble_eval_func, n_meta_samples=200)
    elif method_name == 'F_value':
        method = F_value(n_select)
    elif method_name == 'Relief':
        method = Relief(n_select)
    elif method_name == 'Boruta':
        method = Boruta(n_select, metric_name)
    elif method_name == 'PSO':
        direction = True
        if metric_name == 'RMSE':
            direction = False
        method = PSO(n_select, model, eval_func, direction, maxiter=3)
    elif method_name == 'GFI':
        method = GFI(n_select, metric_name)
    elif method_name == 'MDI':
        method = MDI(n_select, metric_name)
    elif method_name == 'MDA':
        method = MDA(n_select, metric_name)
    elif method_name == 'AutoFeat':
        method = AutoFeat(n_select, metric_name)
    elif method_name == 'SAFE':
        method = SAFE(n_select, metric_name)
    else:
        raise ValueError(f"Method name {method_name} is not supported.")
    return method


def load_model(model_name):
    lgb_param = {'seed': 100, 'n_jobs': 60, 'verbose': -1, 'max_depth': 8, 'reg_alpha': 0.01, 'reg_lambda': 0.5, 'learning_rate': 0.05, 'n_estimators': 300}
    if model_name == 'LogisticRegression':
        model = LogisticRegression()
    elif model_name == 'LGBMClassifier':
        model = LGBMClassifier(**lgb_param)
    elif model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'LGBMRegressor':
        model = LGBMRegressor(**lgb_param)
    else:
        raise ValueError(f"Model name {model_name} is not supported.")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', type=str, default='MetaFS')
    parser.add_argument('--model_name', type=str, default='LogisticRegression')
    parser.add_argument('--n_select', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default='HG')
    parser.add_argument('--target_feature_num', type=int, default=50)
    parser.add_argument('--root_dataset_path', type=str, default='./dataset')
    args = parser.parse_args()

    # load dataset
    train_X, train_y, eval_X, eval_y, test_X, test_y = load_dataset(args.dataset_name, args.target_feature_num,
                                                                    args.root_dataset_path)
    metric_name = dataset_config[args.dataset_name]['metric']
    print(train_X.shape, train_y.shape, eval_X.shape, eval_y.shape, test_X.shape, test_y.shape)

    # load model
    model = load_model(args.model_name)

    # get method
    print(f"Method: {args.method_name}, Model: {args.model_name}, Metric: {metric_name}, n_select: {args.n_select}")
    method = get_method(args.method_name, model, metric_name, args.n_select)
    begin_time = time.time()
    selected_features_list = method.fit(train_X, train_y, eval_X, eval_y)
    end_time = time.time()
    print(f"Time cost: {end_time-begin_time:.2f}s")

    if not args.method_name.startswith('MetaFS'):
        selected_features_list = [selected_features_list]

    print(f"Selected feature list length: {[len(selected_features) for selected_features in selected_features_list]}")

    model_list = []
    for selected_features in selected_features_list:
        model = copy.deepcopy(model)
        model.fit(train_X[:, selected_features], train_y)
        model_list.append(model)

    model_predict_func = get_model_predict_func(args.model_name)
    y_pred_list = []
    for i in range(len(model_list)):
        y_pred = model_predict_func(model, test_X[:, selected_features_list[i]])
        y_pred_list.append(y_pred)

    ensemble_eval_func = get_ensemble_eval_func(metric_name)
    score = ensemble_eval_func(y_pred_list, test_y)
    print(f"Test score: {score:.5f}")





