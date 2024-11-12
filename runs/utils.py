import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mode


def model_predict_func_reg(model, X):
    data_X = np.nan_to_num(X)
    y_pred = model.predict(data_X)
    return y_pred


def model_predict_func_cla(model, X):
    data_X = np.nan_to_num(X)
    y_pred = model.predict_proba(data_X)[:, 1]
    return y_pred


def eval_func_auc(model, X, y):
    y_pred = model_predict_func_cla(model, X)
    score = roc_auc_score(y, y_pred)
    return score


def eval_func_acc(model, X, y):
    y_pred = model_predict_func_reg(model, X)
    y_pred = np.round(y_pred)
    score = np.mean(y_pred == y)
    return score


def eval_func_rmse(model, X, y):
    y_pred = model_predict_func_reg(model, X)
    score = np.sqrt(np.mean((y_pred - y) ** 2))
    return score


def ensemble_eval_func_auc(y_pred_list, y):
    y_pred = np.zeros(y.shape[0])
    for i in range(len(y_pred_list)):
        y_pred += y_pred_list[i]
    y_pred /= len(y_pred_list)
    score = roc_auc_score(y, y_pred)
    return score


def ensemble_eval_func_acc(y_pred_list, y):
    predictions_array = np.stack(y_pred_list, axis=0)
    ensemble_predictions = mode(predictions_array, axis=0)[0].flatten()
    score = np.mean(ensemble_predictions == y)
    return score


def ensemble_eval_func_rmse(y_pred_list, y):
    y_pred = np.zeros(y.shape[0])
    for i in range(len(y_pred_list)):
        y_pred += y_pred_list[i]
    y_pred /= len(y_pred_list)
    score = np.sqrt(np.mean((y_pred - y) ** 2))
    return score















