import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm


class GFI:

    def __init__(self, n_select, metric_name):
        self.lgb_param = {'seed': 100, 'n_jobs': 60, 'verbose': -1, 'max_depth': 8, 'reg_alpha': 0.01, 'reg_lambda': 0.5, 'learning_rate': 0.05, 'n_estimators': 300}
        self.n_select = n_select
        self.cla = True
        if metric_name == 'RMSE':
            self.cla = False

    def fit(self, train_X, train_y, eval_X, eval_y):
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


def featImpMDI(fit, featNames):
    # feat importance based on in-sample mean impurity reduction
    if hasattr(fit, 'estimators_'):  # Check if the fit object has the 'estimators_' attribute
        df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
        df0 = pd.DataFrame.from_dict(df0, orient='index')
        df0.columns = featNames
        df0 = df0.replace(0, np.nan)  # replacing 0 with NaN, assuming it's because max_features=1
        imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** -.5}, axis=1)  # Central Limit Theorem (CLT)
        imp /= imp['mean'].sum()
        return imp
    else:
        raise AttributeError("The fitted model does not have the attribute 'estimators_'. Ensure the correct model is used.")


class MDI:

    def __init__(self, n_select, metric_name):
        self.n_select = n_select
        self.cla = True
        if metric_name == 'RMSE':
            self.cla = False

    def fit(self, train_X, train_y, eval_X, eval_y):
        if self.cla:
            clf = DecisionTreeClassifier(criterion='entropy',
                                         max_features=1,
                                         class_weight='balanced',
                                         min_weight_fraction_leaf=0)
            clf = BaggingClassifier(base_estimator=clf,
                                    n_estimators=1000,
                                    max_features=1.,
                                    max_samples=1.,
                                    oob_score=False,
                                    n_jobs=20)
        else:
            clf = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0)
            clf = BaggingRegressor(base_estimator=clf,
                                   n_estimators=1000,
                                   max_features=1.,
                                   max_samples=1.,
                                   oob_score=False,
                                   n_jobs=20)

        factor_df = pd.DataFrame(train_X, columns=[f"feature_{i}" for i in range(train_X.shape[1])])
        fit = clf.fit(factor_df, train_y)
        imp = featImpMDI(fit, featNames=factor_df.columns)
        imp.sort_values('mean', inplace=True, ascending=False)
        feature_selected = list(imp.index[:self.n_select])
        feature_selected = [int(feature.split('_')[1]) for feature in feature_selected]
        return feature_selected


def featImpMDA_cla(clf, X, y):
    # feat importance based on Out-of-Sample score reduction
    prob = clf.predict_proba(X)  # prediction before shuffles
    scr0 = -log_loss(y, prob, labels=clf.classes_)  # original score
    scr1 = pd.Series(dtype='float64', index=X.columns)

    for j in tqdm(X.columns):
        original_column = X[j].copy()

        shuffled_column = original_column.sample(frac=1.0, random_state=42).reset_index(drop=True)

        X[j] = shuffled_column

        prob_shuffled = clf.predict_proba(X)  # prediction after shuffle
        scr1[j] = -log_loss(y, prob_shuffled, labels=clf.classes_)  # score after shuffle

        X[j] = original_column

    imp = (scr0 - scr1) / -scr1  # importance calculation
    return imp


def featImpMDA_reg(clf, X, y):
    # Feature importance based on Out-of-Sample score reduction
    prob = clf.predict(X)  # prediction before shuffles
    scr0 = mean_squared_error(y, prob)  # original score
    scr1 = pd.Series(dtype='float64', index=X.columns)

    # Iterate over each column to shuffle one at a time
    for j in tqdm(X.columns):
        # Save the original data of the column
        original_column = X[j].copy()

        # Shuffle the column data
        shuffled_column = original_column.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Replace the original column data with shuffled data
        X[j] = shuffled_column

        # Make predictions with the shuffled data and compute the new score
        prob_shuffled = clf.predict(X)
        scr1[j] = mean_squared_error(y, prob_shuffled)

        # Restore the original data back to the DataFrame
        X[j] = original_column

    # Calculate the importance of each feature
    imp = scr1 - scr0
    return imp


class MDA:

    def __init__(self, n_select, metric_name):
        self.n_select = n_select
        self.cla = True
        if metric_name == 'RMSE':
            self.cla = False

    def fit(self, train_X, train_y, eval_X, eval_y):
        if self.cla:
            clf = DecisionTreeClassifier(criterion='entropy',
                                         max_features=1, max_depth=12,
                                         class_weight='balanced',
                                         min_weight_fraction_leaf=0)
            clf = BaggingClassifier(base_estimator=clf,
                                    n_estimators=1000,
                                    max_features=1.,
                                    max_samples=1.,
                                    oob_score=False,
                                    n_jobs=40)
        else:
            clf = DecisionTreeRegressor(max_features=1, min_weight_fraction_leaf=0, max_depth=12)
            clf = BaggingRegressor(base_estimator=clf,
                                   n_estimators=1000,
                                   max_features=1.,
                                   max_samples=1.,
                                   oob_score=False,
                                   n_jobs=40)

        factor_df = pd.DataFrame(train_X, columns=[f"feature_{i}" for i in range(train_X.shape[1])])
        clf.fit(factor_df, train_y)
        factor_df = pd.DataFrame(eval_X, columns=[f"feature_{i}" for i in range(train_X.shape[1])])
        sr_y = pd.Series(eval_y)

        if self.cla:
            imp = featImpMDA_cla(clf, factor_df, sr_y)
        else:
            imp = featImpMDA_reg(clf, factor_df, sr_y)
        imp = imp.sort_values(ascending=False)

        feature_selected = list(imp.index[:self.n_select])
        feature_selected = [int(feature.split('_')[1]) for feature in feature_selected]

        return feature_selected












