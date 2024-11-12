import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
import shap
from tqdm import trange


class PriorInfo:
    def __init__(self, prior_feature_list):

        self.prior_feature_list = prior_feature_list
        self.indicator_list = list(set([feature_pair[0] for feature_pair in self.prior_feature_list]))

        self.lgb_params = {'max_depth': 7, 'num_leaves': 64, 'reg_alpha': 0.2, 'reg_lambda': 0.2,
                           'learning_rate': 0.075, 'seed': 100, 'n_jobs': 30, 'verbose': -1,
                           'n_estimators': 200}

        self.model_type = None
        self.dt_indicator = {}

    def fit(self, train_X, train_y):

        if np.unique(train_y).shape[0] > 100:
            self.model_type = 'reg'
        else:
            self.model_type = 'cla'

        print(f"Calculating prior indicators ...")
        for i in trange(len(self.indicator_list)):
            indicator_name = self.indicator_list[i]
            self.dt_indicator[indicator_name] = self.calculate_indicator(train_X, train_y, indicator_name)

    def transform(self, feature_subsets_list):
        prior_info_list = []
        for indicator_name, agg_operator in self.prior_feature_list:
            ndr_prior_values = self.aggregate_indicator(indicator_name, agg_operator, feature_subsets_list)
            prior_info_list.append(ndr_prior_values)

        ndr_prior_info = np.stack(prior_info_list, axis=1)
        return ndr_prior_info

    def calculate_indicator(self, X, y, indicator_name):
        if indicator_name == 'SFI':
            # split feature importance
            model = LGBMRegressor(**self.lgb_params) if self.model_type == 'reg' else LGBMClassifier(**self.lgb_params)
            model.fit(X, y)
            split_importance = model._Booster.feature_importance(importance_type='split')
            feature_name = model.feature_name_
            importance_df = pd.DataFrame({
                'feature': feature_name,
                'split_importance': split_importance,
            })
            dt_split_imp = {}
            for idx in importance_df.index:
                feature_name = importance_df.loc[idx, 'feature']
                split_imp = importance_df.loc[idx, 'split_importance']
                feature_idx = int(feature_name.split('_')[1])
                dt_split_imp[feature_idx] = split_imp
            return dt_split_imp
        elif indicator_name == 'GFI':
            # gain feature importance
            model = LGBMRegressor(**self.lgb_params) if self.model_type == 'reg' else LGBMClassifier(**self.lgb_params)
            model.fit(X, y)
            gain_importance = model._Booster.feature_importance(importance_type='gain')
            feature_name = model.feature_name_
            importance_df = pd.DataFrame({
                'feature': feature_name,
                'gain_importance': gain_importance,
            })
            dt_gain_imp = {}
            for idx in importance_df.index:
                feature_name = importance_df.loc[idx, 'feature']
                gain_imp = importance_df.loc[idx, 'gain_importance']
                feature_idx = int(feature_name.split('_')[1])
                dt_gain_imp[feature_idx] = gain_imp
            return dt_gain_imp
        elif indicator_name == 'SHAP':
            # Tree shap value
            model = LGBMRegressor(**self.lgb_params) if self.model_type == 'reg' else LGBMClassifier(**self.lgb_params)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:100000])

            # If it's a binary classification problem, shap_values will be a list; take the first element of the list.
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Calculate the average SHAP value (absolute value) for each feature to gauge its importance.
            shap_values_mean = np.abs(shap_values).mean(axis=0)
            feature_shap_dict = {i: shap_values_mean[i] for i in range(len(shap_values_mean))}
            return feature_shap_dict

        elif indicator_name == 'CORR':
            # Correlation with label
            dt_corr = {}
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                dt_corr[i] = corr
            return dt_corr
        else:
            raise ValueError(f"Invalid indicator name: {indicator_name}")

    def aggregate_indicator(self, indicator_name, agg_operator, feature_subsets_list):
        indicator_values = []
        for i in range(len(feature_subsets_list)):
            feature_subset = feature_subsets_list[i]
            if agg_operator == 'mean':
                agg_values = np.nanmean([self.dt_indicator[indicator_name][feature_idx] for feature_idx in feature_subset])
            elif agg_operator == 'std':
                agg_values = np.nanstd([self.dt_indicator[indicator_name][feature_idx] for feature_idx in feature_subset])
            else:
                raise ValueError(f"Invalid aggregation operator: {agg_operator}")
            indicator_values.append(agg_values)
        return np.array(indicator_values)





