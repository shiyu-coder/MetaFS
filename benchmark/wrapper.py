import traceback

import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier, LGBMRegressor
from pyswarm import pso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
sys.path.append("../")
from benchmark.boruta_py import BorutaPy


class Boruta:

    def __init__(self, n_select, metric_name):
        self.n_select = n_select
        self.cla = True
        if metric_name == 'RMSE':
            self.cla = False

    def fit(self, train_X, train_y, eval_X, eval_y):
        if self.cla:
            rf = RandomForestClassifier(n_jobs=60, class_weight='balanced', max_depth=5)
        else:
            rf = RandomForestRegressor(n_jobs=60, max_depth=5)

        # 初始化Boruta
        boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
        boruta_feature_selector.fit(train_X, train_y)
        boruta_rank = np.array(boruta_feature_selector.ranking_)
        boruta_idx = np.argsort(boruta_rank)
        feature_selected = list(boruta_idx[:self.n_select])

        return feature_selected


class PSO:
    def __init__(self, n_select, model, eval_func, direction, maxiter):
        self.n_select = n_select
        self.model = model
        self.eval_func = eval_func
        self.direction = direction
        self.maxiter = maxiter

    def fit(self, train_X, train_y, eval_X, eval_y):
        iter_count = 0

        def feature_selector(x):
            nonlocal iter_count
            selected_indices = np.argsort(x)[-self.n_select:]

            # 创建选择了特定特征的X子集
            X_selected = train_X[:, selected_indices]

            try:
                self.model.fit(X_selected, train_y)
                score = self.eval_func(self.model, eval_X[:, selected_indices], eval_y)
                iter_count += 1
                print(f"Iter-{iter_count} score: {score:.4f}")
                if self.direction:
                    score = -score
            except:
                traceback.print_exc()
                score = 1e9
            return score

        particle_dimension = train_X.shape[1]
        lb = [0] * particle_dimension
        ub = [1] * particle_dimension

        # 运行PSO优化
        xopt, fopt = pso(feature_selector, lb, ub, swarmsize=50, omega=0.5, phip=0.5, phig=0.5, maxiter=self.maxiter, minstep=1e-8, minfunc=1e-8)
        if self.direction:
            fopt = -fopt
        # 输出最优解
        selected_indices = np.argsort(xopt)[-self.n_select:]
        print("Selected feature indices:", selected_indices)
        print("Optimized evaluation score:", fopt)
        return selected_indices




