import copy
import warnings
from itertools import combinations_with_replacement

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from metafs.feature_clustering import feature_space_reduction_clustering
from metafs.util import generate_unique_combinations, ProcessPool
from metafs.prior_indicator import PriorInfo
from metafs.sample_and_select import *


def train_valid_split(X, y, valid_ratio=0.25):
    n = len(X)
    valid_size = int(n * valid_ratio)
    train_size = n - valid_size
    train_X = X[:train_size]
    train_y = y[:train_size]
    valid_X = X[train_size:]
    valid_y = y[train_size:]
    return train_X, train_y, valid_X, valid_y


def meta_sample_eval(cluster_list, feature_list, train_X, train_y, valid_X, valid_y, model, eval_func):
    sample_X = np.nan_to_num(train_X[:, feature_list])
    model.fit(sample_X, train_y)
    score = eval_func(model, valid_X[:, feature_list], valid_y)
    return cluster_list, score


def feature_subset_train(train_X, train_y, eval_X, model, model_predict_func):
    model = copy.deepcopy(model)
    model.fit(train_X, train_y)
    y_pred = model_predict_func(model, eval_X)
    return y_pred


class MetaFS:

    def __init__(self,
                 model,
                 eval_func,
                 model_predict_func,
                 n_select,
                 ensemble_eval_func,
                 ensemble_count=1,
                 n_meta_samples=400,
                 n_tot=1e6,
                 filter_threshold=None,
                 filter_quantile=0.01,
                 n_opt=25,
                 n_clusters=0,
                 direction=True,
                 prior_feature_list=None,
                 cv_n_splits=3,
                 n_sample_paths=6,
                 connectivity_check_threshold=0.2,
                 n_jobs=1,
                 verbose=1):
        """
        MetaFS: A Feature Selection Method Based on Meta-Learning
        :param model: object
            Base model used for training and prediction.
        :param eval_func: callable
            Evaluation function to assess the model's performance.
        :param model_predict_func: callable
            Function to make predictions using the model.
        :param n_select: int
            Number of selected features.
        :param ensemble_eval_func: callable
            Evaluation function for the ensemble model.
        :param ensemble_count: int, optional (default=1)
            Number of ensemble models.
        :param n_meta_samples: int, optional (default=400)
            Number of meta-samples.
        :param n_tot: int, optional (default=1e6)
            Number of random sampled feature subsets.
        :param filter_threshold: float, optional (default=None)
            Threshold for filtering feature subsets. If None, filter_quantile will be used.
        :param filter_quantile: float, optional (default=0.01)
            Quantile for filtering feature subsets. If filter_threshold is None, this parameter will be used.
        :param n_opt: int, optional (default=25)
            Number of optimal feature subsets to select.
        :param n_clusters: int, optional (default=0)
            Number of clusters for feature clustering.
        :param direction: bool, optional (default=True)
            Direction for optimization (True for maximization, False for minimization).
        :param prior_feature_list: list, optional (default=None)
            List of prior features to consider.
        :param cv_n_splits: int, optional (default=3)
            Number of splits for cross-validation.
        :param n_sample_paths: int, optional (default=6)
            Number of sample paths for robust component checking.
        :param connectivity_check_threshold: float, optional (default=0.2)
            Threshold for connectivity check.
        :param n_jobs: int, optional (default=1)
            Number of parallel jobs to run.
        :param verbose: int, optional (default=1)
            Verbosity level.
        """
        self.model = model
        self.ensemble_count = ensemble_count
        self.eval_func = eval_func
        self.model_predict_func = model_predict_func
        self.ensemble_eval_func = ensemble_eval_func
        self.n_select = int(n_select)
        self.n_meta_samples = n_meta_samples
        self.n_tot = int(n_tot)
        self.filter_threshold = filter_threshold
        self.filter_quantile = filter_quantile
        self.n_opt = int(n_opt)
        self.n_clusters = int(n_clusters)
        self.direction = direction
        self.prior_feature_list = prior_feature_list
        if self.prior_feature_list is None:
            self.prior_feature_list = []
        self.n_jobs = n_jobs
        self.n_cluster_sample = 500000
        self.cv_n_splits = cv_n_splits
        self.n_sample_paths = n_sample_paths
        self.connectivity_check_threshold = connectivity_check_threshold
        self.cluster_map = None
        self.n_features = 0
        self.verbose = verbose
        self.meta_learner_param = {'max_depth': 6, 'num_leaves': 16, 'reg_alpha': 0.0001, 'reg_lambda': 0.01, 'learning_rate': 0.1,
                                   'n_jobs': 1, 'verbose': -1, 'n_estimators': 100}
        self.prior_info_generator = None
        self.meta_learner = None
        self.with_prior = False

    def add_prior_feature(self, feature_name, agg_operator):
        self.prior_feature_list.append((feature_name, agg_operator))

    def fit(self, train_X, train_y, valid_X, valid_y):
        self.n_features = train_X.shape[1]

        if self.n_clusters > 0 and self.n_features % self.n_clusters != 0:
            warnings.warn("feature num had better be divisible by n_cluster")

        self.cluster_map = self.feature_clustering(train_X)

        meta_sample_pairs = self.collect_train_info(train_X, train_y, valid_X, valid_y)
        ndr_train_info, ndr_meta_y = self.prepare_train_info(meta_sample_pairs)

        if len(self.prior_feature_list) > 0:
            ndr_prior_info = self.collect_prior_info(train_X, train_y, meta_sample_pairs)
        else:
            ndr_prior_info = None

        meta_X, meta_y = self.prepare_meta_dataset(ndr_train_info, ndr_prior_info, ndr_meta_y)

        if self.verbose:
            print(f"Meta dataset size: {meta_X.shape}")

        self.meta_learner = self.train_meta_learner(meta_X, meta_y)

        sample_subsets_list = self.generate_opt_samples()

        best_subset_list = self.optimize_ensemble(sample_subsets_list, train_X, train_y, valid_X, valid_y)
        return best_subset_list

    def optimize_ensemble(self, sample_subsets_list, train_X, train_y, valid_X, valid_y):
        print("Evaluating candidate feature subsets ...")

        pool = ProcessPool(self.n_jobs, self.verbose)
        for i in range(len(sample_subsets_list)):
            sample_subsets = sample_subsets_list[i]
            subset_X = train_X[:, sample_subsets]
            subset_eval_X = valid_X[:, sample_subsets]
            pool.apply_async(feature_subset_train, args=(subset_X, train_y, subset_eval_X, self.model, self.model_predict_func))

        subset_preds = pool.progress_bar(desc="Training candidate feature subsets")
        pool.close()

        ensemble_indices = list(combinations_with_replacement(list(range(len(sample_subsets_list))), self.ensemble_count))
        score_list = []
        print(f"Evaluating ensemble score ...")
        for i in trange(len(ensemble_indices)):
            indices = ensemble_indices[i]
            y_preds = [subset_preds[j] for j in indices]
            score = self.ensemble_eval_func(y_preds, valid_y)
            score_list.append(score)

        if self.direction:
            best_idx = np.nanargmax(score_list)
        else:
            best_idx = np.nanargmin(score_list)
        best_score = score_list[best_idx]
        best_subset_list = [sample_subsets_list[j] for j in ensemble_indices[best_idx]]
        print(f"Best ensemble score: {best_score:.5f}")
        return best_subset_list

    def feature_clustering(self, train_X):
        """
        Feature Space Reduction Clustering
        """
        if self.n_clusters == 0:
            dt_cluster = {i: [i] for i in range(self.n_features)}
            return dt_cluster

        if self.verbose:
            print(f"Feature clustering, n_cluster: {self.n_clusters}, feature num: {train_X.shape[1]}")
        # 1. Sample n_cluster_sample samples from train_X
        ls_sample_idx = np.random.choice(train_X.shape[0], min(len(train_X), self.n_cluster_sample), replace=False)
        sample_X = train_X[ls_sample_idx]
        sample_X = np.nan_to_num(sample_X)

        if self.verbose:
            print(f"Sampling data for calculating correlation matrix, data size: {len(sample_X)}")

        # 2. Calculate the correlation matrix of sample_X
        corr_mat = np.corrcoef(sample_X, rowvar=False)

        # 3. Cluster the features based on the correlation matrix
        labels = feature_space_reduction_clustering(corr_mat, train_X.shape[1], self.n_clusters)
        dt_cluster = {i: [j for j, label in enumerate(labels) if label == i] for i in range(self.n_clusters)}

        return dt_cluster

    def collect_train_info(self, train_X, train_y, valid_X, valid_y):

        comb_generator = generate_unique_combinations(self.n_features, self.n_select)
        pool = ProcessPool(self.n_jobs, self.verbose)
        for i in range(self.n_meta_samples):
            cluster_idx_list = comb_generator()
            feature_idx_list = []
            for cluster_idx in cluster_idx_list:
                feature_idx_list.extend(self.cluster_map[cluster_idx])

            pool.apply_async(meta_sample_eval, args=(cluster_idx_list, feature_idx_list, train_X, train_y, valid_X, valid_y, self.model, self.eval_func))

        meta_sample_pairs = pool.progress_bar(desc="Collecting training information")
        pool.close()

        return meta_sample_pairs

    def cluster_list_to_feature_list(self, cluster_list):
        feature_subsets_list = []
        for cluster_idx_list in cluster_list:
            feature_idx_list = []
            for cluster_idx in cluster_idx_list:
                feature_idx_list.extend(self.cluster_map[cluster_idx])
            feature_subsets_list.append(feature_idx_list)
        return feature_subsets_list

    def collect_prior_info(self, train_X, train_y, meta_sample_pairs):
        print("Collecting prior information ...")
        cluster_list = [feature_idx_list for feature_idx_list, _ in meta_sample_pairs]
        feature_subsets_list = self.cluster_list_to_feature_list(cluster_list)
        self.prior_info_generator = PriorInfo(self.prior_feature_list)
        self.prior_info_generator.fit(train_X, train_y)
        ndr_prior_info = self.prior_info_generator.transform(feature_subsets_list)
        return ndr_prior_info

    def prepare_train_info(self, meta_sample_pairs):
        feature_subsets_list = [feature_idx_list for feature_idx_list, _ in meta_sample_pairs]
        scores = [score for _, score in meta_sample_pairs]

        train_info_list = []
        for i in range(len(feature_subsets_list)):
            feature_idx_list = feature_subsets_list[i]
            ndr_train_info = np.zeros(self.n_features)
            ndr_train_info[feature_idx_list] = 1
            train_info_list.append(ndr_train_info)
        ndr_train_info = np.stack(train_info_list, axis=0)
        ndr_meta_y = np.array(scores)
        return ndr_train_info, ndr_meta_y

    def train_meta_learner(self, meta_X, meta_y):
        model = LGBMRegressor(**self.meta_learner_param)
        model.fit(meta_X, meta_y)
        return model

    def meta_learner_cv(self, X, y):
        model = LGBMRegressor(**self.meta_learner_param)
        kf = KFold(n_splits=self.cv_n_splits, shuffle=True, random_state=42)
        r2s = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            r2s.append(r2)

        return np.mean(r2s)

    def prepare_meta_dataset(self, ndr_train, ndr_prior, ndr_meta_y):

        r2_wo_prior = self.meta_learner_cv(ndr_train, ndr_meta_y)

        if ndr_prior is not None:
            ndr_meta_X = np.concatenate([ndr_train, ndr_prior], axis=1)
            r2_w_prior = self.meta_learner_cv(ndr_meta_X, ndr_meta_y)
            if self.verbose:
                print(f"Cross Validation on Meta-learner: R2 without prior: {r2_wo_prior:.3f}, R2 with prior: {r2_w_prior:.3f}")
            if r2_wo_prior < r2_w_prior:
                self.with_prior = True
                return ndr_meta_X, ndr_meta_y
        else:
            if self.verbose:
                print(f"Cross Validation on Meta-learner: R2 without prior: {r2_wo_prior:.3f}")

        return ndr_train, ndr_meta_y

    def on_same_component(self, subset1, subset2):
        diff_indexes_0 = list(np.where((subset1 - subset2) == -1)[0])
        diff_indexes_1 = list(np.where((subset1 - subset2) == 1)[0])
        mid_subset = copy.deepcopy(subset1)
        while len(diff_indexes_0) > 0 and len(diff_indexes_1) > 0:
            # Randomly select a position for a 1 and a 0.
            idx0 = np.random.choice(range(len(diff_indexes_0)), 1)[0]
            idx1 = np.random.choice(range(len(diff_indexes_1)), 1)[0]
            pos0, pos1 = diff_indexes_0[idx0], diff_indexes_1[idx1]

            # Swap the values of these two positions.
            mid_subset[pos0], mid_subset[pos1] = mid_subset[pos1], mid_subset[pos0]
            if self.with_prior:
                cluster_list = [np.where(mid_subset == 1)[0]]
                feature_subsets_list = self.cluster_list_to_feature_list(cluster_list)
                ndr_prior_info = self.prior_info_generator.transform(feature_subsets_list)
                mid_subset_X = np.concatenate([mid_subset.reshape(1, -1), ndr_prior_info], axis=1)
            else:
                mid_subset_X = mid_subset.reshape(1, -1)

            if self.meta_learner.predict(mid_subset_X)[0] < self.filter_threshold:
                return False

            # Remove the processed positions from the differential index.
            diff_indexes_0.pop(idx0)
            diff_indexes_1.pop(idx1)

        return True

    def on_same_component_robust(self, subset1, subset2):
        same_count = 0
        for i in range(self.n_sample_paths):
            np.random.seed(100 + i)
            result = self.on_same_component(subset1, subset2)
            same_count += int(result)
        if same_count / self.n_sample_paths < self.connectivity_check_threshold:
            return False
        else:
            return True

    def allocate_to_components(self, ndr_subsets, ndr_pred):
        comp_list = []
        comp_pred_list = []
        if self.verbose:
            print("Allocating feature subsets to components ...")
        for i in range(len(ndr_subsets)):
            ndr_subset = ndr_subsets[i]
            flag = False
            for j in range(len(comp_list)):
                comp = comp_list[j]
                for k in range(len(comp)):
                    comp_subset = comp[k]
                    if self.on_same_component_robust(ndr_subset, comp_subset):
                        comp_list[j].append(ndr_subset)
                        comp_pred_list[j].append(ndr_pred[i])
                        flag = True
                        break
                if flag:
                    break
            if not flag:
                comp_list.append([ndr_subset])
                comp_pred_list.append([ndr_pred[i]])
        # Sort by the largest predicted values in descending order
        comp_max_pred_list = [np.nanmax(pred) for pred in comp_pred_list]
        comp_idx_list = np.argsort(comp_max_pred_list)[::-1]
        comp_list = [comp_list[i] for i in comp_idx_list]
        comp_pred_list = [comp_pred_list[i] for i in comp_idx_list]
        return comp_list, comp_pred_list

    def assign_sample_num_to_components(self, comp_list):
        print(f"Assigning sample num to components ...")
        comp_num = len(comp_list)
        if self.n_opt < comp_num:
            n_sample_list = [1 for _ in range(self.n_opt)] + [0 for _ in range(comp_num - self.n_opt)]
            return n_sample_list

        n_sample_list = [1 for _ in range(comp_num)]
        left_num = self.n_opt - comp_num
        comp_count_list = [len(comp) for comp in comp_list]
        total_count = np.nansum(comp_count_list)
        comp_weight_list = [count / total_count for count in comp_count_list]
        for i in range(len(comp_weight_list)):
            sample_count = min(min(round(comp_weight_list[i] * left_num), left_num), len(comp_list[i])-n_sample_list[i])
            n_sample_list[i] += sample_count
            left_num -= sample_count
            if left_num <= 0:
                break
        return n_sample_list

    def sample_on_component(self, comp_list, comp_pred_list, n_sample_list):
        print(f"Sampling on components ...")
        sample_subsets_list = []
        for i in range(len(comp_list)):
            comp = comp_list[i]
            comp_pred = comp_pred_list[i]
            n_sample = n_sample_list[i]
            if n_sample > 0:
                # First take out the sample with the highest predicted value.
                comp_sample_list = []
                max_idx = np.nanargmax(comp_pred)
                max_pred_subsets = comp[max_idx]
                comp_sample_list.append(list(max_pred_subsets))
                while len(comp_sample_list) < n_sample:
                    # Select the remaining samples according to the FPS algorithm.
                    dis_list = []
                    for j in range(len(comp)):
                        subset_dis_list = []
                        subset = comp[j]
                        for sample_subset in comp_sample_list:
                            subset_dis_list.append(hamming_distance(subset, sample_subset))
                        min_dis = np.nanmin(subset_dis_list)
                        dis_list.append(min_dis)
                    max_dis_idx = np.nanargmax(dis_list)
                    max_dis_subset = comp[max_dis_idx]
                    comp_sample_list.append(list(max_dis_subset))
                sample_subsets_list.extend(comp_sample_list)
        return sample_subsets_list

    def generate_opt_samples(self):
        ndr_tot_subsets = np.array(sample_feature_subset(self.n_features, self.n_select, self.n_tot))

        if self.with_prior:
            cluster_list = []
            for i in range(len(ndr_tot_subsets)):
                cluster_idx_list = np.where(ndr_tot_subsets[i] == 1)[0]
                cluster_list.append(cluster_idx_list)
            feature_subsets_list = self.cluster_list_to_feature_list(cluster_list)
            ndr_prior_info = self.prior_info_generator.transform(feature_subsets_list)
            ndr_tot_X = np.concatenate([ndr_tot_subsets, ndr_prior_info], axis=1)
        else:
            ndr_tot_X = ndr_tot_subsets

        ndr_tot_pred = self.meta_learner.predict(ndr_tot_X)
        if not self.direction:
            ndr_tot_pred = -ndr_tot_pred

        if self.filter_threshold is None:
            self.filter_threshold = np.quantile(ndr_tot_pred, (1-self.filter_quantile/100))
        filtered_indices = np.where(ndr_tot_pred > self.filter_threshold)[0]
        ndr_fil_subsets = ndr_tot_subsets[filtered_indices]
        ndr_fil_pred = ndr_tot_pred[filtered_indices]
        if self.verbose:
            print(f"Filtered threshold: {self.filter_threshold:.5f}, Filtered sample size: {len(ndr_fil_subsets)}")

        comp_list, comp_pred_list = self.allocate_to_components(ndr_fil_subsets, ndr_fil_pred)
        if self.verbose:
            print(f"Component num: {len(comp_list)}")

        n_sample_list = self.assign_sample_num_to_components(comp_list)
        if self.verbose:
            print(f"Sample num : {np.nansum(n_sample_list)}")

        sample_subsets_list = self.sample_on_component(comp_list, comp_pred_list, n_sample_list)
        sample_cluster_list = [[j for j, value in enumerate(seq) if value == 1] for seq in sample_subsets_list]
        sample_features_list = self.cluster_list_to_feature_list(sample_cluster_list)
        return sample_features_list


