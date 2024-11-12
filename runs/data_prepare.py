import pandas as pd
import os
import sys
import numpy as np
from sklearn.datasets import fetch_openml
import argparse
from config import dataset_config
from scipy.sparse import csr_matrix


def load_dataset(dataset_name):
    data_id = dataset_config[dataset_name]['data_id']
    df_X, sr_y = fetch_openml(data_id=data_id, return_X_y=True, as_frame='auto', parser='auto')
    if isinstance(df_X, csr_matrix):
        df_X = df_X.toarray()
        df_X = pd.DataFrame(df_X)
        sr_y = pd.Series(sr_y)

    if dataset_config[dataset_name]['type'] == 'Binclass' or 'Multiclass':
        sr_y = sr_y.astype('category')
        sr_y = sr_y.cat.codes.astype(int)
    for column in df_X.columns:
        # Assume all object types are categorical variables.
        if df_X[column].dtype == 'object' or 'category':
            labels, unique = pd.factorize(df_X[column])
            df_X[column] = labels
    ndr_X = df_X.astype(float).values
    ndr_y = sr_y.values

    train_X, eval_X, test_X = np.split(ndr_X, [int(0.6 * len(ndr_X)), int(0.8 * len(ndr_X))])
    train_y, eval_y, test_y = np.split(ndr_y, [int(0.6 * len(ndr_y)), int(0.8 * len(ndr_y))])
    return train_X, train_y, eval_X, eval_y, test_X, test_y


"""
By randomly combining operators to construct features, expand the features to a given quantity.
"""

dt_operator = {
    'add': 2,
    'minus': 2,
    'mul': 2,
    'div': 2,
    'abs': 1,
    'sqrt': 1,
    'log': 1,
    'sigmoid': 1
}


def random_generate_single_feature(data_df, height):
    ls_operators = list(dt_operator.keys())
    sr_gen_feature = None
    for i in range(height):
        op = np.random.choice(ls_operators, size=1, replace=False)[0]
        if dt_operator[op] == 1:
            if sr_gen_feature is None:
                feature_idx = np.random.choice(range(data_df.shape[1]), size=1, replace=False)[0]
                sr_gen_feature = data_df.iloc[:, feature_idx]
            if op == 'abs':
                sr_gen_feature = sr_gen_feature.abs()
            elif op == 'sqrt':
                sr_gen_feature = np.sqrt(sr_gen_feature.abs())
            elif op == 'log':
                sr_gen_feature = np.log(sr_gen_feature.abs()+1e-10)
            elif op == 'sigmoid':
                sr_gen_feature = sr_gen_feature / (sr_gen_feature.abs().max()+1e-5)
                sr_gen_feature = 1 / (1 + np.exp(-sr_gen_feature))
            else:
                raise ValueError('Operator {} not supported'.format(op))
        elif dt_operator[op] == 2:
            if sr_gen_feature is None:
                feature_idx = np.random.choice(range(data_df.shape[1]), size=1, replace=False)[0]
                sr_gen_feature = data_df.iloc[:, feature_idx]
            feature_idx = np.random.choice(range(data_df.shape[1]), size=1, replace=False)[0]
            sr_feature = data_df.iloc[:, feature_idx]
            if op == 'add':
                sr_gen_feature = sr_gen_feature + sr_feature
            elif op == 'minus':
                sr_gen_feature = sr_gen_feature - sr_feature
            elif op == 'mul':
                sr_gen_feature = sr_gen_feature * sr_feature
            elif op == 'div':
                sr_feature = sr_feature.replace(0, 1e-9)
                sr_gen_feature = sr_gen_feature / sr_feature
            else:
                raise ValueError('Operator {} not supported'.format(op))
    return sr_gen_feature


def random_generate_features(data_X, gen_num):
    data_df = pd.DataFrame(data_X, columns=[f"feature_{i}" for i in range(data_X.shape[1])])
    count = 0
    ls_sr_gen_feature = []
    while count < gen_num:
        height = np.random.randint(1, 4)
        sr_gen_feature = random_generate_single_feature(data_df, height)
        flag = False
        for col in data_df.columns:
            corr = data_df[col].corr(sr_gen_feature)
            if corr > 0.99:
                flag = True
                break
        tor = sr_gen_feature.diff().abs().mean()
        if tor < 1e-5:
            flag = True
        if not flag:
            sr_gen_feature.name = f"gen_feature_{count}"
            count += 1
            ls_sr_gen_feature.append(sr_gen_feature)
            print(f"Iter-{count}/{gen_num}")
    gen_feature_df = pd.concat(ls_sr_gen_feature, axis=1)
    result_df = pd.concat([data_df, gen_feature_df], axis=1)
    gen_data_X = result_df.values
    return gen_data_X


def prepare_dataset(dataset_name, target_feature_num, root_dataset_path):
    train_X, train_y, eval_X, eval_y, test_X, test_y = load_dataset(dataset_name)
    train_size = train_X.shape[0]
    eval_size = eval_X.shape[0]
    data_X = np.concatenate((train_X, eval_X, test_X), axis=0)
    if data_X.shape[1] > target_feature_num:
        data_X = data_X[:, :target_feature_num]
    elif data_X.shape[1] < target_feature_num:
        data_X = random_generate_features(data_X, target_feature_num-data_X.shape[1])

    train_X = data_X[:train_size]
    eval_X = data_X[train_size:train_size+eval_size]
    test_X = data_X[train_size+eval_size:]

    os.makedirs(f"{root_dataset_path}/{dataset_name}_{target_feature_num}", exist_ok=True)
    np.save(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/X_train.npy", train_X)
    np.save(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/X_val.npy", eval_X)
    np.save(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/X_test.npy", test_X)
    np.save(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/y_train.npy", train_y)
    np.save(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/y_val.npy", eval_y)
    np.save(f"{root_dataset_path}/{dataset_name}_{target_feature_num}/y_test.npy", test_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_feature_num', type=int, default=50)
    parser.add_argument('--root_dataset_path', type=str, default='./dataset')
    args = parser.parse_args()

    root_dataset_path = args.root_dataset_path
    target_feature_num = args.target_feature_num

    print(f"Target feature number: {target_feature_num}, root dataset path: {root_dataset_path}")

    for dataset_name in dataset_config.keys():
        print(f"Processing dataset: {dataset_name}")
        prepare_dataset(dataset_name, target_feature_num, root_dataset_path)


