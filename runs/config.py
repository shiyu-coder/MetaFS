

"""
DI dataset should be downloaded from URL:
https://github.com/ZhangTP1996/OpenFE_reproduce
"""
dataset_config = {
    'VE': {
        'data_id': 1242,
        'type': 'Binclass',
        'metric': 'AUC'
    },
    'HG': {
        'data_id': 23512,
        'type': 'Binclass',
        'metric': 'AUC'
    },
    'JC': {
        'data_id': 41027,
        'type': 'Multiclass',
        'metric': 'Acc'
    },
    'CA': {
        'data_id': 44024,
        'type': 'Regression',
        'metric': 'RMSE'
    },
    'JA': {
        'data_id': 41168,
        'type': 'Multiclass',
        'metric': 'Acc'
    },
    'MI': {
        'data_id': 45579,
        'type': 'Regression',
        'metric': 'RMSE'
    },
}










