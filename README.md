# MetaFS: Feature Selection with Meta-learner

## Installation

```bash
pip install -r requirements.txt
```

## Example

```bash
cd examples
python openml.py
```

## Run experiments

### Prepare data

```bash
cd runs
python data_prepare.py --target_feature_num 50 --root_dataset_path ./dataset
```

### Run experiments

```bash
python method_compare.py --method_name MetaFS --model_name LogisticRegression
```

