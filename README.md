# MLP/Catboost/XGBoost

huangjc

## Usage
- Put `train_labels.csv` and `train.csv` in folder `input/predict-student-performance-from-game-play`. About 5GB.
- Run `python preprocess.py`
- Run the training script `train_mlp.py`, `train_catboost.py` and `train_xgboost.py`
- After training, you will see models saved in folder `models/`
- When you have all the models in `models`, run `evaluate.py` to make parameter iteration and compute F1 scores **(on the full training set)**
- `submit.py` is a reference for submission to [Kaggle](https://www.kaggle.com/competitions/predict-student-performance-from-game-play). **Note this is a local script, please upload a dataset and correct the model paths for a real online submission.**

`feature_engineer.py` is some magic.
See https://www.kaggle.com/code/vadimkamaev/catboost-new.
See also https://www.kaggle.com/code/gusthema/student-performance-w-tensorflow-decision-forests.