import numpy as np
import feature_engineer

from sklearn.metrics import f1_score

true = []
for q in range(1, 19):
    true.append(feature_engineer.prepare_label(q).astype(np.uint8).values)
true = np.array(true)
print("The full dataset labels: {}".format(true.shape))

import torch
from mlp import MLP
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def get_preds(target):
    print("==============================================")
    print(f"Eval target: {target}")
    preds = []
    for group, q_list in feature_engineer.LEVEL_QUESTION.items():
        # prepare input for group "0-4" or "5-12" or "13-22"
        # LEVEL_QUESTION {'0-4': [1, 2, 3], '5-12': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], '13-22': [14, 15, 16, 17, 18]}  
        train_data = feature_engineer.prepare_input(group)
        # valid_split = int(0.15 * len(train_data))
        # valid_data = train_data[-valid_split:]
        # valid_data = []
        # train_data = train_data[:-valid_split]
        # print("train data length: {}, valid length: {}".format(len(train_data), len(valid_data)))
        for q in q_list:
            print("***level-group: {}, question: {}".format(group, q))
            FEATURES = feature_engineer.importance_dict[str(q)]
            train_q = train_data[FEATURES].astype(np.float32).values
            # valid_q = valid_data[FEATURES].astype(np.float32).values
            # train_label = feature_engineer.prepare_label(q).astype(np.uint8).values
            # valid_label = train_label[-valid_split:]
            # train_label = train_label[:-valid_split]

            # model = hjc_models[q-1]
            input_dim = train_q.shape[1]
            if target == 'MLP':
                model = MLP(input_dim=input_dim, hidden_dim=256)
                model.load_state_dict(torch.load(f"./models/mlp_model_q{q}.pt"))
                model.eval()
                # model.fit(train_q, train_label, verbose=False, plot=False, eval_set=(valid_q, valid_label))
                # valid_pred = model.predict_proba(valid_q)[:, 1]
                with torch.no_grad():
                    train_pred = model(torch.from_numpy(train_q))
                preds.append(np.array(train_pred))
            elif target == 'CatBoost':
                model = CatBoostClassifier().load_model(f"./models/cat_model_q{q}.bin")
                train_pred = model.predict_proba(train_q)[:, 1]
                preds.append(train_pred)
            elif target == 'XGBoost':
                model = XGBClassifier()
                model.load_model(f"./models/xgb_model_q{q}.bin")
                train_pred = model.predict_proba(train_q)[:, 1]
                preds.append(train_pred)
            else:
                raise Exception("Wrong target (MLP/CatBoost/XGBoost)")
    preds = np.array(preds)
    preds = preds.reshape(18, -1)
    return preds

def evaluate(preds):
    print("Do threshold iteration...")
    scores = []; thresholds = []
    best_score = 0; best_threshold = 0

    for threshold in np.arange(0.4,0.81,0.01):
        # print(f'{threshold:.02f}, ',end='')
        _preds = (preds.reshape((-1))>threshold).astype('int')
        m = f1_score(true.reshape((-1)), _preds, average='macro')   
        scores.append(m)
        thresholds.append(threshold)
        if m>best_score:
            best_score = m
            best_threshold = threshold
    print("Best threshold = {}".format(best_threshold))
    # import matplotlib.pyplot as plt

    # # PLOT THRESHOLD VS. F1_SCORE
    # plt.figure(figsize=(20,5))
    # plt.plot(thresholds,scores,'-o',color='blue')
    # plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
    # plt.xlabel('Threshold',size=14)
    # plt.ylabel('Validation F1 Score',size=14)
    # plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
    # plt.show()

    # print('When using optimal threshold...')

    for k in range(1, 19):
        # COMPUTE F1 SCORE PER QUESTION
        m = f1_score(true[k-1], (preds[k-1]>best_threshold).astype('int'), average='macro')
        print(f'Q{k}: F1 =',m)
        
    # COMPUTE F1 SCORE OVERALL
    m = f1_score(true.reshape((-1)), (preds.reshape((-1))>best_threshold).astype('int'), average='macro')
    print('==> Overall F1 =',m)

evaluate(get_preds("MLP"))
evaluate(get_preds("CatBoost"))
evaluate(get_preds("XGBoost"))