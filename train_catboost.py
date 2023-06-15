import numpy as np
import feature_engineer

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# model_save = []
for group, q_list in feature_engineer.LEVEL_QUESTION.items():
    # prepare input for group "0-4" or "5-12" or "13-22"
    # LEVEL_QUESTION {'0-4': [1, 2, 3], '5-12': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], '13-22': [14, 15, 16, 17, 18]}  
    train_data = feature_engineer.prepare_input(group)
    valid_split = int(0.15 * len(train_data))
    valid_data = train_data[-valid_split:]
    train_data = train_data[:-valid_split]
    print("train data length: {}, valid length: {}".format(len(train_data), len(valid_data)))
    for q in q_list:
        print("==================================================")
        print("***level-group: {}, question: {}".format(group, q))
        FEATURES = feature_engineer.importance_dict[str(q)]
        train_q = train_data[FEATURES].astype(np.float32).values
        valid_q = valid_data[FEATURES].astype(np.float32).values
        train_label = feature_engineer.prepare_label(q).astype(np.uint8).values
        valid_label = train_label[-valid_split:]
        train_label = train_label[:-valid_split]

        model = CatBoostClassifier(
            n_estimators=300,
            learning_rate=0.05,
            depth=6,
            early_stopping_rounds=30,
            custom_metric=['F1', 'Precision', 'Recall'],
            train_dir=f'catboost_log/info_q{q}'
        )
        model.fit(train_q, train_label, verbose=False, plot=False, eval_set=(valid_q, valid_label))
        valid_pred = model.predict(valid_q)
        f1 = f1_score(valid_label, valid_pred, average='macro')
        print("Validation F1 score: {}".format(f1))
        model.save_model(f'./models/cat_model_q{q}.bin')

# for i, m in enumerate( model_save ):
#     m.save_model(f'./models/cat_model_q{i}.bin')