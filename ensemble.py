import numpy as np
import copy
import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import f1_score

import feature_engineer
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from mlp import MLP

class EnsembleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc_prob = nn.ModuleList([nn.Linear(input_size, hidden_size) for i in range(18)])
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(input_size, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, input):
        h = []
        for i in range(18):
            h.append(self.fc_prob[i](input[:, i, :]))
        h = torch.stack(h, dim=1)
        h, _ = self.rnn(h)
        return self.fc2(h)

if __name__ == "__main__":
    """
    pred_mlp = []
    pred_catboost = []
    pred_xgboost = []
    for group, q_list in feature_engineer.LEVEL_QUESTION.items():
        train_data = feature_engineer.prepare_input(group)
        for q in q_list:
            print("***level-group: {}, question: {}".format(group, q))
            FEATURES = feature_engineer.importance_dict[str(q)]
            train_q = train_data[FEATURES].astype(np.float32).values
            train_q_mlp = np.zeros_like(train_q)

            mn = np.min(train_q, axis=0)
            mx = np.max(train_q, axis=0)
            for i in range(len(mx)):
                train_q_mlp[:, i] = train_q[:, i] - mn[i]
                if mx[i] - mn[i] > 1e3:
                    train_q_mlp[:, i] = np.log(train_q_mlp[:, i] + 1) / np.log(mx[i] - mn[i] + 1)
                else:
                    train_q_mlp[:, i] /= mx[i] - mn[i]
            mlp = MLP(input_dim=train_q.shape[1], hidden_dim=2048)
            mlp.load_state_dict(torch.load(f"./models/mlp_model_q{q}.pt"))
            mlp.eval()
            with torch.no_grad():
                pred = torch.softmax(mlp(torch.from_numpy(train_q_mlp)), dim=1)[:, 1]
            pred_mlp.append(np.array(pred))

            catboost = CatBoostClassifier().load_model(f"./models/cat_model_q{q}.bin")
            pred = catboost.predict_proba(train_q)[:, 1]
            pred_catboost.append(pred)

            xgboost = XGBClassifier()
            xgboost.load_model(f"./models/xgb_model_q{q}.bin")
            pred = xgboost.predict_proba(train_q)[:, 1]
            pred_xgboost.append(pred)
    pred_mlp = np.array(pred_mlp).reshape(18, -1)
    pred_catboost = np.array(pred_catboost).reshape(18, -1)
    pred_xgboost = np.array(pred_xgboost).reshape(18, -1)
    labels = []
    for q in range(1, 19):
        labels.append(feature_engineer.prepare_label(q).astype(np.uint8).values)
    labels = np.array(labels)
    inputs = np.stack([pred_mlp, pred_catboost, pred_xgboost], axis=2)
    pickle.dump({
        "inputs": inputs,
        "labels": labels
    }, open("./input/ensemble.pkl", "wb"))
    """
    data = pickle.load(open("./input/ensemble.pkl", "rb"))
    inputs = data["inputs"]
    labels = data["labels"]
    print(inputs.shape, labels.shape)
    train_cutoff = int(0.85 * labels.shape[1])
    threshold = [0.53, 0.64, 0.62]

    inputs[:, :, 0] -= 0.53
    inputs[:, :, 1] -= 0.64
    inputs[:, :, 2] -= 0.62

    #train_input = torch.FloatTensor(inputs[:, :train_cutoff, :]).transpose(0, 1)
    #train_label = torch.LongTensor(labels[:, :train_cutoff]).transpose(0, 1)
    train_input = torch.FloatTensor(inputs[:, train_cutoff:, :]).transpose(0, 1)
    train_label = torch.LongTensor(labels[:, train_cutoff:]).transpose(0, 1)
    #prev_input = torch.cat([torch.zeros(train_label.shape[0], 1), train_label[:, :-1]], dim=1).unsqueeze(2)
    #train_input = torch.cat([train_input, prev_input], dim=2)
    val_input = torch.FloatTensor(inputs[:, train_cutoff:, :]).transpose(0, 1)
    val_label = labels[:, train_cutoff:].transpose()
    #prev_input = torch.cat([torch.zeros(val_label.shape[0], 1), torch.tensor(val_label[:, :-1])], dim=1).unsqueeze(2)
    #val_input = torch.cat([val_input, prev_input], dim=2)
    print(val_input.shape, val_label.shape)
    train_dataset = TensorDataset(train_input, train_label)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
    model = EnsembleRNN(3, 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    device = "cuda:0"
    model.to(device)
    epochs = 200
    best_model = None
    best_val_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y_true in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            if epoch >= 300:
                print(x[0], y_true[0], y_pred[0])
                exit(0)
            loss = loss_fn(y_pred.reshape(-1, 2), y_true.flatten())
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()
        epoch_loss = running_loss / len(train_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        model.eval()
        with torch.no_grad():
            val_pred = model(val_input.to(device)).cpu()
            val_pred = (val_pred[:, :, 1] > val_pred[:, :, 0]).int()
            f1 = f1_score(val_label.reshape((-1)), val_pred.reshape((-1)), average='macro')
            #print(f1)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model = copy.deepcopy(model)
                print(epoch, f1)

    print(best_val_f1)
    val_pred = best_model(val_input.to(device)).cpu()
    val_pred = (val_pred[:, :, 1] > val_pred[:, :, 0]).int()
    for q in range(1, 19):
        f1 = f1_score(val_label[:, q - 1], val_pred[:, q - 1], average='macro')
        print(f'Q{q}: F1 =', f1)