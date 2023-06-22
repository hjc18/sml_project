import numpy as np
import copy
import feature_engineer

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from mlp import MLP

torch.manual_seed(1006)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

device = "cuda:0"
model_save = []
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
        all_q = np.concatenate([train_q, valid_q], axis=0)
        mn = np.min(all_q, axis=0)
        mx = np.max(all_q, axis=0)
        print(len(mn))
        for i in range(len(mx)):
            train_q[:, i] = train_q[:, i] - mn[i]
            valid_q[:, i] = valid_q[:, i] - mn[i]
            if mx[i] - mn[i] > 1e3:
                train_q[:, i] = np.log(train_q[:, i] + 1) / np.log(mx[i] - mn[i] + 1)
                valid_q[:, i] = np.log(valid_q[:, i] + 1) / np.log(mx[i] - mn[i] + 1)
            else:
                train_q[:, i] /= mx[i] - mn[i]
                valid_q[:, i] /= mx[i] - mn[i]

        train_label = feature_engineer.prepare_label(q).astype(np.uint8).values
        valid_label = train_label[-valid_split:]
        train_label = train_label[:-valid_split]

        print("training set shape: {}".format(train_q.shape))
        input_dim = train_q.shape[1]
        model = MLP(input_dim=input_dim, hidden_dim=2048)
        model = model.to(device)
        train_dataset = CustomDataset(train_q, train_label)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.95)
        criteria = nn.CrossEntropyLoss()
        num_epoch = 500
        best_model = None
        best_val_f1 = 0
        patience = 100
        for epoch in range(1, num_epoch):
            # print("current learning rate: {}".format(scheduler.get_last_lr()))
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, lab) in enumerate(train_loader):
                optim.zero_grad()
                inputs = inputs.to(device)
                lab = lab.to(device)
                y = model(inputs)
                loss = criteria(y, lab)
                loss.backward()
                optim.step()
                running_loss += loss.detach().cpu().item()
                #if batch_idx == 0:
                #    print(y, lab)
                """
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                """
            epoch_loss = running_loss / len(train_loader)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}")
            model.eval()
            with torch.no_grad():
                valid_pred = model(torch.from_numpy(valid_q).float().to(device)).cpu()
                valid_pred = (valid_pred[:, 1] > valid_pred[:, 0]).int()
                f1 = f1_score(valid_label, valid_pred, average='macro')
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_model = copy.deepcopy(model)
                    cur_patience = 0
                else:
                    cur_patience += 1
                if cur_patience >= patience:
                    break
                
            #scheduler.step()
        print("Validation F1 score: {}".format(best_val_f1))
        # model_save.append(model)
        torch.save(best_model.state_dict(), f'./models/mlp_model_q{q}.pt')

# for i, m in enumerate( model_save ):
#     torch.save(m.state_dict(), f'./models/mlp_model_q{i}.pt')