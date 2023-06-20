import numpy as np
import feature_engineer

from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from mlp import MLP

torch.manual_seed(1006)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


model_save = []

import time
start_time = time.time()
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

        print("training set shape: {}".format(train_q.shape))
        input_dim = train_q.shape[1]
        model = MLP(input_dim=input_dim, hidden_dim=256)
        train_dataset = CustomDataset(train_q, train_label)
        train_loader = DataLoader(train_dataset, batch_size=128)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.97)
        criteria = nn.BCELoss()
        num_epoch = 80
        for epoch in range(1, num_epoch):
            # print("current learning rate: {}".format(scheduler.get_last_lr()))
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, lab) in enumerate(train_loader):
                optim.zero_grad()
                y = model(inputs)
                loss = criteria(y, lab)
                loss.backward()
                optim.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}")
            # scheduler.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        model.eval()
        with torch.no_grad():
            valid_pred = model(torch.from_numpy(valid_q).float())
            valid_pred = (valid_pred > 0.5).int()
            f1 = f1_score(valid_label, valid_pred, average='macro')
            print("Validation F1 score: {}".format(f1))
        # model_save.append(model)
        torch.save(model.state_dict(), f'./models/mlp_model_q{q}.pt')

end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Training took {elapsed_time:.2f} seconds.")

# for i, m in enumerate( model_save ):
#     torch.save(m.state_dict(), f'./models/mlp_model_q{i}.pt')