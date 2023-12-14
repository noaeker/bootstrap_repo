import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler



class BootstrapDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx, :]
        label = self.y.iloc[idx]
        return features, label


class BootstrapClassification(nn.Module):
    def __init__(self, x, y):
        super(BootstrapClassification, self).__init__()
        self.layer_1 = nn.Linear(x, y)
        self.layer_2 = nn.Linear(100, 50)
        self.layer_out = nn.Linear(50, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(100)
        self.batchnorm2 = nn.BatchNorm1d(50)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))

        return x


def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred.squeeze(), y.float().squeeze())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            test_loss += loss_fn(pred.squeeze(), y.float().squeeze()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def NN_pipeline(dataloader,test_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BootstrapClassification(5, 7).to(device) # instantinate model and move to device

    # Define hyperparameters:
    learning_rate = 0.001
    loss_fn = nn.BCELoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    # Forward pass
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn)
        #test(dataloader, model, loss_fn)
    print("Done!")
    #train(dataloader, model, loss_fn, optimizer, device)



def main():

    data_path = '/Users/noa/Workspace/bootstrap_results/remote_results/full_data/simulations_df_raxml.tsv'
    data  = pd.read_csv(data_path, sep='\t')
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='all')
    data['true_binary_support'] = data['true_support'] == 1
    data = data.sample(n=10000)
    X = data[[col for col in data.columns if 'feature' in col]]#data[['feature_min_ll_diff','feature_partition_branch','feature_mean_parsimony_trees_binary']]
    y = data['true_binary_support'].astype(int)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_dataset = BootstrapDataset(X=X_train, y=y_train)
    test_dataset = BootstrapDataset(X=X_test, y=y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    NN_pipeline(train_dataloader,test_dataloader)

if __name__ == "__main__":
    main()