import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class BootstrapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(35, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        return self.linear3(self.relu(self.linear2(self.dropout(self.relu(self.linear1(x))))))

def NN_pipeline(X,y):
    x_data = torch.tensor(X.values, dtype=torch.float32)
    y_data = torch.tensor(y.values, dtype=torch.float32)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.25, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BootstrapModel().to(device) # instantinate model and move to device

    # Define hyperparameters:
    learning_rate = 0.003
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    # Forward pass
    logits = model(x_train)

    # Logits -> Probabilities b/n 0 and 1
    pred_probab = (torch.sigmoid(logits))

    # Empty loss lists to track values
    epoch_count, train_loss_values, valid_loss_values = [], [], []

    # Loop through the data
    # Number of epochs
    epochs = 10000

    for epoch in range(epochs):

        # Put the model in training mode
        model.train()

        y_logits = model(
            x_train).squeeze()  # forward pass to get predictions; squeeze the logits into the same shape as the labels
        y_pred = (torch.sigmoid(y_logits))  # convert logits into prediction probabilities

        loss = loss_fn(y_logits, y_train)  # compute the loss
        auc = -1#roc_auc_score(y_train, y_pred)  # calculate the accuracy; convert the labels to integers

        optimizer.zero_grad()  # reset the gradients so they don't accumulate each iteration
        loss.backward()  # backward pass: backpropagate the prediction loss
        optimizer.step()  # gradient descent: adjust the parameters by the gradients collected in the backward pass

        # Put the model in evaluation mode
        model.eval()

        with torch.inference_mode():
            valid_logits = model(x_valid).squeeze()
            valid_pred = torch.round(torch.sigmoid(valid_logits))

            valid_loss = loss_fn(valid_logits, y_valid)

            # Print progress a total of 20 times
        if epoch % int(epochs / 20) == 0:
            print(
                f'Epoch: {epoch:4.0f} | Train Loss: {loss:.5f}, Accuracy: {auc:.2f}% | Validation Loss: {valid_loss:.5f}, AUC: {valid_auc:.2f}%')

            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            valid_loss_values.append(valid_loss.detach().numpy())


def main():

    data_path = '/Users/noa/Workspace/bootstrap_results/remote_results/validation_data/simulations_df_raxml.tsv'
    data  = pd.read_csv(data_path, sep='\t')
    data['true_binary_support'] = data['true_support'] == 1
    X = data[[col for col in data.columns if 'feature' in col]]
    y = data['true_binary_support']
    NN_pipeline(X, y)


if __name__ == "__main__":
    main()