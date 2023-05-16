import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data

# i referenced https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/ for parts of init and train/test functions!
class Model(nn.Module):
    def __init__(self, input_size=46, hidden_size=30):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class LSTMModel():
    def __init__(self, time_steps=10, epochs=1000, batch_size=16, hidden_size=30):
        self.epochs = epochs
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def process_data(self, df, x, y):
        dfxarray = df[x].to_numpy()
        dfyarray = df[y].to_numpy()

        self.xs = []
        self.ys = []
        for startentr in range(dfxarray.shape[0] - self.time_steps):
            input = []
            for entr in range(self.time_steps):
                input.append(dfxarray[startentr+entr])
            self.xs.append(input)
            self.ys.append([dfyarray[startentr+self.time_steps]])
        self.xs = torch.tensor(np.array(self.xs)).to(torch.float32)
        self.ys = torch.tensor(np.array(self.ys)).to(torch.float32)
        print("x shape:", list(self.xs.shape), ", y shape", list(self.ys.shape))

        total_rows = self.xs.shape[0]
        self.xtrain, self.ytrain = self.xs[:int(total_rows*0.85)], self.ys[:int(total_rows*0.85)]
        self.xtest, self.ytest = self.xs[int(total_rows*0.85):], self.ys[int(total_rows*0.85):]

        self.model = Model(input_size=self.xs.shape[-1], hidden_size=self.hidden_size)

    def fit(self, xdata, ydata):
        self.errors = []
        optimizer = optim.Adam(self.model.parameters())
        loss_fn = nn.MSELoss()
        loader = data.DataLoader(data.TensorDataset(self.xtrain, self.ytrain), shuffle=True, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Validation
            if epoch % 100 != 0:
                continue
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(self.xtrain)
                train_mse = loss_fn(y_pred, self.ytrain)
                y_pred = self.model(self.xtest)
                test_mse = loss_fn(y_pred, self.ytest)
            self.errors.append({'epoch': epoch, "train_mse": float(train_mse), "test_mse": float(test_mse)})
            print("Epoch %d: train MSE %.4f, test MSE %.4f" % (epoch, train_mse, test_mse))

    def predict(self, xdata, traintest="test"):
        if traintest == "train":
            return torch.flatten(self.model(self.xtrain)[:, -1, :]).detach().numpy()
        else:
            return torch.flatten(self.model(self.xtest)[:, -1, :]).detach().numpy()

if __name__ == "__main__":
    model = LSTMModel()
    a = model.train()
    print('')