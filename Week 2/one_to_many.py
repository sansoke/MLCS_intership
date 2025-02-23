#==========================================#
# Title:  Stock prices prediction with LSTM; one-to-many
# Author: Jaewoong Han modified by Jeanho Kim
# Date:   2025-02-21
#==========================================#
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

seq_length = 60
epochs = 20
hidden_size = 50
num_layers = 2
output_size = 4
batch_size = 1
lr = 0.0001
input_size = 1
ticker = "XRP-USD"

"""
Step1: Preprocess Datasets
"""
data = yf.download(ticker, start="2018-01-01", end="2025-2-21")

# Selecting features
features = ['Close', 'High', 'Low', 'Open']  # Add more if needed
data = data[features]  

print(data.shape)
# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)


def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length, 0])  # Full feature set
        targets.append(data[i + seq_length, :])   # Predicting 'Close' price
    return np.array(sequences), np.array(targets)


X, y = create_sequences(data_normalized, seq_length)

print(X.shape)
print(y.shape)  
# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(y_test.shape)

X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 60, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 4)
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 60, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 4)


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# """
# Step2: Define LSTM Network
# """
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()  # Activation to add non-linearity

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM processes multi-feature input
        out = out[:, -1, :]  # Take the last time step's output
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out  


# """
# Step3: Define model, criterion, optimizer and train
# """
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    train_loss = 0.0
    for sequences, targets in train_loader:
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    print(f'Epoch {str(epoch+1).zfill(2)}/{epochs}, Loss: {train_loss/len(train_loader):.7f}')
torch.save(model.state_dict(), 'one_to_many.pth')

model.load_state_dict(torch.load('one_to_many.pth'))
# """
# Step4: Evaluate model
# """
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for sequences, targets in test_loader:
        output = model(sequences)
        if output.size(1) == 1:
            predictions.append(output.item())
            actuals.append(targets.item())
        else:
            predictions.extend(output.squeeze().tolist())
            actuals.extend(targets.squeeze().tolist())

print(np.array(predictions).shape)
print(np.array(actuals).shape)


predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 4))
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 4))

print(predictions[-5:, :])
print(actuals[-5:, :])


plt.figure(figsize=(12, 6))
plt.title(f'{ticker} Prices Prediction')
plt.xlabel('Date')
plt.ylabel('Feature values')
plt.plot(actuals, label='Actual')   
plt.plot(predictions, label='Predicted')

plt.grid()
plt.legend()
plt.show()