import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into train/test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split X_train_full into 3 parts (clients)
client_splits = np.array_split(X_train_full, 3)
label_splits = np.array_split(y_train_full, 3)

# Scale features
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

client_splits = np.array_split(X_train_full, 3)
label_splits = np.array_split(y_train_full, 3)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define model
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training function
def train_client(X_train, y_train, epochs=50):
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

# Convert client data to tensors
client_tensors = []
for Xc, yc in zip(client_splits, label_splits):
    X_tensor = torch.tensor(Xc, dtype=torch.float32)
    y_tensor = torch.tensor(yc, dtype=torch.long)
    client_tensors.append((X_tensor, y_tensor))

# Train each client
client_models = []
for i, (Xc, yc) in enumerate(client_tensors):
    print(f"\nTraining Client {i+1}...")
    model = train_client(Xc, yc)
    client_models.append(model)

# Federated averaging
global_model = IrisNet()
global_dict = global_model.state_dict()

# Initialize global model params with average of clients
for key in global_dict.keys():
    global_dict[key] = sum([client_models[i].state_dict()[key] for i in range(3)]) / 3.0

global_model.load_state_dict(global_dict)

# Testing combined global model
global_model.eval()
with torch.no_grad():
    outputs = global_model(X_test)
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f"\nGlobal Model Test Accuracy after Federated Averaging: {acc * 100:.2f}%")
