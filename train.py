import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10 # datasets have image file of number 0 to 9
epochs = 15

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

total_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
print(len(total_dataset))
train_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [50000, 10000])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
        self.fc1 = nn.Linear(784, 512) # (input vector, output)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784) # (batch size, 28, 28) -> (batch size, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.fc1 = nn.Linear(784, 512) # (input vector, output)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784) # (batch size, 28, 28) -> (batch size, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x

model_1 = model_1()
model_2 = model_2()

criterion = nn.CrossEntropyLoss()
optimizer_1 = optim.Adam(model_1.parameters())
optimizer_2 = optim.Adam(model_2.parameters())

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model_1(images)
        loss = criterion(outputs, labels)
        
        optimizer_1.zero_grad()
        loss.backward()
        optimizer_1.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{str(epoch+1).zfill(2)}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
print('model_1 training done')

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model_2(images)
        loss = criterion(outputs, labels)
        
        optimizer_2.zero_grad()
        loss.backward()
        optimizer_2.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{str(epoch+1).zfill(2)}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
print('model_2 training done')

torch.save(model_1.state_dict(), 'model_1.pth')
torch.save(model_2.state_dict(), 'model_2.pth')

model_1.eval()
with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model_1(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct/total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
print('model_1 evaluation done')

model_2.eval()
with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model_2(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct/total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
print('model_2 evaluation done')

