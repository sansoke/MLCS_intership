import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

num_classes = 10 # datasets have image file of number 0 to 9
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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

model_1.load_state_dict(torch.load('model_1.pth'))
model_2.load_state_dict(torch.load('model_2.pth'))

criterion = nn.CrossEntropyLoss()

model_1.eval()
with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model_1(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * correct/total
        print(f'test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.2f}%')
print('model_1 evaluation done')

model_2.eval()
with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model_1(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * correct/total
        print(f'test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.2f}%')
print('model_2 evaluation done')
