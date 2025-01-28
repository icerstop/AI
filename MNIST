import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(120),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = MNIST("mnist", train=True, download=True, transform=transform)
test_dataset = MNIST("mnist", train=False, download=True, transform=transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(next(model.parameters()).device)

patience = 5
best_val_accuracy = 0
epochs_no_improve = 0

scaler = torch.amp.GradScaler()

for epoch in range(10000):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        #print(f"Data on: {images.device}")
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Walidacja
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_accuracy = correct / total
    print(f'Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}')
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
            
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
print(f'Test Accuracy: {test_correct / test_total:.4f}')

wrong_examples = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            if predicted[i] != labels[i] and len(wrong_examples) < 10:
                img = images[i].cpu().squeeze().numpy()
                img = (img*0.5)+ 0.5
                wrong_examples.append((img, labels[i].item(), predicted[i].item()))
                
plt.figure(figsize=(15,5))
for i, (img, true_label, pred_label) in enumerate(wrong_examples):
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'True: {true_label}, Pred: {pred_label}')
    plt.axis('off')
plt.tight_layout()
plt.show()
