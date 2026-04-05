import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# load data
train_set = torchvision.datasets.MNIST('./data', train=True, download=True,
                                       transform=transforms.ToTensor())
test_set = torchvision.datasets.MNIST('./data', train=False, download=True,
                                      transform=transforms.ToTensor())

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.view(-1, 784)
        output = model(images)
        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
        total += labels.size(0)

    print(f"epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f} acc={correct / total:.4f}")

# export weights
os.makedirs('weights', exist_ok=True)
for name, param in model.named_parameters():
    p = param.detach()
    if 'weight' in name:
        p = p.T.contiguous()
    p.numpy().tofile(f"weights/{name}.bin")

# export test images and labels
test_images = test_set.data.float().view(10000, 784) / 255.0
test_images.numpy().tofile("weights/test_images.bin")
test_set.targets.numpy().astype('float32').tofile("weights/test_labels.bin")
print("saved 10000 test images and labels")
