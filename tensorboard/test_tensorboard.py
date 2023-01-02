import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参
learning_rate = 1e-3
in_channels = 1
num_classes = 10
batch_size = 64
num_epochs = 5

# 数据集
train_dataset = datasets.MNIST(root="dataset", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型
class CNN(nn.Module):
    def __init__(self, img_channels=3, num_features=8, num_classes=10) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=num_features * 2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * num_features * 2, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

# x = torch.randn(8, 3, 28, 28)
# model = CNN()
# y = model(x)
# print(y.shape)

# 创建模型
model = CNN(img_channels=1, num_features=8, num_classes=10)
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
# tensorboard
# writer = SummaryWriter(f'./run/MNIST/tryingout_tensorboard')
writer = SummaryWriter(f'./run')
global_step = 0

# 训练
for epoch in range(num_epochs):

    losses = []
    accuracies = []

    for step, (batch_images, batch_labels) in enumerate(train_loader):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        predicted_labels = model(batch_images)
        loss = criterion(predicted_labels, batch_labels)

        # gradient 
        optim.zero_grad()
        loss.backward()
        optim.step()

        img_grid = torchvision.utils.make_grid(batch_images)
        writer.add_image('mnist_images', img_grid)
        writer.add_histogram('fc', model.fc.weight)

        losses.append(loss.item())
        _, predicted_index = predicted_labels.max(1)
        num_correct = torch.sum(predicted_index == batch_labels)
        acc = float(num_correct) / float(batch_images.shape[0])
        accuracies.append(acc)

        writer.add_scalar('Training loss', loss, global_step=global_step)
        writer.add_scalar('Training Accuracy', acc, global_step=global_step)
        writer.add_hparams({'lr': learning_rate, 'bsize': batch_size}, {'accuracy': sum(accuracies) / len(accuracies)})

        global_step += 1













    




