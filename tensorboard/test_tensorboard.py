import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == "__main__":
    x = torch.randn(8, 3, 28, 28)
    model = CNN()
    y = model(x)
    print(y.shape)