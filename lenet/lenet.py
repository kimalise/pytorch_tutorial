import torch
import torch.nn as nn

# LeNet: architecture
# channel变化: [1, 32, 32] -> [5, 5] kernel s=1, p=0 -> avg pool s=2, p=0 -> [5, 5], s=1, p=0 -> avg pool s=2, p=0
# 特侦图变化: [32, 32] -> [28, 28] -> [14, 14] -> [10, 10] -> [5, 5] -> [1, 1]
class LeNet(nn.Module):
    def __init__(self, in_channels) -> None:
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x)) # [batch_size, 120, 1, 1]
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    input = torch.ones(4, 1, 32, 32)
    model = LeNet(1)
    output = model(input)
    print(output.shape)
