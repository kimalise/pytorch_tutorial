import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size =1024
num_epochs = 5

model = torchvision.models.vgg16(pretrained=True)
print(model)