import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

writer = SummaryWriter(log_dir="./run")

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset = datasets.MNIST("dataset", train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = torchvision.models.resnet18(pretrained=True)
# print(model)
model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

images, labels = next(iter(dataloader))
print(images.shape)
print(labels.shape)

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()



