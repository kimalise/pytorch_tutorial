import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

batch_size = 64
learning_rate = 1e-3
num_epoch = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepare dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        # input: [batch_size, 1, 28, 28]
        batch_size = x.size()[0]
        x = self.conv1(x) # [1, 28, 28] -> [8, 28, 28]
        x = self.pool(x) # [8, 28, 28] -> [8, 14, 14]
        x = self.conv2(x) # [8, 14, 14] -> [16, 14, 14]
        x = self.pool(x) # [16, 14, 14] -> [16, 7, 7]
        # x =x.view(batch_size, -1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x) # [16, 7, 7] -> [10]

        return x

model = CNN(in_channels=1, num_classes=10)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def test(data_loader, model):
    if data_loader.dataset.train:
        print("Checking accuracy on training dataset")
    else:
        print("Checking accuracy on test dataset")

    golden_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            # labels = labels.to(device)
            pred_logits = model(images)
            _, preds = pred_logits.max(1)

            golden_labels.extend(labels.numpy().tolist())
            pred_labels.extend(preds.cpu().numpy().tolist())
    
    model.train()

    accuracy = accuracy_score(golden_labels, pred_labels)
    f1 = f1_score(golden_labels, pred_labels, average="macro")

    return accuracy, f1


# train
# tqdm_loop = tqdm(enumerate(train_loader))
for epoch in range(num_epoch):
    # for iter, batch in enumerate(train_loader):

    # tqdm_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    tqdm_loop = tqdm(train_loader, total=len(train_loader))
    for (images, labels) in tqdm_loop:
        # images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        pred_labels = model(images)

        loss = criterion(pred_labels, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_loop.set_description(f"epoch: [{epoch} / {num_epoch}]")
        tqdm_loop.set_postfix(loss = loss.cpu().item())

    accuracy, f1 = test(test_loader, model)
    print("[test] accuracy: {:.4f}, f1: {:.4f}".format(accuracy, f1))


# if __name__ == "__main__":
#     cnn = CNN(in_channels=1, num_classes=10)
#     # input = torch.Tensor((4, 1, 28, 28))
#     # input = torch.FloatTensor(4, 1, 28, 28)
#     # print(input)
#     input = torch.ones(4, 1, 28, 28, dtype=torch.float32)
#     output = cnn(input)
#     print(output.size())


