import torch
import torch.nn as nn

# vgg16的channel数量的变化
# VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# linear layers: 4096 * 4096, 1000
VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, type='VGG16') -> None:
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[type])
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x) # [batch_size, 1, 224, 224] -> [batch_size, 512, 7, 7]
        x = x.reshape(x.shape[0], -1) # [batch_size, 512 * 7 * 7]
        x = self.fcs(x) # [batch_size, 512 * 7 * 7] -> [batch_size, 10]
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]

                in_channels = x
            elif x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                ]

        return nn.Sequential(*layers)

if __name__ == "__main__":
    model = VGGNet(type='VGG19')
    print(model)
    # image = torch.ones(4, 3, 224, 224)
    # pred_labels = model(image)
    # print(pred_labels.shape)





