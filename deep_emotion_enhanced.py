import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, max(1, channels // reduction), kernel_size=1)
        self.fc2 = nn.Conv2d(max(1, channels // reduction), channels, kernel_size=1)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        out = F.dropout(out, p=0.3)

        return out


class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)

        # Residual blocks
        self.res_block1 = ResidualBlock(10, 10)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.res_block2 = ResidualBlock(10, 20, downsample=nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=1, stride=1),
            nn.BatchNorm2d(20)
        ))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.res_block3 = ResidualBlock(20, 40, downsample=nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=1, stride=1),
            nn.BatchNorm2d(40)
        ))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.res_block4 = ResidualBlock(40, 80, downsample=nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=1, stride=1),
            nn.BatchNorm2d(80)
        ))
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(80 * 3 * 3, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 7)

        # Localization network for STN
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Fully connected layers for the localization network
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input):
        # Apply STN
        out = self.stn(input)

        # Initial convolutional layer
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)

        # Residual blocks with pooling
        out = self.res_block1(out)
        out = self.pool1(out)

        out = self.res_block2(out)
        out = self.pool2(out)

        out = self.res_block3(out)
        out = self.pool3(out)

        out = self.res_block4(out)
        out = self.pool4(out)

        # Flatten the output
        out = out.view(-1, 80 * 3 * 3)

        # Fully connected layers
        out = self.fc1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


