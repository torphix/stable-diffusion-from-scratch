import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_d, out_d, num_filters=64, num_downs=2, num_bottleneck=2):
        super().__init__()

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        self.in_conv = nn.Conv2d(in_d, num_filters, kernel_size=1)

        # Down path
        for i in range(num_downs):
            self.down_path.append(DownBlock(num_filters, num_filters * 2))
            num_filters = num_filters * 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[MidBlock(num_filters, num_filters) for _ in range(num_bottleneck)]
        )

        # Up path
        in_d = num_filters*2 # As concat is used, the input dimension is doubled
        for i in range(num_downs):
            self.up_path.append(UpBlock(in_d, in_d // 4))
            in_d = in_d // 2

        self.out_conv = nn.Conv2d(in_d//2, out_d, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.in_conv(x)
        for down in self.down_path:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        skips.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, skips[i])
        return self.out_conv(x)


class MidBlock(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_d, out_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_d, out_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_d, out_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_d, out_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_d, out_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_d, out_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(out_d, out_d, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.up(x)
        return x


unet = Unet(3, 3, 64, 4, 3)
x = torch.randn(4, 3, 32, 32)
out = unet(x)
print(out.shape)
