import torchvision
from torch.utils.data import DataLoader


def get_dataloader():
    dataset = torchvision.datasets.MNIST(root="./data/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    return train_dataloader