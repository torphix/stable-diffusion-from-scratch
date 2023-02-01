import sys
import torch
import argparse
from src.noiser import Noiser
import matplotlib.pyplot as plt
from src.data import get_dataloader
from torchvision.utils import save_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    command = sys.argv[1] 
    if command == 'test_noiser':
        dataloader = get_dataloader()
        x, y = next(iter(dataloader))
        amount = torch.linspace(0, 1, x.shape[0])
        noiser = Noiser()
        output = noiser(x, amount)
        save_image(output, 'out.png')