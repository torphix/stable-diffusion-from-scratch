import sys
import torch
import argparse
from src.train import Trainer
from src.noiser import Noiser
import matplotlib.pyplot as plt
from src.inference import Sampler
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

    elif command == 'train':
        parser.add_argument('-e', '--epochs', type=int, default=10)
        args, lf_args = parser.parse_known_args()

        trainer = Trainer()
        trainer(args.epochs)

    elif command == 'inference':
        parser.add_argument('-n', '--n_steps', type=int, default=50)
        parser.add_argument('-m', '--model', type=str, default='./model.pth')
        
        args, lf_args = parser.parse_known_args()

        sampler = Sampler(args.model)
        predictions, step_log = sampler(args.n_steps)
        save_image(predictions, 'out.png')
        save_image(step_log, 'step_log.png')
