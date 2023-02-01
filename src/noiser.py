import torch
import torch.nn as nn


class Noiser(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor, noise_amount:float):
        '''
        To add maximum amount of noise use noise_amount = 1
        to add no noise and keep the raw input use noise_amount = 0
        '''
        noise_amount = noise_amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
        noise = torch.randn_like(x).float() * noise_amount # Scale the magnitude of the noise by the noise_amount
        x = (x * (1-noise_amount)) + noise # Scale the input by 1-noise_amount
        return x 

    