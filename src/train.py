import torch
from src.unet import Unet
from src.noiser import Noiser
from src.data import get_dataloader

class Trainer:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Unet(1, 1, 64, 2, 3).to(self.device)
        self.noiser = Noiser()
        self.dataloader = get_dataloader()
        self.loss = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def __call__(self, epochs: int = 10):
        for e in range(epochs):
            for i, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                # Add noise to samples
                noise_amount = torch.rand(x.shape[0]).to(self.device)
                noisy_x = self.noiser(x, noise_amount)
                # Predict
                y_hat = self.model(noisy_x)
                loss = self.loss(y_hat, x)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if i % 1000 == 0 and i != 0:
                    print(f'Step {(e+1)*i} loss: {round(loss.item()*1000,4)}')

        torch.save(self.model.state_dict(), f'./model.pth')