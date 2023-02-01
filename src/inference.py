import torch
from src.unet import Unet


class Sampler:
    def __init__(self, model_path) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Unet(1, 1, 64, 2, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, n_steps:int):
        '''
        Starts with random noise samples n_steps
        '''
        predictions = []
        step_log = []
        with torch.no_grad():
            noise = torch.randn((8, 1, 28, 28)).to(self.device)
            x = noise
            for step in range(n_steps):
                prediction = self.model(x)
                predictions.append(prediction.detach().cpu())
                step_magnitude = 1/(n_steps - step)
                x = x*(1-step_magnitude) + prediction*step_magnitude # Step in the direction of the prediction re adding noise
                step_log.append(x.detach().cpu())
        return torch.cat(predictions), torch.cat(step_log)

