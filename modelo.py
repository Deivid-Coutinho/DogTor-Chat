import torch
import torch.nn as nn

# Define o modelo CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Função para selecionar o dispositivo com fallback
def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            _ = torch.cuda.memory_allocated()
            return torch.device("cuda")
        except:
            pass
    return torch.device("cpu")

device = get_device()
