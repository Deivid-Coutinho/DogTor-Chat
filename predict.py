import torch
from torchvision import transforms
from PIL import Image
import os
import random

# Caminhos das pastas
doente_dir = 'data/com_doenca'
saudavel_dir = 'data/sem_doenca'

# Função para escolher uma imagem aleatória de cada classe
def escolher_imagens_aleatorias():
    doente_imgs = [os.path.join(doente_dir, f) for f in os.listdir(doente_dir) if f.endswith(('.jpg', '.png'))]
    saudavel_imgs = [os.path.join(saudavel_dir, f) for f in os.listdir(saudavel_dir) if f.endswith(('.jpg', '.png'))]
    
    if not doente_imgs or not saudavel_imgs:
        raise FileNotFoundError("Certifique-se de que as pastas contêm imagens válidas.")

    return [random.choice(doente_imgs), random.choice(saudavel_imgs)]

# Mesmo pré-processamento usado no treino
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Mesmo modelo do treino
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32 * 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Carrega modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
model.eval()

# Rótulos
labels = ['Doente', 'Saudavel']

# Seleciona imagens e faz predição
image_paths = escolher_imagens_aleatorias()
for path in image_paths:
    image = Image.open(path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
        print(f'{path} -> Predição: {labels[pred]}')
