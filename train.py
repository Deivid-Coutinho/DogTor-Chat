import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelo import CNN, device

# Transforms para as imagens
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])


# Dataset e DataLoader
dataset = datasets.ImageFolder(root='data', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Instancia o modelo e move para a GPU (se disponível)
model = CNN().to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
num_epochs = 15
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Época {epoch+1} | Loss: {running_loss / len(dataloader):.4f}')

# Salva o modelo
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/cnn_model.pth')
print("Modelo salvo com sucesso em 'models/cnn_model.pth'")
