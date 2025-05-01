import os
import csv
import random
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from modelo import CNN, device

# Caminho do modelo e imagens
model_path = 'models/cnn_model.pth'
image_folder = 'predict_imagens'
labels = ['Doente', 'Saudavel']

# Pré-processamento
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Carrega modelo
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Lista de imagens da pasta
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Seleciona 2 imagens aleatórias
selected_files = random.sample(image_files, 2)

# Lista de resultados
resultados = []

# Cria figura
plt.figure(figsize=(10, 5))

for i, filename in enumerate(selected_files):
    path = os.path.join(image_folder, filename)
    image = Image.open(path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = labels[predicted.item()]

    # Adiciona ao CSV
    resultados.append([filename, label])

    # Mostra a imagem com o rótulo
    plt.subplot(1, 2, i+1)
    plt.imshow(image)
    plt.title(f'{filename}\n{label}')
    plt.axis('off')

# Salva CSV
with open('resultados.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['imagem', 'previsao'])
    writer.writerows(resultados)

print("Resultados salvos em 'resultados.csv'")
plt.tight_layout()
plt.show()
