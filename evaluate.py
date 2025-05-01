import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelo import CNN, device
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Caminho para os dados e modelo
data_path = 'data'
model_path = 'models/cnn_model.pth'

# Transformação
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset e DataLoader
dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Carregar modelo
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Avaliação
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Relatório de classificação
class_names = dataset.classes
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('evaluation_results.csv')
print("Relatório de classificação salvo como 'evaluation_results.csv'")

# Matriz de confusão
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
