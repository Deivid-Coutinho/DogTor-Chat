import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelo import CNN, device
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Caminhos e configurações
val_path = 'val'  # pasta de validação
model_path = 'models/cnn_model.pth'
batch_size = 16

# Transformações
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset de validação
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Carrega o modelo
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Avaliação
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Nomes das classes
class_names = val_dataset.classes

# Relatório de validação
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('validation_report.csv')
print("✅ Relatório de validação salvo como 'validation_report.csv'")

# Matriz de confusão
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Oranges')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Validação')
plt.tight_layout()
plt.savefig('validation_confusion_matrix.png')
plt.show()
