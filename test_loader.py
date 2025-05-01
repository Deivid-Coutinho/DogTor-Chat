from utils import get_dataloaders
import matplotlib.pyplot as plt
import torchvision

loader, classes = get_dataloaders('data')

# Mostrar 1 batch de imagens
images, labels = next(iter(loader))
grid = torchvision.utils.make_grid(images)

plt.imshow(grid.permute(1, 2, 0))
plt.title([classes[label] for label in labels])
plt.axis('off')
plt.show()

