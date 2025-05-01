from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='data', batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, dataset.classes
