from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

def test_euro_data(data_path='data/'):
    """
    Check the shape of the EuroSAT dataset for the model.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    image_dataset = ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(image_dataset, batch_size=200, shuffle=True)
    images, labels = next(iter(dataloader))
    print(f"Images shape: {images.size()}")
    print(f"Labels shape: {labels.size()}")
    
    
    
if __name__ == '__main__':
    test_euro_data(data_path='../data/')