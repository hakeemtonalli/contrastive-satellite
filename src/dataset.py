from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

# create dataset using imageFolder base class
# Extend it so it returns the proper triplets for training
# and so that it is consistent with SimCLR process

class EuroSATDataset(ImageFolder):
    
    def __init__(self, train_transform=True):
        super(EuroSATDataset, self).__init__()
        
        if train_transform:
            self.transform = self._train_transform
        else:
            self.transform = self._test_transform
            
    def _train_transform(self):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        return transform 
    
    def _test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        return transform
        
        

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