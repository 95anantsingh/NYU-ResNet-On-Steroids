
from torchvision import transforms
from torch.utils.data import Dataset


# Custom Dataset Class for Data transforming
class DataTransformer(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.dataset)


# Transformers
class Transforms:

    # Base Transform
    def ToTensor():
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        return transform

    # Test Transforms
    def TestTransform_1():
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        return transform

    # Train Transforms 
    def TrainTransform_1():
        crop_size = 32
        crop_padding = 4
        flip_prob = 0.5
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=crop_padding),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        return transform
    
    def TrainTransform_2():
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
                transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=0.2),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
                transforms.RandomErasing()
        ])

        return transform
