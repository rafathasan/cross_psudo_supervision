import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from dataclasses import dataclass
from utils import download_data

@dataclass
class BingRGB(pl.LightningDataModule):
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    download: dict
    train_sup_data_path: str
    train_unsup_data_path: str
    val_data_path: str
    test_data_path: str
    transforms: str
    num_workers: int

    def __post_init__(self):
        super().__init__()

    def prepare_data(self):
        # Use this method to perform any setup that requires downloading or preprocessing data.
        download_data(**self.download.data)
        download_data(**self.download.weight)

    def setup(self, stage=None):
        # Use this method to initialize datasets and perform any necessary setup.
        # The stage argument specifies the stage of training, either 'fit', 'validate', or 'test'.
        # If stage is None, this method is called for all stages.

        if stage in (None, 'fit'):
            train_transforms = self.train_transforms()
            self.train_dataset = MyDataset(self.train_sup_data_path, self.train_unsup_data_path, train_transforms)

            val_transforms = self.val_transforms()
            self.val_dataset = MyDataset(self.val_data_path, None, val_transforms)

        if stage in (None, 'test'):
            test_transforms = self.test_transforms()
            self.test_dataset = MyDataset(self.test_data_path, None, test_transforms)

    def train_dataloader(self):
        # Use this method to return a PyTorch DataLoader instance for the training dataset.
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Use this method to return a PyTorch DataLoader instance for the validation dataset.
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Use this method to return a PyTorch DataLoader instance for the test dataset.
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        # Use this method to return a PyTorch DataLoader instance for the prediction dataset.
        return self.test_dataloader()

    def train_transforms(self):
        # Use this method to return a list of data augmentations to apply to the training data.
        return transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.303, 0.313, 0.220], std=[0.108, 0.088, 0.075])
        ])

    def val_transforms(self):
        # Use this method to return a list of data augmentations to apply to the validation data.
        return transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transforms(self):
        # Use this method to return a list of data augmentations to apply to the test data.
        return transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
@dataclass
class MyDataset(Dataset):
    sup_data_path: str
    unsup_data_path: str = None
    transforms: object = None
    def __post_init__(self):
        # Get a list of image filenames in the data directory
        self.sup_filenames = []
        for filename in os.listdir(self.sup_data_path): # slicing large than test image will give error. if slice value > min(train_size, test_size) 
            if 'gt' not in filename:
                self.sup_filenames.append(filename)

        if self.unsup_data_path:
            self.unsup_filenames = []
            for filename in os.listdir(self.unsup_data_path):
                if 'gt' not in filename:
                    self.unsup_filenames.append(filename)

    def __len__(self):
        # Return the number of images in the dataset
        return max(len(self.sup_filenames), len(self.unsup_filenames)) if self.unsup_data_path else len(self.sup_filenames)

    def __getitem__(self, index):
        # Load an image and its corresponding label, and apply the specified transforms (if any)
        image_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_filename = self.sup_filenames[index%len(self.sup_filenames)]

        image_path = os.path.join(self.sup_data_path, image_filename)
        
        image = Image.open(image_path).convert('RGB')
        mask = np.array(Image.open(image_path.replace('.png','_gt.png')).convert('L'))

        if self.transforms is not None:
            image = self.transforms(image)
            image = image_transform(image)
            mask = self.transforms(mask)*255
        
        image[:,(mask == 0).squeeze()] = 0

        if self.unsup_data_path:
            unsup_filename = self.unsup_filenames[index%len(self.unsup_filenames)]
            unsup_path = os.path.join(self.unsup_data_path, unsup_filename)
            unsup = Image.open(unsup_path).convert('RGB')
            if self.transforms is not None:
                unsup = self.transforms(unsup)
                unsup = image_transform(unsup)

            return unsup, image, mask.long()

        return image, mask.long()