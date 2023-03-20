import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

train_path = "/src/data/_BingRGB/train"
val_path = "/src/data/_BingRGB/val"
test_path = "/src/data/_BingRGB/test"

class BDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data_path = train_path, val_data_path = val_path, test_data_path = test_path):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size

    def prepare_data(self):
        # Use this method to perform any setup that requires downloading or preprocessing data.
        pass

    def setup(self, stage=None):
        # Use this method to initialize datasets and perform any necessary setup.
        # The stage argument specifies the stage of training, either 'fit', 'validate', or 'test'.
        # If stage is None, this method is called for all stages.

        if stage in (None, 'fit'):
            train_transforms = self.train_transforms()
            self.train_dataset = MyDataset(self.train_data_path, train_transforms, unsup=True)

            val_transforms = self.val_transforms()
            self.val_dataset = MyDataset(self.val_data_path, val_transforms)

        if stage in (None, 'test'):
            test_transforms = self.test_transforms()
            self.test_dataset = MyDataset(self.test_data_path, test_transforms)

    def train_dataloader(self):
        # Use this method to return a PyTorch DataLoader instance for the training dataset.
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    # def val_dataloader(self):
    #     # Use this method to return a PyTorch DataLoader instance for the validation dataset.
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    # def test_dataloader(self):
    #     # Use this method to return a PyTorch DataLoader instance for the test dataset.
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        # Use this method to return a PyTorch DataLoader instance for the prediction dataset.
        return self.test_dataloader()

    def train_transforms(self):
        # Use this method to return a list of data augmentations to apply to the training data.
        return transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

class MyDataset(Dataset):
    def __init__(self, data_path, transforms=None, unsup=False):
        self.data_path = data_path
        self.transforms = transforms
        self.unsup = unsup

        # Get a list of image filenames in the data directory
        self.filenames = []


        if self.unsup:
            for filename in os.listdir(self.data_path):
                if 'jpg' in filename:
                    self.filenames.append(filename)
        else:
            for filename in os.listdir(self.data_path):
                if 'jpg' in filename:
                    self.filenames.append(filename)

        if self.unsup:
            self.unsup_filenames, self.image_filenames = np.split(self.filenames, [int(len(self.filenames)*0.8)])

    def __len__(self):
        # Return the number of images in the dataset
        return max(len(self.image_filenames), len(self.unsup_filenames)) if self.unsup else len(self.filenames)

    def __getitem__(self, index):
        # Load an image and its corresponding label, and apply the specified transforms (if any)
        if self.unsup:
            image_filename = self.image_filenames[index%len(self.image_filenames)]
            unsup_filename = self.unsup_filenames[index]

            image_path = os.path.join(self.data_path, image_filename)
            unsup_path = os.path.join(self.data_path, unsup_filename)
            
            image = Image.open(image_path).convert('RGB')
            unsup = Image.open(unsup_path).convert('RGB')
            mask = Image.open(image_path.replace('jpg','png')).convert('L')

            if self.transforms is not None:
                image = self.transforms(image)
                unsup = self.transforms(unsup)
                mask = self.transforms(mask)

            return unsup, image, mask.long()

        image_filename = self.filenames[index]

        image_path = os.path.join(self.data_path, image_filename)
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(image_path.replace('jpg','png')).convert('L')

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask.long()