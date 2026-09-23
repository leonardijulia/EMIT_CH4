from typing import Optional, Any
from torch.utils.data import DataLoader
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from data.dataset import MethaneDataset

class MethaneDataModule:
    """
    DataModule for the Methane dataset.
    
    This class is responsible for loading the Methane dataset and providing DataLoaders for training, validation, and testing.
    It uses the MethaneDataset class to handle the dataset and applies any specified transformations.
    """
    
    def __init__(
        self, 
        batch_size: int = 8, 
        data_dir: str = './emit_ch4',
        num_workers: int = 4,
    ) -> None:
        """
        Orchestrates Train, Val, and Test splits using the predefined split text files.

        Args:
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 8.
            patch_size (Union[int, tuple[int, int]], optional): Size of the patches to extract from the images. Defaults to 128.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[MethaneDataset] = None
        self.val_dataset: Optional[MethaneDataset] = None
        self.test_dataset: Optional[MethaneDataset] = None
        
        self.train_aug = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
        ])
        
        self.val_test_aug = v2.Compose([
            v2.Resize((128, 128)),
        ])
        
    def train_transform(self, sample: dict) -> dict:
        """Applies training augmentations and manages PyTorch/numpy conversions."""
        # HSI: Shape changes from (C, H, W) -> (H, W, C) for Albumentations
        wrapped = {
            "image": tv_tensors.Image(sample["image"]),
            "label": tv_tensors.Mask(sample["label"])
        }
        
        # Apply the pipeline
        transformed = self.train_aug(wrapped)
        
        # Unwrap and cast back to standard PyTorch Tensors
        return {
            "image": transformed["image"].as_subclass(torch.Tensor).float(),
            "label": transformed["label"].as_subclass(torch.Tensor).long()
        }
    
    def val_test_transform(self, sample: dict) -> dict:
        """Applies validation/test scaling and manages PyTorch/numpy conversions."""
        wrapped = {
            "image": tv_tensors.Image(sample["image"]),
            "label": tv_tensors.Mask(sample["label"])
        }
        
        transformed = self.val_test_aug(wrapped)
        
        return {
            "image": transformed["image"].as_subclass(torch.Tensor).float(),
            "label": transformed["label"].as_subclass(torch.Tensor).long()
        }
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiates the dataset splits based on the requested execution stage.
        """
        if stage is None or stage == "fit":
            self.train_dataset = MethaneDataset(
                root=self.data_dir, 
                split='train', 
                transforms=self.train_transform
            )
            self.val_dataset = MethaneDataset(
                root=self.data_dir, 
                split='val', 
                transforms=self.val_test_transform
            )

        if stage is None or stage == "test":
            self.test_dataset = MethaneDataset(
                root=self.data_dir, 
                split='test', 
                transforms=self.val_test_transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,          
            num_workers=self.num_workers,
            pin_memory=True,       # Speeds up tensor transfer to GPU
            drop_last=True         # Keeps batch sizes stable for batch norm layers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,         # Never shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )