import os
from pathlib import Path
from typing import Callable, Optional, List
import torch
from torch import Tensor
import rasterio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class MethaneDataset(Dataset):
    """
    Dataset class for the methane dataset curated from the EMIT data and methane L2B product.
    
    This dataset pairs EMIT hyperspectral imagery with corresponding methane segmentation labels given by the L2B EMIT product provided by NASA.
    The labels are binary masks indicating the presence of methane plumes in the hyperspectral images.
    Expects a directory layout:
        root/
            EMIT_L2A/
            plumes_masks/
            splits/
                train.txt
                val.txt
                test.txt
    """
    
    VALID_SPLITS = ['train', 'val', 'test']
    IMAGE_ROOT = 'EMIT_L2A'
    LABEL_ROOT = 'plumes_masks'
    rgb_indices = [54, 36, 21]
    
    def __init__(
        self, 
        root: str = './emit_ch4', 
        split: str = 'train',
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None
    ) -> None:
        """Initialize the EMIT Methane dataset.

        Args:
            root (str, optional): Root directory where the dataset is stored. Defaults to './emit_ch4'.
            split (str, optional): Data split to use ('train', 'val', or 'test'). Defaults to 'train'.
            transforms (Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]], optional): Transformations to apply to the data. Defaults to None.
        Raises:
            ValueError: If split is invalid or split file is not found.
        """
        super().__init__()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Valid options are: {self.VALID_SPLITS}")
        
        self.split = split
        self.root = Path(root)
        self.transforms = transforms
        
        self.split_file = self.root / 'splits' / f'{split}.txt'
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file '{self.split_file}' not found. Please ensure the dataset is properly set up.")
        
    def read_split_file(self) -> List:
        """Read the split file and return a list of tuples containing image and label paths.

        Returns:
            List[Tuple[str, str]]: List of (image_path, label_path) tuples.
        """
        with open(self.split_file, 'r') as f:
            sample_ids = [x.strip() for x in f.readlines()]
        sample_collection = [
            (
                self.root / self.IMAGE_ROOT / sample_id,
                self.root / self.LABEL_ROOT / sample_id
            )
            for sample_id in sample_ids
        ]
        return sample_collection

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_collection)

    def __getitem__(self, idx):
        """Return the sample at the given index."""
        
        img_path, label_path = self.sample_collection[idx]
        sample = {
            "image": self._load_image(img_path),
            "label": self._load_label(label_path)
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
    
    def _load_image(self, path: str) -> Tensor:
        """Load the EMIT image using rasterio."""
        with rasterio.open(path) as src:
            img = src.read()  # (285, H, W)
            img_tensor = torch.from_numpy(img).float()  # Convert to tensor
        return img_tensor
    
    def _load_label(self, path: str) -> Tensor:
        """Load the methane segmentation label using rasterio."""
        with rasterio.open(path) as src:
            label = src.read()  # shape: (1, H, W)
            label_tensor = torch.from_numpy(label).long()  # Convert to tensor
        return label_tensor
    
    def plot_sample(
        self, 
        sample: dict[str, Tensor],
        show_titles: bool = True
    ) -> None:
        """Plot the sample image and label """
        ncols = 2
        image = sample['image'][self.rgb_indices].numpy()
        image = np.transpose(image, (1, 2, 0))  # Convert to HWC format for plotting
        image = (image - image.min()) / (image.max() - image.min())  # Normalize for visualization
        
        label = sample['label'].squeeze(0).numpy()
        
        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0).numpy()
            ncols = 3
        
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 6))
        # Assuming the image has multiple channels, we can visualize the first channel
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].imshow(label, cmap='magma')
        axes[1].axis('off')
        if show_titles:
            axes[0].set_title("EMIT Image")
            axes[1].set_title("Methane Segmentation Label")
            
        if showing_predictions:
            axes[2].imshow(pred, cmap='magma')
            axes[2].axis('off')
            if show_titles:
                axes[2].set_title("Model Prediction")
        return fig