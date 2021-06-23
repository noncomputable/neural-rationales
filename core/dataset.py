import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

from PIL import Image
import pandas as pd

class Dataset(Dataset):
    def __init__(self, img_dir, annotations_file, label_names_file, only_class_id = None, transform=None):
        """
        Args:
        only_class_id - Only load data for the class with this id.
        """
        
        self.img_labels = pd.read_csv(f"{img_dir}/{annotations_file}")    
        if only_class_id is not None:
            self.img_labels = self.img_labels.loc[self.img_labels["img_class"] == only_class_id]
        self.only_class_id = only_class_id     
        label_names = pd.read_csv(f"{img_dir}/{label_names_file}")
        self.class_idx_to_name = list(label_names["class_name"])
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):        
        class_idx, class_img_idx = self.img_labels.iloc[idx]
        image, class_idx = self.get_class_item(class_idx, class_img_idx)
        
        return image, class_idx

    def get_class_item(self, class_idx, class_img_idx):
        class_name = self.class_idx_to_name[class_idx]
        img_path = f"{self.img_dir}/{class_name}/{class_name}_{class_img_idx}.png"
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
