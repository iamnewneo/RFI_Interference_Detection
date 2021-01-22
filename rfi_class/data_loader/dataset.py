import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RFIDataset(Dataset):
    def __init__(self, df, resize):
        self.image_paths = df["path"]
        self.targets = df["target"]
        self.ids = df["id"]
        self.resize = resize

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = Image.open(self.image_paths[idx])
        target = self.targets[idx]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "target": torch.tensor(target).type(torch.LongTensor),
            "id": id,
        }

    def transform(self, image):
        return image