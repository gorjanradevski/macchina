from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Tuple
import torch
import json
from transformers import BertTokenizer
from torchvision import transforms


class ImagesDataset(Dataset):
    def __init__(self, images_path: str, transform=None):
        self.images_paths = [
            os.path.join(images_path, image_path)
            for image_path in os.listdir(images_path)
            if image_path.endswith(".jpg")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.images_paths[idx]
        image_name = image_path.split("/")[-1][:-4]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, image_name


class TextsDataset(Dataset):
    pass


class JsonDataset(Dataset):
    def __init__(self, json_file_path: str, images_dir_path: str):
        data = json.load(open(json_file_path))
        self.images_dir_path = images_dir_path
        self.image_paths = [element["figure"] for element in data]
        self.captions = [element["caption"] for element in data]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(
            os.path.join(self.images_dir_path, self.image_paths[idx])
        ).convert("RGB")
        image_transformed = self.transform(image)
        caption = torch.tensor(
            self.tokenizer.encode(self.captions[idx], add_special_tokens=True)
        )

        return image_transformed, caption


class Subset:
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def collate_pad_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    images, sentences = zip(*batch)
    images = torch.stack(images, 0)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    return images, padded_sentences
