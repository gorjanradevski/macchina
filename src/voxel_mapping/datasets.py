from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import torch
from typing import Tuple
import random
from torchvision import transforms
from PIL import Image
import re


class VoxelSentenceMappingDataset:
    # Assumes that the dataset is: {
    # "sentence": str,
    # "keywords": List[str, str, ...],
    # "location_map": List[[float, float, float], [float, float, float],...],
    # "bounding_box": List[[float, float], [float, float], [float, float]]
    # }
    def __init__(self, json_path: str, bert_tokenizer_path_or_name: str):
        self.json_data = json.load(open(json_path))
        self.sentences, self.mappings, self.keywords, self.bounding_boxes = (
            [],
            [],
            [],
            [],
        )
        for element in self.json_data:
            if len(element["text"]) > 200:
                continue
            self.sentences.append(element["text"])
            self.mappings.append(element["centers"])
            self.keywords.append(element["keywords"])
            self.bounding_boxes.append(element["bboxes"])
        self.tokenizer = self.tokenizer = BertTokenizer.from_pretrained(
            bert_tokenizer_path_or_name
        )


class VoxelSentenceMappingTrainDataset(VoxelSentenceMappingDataset, Dataset):
    def __init__(self, json_path: str, bert_tokenizer_path_or_name: str):
        super().__init__(json_path, bert_tokenizer_path_or_name)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        mask = {word: random.choice([0, 1]) for word in self.keywords[idx]}
        masked_sentence = " ".join(
            [
                "[MASK]" if word in mask and mask[word] == 1 else word
                for word in re.findall(r"[\w']+|[.,!?;]", self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        )
        mapping = torch.tensor(self.mappings[idx])
        bounding_box = torch.tensor(self.bounding_boxes[idx])
        num_organs = len(mapping)

        return (tokenized_sentence, mapping, num_organs, bounding_box)


class VoxelSentenceMappingTestDataset(VoxelSentenceMappingDataset, Dataset):
    def __init__(self, json_path: str, bert_tokenizer_path_or_name: str):
        super().__init__(json_path, bert_tokenizer_path_or_name)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(self.sentences[idx], add_special_tokens=True)
        )
        mapping = torch.tensor(self.mappings[idx])
        bounding_box = torch.tensor(self.bounding_boxes[idx])
        num_organs = len(mapping)

        return (tokenized_sentence, mapping, num_organs, bounding_box)


class VoxelSentenceMappingTestMaskedDataset(VoxelSentenceMappingDataset, Dataset):
    def __init__(self, json_path: str, bert_tokenizer_path_or_name: str):
        super().__init__(json_path, bert_tokenizer_path_or_name)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        mask = {word for word in self.keywords[idx]}
        masked_sentence = " ".join(
            [
                "[MASK]" if word in mask else word
                for word in re.findall(r"[\w']+|[.,!?;]", self.sentences[idx])
            ]
        )
        tokenized_sentence = torch.tensor(
            self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        )
        mapping = torch.tensor(self.mappings[idx])
        bounding_box = torch.tensor(self.bounding_boxes[idx])
        num_organs = len(mapping)

        return (tokenized_sentence, mapping, num_organs, bounding_box)


def collate_pad_sentence_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    sentences, mappings, num_organs, bounding_boxes = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_mappings = torch.nn.utils.rnn.pad_sequence(mappings, batch_first=True)
    num_organs = torch.tensor([*num_organs])

    # IDK why num_organs and bounding_boxes is a Tuple
    return padded_sentences, padded_mappings, num_organs, bounding_boxes


class VoxelImageMappingDataset:
    # Assumes that the dataset is: {
    # "image_path": str,
    # "centers": List[[float, float, float], [float, float, float],...],
    # "bboxes": List[[float, float], [float, float], [float, float]]
    # }
    def __init__(
        self,
        json_path: str,
        ind2organ_path: str,
        organ2center_path: str,
        organ2bbox_path: str,
    ):
        # Load json files
        self.json_data = json.load(open(json_path))
        self.ind2organ = json.load(open(ind2organ_path))
        self.organ2center = json.load(open(organ2center_path))
        self.organ2bbox = json.load(open(organ2bbox_path))
        # Obtain image_paths, mappings, bounding_boxes
        self.image_paths = [element["image_path"] for element in self.json_data]
        self.organs = [element["organ"] for element in self.json_data]
        self.organ_per_image = [element["organ"] for element in self.json_data]

        self.mappings = []
        self.bounding_boxes = []
        for indexes in self.organs:
            tmp_mappings = []
            tmp_bbox = []
            for index in indexes:
                tmp_mappings.append(self.organ2center[self.ind2organ[str(index)]])
                tmp_bbox.append(self.organ2bbox[self.ind2organ[str(index)]])
            self.mappings.append(tmp_mappings)
            self.bounding_boxes.append(tmp_bbox)

        self.all_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class VoxelImageMappingTrainDataset(VoxelImageMappingDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        ind2organ_path: str,
        organ2center_path: str,
        organ2bbox_path: str,
    ):
        super().__init__(json_path, ind2organ_path, organ2center_path, organ2bbox_path)
        self.train_transforms = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_train_transformed = self.train_transforms(image)
        image_all_transformed = self.all_transforms(image_train_transformed)

        mapping = torch.tensor(self.mappings[idx])
        bounding_box = torch.tensor(self.bounding_boxes[idx])
        num_organs = len(mapping)

        return (image_all_transformed, mapping, num_organs, bounding_box)


class VoxelImageMappingTestDataset(VoxelImageMappingDataset, Dataset):
    def __init__(
        self,
        json_path: str,
        ind2organ_path: str,
        organ2center_path: str,
        organ2bbox_path: str,
    ):
        super().__init__(json_path, ind2organ_path, organ2center_path, organ2bbox_path)
        self.test_transforms = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_test_transformed = self.test_transforms(image)
        image_all_transformed = self.all_transforms(image_test_transformed)

        mapping = torch.tensor(self.mappings[idx])
        bounding_box = torch.tensor(self.bounding_boxes[idx])
        num_organs = len(mapping)

        return (image_all_transformed, mapping, num_organs, bounding_box)


def collate_pad_image_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    images, mappings, num_organs, bounding_boxes = zip(*batch)
    images = torch.stack(images, 0)
    padded_mappings = torch.nn.utils.rnn.pad_sequence(mappings, batch_first=True)
    num_organs = torch.tensor([*num_organs])

    # IDK why num_organs and bounding_boxes is a Tuple
    return images, padded_mappings, num_organs, bounding_boxes
