from transformers import BertModel
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet152


class SentenceMappingsProducer(nn.Module):
    def __init__(
        self, bert_path_or_name: str, joint_space: int, finetune: bool = False
    ):
        super(SentenceMappingsProducer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path_or_name)
        self.bert.eval()
        self.projector = Projector(768, joint_space)
        self.finetune = finetune

        for param in self.bert.parameters():
            param.requires_grad = finetune

    def forward(self, sentences: torch.Tensor):
        # https://arxiv.org/abs/1801.06146
        hidden_states = self.bert(sentences)
        last_state = hidden_states[0][:, 0, :]

        return self.projector(last_state)

    def train(self, mode: bool = True):
        if self.finetune and mode:
            self.bert.train(True)
            self.projector.train(True)
        elif mode:
            self.projector.train(True)
        else:
            self.bert.train(False)
            self.projector.train(False)


class ImageMappingsProducer(nn.Module):
    def __init__(self, joint_space: int, finetune: bool = False):
        super(ImageMappingsProducer, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )
        self.resnet.eval()
        self.projector = Projector(2048, joint_space)
        self.finetune = finetune

        for param in self.resnet.parameters():
            param.requires_grad = finetune

    def forward(self, images: torch.Tensor):
        embedded_images = torch.flatten(self.resnet(images), start_dim=1)

        return self.projector(embedded_images)

    def train(self, mode: bool = True):
        if self.finetune and mode:
            self.resnet.train(True)
            self.projector.train(True)
        elif mode:
            self.projector.train(True)
        else:
            self.resnet.train(False)
            self.projector.train(False)


class Projector(nn.Module):
    def __init__(self, input_space, joint_space: int):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_space, joint_space)
        self.bn = nn.BatchNorm1d(joint_space)
        self.fc2 = nn.Linear(joint_space, 3)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected_embeddings = self.fc2(self.bn(F.relu(self.fc1(embeddings))))

        return projected_embeddings