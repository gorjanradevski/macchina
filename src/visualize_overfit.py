import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.losses import OrganDistanceLoss

colors = mcolors.CSS4_COLORS
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def visualize_mappings(
    samples: List,
    ind2organ: Dict,
    organ2voxels: Dict,
    model: nn.Module,
    device: torch.device,
):

    organ_indices = [sample["organ_indices"] for sample in samples]
    organ_indices = [item for sublist in organ_indices for item in sublist]
    organ_indices = list(set(organ_indices))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    maxes = np.array([[0, 0, 0]])
    mins = np.array([[0, 0, 0]])
    for i, organ_index in enumerate(organ_indices):
        # points = organ2voxels[organ]
        # points = random.sample(points, int(len(points) / 500))
        # points = np.array(points)
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=".", alpha=0.05)
        color = colors[list(colors.keys())[organ_index + len(ind2organ) + 10]]
        points = np.array(organ2voxels[ind2organ[str(organ_index)]])

        # To determine axis bounds
        maxes = np.max(np.concatenate((maxes, points), axis=0), axis=0)[None, :]
        mins = np.min(np.concatenate((mins, points), axis=0), axis=0)[None, :]

        hull = ConvexHull(points)
        faces = hull.simplices

        organ_hull = np.array(
            [
                [
                    [points[s[0], 0], points[s[0], 1], points[s[0], 2]],
                    [points[s[1], 0], points[s[1], 1], points[s[1], 2]],
                    [points[s[2], 0], points[s[2], 1], points[s[2], 2]],
                ]
                for s in faces
            ]
        )
        ax.add_collection3d(Poly3DCollection(organ_hull, alpha=0.025, color=color))

    for sample in samples:
        sentence_vector = torch.tensor(sample["vector"]).unsqueeze(0).to(device)
        color = colors[
            list(colors.keys())[np.array(sample["organ_indices"]).sum() + 10]
        ]
        label = "_".join(
            [ind2organ[str(organ_ind)] for organ_ind in sample["organ_indices"]]
        )
        coordinates = model(sentence_vector).cpu().numpy() * np.array(VOXELMAN_CENTER)
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=color,
            s=100,
            marker="*",
            edgecolor="k",
            label=label,
        )

    # Expand axis bounds a bit
    maxes = maxes[0]
    mins = mins[0]
    maxes = maxes + 0.2 * np.abs(maxes)
    mins = mins - 0.2 * np.abs(mins)

    ax.set_xlim((mins[0], maxes[0]))
    ax.set_ylim((mins[1], maxes[1]))
    ax.set_zlim((mins[2], maxes[2]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.tanh(output)
        return output


class SentenceVectorDataset(Dataset):
    def __init__(
        self, samples: List, ind2organ: str, organ2voxels: str, num_anchors: int = 50
    ):
        self.samples = samples
        self.sentence_vectors, self.indices = ([], [])
        for element in self.samples:
            self.sentence_vectors.append(element["vector"])
            self.indices.append(element["organ_indices"])
        self.ind2organ = ind2organ
        self.organ2voxels = organ2voxels
        self.num_anchors = num_anchors
        self.center = torch.from_numpy(VOXELMAN_CENTER)

    def __len__(self):
        return len(self.sentence_vectors)

    def __getitem__(self, idx: int):
        sentence_vector = torch.tensor(self.sentence_vectors[idx])
        mapping = (
            torch.tensor(
                [
                    random.sample(
                        self.organ2voxels[self.ind2organ[str(index)]], self.num_anchors
                    )
                    for index in self.indices[idx]
                ]
            )
            / self.center
        )
        num_organs = len(mapping)

        return sentence_vector, mapping, num_organs


def collate_pad_batch(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    sentence_vectors, mappings, num_organs = zip(*batch)

    sentence_vectors = torch.stack(sentence_vectors)
    padded_mappings = torch.nn.utils.rnn.pad_sequence(
        mappings, batch_first=True, padding_value=0
    )
    num_organs = torch.tensor([*num_organs])

    return sentence_vectors, padded_mappings, num_organs


def embed_sample_sentences(samples_json_path: str):
    nlp = spacy.load("en_core_web_md")
    samples = json.load(open(samples_json_path))

    for sample in samples:
        sentence = sample["text"]
        doc = nlp(sentence)
        sample["vector"] = doc.vector.tolist()

    return samples


def test_loss_function(
    samples_json_path,
    organs_dir_path,
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=0,
    num_anchors=100,
    voxel_temperature=1,
    organ_temperature=1,
):

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = embed_sample_sentences(samples_json_path)
    ind2organ_path = os.path.join(organs_dir_path, "ind2organ.json")
    organ2voxels_path = os.path.join(organs_dir_path, "organ2voxels.json")

    ind2organ = json.load(open(ind2organ_path))
    organ2voxels = json.load(open(organ2voxels_path))

    input_size = len(samples[0]["vector"])
    hidden_size = input_size // 2
    output_size = 3

    model = Feedforward(input_size, hidden_size, output_size)
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    criterion = OrganDistanceLoss(
        voxel_temperature=voxel_temperature, organ_temperature=organ_temperature
    )
    logging.warning(f"Using {num_anchors} voxel points!")
    dataset = SentenceVectorDataset(samples, ind2organ, organ2voxels, num_anchors)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_pad_batch,
    )
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        with tqdm(total=len(dataloader)) as pbar:
            for sentence_vectors, true_mappings, num_organs in dataloader:

                optimizer.zero_grad()

                sentence_vectors, true_mappings, num_organs = (
                    sentence_vectors.to(device),
                    true_mappings.to(device),
                    num_organs.to(device),
                )

                mappings = model(sentence_vectors)
                loss = criterion(mappings, true_mappings, num_organs)

                loss.backward()

                optimizer.step()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})
        if not epoch % 20:
            visualize_mappings(samples, ind2organ, organ2voxels, model, device)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize and test the loss function in an overfit experiment."
    )
    parser.add_argument(
        "--samples_json_path", type=str, help="Path to json file with test samples."
    )
    parser.add_argument(
        "--organs_dir_path", type=str, help="Path to the directory with organ info."
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay parameter."
    )
    parser.add_argument(
        "--num_anchors", type=int, default=1, help="The number of anchor points to use."
    )
    parser.add_argument(
        "--voxel_temperature", type=float, default=1.0, help="The voxel temperature."
    )
    parser.add_argument(
        "--organ_temperature", type=float, default=1.0, help="The organ temperature."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_loss_function(
        args.samples_json_path,
        args.organs_dir_path,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.num_anchors,
        args.voxel_temperature,
        args.organ_temperature,
    )


if __name__ == "__main__":
    main()
