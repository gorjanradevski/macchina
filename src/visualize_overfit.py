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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.losses import OrganDistanceLoss, BaselineRegLoss

colors = mcolors.CSS4_COLORS
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def visualize_mappings_2D(
    samples: List,
    ind2organ: Dict,
    organ2voxels: Dict,
    model: nn.Module,
    device: torch.device,
):

    organ2ind = dict(zip(ind2organ.values(), ind2organ.keys()))
    for organ, ind in organ2ind.items():
        organ2ind[organ] = int(ind)
    organ_indices = [sample["organ_indices"] for sample in samples]
    organ_indices = [item for sublist in organ_indices for item in sublist]
    organ_indices = list(set(organ_indices))
    organ_names = [ind2organ[str(organ_index)] for organ_index in organ_indices]

    organ_points_dict = {}
    for i, organ_name in enumerate(organ_names):
        points = organ2voxels[organ_name]
        points = random.sample(points, int(len(points) / 250))
        if organ_name not in organ_points_dict:
            organ_points_dict[organ_name] = []
        organ_points_dict[organ_name].extend(points)

    organ_coords_dict = {}
    organ_colors_dict = {}
    for sample in tqdm(samples):
        sentence_vector = torch.tensor(sample["vector"]).unsqueeze(0).to(device)
        color = colors[
            list(colors.keys())[np.array(sample["organ_indices"]).sum() + 10]
        ]
        label = "_".join(
            [ind2organ[str(organ_ind)] for organ_ind in sample["organ_indices"]]
        )
        if label not in organ_coords_dict:
            organ_coords_dict[label] = []
            organ_colors_dict[label] = color
        coordinates = model(sentence_vector).cpu().squeeze().numpy() * np.array(
            VOXELMAN_CENTER
        )
        organ_coords_dict[label].append(coordinates.tolist())

    """Perform PCA"""
    all_points = []
    for organ, organ_points in organ_points_dict.items():
        all_points.extend(organ_points)

    # Don't use the projected samples so that the transform is always constant
    # for label, sample_points in organ_coords_dict.items():
    #     all_points.extend(sample_points)

    all_points = np.array(all_points)
    pca = PCA(n_components=2)
    pca_transform = pca.fit(all_points)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    for organ, points in organ_points_dict.items():
        points = np.array(points)
        points = pca_transform.transform(points)
        ax.scatter(points[:, 0], points[:, 1], marker=".", alpha=0.125, label=organ)

    for label, coordinates in organ_coords_dict.items():
        coordinates = np.array(coordinates)
        coordinates = pca_transform.transform(coordinates)
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=organ_colors_dict[label],
            s=100,
            marker="*",
            edgecolor="k",
            label=label,
        )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


@torch.no_grad()
def visualize_mappings_3D(
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
    save_dir,
    num_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    num_anchors,
    loss_type,
    voxel_temperature,
    organ_temperature,
    visualize_every,
    visualize_3D,
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

    if loss_type == "organ_loss":
        criterion = OrganDistanceLoss(
            voxel_temperature=voxel_temperature, organ_temperature=organ_temperature
        )
        logging.warning("Using SSL loss!")
    elif loss_type == "baseline_loss":
        criterion = BaselineRegLoss()
        logging.warning("Using baseline REG loss!")
    else:
        raise ValueError(f"Invalid loss type {loss_type}")

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

                if loss_type == "organ_loss":
                    loss = criterion(mappings, true_mappings, num_organs)
                elif loss_type == "baseline_loss":
                    loss = criterion(mappings, true_mappings)
                else:
                    raise ValueError(f"Invalid loss type: {loss_type}")

                loss.backward()

                optimizer.step()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        if visualize_every > 0 and not epoch % visualize_every:
            if visualize_3D:
                visualize_mappings_3D(samples, ind2organ, organ2voxels, model, device)
            else:
                visualize_mappings_2D(samples, ind2organ, organ2voxels, model, device)

    if visualize_3D:
        visualize_mappings_3D(samples, ind2organ, organ2voxels, model, device)
    else:
        visualize_mappings_2D(samples, ind2organ, organ2voxels, model, device)

    if visualize_every < 0 and save_dir:
        plt.savefig(
            os.path.join(
                save_dir, f"{loss_type}_vtemp_{voxel_temperature}_otemp_{organ_temperature}.png"
            )
        )
    else:
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
    parser.add_argument(
        "--save_dir", type=str, default="", help="Directory where the figure is saved."
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
        "--num_anchors", type=int, default=100, help="The number of anchor points to use."
    )
    parser.add_argument(
        "--loss_type", type=str, default="organ_loss", help="The loss type"
    )
    parser.add_argument(
        "--voxel_temperature", type=float, default=1.0, help="The voxel temperature."
    )
    parser.add_argument(
        "--organ_temperature", type=float, default=1.0, help="The organ temperature."
    )
    parser.add_argument(
        "--visualize_every",
        type=int,
        default=-1,
        help="Number of epochs after which the visualization is made.",
    )
    parser.add_argument(
        "--visualize_3D",
        action="store_true",
        help="Whether to visualize in 3D as opposed to the standard 2D visualization."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    test_loss_function(
        args.samples_json_path,
        args.organs_dir_path,
        args.save_dir,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.num_anchors,
        args.loss_type,
        args.voxel_temperature,
        args.organ_temperature,
        args.visualize_every,
        args.visualize_3D,
    )


if __name__ == "__main__":
    main()
