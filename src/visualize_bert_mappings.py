import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import random
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.models import RegModel

colors = mcolors.CSS4_COLORS


@torch.no_grad()
def visualize_mappings_2D(
    samples: List,
    ind2organ: Dict,
    organ2voxels: Dict,
    model,
    tokenizer,
    device: torch.device,
):

    organ2ind = dict(zip(ind2organ.values(), ind2organ.keys()))
    for organ, ind in organ2ind.items():
        organ2ind[organ] = int(ind)
    organ_indices = [sample["organ_indices"] for sample in samples]
    organ_indices = [item for sublist in organ_indices for item in sublist]
    organ_indices = list(set(organ_indices))

    organ_points_dict = {}
    for i, organ_index in enumerate(organ_indices):
        points = organ2voxels[organ]
        points = random.sample(points, int(len(points) / 250))
        if ind2organ[str(organ_index)] not in organ_points_dict:
            organ_points_dict[ind2organ[str(organ_index)]] = []
        organ_points_dict[ind2organ[str(organ_index)]].extend(points)

    organ_coords_dict = {}
    organ_colors_dict = {}
    for sample in tqdm(samples):
        sentence = sample["text"]
        color = colors[list(colors.keys())[np.array(sample["organ_indices"]).sum()]]
        label = ", ".join(
            [ind2organ[str(organ_ind)] for organ_ind in sample["organ_indices"]]
        )
        if label not in organ_coords_dict:
            organ_coords_dict[label] = []
            organ_colors_dict[label] = color
        encoded = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        attn_mask = torch.ones_like(encoded)
        coordinates = model(encoded, attn_mask).cpu().numpy() * VOXELMAN_CENTER
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
    model,
    tokenizer,
    device: torch.device,
):

    organ_indices = [sample["organ_indices"] for sample in samples]
    organ_indices = [item for sublist in organ_indices for item in sublist]
    organ_indices = list(set(organ_indices))

    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(111, projection="3d")

    maxes = np.array([[0, 0, 0]])
    mins = np.array([[0, 0, 0]])
    for i, organ_index in enumerate(organ_indices):
        color = colors[list(colors.keys())[organ_index + len(ind2organ)]]
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
        ax.add_collection3d(Poly3DCollection(organ_hull, alpha=0.05, color=color))

    organ_coords_dict = {}
    for sample in tqdm(samples):
        sentence = sample["text"]
        color = colors[list(colors.keys())[np.array(sample["organ_indices"]).sum()]]
        label = ", ".join(
            [ind2organ[str(organ_ind)] for organ_ind in sample["organ_indices"]]
        )
        if label not in organ_coords_dict:
            organ_coords_dict[label] = []
        encoded = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
        attn_mask = torch.ones_like(encoded)
        coordinates = model(encoded, attn_mask).cpu().numpy() * VOXELMAN_CENTER
        organ_coords_dict[label].append(coordinates.tolist())
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


def visualize_bert_mappings(
    organs_dir_path: str,
    samples_json_path: str,
    save_path: str,
    bert_name: str,
    checkpoint_path: str,
    visualize_3D: bool,
):

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = json.load(open(samples_json_path))

    ind2organ_path = os.path.join(organs_dir_path, "ind2organ.json")
    organ2voxels_path = os.path.join(organs_dir_path, "organ2voxels.json")

    ind2organ = json.load(open(ind2organ_path))
    organ2voxels = json.load(open(organ2voxels_path))

    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(RegModel(bert_name, config, final_project_size=3)).to(
        device
    )
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    if visualize_3D:
        visualize_mappings_3D(
            samples, ind2organ, organ2voxels, model, tokenizer, device
        )
    else:
        visualize_mappings_2D(
            samples, ind2organ, organ2voxels, model, tokenizer, device
        )

    if save_path:
        plt.savefig(os.path.join(save_path))
    plt.show()


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Visualize mappings.")
    parser.add_argument(
        "--organs_dir_path", type=str, help="Path to the data organs directory path."
    )
    parser.add_argument(
        "--samples_json_path", type=str, help="Path to the json file with the samples."
    )
    parser.add_argument(
        "--save_path", type=str, default="", help="Directory where the figure is saved."
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="Should be one of [bert-base-uncased, allenai/scibert_scivocab_uncased,"
        "monologg/biobert_v1.1_pubmed, emilyalsentzer/Bio_ClinicalBERT,"
        "google/bert_uncased_L-4_H-512_A-8].",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint.",
    )
    parser.add_argument(
        "--visualize_3D",
        action="store_true",
        help="Whether to visualize in 3D as opposed to the standard 2D visualization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_bert_mappings(
        args.organs_dir_path,
        args.samples_json_path,
        args.save_path,
        args.bert_name,
        args.checkpoint_path,
        args.visualize_3D,
    )


if __name__ == "__main__":
    main()
