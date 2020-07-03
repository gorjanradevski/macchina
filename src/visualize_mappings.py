import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.models import model_factory

colors = mcolors.CSS4_COLORS


@torch.no_grad()
def visualize_mappings_2D(
    samples: List,
    ind2organ: Dict,
    model,
    tokenizer,
    use_tsne: bool,
    device: torch.device,
):
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
        coordinates = model(encoded, attn_mask).cpu().squeeze().numpy()
        organ_coords_dict[label].append(coordinates.tolist())

    """Perform PCA"""
    all_points = []
    for label, sample_points in organ_coords_dict.items():
        all_points.extend(sample_points)

    all_points = np.array(all_points)
    if use_tsne:
        dim_reduction = TSNE(n_components=2)
    else:
        dim_reduction = PCA(n_components=2)
    dim_reduction_transform = dim_reduction.fit(all_points)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    for label, coordinates in organ_coords_dict.items():
        coordinates = np.array(coordinates)
        coordinates = dim_reduction_transform.transform(coordinates)
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
    samples: List, ind2organ: Dict, model, tokenizer, device: torch.device
):

    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(111, projection="3d")

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
    model_name: str,
    bert_name: str,
    checkpoint_path: str,
    visualize_3D: bool,
    use_tsne: bool,
):

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = json.load(open(samples_json_path))

    ind2organ_path = os.path.join(organs_dir_path, "ind2organ.json")

    ind2organ = json.load(open(ind2organ_path))

    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(
        model_factory(model_name, bert_name, config, final_project_size=3)
    ).to(device)
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    if visualize_3D:
        visualize_mappings_3D(samples, ind2organ, model, tokenizer, device)
    else:
        visualize_mappings_2D(samples, ind2organ, model, tokenizer, use_tsne, device)

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
        "--model_name", type=str, default="reg_model", help="The model name."
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
    parser.add_argument(
        "--use_tsne",
        action="store_true",
        help="Whether to use TSNE (if not it uses PCA).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_bert_mappings(
        args.organs_dir_path,
        args.samples_json_path,
        args.save_path,
        args.model_name,
        args.bert_name,
        args.checkpoint_path,
        args.visualize_3D,
        args.use_tsne,
    )


if __name__ == "__main__":
    main()
