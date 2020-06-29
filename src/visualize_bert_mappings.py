import argparse
import json
import os
import random
from typing import List, Dict

import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.models import RegModel

colors = mcolors.CSS4_COLORS


@torch.no_grad()
def visualize_mappings(
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
    organs = [ind2organ[str(organ_index)] for organ_index in organ_indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, organ in enumerate(organs):
        points = organ2voxels[organ]
        points = random.sample(points, int(len(points) / 500))
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=".", alpha=0.05)

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

    # for label, coordinates in organ_coords_dict.items():
    #     organs = label.split(", ")
    #     organ_indices = [organ2ind[organ] for organ in organs]
    #     color = colors[list(colors.keys())[np.array(organ_indices).sum()]]
    #     coordinates = np.array(coordinates).mean(axis=0)
    #     ax.scatter(
    #         coordinates[:, 0],
    #         coordinates[:, 1],
    #         coordinates[:, 2],
    #         c=color,
    #         s=100,
    #         marker="*",
    #         edgecolor="r",
    #         label=label + " average",
    #     )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()


def visualize_bert_mappings(
    organs_dir_path: str, samples_json_path: str, bert_name: str, checkpoint_path: str
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

    visualize_mappings(samples, ind2organ, organ2voxels, model, tokenizer, device)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Visualize mappings.")
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs",
        help="Path to the data organs directory path.",
    )
    parser.add_argument(
        "--samples_json_path", type=str, help="Path to the json file with the samples."
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
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_bert_mappings(
        args.organs_dir_path,
        args.samples_json_path,
        args.bert_name,
        args.checkpoint_path,
    )


if __name__ == "__main__":
    main()