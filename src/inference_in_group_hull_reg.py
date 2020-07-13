import argparse
import os
import pickle

import torch
from scipy.spatial import Delaunay
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.datasets import (
    VoxelSentenceMappingTestRegDataset,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import RegModel


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def inference(
    group_name: str,
    test_json_path: str,
    organs_dir_path: str,
    bodyhull_dir_path: str,
    batch_size: int,
    bert_name: str,
    checkpoint_path: str,
):
    assert group_name in [
        "respiratory",
        "digestive",
        "cardiovascular",
        "urinary",
        "reproductive",
        "thyroid gland",
        "spleen",
    ], "Wrong organ group name"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    group2hull = pickle.load(
        open(os.path.join(bodyhull_dir_path, "group2hull_small.pkl"), "rb")
    )
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    test_dataset = VoxelSentenceMappingTestRegDataset(test_json_path, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(
        RegModel(bert_name, config, final_project_size=3).to(device)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)

    center = torch.from_numpy(VOXELMAN_CENTER)

    corrects = 0
    totals = 0
    with torch.no_grad():
        for sentences, attn_mask, organs_indices, _ in tqdm(test_loader):

            sentences, attn_mask = (sentences.to(device), attn_mask.to(device))
            output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
            output_mappings = output_mappings.cpu() * center

            for output_mapping, organ_indices in zip(output_mappings, organs_indices):
                totals += 1
                corrects += int(
                    in_hull(output_mapping.cpu().numpy(), group2hull[group_name])
                )

        print(f"The avg IOR on the test set is: {corrects / totals * 100:.2f}%")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.group_name,
        args.test_json_path,
        args.organs_dir_path,
        args.bodyhull_dir_path,
        args.batch_size,
        args.bert_name,
        args.checkpoint_path,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Inference or regression model based on IOR within organ group."
    )
    parser.add_argument(
        "--group_name",
        type=str,
        help="Name of the organ group - one of (respiratory, digestive, cardiovascular, urinary, reproductive, thyroid gland, spleen).",
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs",
        help="Path to the data organs directory path.",
    )
    parser.add_argument(
        "--bodyhull_dir_path",
        type=str,
        default="data/data_organs",
        help="Path to the body hulls directory path.",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_test_fixd.json",
        help="Path to the test set",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The pre-trained Bert model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
