import argparse
import json
import os
import pickle

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestClassDataset,
    collate_pad_sentence_class_batch,
)
from voxel_mapping.models import ClassModel


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

    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    group2hull = pickle.load(
        open(os.path.join(bodyhull_dir_path, "group2hull_small.pkl"), "rb")
    )
    organ2center = json.load(open(os.path.join(organs_dir_path, "organ2center.json")))
    num_classes = max([int(index) for index in ind2organ.keys()]) + 1
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    test_dataset = VoxelSentenceMappingTestClassDataset(
        test_json_path, tokenizer, num_classes
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_pad_sentence_class_batch
    )
    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(
        ClassModel(bert_name, config, final_project_size=num_classes).to(device)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)

    corrects = 0
    totals = 0
    with torch.no_grad():
        for sentences, attn_mask, organs_indices, _ in tqdm(test_loader):
            sentences, attn_mask = sentences.to(device), attn_mask.to(device)
            output_mappings = model(input_ids=sentences, attention_mask=attn_mask)
            y_pred = torch.argmax(output_mappings, dim=-1)
            pred_points = np.array(
                [organ2center[ind2organ[str(ind.item())]] for ind in y_pred]
            )
            for pred_point, organ_indices in zip(pred_points, organs_indices):
                totals += 1
                corrects += int(in_hull(pred_point, group2hull[group_name]))

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
        description="Inference or classification model based on IOR within organ group."
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
