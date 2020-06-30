import argparse
import json
import logging
import os
import random

import seaborn as sns  # noqa: F401
from matplotlib import pyplot as plt  # noqa: F401
from sklearn.model_selection import train_test_split as dataset_split
from tqdm import tqdm
from transformers import BertTokenizer


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def subsample_all_mesh_dataset(
    src_dset_path: str,
    dst_dset_path: str,
    organs_dir_path: str,
    organ_cap_single: int,
    organ_cap_multi: int,
    train_percentage: float,
    split: bool = False,
    fit_to_size: bool = False,
):
    """Subsample a dataset - select a portion of the samples and potentially remove long ones.
    Arguments:
        src_dset_path (str): Path to the source dataset.
        dst_dset_path (str): Path under which the dataset is saved.
        organs_dir_path (str): Path to the directory with organ dictionaries.
        organ_cap (int): Maximum number of organ occurrences in dataset subset.
        train_percentage (float): Percentage of training set samples.
        split (bool): Whether to split in the train, val and test dataset.
        fit_to_size (bool): Whether to only take samples shorter than 512 tokens
    """

    if not os.path.exists(os.path.dirname(dst_dset_path)):
        os.makedirs(os.path.dirname(dst_dset_path))

    dset = json.load(open(src_dset_path))

    if fit_to_size:
        print("Taking only short abstracts...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dset = [
            sample for sample in dset if len(tokenizer.encode(sample["text"])) < 512
        ]

    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))

    organ_names = list(organ2ind.keys())

    print("Taking samples with single organ occurrence...")
    organs_singles = {}

    for organ_name in tqdm(organ_names):
        organs_singles[organ_name] = [
            sample for sample in dset if sample["organ_names"] == [organ_name]
        ]

    print("Taking samples with multiple organ occurrences...")
    organs_multis = {}

    for organ_name in tqdm(organ_names):
        organs_multis[organ_name] = [
            sample
            for sample in dset
            if organ_name in sample["organ_names"] and len(sample["organ_names"]) > 1
        ]

    for organ_name in tqdm(organ_names):
        organs_singles[organ_name] = (
            random.sample(organs_singles[organ_name], organ_cap_single)
            if len(organs_singles[organ_name]) > organ_cap_single
            else organs_singles[organ_name]
        )

    for organ_name in tqdm(organ_names):
        organs_multis[organ_name] = (
            random.sample(organs_multis[organ_name], organ_cap_multi)
            if len(organs_multis[organ_name]) > organ_cap_multi
            else organs_multis[organ_name]
        )

    dset_sample = []

    for organ_name in tqdm(organ_names):
        dset_sample.extend(organs_singles[organ_name])
        dset_sample.extend(organs_multis[organ_name])

    if split:
        dset_train, dset_val_test = dataset_split(
            dset_sample, train_size=train_percentage
        )
        dset_val, dset_test = dataset_split(dset_val_test, test_size=0.5)

    json.dump(dset_sample, open(dst_dset_path, "w"))

    if split:
        json.dump(
            dset_train,
            open(
                os.path.splitext(dst_dset_path)[0]
                + "_train"
                + os.path.splitext(dst_dset_path)[1],
                "w",
            ),
        )

        json.dump(
            dset_val,
            open(
                os.path.splitext(dst_dset_path)[0]
                + "_val"
                + os.path.splitext(dst_dset_path)[1],
                "w",
            ),
        )

        json.dump(
            dset_test,
            open(
                os.path.splitext(dst_dset_path)[0]
                + "_test"
                + os.path.splitext(dst_dset_path)[1],
                "w",
            ),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subsample a given dataset, pick only abstracts shorter than 512 tokens."
    )
    parser.add_argument("--src_dset_path", type=str, help="Path to the source dataset.")
    parser.add_argument(
        "--dst_dset_path", type=str, help="Path under which the dataset is saved."
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        help="Path to the directory with organ dictionaries.",
    )
    parser.add_argument(
        "--organ_cap_single",
        type=int,
        default=750,
        help="Soft cap to the number of organ appearances in the dataset.",
    )
    parser.add_argument(
        "--organ_cap_multi",
        type=int,
        default=500,
        help="Soft cap to the number of organ appearances in the dataset.",
    )
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=0.7,
        help="Percentage of training set samples.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Whether to split the dataset subsequently.",
    )
    parser.add_argument(
        "--fit_to_size",
        action="store_true",
        help="Whether to split the dataset subsequently.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    subsample_all_mesh_dataset(
        args.src_dset_path,
        args.dst_dset_path,
        args.organs_dir_path,
        args.organ_cap_single,
        args.organ_cap_multi,
        args.train_percentage,
        args.split,
        args.fit_to_size,
    )


if __name__ == "__main__":
    main()
