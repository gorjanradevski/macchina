import argparse
import json
import os
from typing import Dict

import seaborn as sns  # noqa: F401
from matplotlib import pyplot as plt  # noqa: F401
from tqdm import tqdm

from utils.text import count_occurrences, detect_occurrences


def fix_keyword_detection_issues(dset: str, organ2ind: Dict):

    # # SOLVE CARDIA PROBLEM

    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            "cardiac" in keywords
            and "stomach" in occ_organ_names
            and any(
                [
                    item in organ_names
                    for item in ["atrium", "ventricle", "myocardium", "pericardium"]
                ]
            )
        ):
            occ_organ_indices.remove(organ2ind["stomach"])
            occ_organ_names.remove("stomach")
        if (
            "cardia" in keywords
            and "myocardium" in occ_organ_names
            and any([item in organ_names for item in ["stomach"]])
        ):
            occ_organ_indices.remove(organ2ind["myocardium"])
            occ_organ_names.remove("myocardium")
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names

    inds = []
    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if "cardiac" in keywords and "stomach" in occ_organ_names:
            inds.append(ind)

    # # SOLVE THE LIVER - DELIVER PROBLEM

    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            any(
                [
                    keyword in keywords
                    for keyword in ["delivery", "delivered", "deliver", "delivering"]
                ]
            )
            and "liver" not in organ_names
        ):
            occ_organ_indices.remove(organ2ind["liver"])
            occ_organ_names.remove("liver")
        keywords = [
            keyword
            for keyword in keywords
            if keyword not in ["delivery", "delivered", "deliver", "delivering"]
        ]
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names
        abstract["keywords"] = keywords

    # # SOLVE THE COLON - COLONISE PROBLEM

    inds = []
    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            any(
                [
                    keyword in keywords
                    for keyword in [
                        "colonize",
                        "colonise",
                        "colonized",
                        "colonised",
                        "colonies",
                    ]
                ]
            )
            and "colon" not in organ_names
        ):
            occ_organ_indices.remove(organ2ind["colon"])
            occ_organ_names.remove("colon")
        keywords = [
            keyword
            for keyword in keywords
            if keyword
            not in ["colonize", "colonise", "colonized", "colonised", "colonies"]
        ]
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names
        abstract["keywords"] = keywords

    # # SOLVE THE BLADDER - GALLBLADDER PROBLEM

    """Gallbladder doesn't cause the bladder keyword"""
    """Bladder does cause problems"""

    for ind, abstract in tqdm(enumerate(dset)):
        keywords = abstract["keywords"]
        occ_organ_indices = abstract["occ_organ_indices"]
        occ_organ_names = abstract["occ_organ_names"]
        organ_names = abstract["organ_names"]
        if (
            any([keyword in keywords for keyword in ["bladder", "bladders"]])
            and any(
                [
                    keyword in keywords
                    for keyword in [
                        "gall",
                        "gallbladder",
                        "gall-bladder",
                        "gallbladders",
                        "gall-bladders",
                    ]
                ]
            )
            and "gallbladder" in organ_names
        ):
            occ_organ_indices.remove(organ2ind["urinary bladder"])
            occ_organ_names.remove("urinary bladder")
            keywords = [
                keyword
                for keyword in keywords
                if keyword not in ["bladder", "bladders"]
            ]
        abstract["occ_organ_indices"] = occ_organ_indices
        abstract["occ_organ_names"] = occ_organ_names
        abstract["keywords"] = keywords

    return dset


def generate_occurrences(src_dset_path: str, dst_dset_path: str, organs_dir_path: str):

    dset = json.load(open(src_dset_path))

    organ2alias = json.load(open(os.path.join(organs_dir_path, "organ2alias.json")))
    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))

    print("Generating maskwords...")
    all_aliases = [item for sublist in organ2alias.values() for item in sublist]
    for abstract in tqdm(dset):
        abstract["keywords"] = detect_occurrences(abstract["text"], all_aliases)

    print("Counting organ occurrences...")
    for abstract in tqdm(dset):
        text = abstract["text"]
        occ_organ_names = []
        occ_organ_indices = []
        for organ, aliases in organ2alias.items():
            if count_occurrences(text, aliases):
                occ_organ_names.append(organ)
                occ_organ_indices.append(organ2ind[organ])
        abstract["occ_organ_names"] = occ_organ_names
        abstract["occ_organ_indices"] = occ_organ_indices

    occ_organ_count_dict = {}
    for abstract in tqdm(dset):
        organ_names = abstract["occ_organ_names"]
        for organ_name in organ_names:
            if organ_name not in occ_organ_count_dict:
                occ_organ_count_dict[organ_name] = 1
            else:
                occ_organ_count_dict[organ_name] += 1
    print("Organ occurrence counts in dataset...")
    print(occ_organ_count_dict)

    print("Fixing keyword detection...")
    dset = fix_keyword_detection_issues(dset, organ2ind)

    json.dump(dset, open(dst_dset_path, "w"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate occurrence annotation and maskwords."
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
    return parser.parse_args()


def main():
    args = parse_args()
    generate_occurrences(args.src_dset_path, args.dst_dset_path, args.organs_dir_path)


if __name__ == "__main__":
    main()
