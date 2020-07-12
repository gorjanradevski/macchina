import argparse
import json
import os
from typing import Dict

from tqdm.notebook import tqdm

from utils.text import detect_occurrences


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


def remove_names_and_indices(samples, organs_to_remove, organ2ind):

    for sample in tqdm(samples):
        organ_names = sample["organ_names"]
        occ_organ_names = sample["occ_organ_names"]

        sample["organ_names"] = [
            name for name in organ_names if name not in organs_to_remove
        ]
        sample["occ_organ_names"] = [
            name for name in occ_organ_names if name not in organs_to_remove
        ]
        sample["organ_indices"] = [organ2ind[organ] for organ in sample["organ_names"]]
        sample["occ_organ_indices"] = [
            organ2ind[organ] for organ in sample["occ_organ_names"]
        ]

    return samples


def remove_dataset_organs(
    src_dset, dst_dset, organs_dir, organs_to_remove, redo_occurrences
):

    organ2ind = json.load(open(os.path.join(organs_dir, "organ2ind.json")))
    organ2alias = json.load(open(os.path.join(organs_dir, "organ2alias.json")))

    all_aliases = [
        organ_alias
        for organ_aliases in organ2alias.values()
        for organ_alias in organ_aliases
    ]

    aliases_to_remove = []
    for organ_to_remove in organs_to_remove:
        organ_aliases = organ2alias[organ_to_remove]
        aliases_to_remove.extend(organ_aliases)

    new_aliases = list(set(all_aliases).difference(set(organ_aliases)))

    samples = json.load(open(src_dset))

    samples = remove_names_and_indices(samples, organs_to_remove, organ2ind)
    samples = [sample for sample in samples if sample["organ_names"]]

    if redo_occurrences:
        for sample in tqdm(samples):
            sample["keywords"] = detect_occurrences(sample["text"], new_aliases)
        samples = fix_keyword_detection_issues(samples, organ2ind)

    with open(dst_dset, "w") as outfile:
        json.dump(samples, outfile)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove a set of organs from the dataset."
    )
    parser.add_argument("--src_dset", type=str, help="Path to the original dataset.")
    parser.add_argument(
        "--dst_dset", type=str, help="Path under which the new dataset is stored."
    )
    parser.add_argument(
        "--organs_dir",
        type=str,
        help="Path to the directory with the atlas dictionaries that contain the indices and aliases of organs.",
    )
    parser.add_argument(
        "--organs_to_remove",
        type=str,
        action="append",
        help="A set of organ names that will be removed",
    )
    parser.add_argument(
        "--redo_occurrences",
        action="store_true",
        help="Whether to recompute the keywords.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    remove_dataset_organs(
        args.src_dset,
        args.dst_dset,
        args.organs_dir,
        args.organs_to_remove,
        args.redo_occurrences,
    )


if __name__ == "__main__":
    main()
