import argparse
import json
import os

import ndjson
import seaborn as sns  # noqa: F401
from matplotlib import pyplot as plt  # noqa: F401
from sklearn.model_selection import train_test_split as dataset_split
from tqdm import tqdm


def create_all_mesh_dataset(
    dset_path: str,
    all_mesh_path: str,
    organs_dir_path: str,
    train_percentage: float,
    split: bool = False,
):
    """Create a dataset based on a directory containing json files with allMesh dataset abstracts
    Arguments:
        dset_path (str): Path to store dataset.
        all_mesh_path (str): Path to the directory with json files constaining allMesh dataset abstracts.
        organs_dir_path (str): Path to the directory with organ dictionaries.
        organ_cap (int): Maximum number of organ occurrences in dataset subset.
        train_percentage (float): Percentage of training set samples.
        split (bool): Whether to split in the train, val and test dataset.
    """

    if not os.path.exists(os.path.dirname(dset_path)):
        os.makedirs(os.path.dirname(dset_path))

    organ2alias = json.load(open(os.path.join(organs_dir_path, "organ2alias.json")))
    organ2ind = json.load(open(os.path.join(organs_dir_path, "organ2ind.json")))

    # for organ, aliases in organ2alias.items():
    #     organ2alias[organ] = [alias.strip() for alias in aliases]

    dset = []
    for json_file in tqdm(os.listdir(all_mesh_path)):
        abstracts = ndjson.load(open(os.path.join(all_mesh_path, json_file)))
        dset.extend(abstracts)

    all_aliases = list(organ2alias.values())
    all_aliases = [item for sublist in all_aliases for item in sublist]
    mesh_term_count_dict = {}
    for abstract in tqdm(dset):
        mesh_terms = abstract["meshMajor"]
        count = len([item for item in mesh_terms if item.lower() in all_aliases])
        if count not in mesh_term_count_dict:
            mesh_term_count_dict[count] = 1
        else:
            mesh_term_count_dict[count] += 1
        abstract["organMeshCount"] = count

    num_single_organ = len(
        [abstract for abstract in dset if abstract["organMeshCount"] == 1]
    )

    num_multiple_organ = len(
        [abstract for abstract in dset if abstract["organMeshCount"] > 1]
    )

    print(f"Number of abstracts pertaining to one organ: {num_single_organ}")
    print(
        f"Number of abstracts pertaining to more than one organ: {num_multiple_organ}"
    )

    for abstract in tqdm(dset):
        organ_names = []
        organ_indices = []
        mesh_terms = abstract["meshMajor"]
        for organ, aliases in organ2alias.items():
            if any([mesh_term.lower() in aliases for mesh_term in mesh_terms]):
                organ_names.append(organ)
                organ_indices.append(organ2ind[organ])
        if "organMeshCount" in abstract:
            del abstract["organMeshCount"]
        abstract["text"] = abstract["abstractText"]
        del abstract["abstractText"]
        abstract["organ_names"] = organ_names
        abstract["organ_indices"] = organ_indices
        abstract["mesh_terms"] = abstract["meshMajor"]
        abstract["keywords"] = []
        del abstract["meshMajor"]

    """Remove abstracts with animal related mesh terms"""
    animal_mesh_terms = [
        "Animals",
        "Rats",
        "Mice",
        "Rats, Sprague-Dawley",
        "Rats, Wistar",
        "Mice, Inbred C57BL",
        "Rats, Inbred Strains",
        "Disease Models, Animal",
        "Dogs",
        "Rabbits",
        "Swine",
        "Mice, Inbred BALB C",
        "Guinea Pigs",
        "Mice, Knockout",
        "Cattle",
        "Animals, Newborn",
        "Mice, Transgenic",
        "Chickens",
        "Sheep",
        "Mice, Inbred Strains",
        "Rats, Inbred F344",
    ]
    dset = [
        item
        for item in dset
        if not any([mesh_term in animal_mesh_terms for mesh_term in item["mesh_terms"]])
    ]

    """Count organ appearances via mesh terms"""
    organ_count_dict = {}
    for abstract in tqdm(dset):
        organ_names = abstract["organ_names"]
        for organ_name in organ_names:
            if organ_name not in organ_count_dict:
                organ_count_dict[organ_name] = 1
            else:
                organ_count_dict[organ_name] += 1
    print("Organ mesh term appearance counts in dataset...")
    print(organ_count_dict)

    dset_train, dset_val_test = dataset_split(dset, test_size=train_percentage)
    dset_val, dset_test = dataset_split(dset_val_test, test_size=0.5)

    json.dump(dset, open(dset_path, "w"))

    if split:
        json.dump(
            dset_train,
            open(
                os.path.splitext(dset_path)[0]
                + "_train"
                + os.path.splitext(dset_path)[1],
                "w",
            ),
        )

        json.dump(
            dset_val,
            open(
                os.path.splitext(dset_path)[0]
                + "_val"
                + os.path.splitext(dset_path)[1],
                "w",
            ),
        )

        json.dump(
            dset_test,
            open(
                os.path.splitext(dset_path)[0]
                + "_test"
                + os.path.splitext(dset_path)[1],
                "w",
            ),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a dataset based on a directory containing json files with allMesh dataset abstracts and a directory of organs."
    )
    parser.add_argument(
        "--dset_path", type=str, help="Path under which the dataset is saved."
    )
    parser.add_argument(
        "--all_mesh_path",
        type=str,
        help="Path to the directory with json files containing abstracts.",
    )
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        help="Path to the directory with organ dictionaries.",
    )
    parser.add_argument(
        "--organ_cap",
        type=int,
        default=1000,
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

    return parser.parse_args()


def main():
    args = parse_args()
    create_all_mesh_dataset(
        args.dset_path,
        args.all_mesh_path,
        args.organs_dir_path,
        args.organ_cap,
        args.train_percentage,
        args.split,
    )


if __name__ == "__main__":
    main()
