import argparse
import json
import os
import random
from tqdm import tqdm
import numpy as np
from voxel_mapping.evaluator import InferenceEvaluatorPerOrgan


def occurrence_evaluation_simple(dset_path: str):
    test_dataset = json.load(open(dset_path))

    hits = []
    for item in tqdm(test_dataset):
        gt_organs = item["organ_names"]
        occ_organs = item["occ_organ_names"]

        if occ_organs:
            pred_organ = random.sample(occ_organs, 1)[0]
            if pred_organ in gt_organs:
                hits.append(1)
            else:
                hits.append(0)
        else:
            hits.append(0)

    hits = np.array(hits)
    print(
        f"Hit rate: {hits.sum()/hits.size * 100:.1f} +/- {np.std(hits, ddof=1) / np.sqrt(hits.size) * 100:.1f}"
    )


def occurrence_evaluation_complex(
    dset_path: str, organs_dir_path: str, voxelman_images_path: str
):
    test_dataset = json.load(open(dset_path))
    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    organ2label = json.load(open(os.path.join(organs_dir_path, "organ2label.json")))
    organ2voxels = json.load(open(os.path.join(organs_dir_path, "organ2summary.json")))
    organ2center = json.load(open(os.path.join(organs_dir_path, "organ2center.json")))
    evaluator = InferenceEvaluatorPerOrgan(
        ind2organ, organ2label, organ2voxels, voxelman_images_path, len(test_dataset)
    )

    for item in tqdm(test_dataset):
        gt_indices = item["organ_indices"]
        occ_organs = item["occ_organ_names"]

        if occ_organs:
            pred_organ = random.sample(occ_organs, 1)[0]
            pred = np.array(random.sample(organ2voxels[pred_organ], 1))[0]
        else:
            pred = np.array([0.0, 0.0, 0.0])

        evaluator.update_counters(pred, np.array(gt_indices))

    print(
        "The avg IOR on the test set is: "
        f"{evaluator.get_current_ior()} +/- {evaluator.get_ior_error_bar()}"
    )
    print(
        "The avg distance on the test set is: "
        f"{evaluator.get_current_distance()} +/- {evaluator.get_distance_error_bar()}"
    )
    print(
        "The avg miss distance on the test set is: "
        f"{evaluator.get_current_miss_distance()} +/- {evaluator.get_miss_distance_error_bar()}"
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs naive baseline evaluation.")
    parser.add_argument("--test_json_path", type=str, help="Path to the test set.")
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs",
        help="Path to the data organs directory.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxelman images.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Whether to perform a simple baseline without computing distances.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.simple:
        occurrence_evaluation_simple(args.test_json_path)
    else:
        occurrence_evaluation_complex(
            args.test_json_path, args.organs_dir_path, args.voxelman_images_path
        )


if __name__ == "__main__":
    main()
