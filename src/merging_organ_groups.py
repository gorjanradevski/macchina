import argparse
import json
import os
import random
from typing import List

import natsort
import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import ConvexHull

from skimage.measure import label
from utils.constants import VOXELMAN_CENTER


def get_images(folder, extension):
    """Return file names of image files inside a folder.

    Args:
        folder: str - path to folder
        extension: str - acceptable extension of files
    """
    return natsort.natsorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.endswith(extension)
        ]
    )[::-1]


def read_images(folder, extension=".tif"):
    """Return a 3D numpy array of stacked images in folder

        Args:
            folder: str - path to folder
            extension: str - acceptable extension of files
        """
    image_files = get_images(folder, extension)
    images = tifffile.imread(image_files)
    images = images.transpose(1, 2, 0)
    return images


def getLargestCC(points):
    labels = label(points)
    assert labels.max() != 0, "No connected regions"
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return np.where(largestCC == True)  # noqa: E712


def get_center_of_mass(labels, images_path):

    images_in = read_images(images_path, extension=".tif")

    images = np.zeros(images_in.shape, dtype=int)
    for _label in labels:
        images[images_in == _label] = 1

    erosion_mask = generate_binary_structure(3, 1)
    i = 0
    while True:
        last_points = np.where(images != 0)
        images = binary_erosion(images, erosion_mask).astype(int)
        i += 1
        if not images.sum():
            print(f"Eroded all voxels after {i} erosions")
            break
    images[last_points] = 1
    last_points = getLargestCC(images)
    mass_center = np.array(last_points).transpose().mean(axis=0)
    mass_center = mass_center - VOXELMAN_CENTER
    return mass_center.tolist()


def point_within_organ(point, labels, images_path):
    images = read_images(images_path, extension=".tif")
    point = np.round(point + VOXELMAN_CENTER)
    x, y, z = point.astype(int)
    correct = int(images[x, y, z] in labels)
    return correct


def get_organ2summary(organ2voxels: str, num_anchors: int = 1000):
    organ2summary = {}

    for organ, voxels in organ2voxels.items():
        if len(voxels) > num_anchors:
            organ2summary[organ] = random.sample(voxels, num_anchors)
        else:
            organ2summary[organ] = np.array(voxels)[
                np.random.choice(range(len(voxels)), num_anchors)
            ].tolist()

    return organ2summary


def organ_density(points) -> float:
    points = np.array(points)
    hull = ConvexHull(points)
    return len(points) / hull.volume


def merge_organs(
    organ_names: list, organ2voxels: list, downweigh_dense: bool = False
) -> list:

    organ_point_counts = []
    densities = []
    for organ in organ_names:
        organ_point_counts.append(len(organ2voxels[organ]))
        densities.append(organ_density(organ2voxels[organ]))
    smallest_point_count = min(organ_point_counts)

    densities = np.array(densities)
    inverse_densities = (densities.sum() - densities) / densities.sum()

    voxels = []
    for i, organ in enumerate(organ_names):
        organ_voxels = random.sample(organ2voxels[organ], smallest_point_count)
        if downweigh_dense:
            organ_voxels = random.sample(
                organ_voxels, int(inverse_densities[i] * len(organ_voxels))
            )
        voxels.extend(organ_voxels)
    return voxels


def merge_organ_groups(
    src_dir,
    dst_dir,
    organ_groups: List[List],
    superorgan_names: List,
    superorgan_indices: List,
    images_path,
):
    assert any(
        isinstance(item, List) for item in organ_groups
    ), "Organ groups need to be a list of lists"
    assert isinstance(superorgan_names, List), "Superorgan names needs to be a list"
    assert isinstance(superorgan_indices, List), "Superorgan indices needs to be a list"
    assert (
        len(organ_groups) == len(superorgan_names) == len(superorgan_indices)
    ), "The number of organ groups, superorgan names and superorgan indices needs to match"

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    ind2organ = json.load(open(os.path.join(src_dir, "ind2organ.json")))
    organ2ind = json.load(open(os.path.join(src_dir, "organ2ind.json")))
    organ2label = json.load(open(os.path.join(src_dir, "organ2label.json")))
    organ2alias = json.load(open(os.path.join(src_dir, "organ2alias.json")))
    organ2center = json.load(open(os.path.join(src_dir, "organ2center.json")))
    organ2voxels = json.load(open(os.path.join(src_dir, "organ2voxels.json")))

    for organs_to_merge, superorgan_name, superorgan_index in zip(
        organ_groups, superorgan_names, superorgan_indices
    ):

        print(f"Merging: {organs_to_merge} into a superorgan: {superorgan_name}")

        ind2organ[superorgan_index] = superorgan_name
        organ2ind[superorgan_name] = int(superorgan_index)

        aliases = []
        labels = []

        for organ_to_merge in organs_to_merge:
            aliases = aliases + organ2alias[organ_to_merge]
            labels = labels + organ2label[organ_to_merge]

        voxels = merge_organs(organs_to_merge, organ2voxels, downweigh_dense=True)

        organ2alias[superorgan_name] = aliases
        organ2label[superorgan_name] = labels
        organ2voxels[superorgan_name] = voxels
        organ2center[superorgan_name] = get_center_of_mass(
            organ2label[superorgan_name], images_path
        )

        if point_within_organ(
            organ2center[superorgan_name], organ2label[superorgan_name], images_path
        ):
            print("Center of mass is inside merged organ")
        else:
            print("Center of mass is not inside merged organ, that is an error")

        for organ_to_merge in organs_to_merge:
            del ind2organ[str(organ2ind[organ_to_merge])]
            del organ2ind[organ_to_merge]
            del organ2label[organ_to_merge]
            del organ2alias[organ_to_merge]
            del organ2center[organ_to_merge]
            del organ2voxels[organ_to_merge]

    organ2summary = get_organ2summary(organ2voxels, num_anchors=1000)

    json.dump(ind2organ, open(os.path.join(dst_dir, "ind2organ.json"), "w"))
    json.dump(organ2ind, open(os.path.join(dst_dir, "organ2ind.json"), "w"))
    json.dump(organ2label, open(os.path.join(dst_dir, "organ2label.json"), "w"))
    json.dump(organ2alias, open(os.path.join(dst_dir, "organ2alias.json"), "w"))
    json.dump(organ2center, open(os.path.join(dst_dir, "organ2center.json"), "w"))
    json.dump(organ2voxels, open(os.path.join(dst_dir, "organ2voxels.json"), "w"))
    json.dump(organ2summary, open(os.path.join(dst_dir, "organ2summary.json"), "w"))


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Group organs into superorgans.")
    parser.add_argument(
        "--src_dir", type=str, help="Path to the source data organs directory path."
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Path to the destination data organs directory path.",
    )
    parser.add_argument(
        "-og",
        "--organ_groups",
        type=str,
        nargs="+",
        action="append",
        help="List of lists - a list of organ groups that will be merged",
    )
    parser.add_argument(
        "-sn",
        "--superorgan_names",
        type=str,
        action="append",
        help="List of names of superorgans that will be formed",
    )
    parser.add_argument(
        "-si",
        "--superorgan_indices",
        type=str,
        action="append",
        help="List of indices of superorgans that will be formed",
    )
    parser.add_argument("--images_path", type=str, help="Path to the voxel-man images")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_organ_groups(
        args.src_dir,
        args.dst_dir,
        args.organ_groups,
        args.superorgan_names,
        args.superorgan_indices,
        args.images_path,
    )


"""
respiratory_organs = ["bronchi", "diaphragm", "trachea", "lung", "larynx"]
digestive_organs_solid = ["gallbladder", "liver", "pancreas"]
digestive_organs_hollow = ["ampulla", "ascending colon", "duodenum", "cystic duct", "rectum", "sigmoid colon", "stomach", "transverse colon", "small intestine", "descending colon", "caecum"]
"""

if __name__ == "__main__":
    main()
