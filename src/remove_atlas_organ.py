import os
import json
import argparse


def remove_organs_from_atlas(src_dir, dst_dir, organs_to_remove, images_path):

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    ind2organ = json.load(open(os.path.join(src_dir, "ind2organ.json")))
    organ2ind = json.load(open(os.path.join(src_dir, "organ2ind.json")))
    organ2label = json.load(open(os.path.join(src_dir, "organ2label.json")))
    organ2alias = json.load(open(os.path.join(src_dir, "organ2alias.json")))
    organ2center = json.load(open(os.path.join(src_dir, "organ2center.json")))
    organ2voxels = json.load(open(os.path.join(src_dir, "organ2voxels.json")))

    for organ_to_remove in organs_to_remove:
        del ind2organ[str(organ2ind[organ_to_remove])]
        del organ2ind[organ_to_remove]
        del organ2label[organ_to_remove]
        del organ2alias[organ_to_remove]
        del organ2center[organ_to_remove]
        del organ2voxels[organ_to_remove]

    json.dump(ind2organ, open(os.path.join(dst_dir, "ind2organ.json"), "w"))
    json.dump(organ2ind, open(os.path.join(dst_dir, "organ2ind.json"), "w"))
    json.dump(organ2label, open(os.path.join(dst_dir, "organ2label.json"), "w"))
    json.dump(organ2alias, open(os.path.join(dst_dir, "organ2alias.json"), "w"))
    json.dump(organ2center, open(os.path.join(dst_dir, "organ2center.json"), "w"))
    json.dump(organ2voxels, open(os.path.join(dst_dir, "organ2voxels.json"), "w"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove a set of organs from the human atlas dictionaries."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        help="Path to the directory with the source organ dictionaries.",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Path to the directory where the new organ dictionaries will be stored.",
    )
    parser.add_argument(
        "--organs_to_remove",
        type=str,
        action="append",
        help="A set of organ names that will be removed",
    )
    parser.add_argument(
        "--images_path", type=str, help="Path to the directory with voxelman images."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    remove_organs_from_atlas(
        args.src_dir, args.dst_dir, args.organs_to_remove, args.images_path
    )


if __name__ == "__main__":
    main()
