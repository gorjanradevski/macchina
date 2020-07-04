import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import os
import json
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestRegDataset,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import RegModel
from voxel_mapping.evaluator import InferenceEvaluatorPerOrgan
from utils.constants import VOXELMAN_CENTER


def inference(
    organs_dir_path: str,
    voxelman_images_path: str,
    test_json_path: str,
    batch_size: int,
    bert_name: str,
    checkpoint_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    test_dataset = VoxelSentenceMappingTestRegDataset(test_json_path, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    # Create model
    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(RegModel(bert_name, config, final_project_size=3)).to(
        device
    )
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    # Prepare paths
    ind2organ = json.load(open(os.path.join(organs_dir_path, "ind2organ.json")))
    organ2label = json.load(open(os.path.join(organs_dir_path, "organ2label.json")))
    organ2voxels = json.load(open(os.path.join(organs_dir_path, "organ2voxels.json")))
    # Create evaluator
    evaluator = InferenceEvaluatorPerOrgan(
        ind2organ, organ2label, organ2voxels, voxelman_images_path, len(test_dataset),
    )
    center = torch.from_numpy(VOXELMAN_CENTER)
    with torch.no_grad():
        # Restart counters
        evaluator.reset_counters()
        for sentences, attn_mask, organs_indices, _ in tqdm(test_loader):
            sentences, attn_mask = sentences.to(device), attn_mask.to(device)
            output_mappings = (
                model(input_ids=sentences, attention_mask=attn_mask).cpu() * center
            )
            for output_mapping, organ_indices in zip(output_mappings, organs_indices):
                evaluator.update_counters(output_mapping.numpy(), organ_indices.numpy())

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
        print("============================================")
        for organ_name in evaluator.organ_names:
            if evaluator.get_current_ior_for_organ(organ_name) > -1:
                print(
                    f"The avg IOR for {organ_name} is: {evaluator.get_current_ior_for_organ(organ_name)}"
                )
                print(
                    f"The avg NVD for {organ_name} is: {evaluator.get_current_distance_for_organ(organ_name)}"
                )
                print("============================================")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.organs_dir_path,
        args.voxelman_images_path,
        args.test_json_path,
        args.batch_size,
        args.bert_name,
        args.checkpoint_path,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs class mapping inference.")
    parser.add_argument(
        "--organs_dir_path",
        type=str,
        default="data/data_organs",
        help="Path to the data organs directory path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxelman images",
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
