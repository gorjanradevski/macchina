import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from copy import deepcopy

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestDataset,
    VoxelSentenceMappingTestMaskedDataset,
    collate_pad_sentence_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.evaluator import Evaluator


def inference(
    ind2organ_path: str,
    organ2label_path: str,
    voxelman_images_path: str,
    test_json_path: str,
    batch_size: int,
    bert_path_or_name: str,
    checkpoint_path: str,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = VoxelSentenceMappingTestDataset(test_json_path, bert_path_or_name)
    test_masked_dataset = VoxelSentenceMappingTestMaskedDataset(
        test_json_path, bert_path_or_name
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_batch,
    )
    test_masked_loader = DataLoader(
        test_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_batch,
    )
    # Create model
    model = nn.DataParallel(
        SentenceMappingsProducer(bert_path_or_name, joint_space, finetune=False)
    ).to(device)
    # Load model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    # Create evaluator
    evaluator = Evaluator(ind2organ_path, organ2label_path, voxelman_images_path)
    with torch.no_grad():
        # Restart counters
        evaluator.reset_counters()
        for sentences, _, _, organs_indices in tqdm(test_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences).cpu().numpy()
            for output_mapping, organ_indices in zip(output_mappings, organs_indices):
                evaluator.update_counters(output_mapping, organ_indices.numpy())

        print(
            f"The accuracy on the masked validation set is {evaluator.get_current_accuracy()}"
        )
        # Restart counters
        evaluator.reset_counters()
        for sentences, _, _, organs_indices in tqdm(test_masked_loader):
            sentences = sentences.to(device)
            output_mappings = model(sentences).cpu().numpy()
            for output_mapping, organ_indices in zip(output_mappings, organs_indices):
                evaluator.update_counters(output_mapping, organ_indices.numpy())

        print(
            f"The accuracy on the masked validation set is {evaluator.get_current_accuracy()}"
        )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.ind2organ_path,
        args.organ2label_path,
        args.voxelman_images_path,
        args.test_json_path,
        args.batch_size,
        args.bert_path_or_name,
        args.checkpoint_path,
        args.joint_space,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs mapping inference.")
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs/ind2organ.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--organ2label_path",
        type=str,
        default="data/data_organs/organ2label.json",
        help="Path to the organ2label path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman/",
        help="Path to the voxel-man images",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/val_dataset.json",
        help="Path to the validation set",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--joint_space",
        type=int,
        default=512,
        help="The joint space where the encodings will be projected.",
    )
    parser.add_argument(
        "--bert_path_or_name",
        type=str,
        default="bert-base-uncased",
        help="The name or path to a pretrained bert model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained.pt",
        help="Path to a pretrained checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
