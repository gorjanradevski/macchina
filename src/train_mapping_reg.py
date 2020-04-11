import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch import nn
from tqdm import tqdm
import json
from typing import Dict
import numpy as np
import random
import sys
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTrainRegDataset,
    VoxelSentenceMappingTestRegDataset,
    VoxelSentenceMappingTestMaskedRegDataset,
    collate_pad_sentence_reg_train_batch,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import SentenceMappingsProducer
from voxel_mapping.losses import MinDistanceLoss, OrganDistanceLoss
from voxel_mapping.evaluator import TrainingRegEvaluator


def create_ind2anchors(
    organ2ind_path: str, organ2voxels_path: str, num_anchors: int = 1000
) -> Dict:
    """CREATING A MAPPING FROM INDEX TO A SET OF ORGAN POINTS"""
    """RANDOM SAMPLING ORGAN VOXELS - DONE IN THE BEGINING OF EVERY EPOCH"""

    organ2ind = json.load(open(organ2ind_path))
    organ2voxels = json.load(open(organ2voxels_path))
    ind2voxels = {}

    for organ, ind in organ2ind.items():
        if len(organ2voxels[organ]) > num_anchors:
            ind2voxels[ind] = random.sample(organ2voxels[organ], num_anchors)
        else:
            ind2voxels[ind] = np.array(organ2voxels[organ])[
                np.random.choice(range(len(organ2voxels[organ])), num_anchors)
            ].tolist()

    ind2voxels[-1] = np.zeros(shape=(num_anchors, 3), dtype=np.float)

    return ind2voxels


def train(
    organ2ind_path: str,
    organ2voxels_path: str,
    ind2organ_path: str,
    organ2label_path: str,
    organ2summary_path: str,
    voxelman_images_path: str,
    train_json_path: str,
    val_json_path: str,
    use_all_voxels: bool,
    epochs: int,
    batch_size: int,
    bert_path_or_name: str,
    checkpoint_path: str,
    save_model_path: str,
    save_intermediate_model_path: str,
    learning_rate: float,
    clip_val: float,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check for the type of loss
    ind2anchors = None
    if use_all_voxels:
        ind2anchors = create_ind2anchors(organ2ind_path, organ2voxels_path, 1000)
        criterion = OrganDistanceLoss()
        print("Using all organ points!")
    else:
        print("Using only one organ center!")
        criterion = MinDistanceLoss()

    tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
    organ_names = [organ_name for organ_name in json.load(open(organ2ind_path)).keys()]
    train_dataset = Subset(
        VoxelSentenceMappingTrainRegDataset(
            train_json_path, tokenizer, organ_names, ind2anchors
        ),
        list(range(20)),
    )
    val_dataset = Subset(
        VoxelSentenceMappingTestRegDataset(val_json_path, tokenizer, ind2anchors),
        list(range(20)),
    )
    val_masked_dataset = Subset(
        VoxelSentenceMappingTestMaskedRegDataset(val_json_path, tokenizer, ind2anchors),
        list(range(20)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_train_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    val_masked_loader = DataLoader(
        val_masked_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    config = BertConfig.from_pretrained(bert_path_or_name)
    model = nn.DataParallel(SentenceMappingsProducer(bert_path_or_name, config)).to(
        device
    )
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load model
    cur_epoch = 0
    best_avg_distance = sys.maxsize
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_avg_distance = checkpoint["best_distance"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint
        print(
            f"Starting training from checkpoint {checkpoint_path} with starting epoch {cur_epoch}!"
        )
        print(f"The previous best distance was: {best_avg_distance}!")

    # Create evaluator
    evaluator = TrainingRegEvaluator(
        ind2organ_path,
        organ2label_path,
        voxelman_images_path,
        organ2summary_path,
        len(val_dataset),
        best_avg_distance,
    )

    for epoch in range(cur_epoch, cur_epoch + epochs):
        print(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for sentences, true_mappings, num_organs, true_labels in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                sentences, true_mappings, num_organs = (
                    sentences.to(device),
                    true_mappings.to(device),
                    num_organs.to(device),
                )
                output_mappings = model(sentences)
                loss = criterion(output_mappings, true_mappings, num_organs, device)
                # backward
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        # Reset current average distance
        evaluator.reset_current_average_distance()
        with torch.no_grad():
            # Restart counters
            evaluator.reset_counters()
            for sentences, organs_indices in tqdm(val_loader):
                sentences = sentences.to(device)
                output_mappings = model(sentences).cpu().numpy()
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            print(
                f"The IOR on the non-masked validation set is {evaluator.get_current_ior()}"
            )
            print(
                f"The distance on the non-masked validation set is {evaluator.get_current_distance()}"
            )
            evaluator.update_current_average_distance()
            # Restart counters
            evaluator.reset_counters()
            for sentences, organs_indices in tqdm(val_masked_loader):
                sentences = sentences.to(device)
                output_mappings = model(sentences).cpu().numpy()
                for output_mapping, organ_indices in zip(
                    output_mappings, organs_indices
                ):
                    evaluator.update_counters(output_mapping, organ_indices.numpy())

            print(
                f"The IOR on the masked validation set is {evaluator.get_current_ior()}"
            )
            print(
                f"The distance on the masked validation set is {evaluator.get_current_distance()}"
            )
            evaluator.update_current_average_distance()

            if evaluator.is_best_avg_distance():
                evaluator.update_best_avg_distance()
                print("======================")
                print(
                    f"Found new best with avg distance: "
                    f"{evaluator.best_avg_distance} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                print("======================")
                # torch.save(model.state_dict(), save_model_path)
            else:
                print(
                    f"Avg distance on epoch {epoch+1} is: "
                    f"{evaluator.current_average_distance}"
                )
            print("Saving intermediate checkpoint...")
            """
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_distance": evaluator.best_avg_distance,
                },
                save_intermediate_model_path,
            )
            """


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    train(
        args.organ2ind_path,
        args.organ2voxels_path,
        args.ind2organ_path,
        args.organ2label_path,
        args.organ2summary_path,
        args.voxelman_images_path,
        args.train_json_path,
        args.val_json_path,
        args.use_all_voxels,
        args.epochs,
        args.batch_size,
        args.bert_path_or_name,
        args.checkpoint_path,
        args.save_model_path,
        args.save_intermediate_model_path,
        args.learning_rate,
        args.clip_val,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains atlas reg mapping model.")
    parser.add_argument(
        "--organ2voxels_path",
        type=str,
        default="data/data_organs_sages/organ2voxels_eroded.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--organ2summary_path",
        type=str,
        default="data/data_organs_sages/organ2summary.json",
        help="Path to the organ2label file.",
    )
    parser.add_argument(
        "--organ2ind_path",
        type=str,
        default="data/data_organs_sages/organ2ind.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--ind2organ_path",
        type=str,
        default="data/data_organs_sages/ind2organ.json",
        help="Path to the ind2organ path.",
    )
    parser.add_argument(
        "--organ2label_path",
        type=str,
        default="data/data_organs_sages/organ2label.json",
        help="Path to the organ2label path.",
    )
    parser.add_argument(
        "--voxelman_images_path",
        type=str,
        default="data/voxelman_images",
        help="Path to the voxel-man images",
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_train_fixd.json",
        help="Path to the training set",
    )
    parser.add_argument(
        "--val_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_val_fixd.json",
        help="Path to the validation set",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/sentence_mapping_regressor.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--use_all_voxels", action="store_true", help="Whether to use the all voxels."
    )
    parser.add_argument(
        "--save_intermediate_model_path",
        type=str,
        default="models/intermediate_sentence_mapping_regressor.pt",
        help="Where to save the intermediate checkpoint model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="The number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="The learning rate."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
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
        default=None,
        help="If resuming training, start from here.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
