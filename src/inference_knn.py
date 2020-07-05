import argparse
from collections import Counter
from itertools import groupby
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from utils.constants import VOXELMAN_CENTER
from voxel_mapping.datasets import (
    VoxelSentenceMappingTestRegDataset,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import model_factory
from voxel_mapping.retrieval_utils import EmbeddedDoc


def multimode(lst):
    # group most_common output by frequency
    freqs = groupby(Counter(lst).most_common(), lambda x: x[1])
    # pick off the first group (highest frequency)
    return [val for val, count in next(freqs)[1]]


def get_neighbor_vote(
    doc_distances: List[Tuple[np.ndarray, float]], k: int, sorted_by_dist: bool = False
):
    if not sorted_by_dist:
        doc_distances = sorted(doc_distances, key=lambda tup: tup[1])
        sorted_by_dist = True

    neighbor_annotations = [
        tuple(doc_distance[0]) for doc_distance in doc_distances[:k]
    ]
    most_common_indices = multimode(neighbor_annotations)
    if len(most_common_indices) > 1:
        return get_neighbor_vote(doc_distances, k - 1, sorted_by_dist)
    else:
        return np.array(most_common_indices[0])


def inference(
    train_json_path: str,
    test_json_path: str,
    model_name: str,
    batch_size: int,
    bert_name: str,
    project_size: int,
    checkpoint_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    train_dataset = VoxelSentenceMappingTestRegDataset(train_json_path, tokenizer)
    test_dataset = VoxelSentenceMappingTestRegDataset(test_json_path, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, collate_fn=collate_pad_sentence_reg_test_batch
    )
    # Create and load model, then set it to eval mode
    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(
        model_factory(model_name, bert_name, config, project_size)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    # Get voxelman center
    center = torch.from_numpy(VOXELMAN_CENTER)

    print("Embedding training set documents...")
    embedded_train_docs = []
    with torch.no_grad():
        for sentences, attn_mask, organs_indices, docs_ids in tqdm(train_loader):
            sentences, attn_mask = sentences.to(device), attn_mask.to(device)
            output_mappings = (
                model(input_ids=sentences, attention_mask=attn_mask).cpu() * center
            )
            for output_mapping, organ_indices, doc_id in zip(
                output_mappings, organs_indices, docs_ids
            ):
                # Get only non -1 indices
                organ_indices = organ_indices[: (organ_indices >= 0).sum()]
                embedded_train_docs.append(
                    EmbeddedDoc(
                        doc_id, np.sort(organ_indices.numpy()), output_mapping.numpy()
                    )
                )

    K = {"1": 0, "5": 0, "10": 0}
    print("Evaluating test set documents...")
    with torch.no_grad():
        for sentences, attn_mask, organs_indices, docs_ids in tqdm(test_loader):
            sentences, attn_mask = sentences.to(device), attn_mask.to(device)
            output_mappings = (
                model(input_ids=sentences, attention_mask=attn_mask).cpu() * center
            )
            organ_indices = organ_indices[: (organ_indices >= 0).sum()]
            embedded_doc = EmbeddedDoc(
                docs_ids[0], np.sort(organ_indices.numpy()), output_mapping.numpy()
            )

            cur_doc_distances = []
            for embedded_train_doc in embedded_train_docs:
                if embedded_doc.doc_id == embedded_train_doc.doc_id:
                    continue
                cur_doc_distances.append(
                    (
                        embedded_train_doc.organ_indices,
                        embedded_doc.docs_distance(embedded_train_doc),
                    )
                )
            cur_doc_distances = sorted(cur_doc_distances, key=lambda tup: tup[1])
            for k in K.keys():
                most_voted_indices = get_neighbor_vote(
                    cur_doc_distances, int(k), sorted_by_dist=True
                )
                if most_voted_indices.shape == embedded_doc.organ_indices.shape:
                    if (most_voted_indices == embedded_doc.organ_indices).all():
                        K[k] += 1

    for k, corrects in K.items():
        print(f"KNN accuracy at k={k} is: {round(corrects/len(test_dataset) * 100, 1)}")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.train_json_path,
        args.test_json_path,
        args.model_name,
        args.batch_size,
        args.bert_name,
        args.project_size,
        args.checkpoint_path,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluates KNN classification accuracy."
    )
    parser.add_argument("--train_json_path", type=str, help="Path to the train set")
    parser.add_argument("--test_json_path", type=str, help="Path to the test set")
    parser.add_argument(
        "--model_name", type=str, default="reg_model", help="The model name."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The size of the batch."
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="Should be one of [bert-base-uncased, allenai/scibert_scivocab_uncased,"
        "monologg/biobert_v1.1_pubmed, emilyalsentzer/Bio_ClinicalBERT,"
        "google/bert_uncased_L-4_H-512_A-8]",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint.",
    )
    parser.add_argument(
        "--project_size", type=int, default=3, help="The projection size."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
