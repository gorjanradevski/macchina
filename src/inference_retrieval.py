import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from voxel_mapping.datasets import (
    VoxelSentenceMappingTestRegDataset,
    collate_pad_sentence_reg_test_batch,
)
from voxel_mapping.models import model_factory
from voxel_mapping.retrieval_utils import EmbeddedDoc
from utils.constants import VOXELMAN_CENTER


def inference(
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
    test_dataset = VoxelSentenceMappingTestRegDataset(test_json_path, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_sentence_reg_test_batch,
    )
    # Create and load model, then set it to eval mode
    config = BertConfig.from_pretrained(bert_name)
    model = nn.DataParallel(
        model_factory(model_name, bert_name, config, project_size)
    ).to(device)
    assert (
        model.module.bert.embeddings.word_embeddings.num_embeddings
        == tokenizer.vocab_size
    )
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Not loading from checkpoint! Inference from {bert_name}")
    model.train(False)
    # Get voxelman center
    center = torch.from_numpy(VOXELMAN_CENTER)
    embedded_docs = []
    with torch.no_grad():
        for sentences, attn_mask, organs_indices, docs_ids in tqdm(test_loader):
            sentences, attn_mask = sentences.to(device), attn_mask.to(device)
            output_mappings = model(input_ids=sentences, attention_mask=attn_mask).cpu()
            if model_name == "reg_model":
                # The reg_model normalizes the embeddings between -1 and 1
                output_mappings *= center
            for output_mapping, organ_indices, doc_id in zip(
                output_mappings, organs_indices, docs_ids
            ):
                # Get only non -1 indices
                organ_indices = organ_indices[: (organ_indices >= 0).sum()]
                embedded_docs.append(
                    EmbeddedDoc(doc_id, organ_indices.numpy(), output_mapping.numpy())
                )

    recalls = {"1": 0, "5": 0, "10": 0}
    precisions = {"1": 0, "5": 0, "10": 0}
    for document1 in tqdm(embedded_docs):
        cur_doc_distances = []
        for document2 in embedded_docs:
            if document1.doc_id == document2.doc_id:
                continue
            cur_doc_distances.append(
                (document2.organ_indices, document1.docs_distance(document2))
            )
        cur_doc_distances_sorted = sorted(cur_doc_distances, key=lambda tup: tup[1])
        for k in recalls.keys():
            for cur_doc in cur_doc_distances_sorted[: int(k)]:
                if cur_doc[0].shape == document1.organ_indices.shape:
                    if (cur_doc[0] == document1.organ_indices).all():
                        recalls[k] += 1
                        break
        for k in precisions.keys():
            cur_precision = 0
            for cur_doc in cur_doc_distances_sorted[: int(k)]:
                if cur_doc[0].shape == document1.organ_indices.shape:
                    if (cur_doc[0] == document1.organ_indices).all():
                        cur_precision += 1
            cur_precision /= int(k)
            precisions[k] += cur_precision

    for k, recall in recalls.items():
        print(f"The recall at {k} is: {round(recall/len(embedded_docs) * 100, 1)}")

    for k, precision in precisions.items():
        print(
            f"The precision at {k} is: {round(precision/len(embedded_docs) * 100, 1)}"
        )


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
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
    parser = argparse.ArgumentParser(description="Evaluates recall at K retrieval.")
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="data/dataset_text_atlas_mapping_test_fixd.json",
        help="Path to the test set",
    )
    parser.add_argument(
        "--model_name", type=str, default="reg_model", help="The model name.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
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
    parser.add_argument(
        "--project_size", type=int, default=3, help="The projection size."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
