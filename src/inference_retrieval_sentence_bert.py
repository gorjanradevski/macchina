import argparse
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from voxel_mapping.retrieval_utils import EmbeddedDoc


def inference(test_json_path: str, model_path: str):
    test_set = json.load(open(test_json_path))
    model = SentenceTransformer(model_path)
    corpus = [item["text"] for item in test_set]
    embeddings = model.encode(corpus, show_progress_bar=True)

    embedded_docs = []
    for item, embedding in tqdm(zip(test_set, embeddings)):
        doc_id = item["pmid"]
        organ_indices = np.array(item["organ_indices"])
        embedded_docs.append(EmbeddedDoc(doc_id, organ_indices, np.array(embedding)))

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
    args = parse_args()
    inference(args.test_json_path, args.model_path)


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
        "--model_path", type=str, help="The path to the pretrained sentence bert."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
