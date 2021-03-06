import json
import os
from typing import List, Union

import natsort
import numpy as np
import tifffile

from utils.constants import VOXELMAN_CENTER


class Evaluator:
    def __init__(
        self,
        ind2organ_path: str,
        organ2label_path: str,
        organ2summary_path: str,
        voxelman_images_path: str,
        total_samples: int,
    ):
        self.total_samples = total_samples
        self.organ2label = json.load(open(organ2label_path))
        self.ind2organ = json.load(open(ind2organ_path))
        self.organ2summary = json.load(open(organ2summary_path))
        image_files = natsort.natsorted(
            [
                os.path.join(voxelman_images_path, f)
                for f in os.listdir(voxelman_images_path)
                if os.path.isfile(os.path.join(voxelman_images_path, f))
                and f.endswith(".tif")
            ]
        )[::-1]
        self.voxelman = tifffile.imread(image_files).transpose(1, 2, 0)
        self.corrects = np.zeros(self.total_samples)
        self.distances = np.zeros(self.total_samples)
        self.index = 0

    def reset_counters(self):
        self.index = 0
        self.corrects = np.zeros(self.total_samples)
        self.distances = np.zeros(self.total_samples)

    def update_counters(self, output_mapping: np.ndarray, organ_indices: np.ndarray):
        self.corrects[self.index] = self.voxels_inside(output_mapping, organ_indices)
        self.distances[self.index] = self.voxels_distance(output_mapping, organ_indices)
        self.index += 1

    def get_current_ior(self):
        return np.round((np.sum(self.corrects) / self.total_samples) * 100, decimals=2)

    def get_current_distance(self):
        return np.round((np.sum(self.distances) / self.total_samples) / 10, decimals=2)

    def get_current_miss_distance(self):
        return np.round(
            (np.sum(self.distances) / np.count_nonzero(self.distances)) / 10, decimals=2
        )

    def voxels_inside(
        self, pred: np.ndarray, organ_indices: Union[List, np.ndarray]
    ) -> int:
        """
        :param pred: prediction for one sample, numpy array or shape (3,)
        :param organ_indices: set of true organ indices for the sample, list or numpy array
        :return: Array with k entries
         1.0 at i-th entry - pred is inside the voxels of the i-th organ,
         0.0 at i-th entry - pred is outside of the voxels of the i-th organ
        """
        organ_indices = np.array(organ_indices)
        corrects = np.zeros(organ_indices.size)
        pred = np.round(pred + VOXELMAN_CENTER)
        pred = np.clip(pred, a_min=[0, 0, 0], a_max=(VOXELMAN_CENTER * 2) - 1)
        x, y, z = pred.astype(int)
        for i, organ_index in enumerate(organ_indices):
            if organ_index < 0:
                continue
            labels = self.organ2label[self.ind2organ[str(organ_index)]]
            corrects[i] = int(self.voxelman[x, y, z] in labels)

        return 1 if np.count_nonzero(corrects) > 0 else 0

    def voxels_distance(
        self, pred: np.ndarray, organ_indices: Union[List, np.ndarray]
    ) -> np.ndarray:
        # Setting distances to large number so that we can take minimum
        distances = np.ones(organ_indices.size, dtype=np.float) * 1e15
        pred_ind = np.round(pred + VOXELMAN_CENTER)
        pred_ind = np.clip(pred_ind, a_min=[0, 0, 0], a_max=(VOXELMAN_CENTER * 2) - 1)
        x, y, z = pred_ind.astype(int)
        for i, organ_index in enumerate(organ_indices):
            if organ_index < 0:
                continue
            str_organ_index = str(organ_index)
            if (
                self.voxelman[x, y, z]
                in self.organ2label[self.ind2organ[str_organ_index]]
            ):
                # If it is inside, the minimum will always be 0
                return 0.0
            else:
                summary_points = np.array(
                    self.organ2summary[self.ind2organ[str_organ_index]]
                )
                distances[i] = np.linalg.norm(pred - summary_points, axis=-1).min()

        return distances.min()


class InferenceEvaluator(Evaluator):
    def __init__(
        self,
        ind2organ_path: str,
        organ2label_path: str,
        organ2summary_path: str,
        voxelman_images_path: str,
        total_samples: int,
    ):
        super().__init__(
            ind2organ_path,
            organ2label_path,
            organ2summary_path,
            voxelman_images_path,
            total_samples,
        )

    def get_ior_error_bar(self):
        return np.round(
            np.std(self.corrects, ddof=1) / np.sqrt(self.total_samples) * 100,
            decimals=2,
        )

    def get_distance_error_bar(self):
        return np.round(
            np.std(self.distances, ddof=1) / np.sqrt(self.total_samples) / 10,
            decimals=2,
        )

    def get_miss_distance_error_bar(self):
        return np.round(
            np.std(self.distances[np.nonzero(self.distances)], ddof=1)
            / np.sqrt(np.count_nonzero(self.distances))
            / 10,
            decimals=2,
        )


class TrainingRegEvaluator(Evaluator):
    def __init__(
        self,
        ind2organ_path: str,
        organ2label_path: str,
        organ2summary_path: str,
        voxelman_images_path: str,
        total_samples: int,
        best_avg_distance: float,
    ):
        super().__init__(
            ind2organ_path,
            organ2label_path,
            organ2summary_path,
            voxelman_images_path,
            total_samples,
        )
        self.best_avg_distance = best_avg_distance
        self.current_average_distance = 0

    def reset_current_average_distance(self):
        self.current_average_distance = 0

    def update_current_average_distance(self):
        self.current_average_distance += self.get_current_distance()

    def finalize_current_average_distance(self):
        self.current_average_distance /= 2

    def is_best_avg_distance(self):
        return self.current_average_distance < self.best_avg_distance

    def update_best_avg_distance(self):
        self.best_avg_distance = self.current_average_distance
