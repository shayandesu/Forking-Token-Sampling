from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn.functional as F
import numpy as np
import scipy
from safetensors.torch import load_file
from sklearn import preprocessing
from tqdm import tqdm


class Vendi:
    @staticmethod
    def entropy_q(p, q=1):
        p_ = p[p > 0]
        if q == 1:
            return -(p_ * np.log(p_)).sum()
        if q == "inf":
            return -np.log(np.max(p))
        return np.log((p_ ** q).sum()) / (1 - q)

    @staticmethod
    def compute_reverse_similarity_matrix(data, normalize=True):
        if normalize:
            data = torch.Tensor(preprocessing.normalize(data.cpu(), axis=1)).cuda()

        return data.T @ data

    @staticmethod
    def compute_vendi_score(sim_matrix, n):
        w = scipy.linalg.eigvalsh(sim_matrix.cpu() / n)
        return np.exp(Vendi.entropy_q(w, q=1)).item()


class GradientVendi:
    @staticmethod
    def load_all_gradients(gradient_files: List[Path] | Path, device=0) -> Tuple[List, torch.Tensor]:
        if isinstance(gradient_files, Path):
            # gradient_files is the parent folder of safetensor files
            gradient_files = list(gradient_files.glob("*.safetensors"))

        all_gradient_dict = {}
        for gradient_file in gradient_files:
            assert gradient_file.suffix == ".safetensors", \
                "`gradient_files` should only include `.safetensor` files."
            all_gradient_dict.update(load_file(gradient_file, device=device))

        # convert into two lists - one for id, one for gradient
        sample_ids, sample_gradients = [], []
        for sample_id, sample_gradient in all_gradient_dict.items():
            sample_ids.append(sample_id)
            sample_gradients.append(sample_gradient)

        # convert `sample_gradients` into a single tensor
        sample_gradients = torch.stack(sample_gradients, dim=0)

        return sample_ids, sample_gradients

    @staticmethod
    def load_gradients_for_sample_ids(gradient_files: List[Path] | Path, sample_ids: List[str], device=0) -> Tuple[List, torch.Tensor]:
        if isinstance(gradient_files, Path):
            # gradient_files is the parent folder of safetensor files
            gradient_files = list(gradient_files.glob("*.safetensors"))

        set_sample_ids = set(sample_ids)
        all_gradient_dict = {}
        for gradient_file in gradient_files:
            assert gradient_file.suffix == ".safetensors", \
                "`gradient_files` should only include `.safetensor` files."
            loaded_gradients = load_file(gradient_file, device=device)
            all_gradient_dict.update({s_id: g for s_id, g in loaded_gradients.items() if s_id in set_sample_ids})

        # arrange gradients in the same order as `sample_ids`
        sample_gradients = [all_gradient_dict[sample_id] for sample_id in sample_ids]

        # convert `sample_gradients` into a single tensor
        sample_gradients = torch.stack(sample_gradients, dim=0)

        return sample_ids, sample_gradients

    @staticmethod
    def cluster_kmeans(data: torch.Tensor, k: int, num_iter: int, use_tqdm: bool = False) -> Tuple:
        """
        K-Means algorithm optimized for VRAM usage.
        Input:
            data: torch.Tensor, sized (n, grad_dim). data must be in cuda, and normalized.
            k: number of clusters
            num_iter: number of iterations for Lloyd's algorithm
        Output:
            labels: torch.Tensor, sized (n,). Cluster assignment for each point in `data`.
            centroids: torch.Tensor, sized (k, grad_dim). Centroids for the k clusters.
        """
        assert data.is_cuda, "`data` should be in CUDA device and normalized."

        data = F.normalize(data, dim=1)

        # -- initialize centroids and centroid labels -- #
        centroids = data[:k, :].clone()  # (k, grad_dim)
        labels = None  # (n,)

        # -- loop in Lloyd's algorithm -- #
        for _ in tqdm(range(num_iter), disable=not use_tqdm, desc="Cluster-Kmeans"):
            # -- E step: assign points to the closest cluster according to cosine similarity -- #
            labels = GradientVendi._calculate_sim_matrix_and_label(data, centroids)

            # -- M step: update the centroids to the normalized cluster average -- #
            # compute the sum of samples per cluster - we don't use scatter_add_ to save memory :)
            centroids.zero_()
            for cluster_idx in range(k):
                centroids[cluster_idx] = torch.sum(data[labels == cluster_idx], dim=0)

            # normalize centroids
            centroids = F.normalize(centroids, dim=1)

        return labels, centroids

    @staticmethod
    def _calculate_sim_matrix_and_label(data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient subroutine for cluster_kmeans.
        Given data of size (n, embed_dim) and centroids (k, embed_size),
        (1) compute the similarity matrix sized (n, embed_dim),
        (2) compute the labels for each sample in data sized (n,)
        and return the labels (n,)
        """
        max_batch_num_centroids = 90

        max_values_list, max_indices_list = [], []
        for batch_start_idx in range(0, centroids.size(0), max_batch_num_centroids):
            batch_similarity_matrix = data @ centroids[
                                             batch_start_idx:batch_start_idx + max_batch_num_centroids].T  # (n, batch_num_centroids)
            max_values, max_indices = batch_similarity_matrix.max(dim=1)
            max_indices += batch_start_idx  # index should represent actual centroid index across batches
            max_values_list.append(max_values)
            max_indices_list.append(max_indices)

            del batch_similarity_matrix

        max_values_in_each_batch = torch.stack(max_values_list, dim=1)  # (n, num_batches)
        max_indices_in_each_batch = torch.stack(max_indices_list, dim=1)  # (n, num_batches)
        indices_of_max_values_across_batches = max_values_in_each_batch.argmax(
            dim=-1)  # (n,), where i-th element is the index of the batch whose max value was the largest across all batches
        labels = max_indices_in_each_batch[
            torch.arange(max_indices_in_each_batch.size(0)), indices_of_max_values_across_batches]

        return labels

    @staticmethod
    def compute_gradient_vendi(gradients: torch.Tensor):
        """
        Compute Vendi-score using gradients
        """
        # cluster gradients with large k
        # we empirically found that this mitigates the effect of noise from outliers while being more efficient
        k = int(gradients.size(0) / 10)
        _, centroids = GradientVendi.cluster_kmeans(gradients, k, 20, use_tqdm=gradients.size(0) > 1e5)

        # compute reverse-sim matrix
        reverse_sim_matrix = Vendi.compute_reverse_similarity_matrix(F.normalize(gradients, dim=1))

        # compute vendi score
        score = Vendi.compute_vendi_score(reverse_sim_matrix, n=gradients.shape[0])

        return score
