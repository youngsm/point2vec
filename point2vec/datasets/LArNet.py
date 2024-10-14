import os
import torch
import numpy as np
import atexit
import h5py
from glob import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from torch.nn.utils.rnn import pad_sequence


class LArNet(Dataset):
    def __init__(
        self,
        data_path: str,
        emin: float = 1.0e-6,
        emax: float = 20.0,
        energy_threshold: float = 0.13,
        normalize: bool = True,
        remove_low_energy_scatters: bool = False,
    ):
        self.data_path = data_path
        self.h5_files = glob(data_path)
        self.emin = emin
        self.emax = emax
        self.energy_threshold = energy_threshold
        self.normalize = normalize
        self.remove_low_energy_scatters = remove_low_energy_scatters

        print(f"[DATASET] {self.emin=}, {self.emax=}, {self.energy_threshold=}, {self.normalize=}, {self.remove_low_energy_scatters=}")

        self.lengths = []

        print(f"[DATASET] Building index")
        self._build_index()
        print(f"[DATASET] {len(self.h5_files)} files were loaded")
        self.h5data = []

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _build_index(self):
        self.cumulative_lengths = []
        indices = []
        for h5_file in self.h5_files:
            index = np.load(h5_file.replace(".h5", "_gt2048.npy"))
            self.cumulative_lengths.append(index.shape[0])
            indices.append(index)
        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        self.indices = indices
        print(f"[DATASET] {self.cumulative_lengths[-1]} point clouds were loaded")

    def h5py_worker_init(self):
        print(f"[DATASET] Initializing h5py workers")
        self.h5data = []
        for h5_file in self.h5_files:
            self.h5data.append(h5py.File(h5_file, mode="r", libver="latest", swmr=True))
        atexit.register(self.cleanup)

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        # centroid = np.mean(pc, axis=0)
        centroid = np.array([760.0, 760.0, 760.0]) / 2  # Center of the box
        pc[:, :3] = pc[:, :3] - centroid
        # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))        # max distance between points
        m = (
            760.0 * np.sqrt(3) / 2
        )  # Diagonal size is 760*sqrt(3), so max distance is sqrt(3)*760/2
        pc[:, :3] = pc[:, :3] / m
        return pc

    def transform_energy(self, pc):
        """tranforms energy to logarithmic scale on [-1,1]"""
        energy_mask = None
        if self.energy_threshold > 0.0:
            energy_mask = pc[:, 3] > self.energy_threshold
            self.emin = self.energy_threshold
        # pc[:, 3] = log_transform(pc[:, 3], self.emax, self.emin)
        return pc, energy_mask

    def __getitem__(self, idx):
        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        h5_file = self.h5data[h5_idx]
        idx = idx - self.cumulative_lengths[h5_idx]
        idx = self.indices[h5_idx][idx]
        data = h5_file["point"][idx].reshape(-1, 8)[:, :4]
        cluster_size, semantic_id = h5_file["cluster"][idx].reshape(-1, 5)[:, [0, -1]].T

        # remove first particle from data, i.e. low energy scatters
        if self.remove_low_energy_scatters:
            data = data[cluster_size[0] :]
            semantic_id, cluster_size = semantic_id[1:], cluster_size[1:]
        data_semantic_id = np.repeat(semantic_id, cluster_size)

        # Normalize
        if self.normalize:
            data = self.pc_norm(data)
        data, energy_mask = self.transform_energy(data)

        if energy_mask is not None:
            data = data[energy_mask]
            data_semantic_id = data_semantic_id[energy_mask]

        data = torch.from_numpy(data).float()
        data_semantic_id = torch.from_numpy(data_semantic_id).unsqueeze(1).long()
        return data, data_semantic_id

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        for h5_file in self.h5data:
            h5_file.close()

    @staticmethod
    def init_worker_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.h5py_worker_init()

    @staticmethod
    def collate_fn(batch):
        data = [item[0] for item in batch]
        semantic_id = [item[1] for item in batch]
        lengths = torch.tensor(
            [points.size(0) for points in data], dtype=torch.long
        )  # Shape: (B,)
        padded_points = pad_sequence(data, batch_first=True)  # Shape: (B, N_max, 4)
        padded_semantic_id = pad_sequence(
            semantic_id, batch_first=True, padding_value=-1
        )  # Shape: (B, N_max)

        return (
            padded_points,
            lengths,
            padded_semantic_id,
        )


class LArNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "/sdf/home/y/youngsam/data/dune/larnet/h5/DataAccessExamples/train/generic_v2*.h5",
        batch_size: int = 512,
        num_workers: int = 8,
        dataset_kwargs: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.persistent_workers = True if num_workers > 0 else False

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LArNet(self.hparams.data_path, **self.hparams.dataset_kwargs)
        test_dir = self.hparams.data_path.replace("train", "val")
        self.test_dataset = LArNet(test_dir, **self.hparams.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=LArNet.collate_fn,
            worker_init_fn=LArNet.init_worker_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=LArNet.collate_fn,
            worker_init_fn=LArNet.init_worker_fn,
        )


def log_transform(x, xmax=1, eps=1e-7):
    y0 = np.log10(eps)
    y1 = np.log10(eps + xmax)
    return 2 * (np.log10(x + eps) - y0) / (y1 - y0) - 1


def inv_log_transform(x, xmax=1, eps=1e-7):
    y0 = np.log10(eps)
    y1 = np.log10(xmax + eps)
    x = (x + 1) / 2
    return 10 ** (x * (y1 - y0) + y0) - eps


def pad_with_first_point(batch):
    # Step 1: Compute the maximum length
    max_length = max(tensor.size(0) for tensor in batch)

    padded_tensors = []
    for tensor in batch:
        current_length = tensor.size(0)
        pad_size = max_length - current_length

        if pad_size > 0:
            # Step 2: Create padding tensor by repeating the first point
            pad_tensor = tensor[0].unsqueeze(0).expand(pad_size, -1)
            # Step 3: Concatenate the original tensor with the padding tensor
            padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        else:
            padded_tensor = tensor

        padded_tensors.append(padded_tensor)

    # Step 4: Stack all padded tensors
    return torch.stack(padded_tensors, dim=0)
