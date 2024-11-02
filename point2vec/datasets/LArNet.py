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
        endpoints: bool = False,
        maxlen: int = -1,
    ):
        self.data_path = data_path
        self.h5_files = glob(data_path)
        self.emin = emin
        self.emax = emax
        self.energy_threshold = energy_threshold
        self.normalize = normalize
        self.remove_low_energy_scatters = remove_low_energy_scatters
        self.endpoints = endpoints
        self.maxlen = maxlen
        self.initted = False

        print(f"[DATASET] {self.emin=}, {self.emax=}, {self.energy_threshold=}, {self.normalize=}, {self.remove_low_energy_scatters=}")

        self.lengths = []

        self._build_index()
        self.h5data = []

    def __len__(self):
        if self.maxlen > 0:
            return min(self.maxlen, self.cumulative_lengths[-1])
        return self.cumulative_lengths[-1]

    def _build_index(self):
        print("[DATASET] Building index")
        self.cumulative_lengths = []
        indices = []
        for h5_file in self.h5_files:
            index = np.load(h5_file.replace(".h5", "_gt2048.npy"))
            self.cumulative_lengths.append(index.shape[0])
            indices.append(index)
        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        self.indices = indices
        print(f"[DATASET] {self.cumulative_lengths[-1]} point clouds were loaded")
        print(f"[DATASET] {len(self.h5_files)} files were loaded")

    def h5py_worker_init(self):
        print(f"[DATASET] Initializing h5py workers")
        self.h5data = []
        for h5_file in self.h5_files:
            self.h5data.append(h5py.File(h5_file, mode="r", libver="latest", swmr=True))
        atexit.register(self.cleanup)
        self.initted = True

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
        threshold_mask = None
        if self.energy_threshold > 0.0:
            threshold_mask = pc[:, 3] > self.energy_threshold
            self.emin = self.energy_threshold
        pc[:, 3] = log_transform(pc[:, 3], self.emax, self.emin)
        return pc, threshold_mask
    
    def compute_endpoints(self, pc, cluster_id, semantic_id):
        """compute endpoints for each cluster
        
        these act as labels for line segments corresponding to the start and end points of each particle.
        if you want to train a model to predict these endpoints, you can use these as labels.

        start and end points are unordered, with the expectation that the loss will be a chamfer distance,
        which is symmetric to ordering.

        pc: Nx4, where each point is (x,y,z,e)
        cluster_id: N, where each point is the cluster id of the particle it belongs to. cluster ids should be in order,
                    i.e. [0,0,0,1,1,2,3,3,3,...]
        semantic_id: N, where each point is the semantic id of the particle it belongs to.
                    i.e. [0,0,0,1,1,2,3,3,3,...]. each cluster has the same semantic id.

        returns: Nx6, where each point is (x1,y1,z1,x2,y2,z2)
        """
        cluster_size = np.bincount(cluster_id) # (C,)
        endpoints = np.zeros((pc.shape[0], 6)) # (N, 6)
        i = 0
        for c in cluster_size:
            if c > 1:
                p1p2 = compute_endpoints(pc[i : i + c], semantic_id[i]) # (6,)
                endpoints[i : i + c] = np.tile(p1p2, (c, 1)) # (N_C, 6)
            i += c
        return endpoints

    def __getitem__(self, idx):
        if not self.initted:
            self.h5py_worker_init()

        h5_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")

        h5_file = self.h5data[h5_idx]
        idx = idx - self.cumulative_lengths[h5_idx]
        idx = self.indices[h5_idx][idx]
        data = h5_file["point"][idx].reshape(-1, 8)[:, [0,1,2,3,5]] # (x,y,z,e,t)
        cluster_size, semantic_id = h5_file["cluster"][idx].reshape(-1, 5)[:, [0, -1]].T

        # remove first particle from data, i.e. low energy scatters
        if self.remove_low_energy_scatters:
            data = data[cluster_size[0] :]
            semantic_id, cluster_size = semantic_id[1:], cluster_size[1:]

        data_semantic_id = np.repeat(semantic_id, cluster_size)
        cluster_id = np.repeat(np.arange(len(cluster_size)), cluster_size)

        # Normalize
        if self.normalize:
            data = self.pc_norm(data)
        data, threshold_mask = self.transform_energy(data)

        if threshold_mask is not None:
            data = data[threshold_mask]
            data_semantic_id = data_semantic_id[threshold_mask]
            cluster_id = cluster_id[threshold_mask]

        # Compute endpoints if needed (for line detection)
        endpoints = None
        if self.endpoints:
            endpoints = self.compute_endpoints(data, cluster_id, data_semantic_id)
            endpoints = torch.from_numpy(endpoints).float()

        data = torch.from_numpy(data[:,:4]).float()
        data_semantic_id = torch.from_numpy(data_semantic_id).unsqueeze(1).long()
        return data, data_semantic_id, endpoints # (N, 4), (N, 1), (N, 6)

    def __del__(self):
        if self.initted:
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
        endpoints = [item[2] for item in batch]
        lengths = torch.tensor(
            [points.size(0) for points in data], dtype=torch.long
        )  # Shape: (B,)
        padded_points = pad_sequence(data, batch_first=True)  # Shape: (B, N_max, 4)
        padded_semantic_id = pad_sequence(
            semantic_id, batch_first=True, padding_value=-1
        )  # Shape: (B, N_max)

        if endpoints[0] is not None:
            padded_endpoints = pad_sequence(endpoints, batch_first=True) # Shape: (B, N_max, 6)
        else:
            padded_endpoints = None

        return (
            padded_points,
            lengths,
            padded_semantic_id,
            padded_endpoints,
        )


class LArNetDataModule(pl.LightningDataModule):
    _class_weights = None
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

        # from datasets/ShapeNetPart.py
        self._category_to_seg_classes = {
            "shower": [0],
            "track": [1],
            "michel": [2],
            "delta": [3],
            "low energy deposit": [4],
        }
        # inverse mapping

        self._seg_class_to_category = {}
        for cat in self._category_to_seg_classes.keys():
            for cls in self._category_to_seg_classes[cat]:
                self._seg_class_to_category[cls] = cat

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LArNet(self.hparams.data_path, **self.hparams.dataset_kwargs)
        test_dir = self.hparams.data_path.replace("train", "val")
        self.test_dataset = LArNet(test_dir, **self.hparams.dataset_kwargs)

        if self.train_dataset.remove_low_energy_scatters:
            self._category_to_seg_classes.pop("low energy deposit")
            self._seg_class_to_category.pop(4)

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

    @property
    def category_to_seg_classes(self):
        return self._category_to_seg_classes

    @property
    def seg_class_to_category(self):
        return self._seg_class_to_category

    @property
    def num_seg_classes(self):
        return len(self._category_to_seg_classes)

    @property
    def class_weights(self):
        """
        inverse class weights, computed on validation set (~300 events),
        in the same order as category_to_seg_classes
        """
        class_counts = torch.tensor([13151438.0, 13294331.0, 204836.0, 598025.0])
        return class_counts.sum() / class_counts

def log_transform(x, xmax=1, eps=1e-7):
    # [eps, xmax] -> [-1,1]
    y0 = np.log10(eps)
    y1 = np.log10(eps + xmax)
    return 2 * (np.log10(x + eps) - y0) / (y1 - y0) - 1


def inv_log_transform(x, xmax=1, eps=1e-7):
    # [-1,1] -> [eps, xmax]
    y0 = np.log10(eps)
    y1 = np.log10(xmax + eps)
    x = (x + 1) / 2
    return 10 ** (x * (y1 - y0) + y0) - eps

# def compute_endpoints(points):
#     # get centroid
#     c = np.mean(points, axis=0)
#     centered = points - c

#     # get primary direction via pca
#     cov = np.cov(centered, rowvar=False)
#     eval, evec = np.linalg.eig(cov)
#     dir = evec[:, eval.argmax()]

#     # project points onto primary direction to find endpoints
#     projections = np.dot(centered, dir)
#     p1, p2 = points[projections.argmin()], points[projections.argmax()]
#     return np.concatenate([p1, p2])

def compute_endpoints(points: np.ndarray, semantic_id: int):
    """
    compute endpoints for a cluster of points based on
    the time ordering of those points. we take first and last
    point in time as the endpoints.

    if the cluster is a shower, we take the first point twice,
    because there is no well-defined 'final point'.

    points: Nx5, where each point is (x,y,z,e,t)
    semantic_id: int, the semantic id of the cluster
    """

    points = points[points[:, -1].argsort()]
    first_point, last_point = points[0, :3], points[-1, :3]
    if semantic_id == 0: # shower has only 1 end point
        return np.concatenate([first_point, first_point])
    return np.concatenate([first_point, last_point])