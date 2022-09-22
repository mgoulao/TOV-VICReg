import os
import gzip
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tov_vicreg.utils.pytorch_utils import *


class DQNReplayDataset(Dataset):
    """
    A dataset of observations from a one checkpoint of one game.
    It saves a tensor of dimension: (dataset_size, h, w)
    and given an index i returns a slice starting at i and
    ending in i plus a number of frames: (slice_size, h, w).
    The slice size should be equivalent to the number of frames stacked
    during the RL phase.
    In add adjacent mode the dataset returns three stacked observations
    the observation before i the observation i and the observation after i.
    (3, slice_size, h, w)
    """

    def __init__(
        self,
        data_path: Path,
        game: str,
        checkpoint: int,
        frames: int,
        max_size: int,
        transform: object,
        add_adjacent=False,
        adjacent_transform=None,
        actions=False,
        start_index=0,
    ) -> None:
        self.add_adjacent = add_adjacent
        self.actions = None
        data = torch.tensor([])
        self.start_index = start_index

        filename = Path(data_path / f"{game}/observation_{checkpoint}.gz")
        print(f"Loading {filename}")

        zipFile = gzip.GzipFile(filename=filename)
        loaded_data = np.load(zipFile)
        loaded_data_capped = np.copy(
            loaded_data[self.start_index : self.start_index + max_size]
        )

        print(f"Using {loaded_data.size * loaded_data.itemsize} bytes")
        print(f"Shape {loaded_data.shape}")

        data = torch.from_numpy(loaded_data_capped)
        setattr(self, "observation", data)

        del loaded_data
        del zipFile
        del loaded_data_capped

        if actions:
            actions_filename = Path(data_path / f"{game}/action_{checkpoint}.gz")
            actions_zipFile = gzip.GzipFile(filename=actions_filename)
            actions_loaded_data = np.load(actions_zipFile)
            actions_data_capped = np.copy(
                actions_loaded_data[self.start_index : self.start_index + max_size]
            )
            data = torch.from_numpy(actions_data_capped)
            setattr(self, "actions", data)

        self.size = min(data.shape[0], max_size)
        self.game = game
        self.frames = frames
        self.effective_size = self.size - self.frames + 1
        self.transform = transform
        self.adjacent_transform = adjacent_transform

    def __len__(self):
        return self.effective_size

    def __getitem__(self, index: int) -> torch.Tensor:
        time_ind = index % self.effective_size
        if self.frames <= 1:
            obs = self.observation[time_ind]
        else:
            sl = slice(time_ind, time_ind + self.frames)
            obs = self.observation[sl]
        res_action = self.actions[time_ind] if self.actions is not None else 0
        res_obs = None
        if self.add_adjacent:
            before_index = max(0, index - self.frames)
            after_index = min(self.effective_size - 1, index + self.frames)
            if self.frames <= 1:
                before_obs = self.observation[before_index]
                after_obs = self.observation[after_index]
            else:
                before_slice = slice(before_index, before_index + self.frames)
                after_slice = slice(after_index, after_index + self.frames)
                before_obs = self.observation[before_slice]
                # now_obs = self.observation[sl]
                after_obs = self.observation[after_slice]
            if self.transform is not None:
                transformed_obs = self.transform(obs)
                if not isinstance(transformed_obs, (list, tuple)):
                    transformed_obs = [transformed_obs]
                if self.adjacent_transform is not None:
                    before_obs = self.adjacent_transform(before_obs)
                    # now_obs = self.adjacent_transform(now_obs)
                    after_obs = self.adjacent_transform(after_obs)
                transformed_obs.extend([before_obs, after_obs])
                res_obs = transformed_obs
            else:
                res_obs = [obs, before_obs, after_obs]
        else:
            res_obs = self.transform(obs) if self.transform is not None else obs
        return res_obs, res_action


class MultiDQNReplayDataset(Dataset):
    """
    This dataset corresponds to the concatenation of several DQNReplayDataset.
    Meaning that it contains several checkpoints from several games.
    """

    def __init__(
        self,
        data_path: Path,
        games: List[str],
        checkpoints: List[int],
        frames: int,
        max_size: int,
        transform: object,
        add_adjacent=False,
        adjacent_transform=None,
        actions=False,
        start_index=0,
    ) -> None:
        self.actions = actions
        self.n_checkpoints_per_game = len(checkpoints)
        self.add_adjacent = add_adjacent
        self.datasets = [
            DQNReplayDataset(
                data_path,
                game,
                ckpt,
                frames,
                max_size,
                transform,
                add_adjacent,
                adjacent_transform,
                actions,
                start_index,
            )
            for ckpt in checkpoints
            for game in games
        ]

        self.n_datasets = len(self.datasets)
        self.single_dataset_size = len(self.datasets[0])

    def get_seq_samples(self, seq_len, n_games):
        start_index = 100
        res = []
        for i in range(n_games):
            dataset_index = (
                i * self.n_checkpoints_per_game + 1
                if self.n_checkpoints_per_game > 1
                else i * self.n_checkpoints_per_game
            )
            for j in range(start_index, start_index + seq_len):
                datapoint, _ = self.datasets[dataset_index][j]
                if isinstance(
                    datapoint, (list, tuple)
                ):  # add_adjacent and transform might return lists
                    datapoint = datapoint[0]
                res.append(datapoint)
        return torch.stack(res)

    def __len__(self) -> int:
        return self.n_datasets * self.single_dataset_size

    def __getitem__(self, index: int) -> torch.Tensor:
        multidataset_index = index % self.n_datasets
        dataset_index = index // self.n_datasets
        res_obs, res_action = self.datasets[multidataset_index][dataset_index]
        return [res_obs, res_action]


def _get_DQN_Replay_loader(
    data_path: Path,
    games: List[str],
    checkpoints: List[int],
    frames: int,
    max_size_per_single_dataset: int,
    batch_size: int,
    num_workers: int,
    transform,
) -> DataLoader:
    dataset = MultiDQNReplayDataset(
        data_path,
        games,
        checkpoints,
        frames,
        max_size_per_single_dataset,
        transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    return dataloader


def get_DQN_Replay_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    sampler=None,
) -> DataLoader:

    if sampler == None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=True,
            num_workers=num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            sampler=sampler,
            num_workers=num_workers,
        )

    return dataloader


def get_DQN_Replay_dataset(
    data_path=Path("/media/msgstorage/dqn"),
    games=["Alien"],
    checkpoints=["1"],
    frames=3,
    max_size_per_single_dataset=1000,
    transform=None,
) -> MultiDQNReplayDataset:

    return MultiDQNReplayDataset(
        data_path,
        games,
        checkpoints,
        frames,
        max_size_per_single_dataset,
        transform,
    )
