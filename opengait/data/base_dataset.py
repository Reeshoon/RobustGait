# datasets/base_dataset.py

import os
import pickle
import os.path as osp
import json
from typing import List, Sequence, Tuple

import numpy as np
import torch.utils.data as tordata

from utils import get_msg_mgr


class BaseDataset(tordata.Dataset):
    """
    Base class for all *robust* dataset variants used in RobustGait.

    It handles:
    - Loading a partition file (TRAIN_SET / TEST_SET)
    - Building `self.seqs_info` as [label, type, view, paths]
    - Optional caching of loaded sequences
    - A robust PKL loader that skips empty / malformed files

    Subclasses should override:
        - `_build_seqs_info_for_labels(label_set, data_config, training)`
    to implement their own path logic (clean gallery, perturbed gallery, etc.).
    """

    def __init__(self, data_cfg: dict, training: bool):
        super().__init__()
        self.data_cfg = data_cfg
        self.training = training

        self.cache = bool(data_cfg.get("cache", False))
        self.msg_mgr = get_msg_mgr()

        # Let subclass define self.seqs_info
        self.seqs_info: List[Sequence] = self._build_seqs_info(data_cfg, training)

        # Metadata: labels, types, views
        self.label_list = [s[0] for s in self.seqs_info]
        self.types_list = [s[1] for s in self.seqs_info]
        self.views_list = [s[2] for s in self.seqs_info]

        self.label_set = sorted(set(self.label_list))
        self.types_set = sorted(set(self.types_list))
        self.views_set = sorted(set(self.views_list))

        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)

        if self.cache:
            self._load_all_data()

    # --------------------------------------------------------------------- #
    # Public dataset API
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.seqs_info)

    def __getitem__(self, idx: int):
        if not self.cache:
            data_list = self._loader(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self._loader(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]

        # Skip sequences that failed to load
        if data_list is None or any(len(d) == 0 for d in data_list):
            return None

        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    # --------------------------------------------------------------------- #
    # Hooks for subclasses
    # --------------------------------------------------------------------- #

    def _build_seqs_info(self, data_config: dict, training: bool) -> List[Sequence]:
        """
        Template method:
        1. Reads partition file.
        2. Chooses TRAIN_SET or TEST_SET.
        3. Delegates to `_build_seqs_info_for_labels`.

        Subclasses typically only override `_build_seqs_info_for_labels`.
        """
        dataset_root = data_config["dataset_root"]
        try:
            data_in_use = data_config["data_in_use"]
        except KeyError:
            data_in_use = None

        with open(data_config["dataset_partition"], "rb") as f:
            partition = json.load(f)

        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)

        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [
            label for label in label_list
            if label not in (train_set + test_set)
        ]

        if len(miss_pids) > 0:
            self.msg_mgr.log_debug("-------- Miss Pid List --------")
            self.msg_mgr.log_debug(miss_pids)

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                self.msg_mgr.log_info("[{}, {}, ..., {}]".format(
                    pid_list[0], pid_list[1], pid_list[-1]
                ))
            else:
                self.msg_mgr.log_info(pid_list)

        if training:
            self.msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
            label_set = train_set
        else:
            self.msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)
            label_set = test_set

        return self._build_seqs_info_for_labels(
            label_set=label_set,
            dataset_root=dataset_root,
            data_in_use=data_in_use,
            data_config=data_config,
            training=training,
        )

    def _build_seqs_info_for_labels(
        self,
        label_set: Sequence[str],
        dataset_root: str,
        data_in_use: Sequence[bool],
        data_config: dict,
        training: bool,
    ) -> List[Sequence]:
        """
        Default: original OpenGait behaviour with additional robustness:

        - builds [label, type, view, [paths...]] from dataset_root
        - filters empty / malformed PKL files for CCPG / SUSTech1K

        Subclasses may override this to implement custom path logic.
        """
        skipped_files = []
        seqs_info_list: List[Sequence] = []

        for lab in label_set:
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]
                    seq_path = osp.join(dataset_root, *seq_info)
                    seq_dirs = sorted(os.listdir(seq_path))

                    if not seq_dirs:
                        self.msg_mgr.log_debug(
                            "Find no .pkl file in %s-%s-%s." % (lab, typ, vie)
                        )
                        continue

                    seq_dirs = [osp.join(seq_path, d) for d in seq_dirs]
                    if data_in_use is not None:
                        seq_dirs = [
                            d for d, use_bl in zip(seq_dirs, data_in_use) if use_bl
                        ]

                    seq_dirs_new = []
                    for item in seq_dirs:
                        if os.path.getsize(item) == 0:
                            print(f"Error: File is empty - {item}")
                            continue
                        item_pkl = self._load_pkl(item)
                        if len(item_pkl) == 0:
                            skipped_files.append([*seq_info, seq_dirs, "empty"])
                            continue
                        if (
                            data_config.get("dataset_name") in ["CCPG", "SUSTech1K"]
                            and getattr(item_pkl, "ndim", getattr(item_pkl, "ndim", 0)) not in (3, )
                        ):
                            skipped_files.append(
                                [*seq_info, seq_dirs, getattr(item_pkl, "ndim", "na")]
                            )
                            continue
                        seq_dirs_new.append(item)

                    if seq_dirs_new:
                        seqs_info_list.append([*seq_info, seq_dirs_new])
                    else:
                        self.msg_mgr.log_debug(
                            ".pkl files empty in %s-%s-%s." % (lab, typ, vie)
                        )

        if skipped_files:
            print(f"Skipped a total of {len(skipped_files)} files")

        return seqs_info_list

    # --------------------------------------------------------------------- #
    # Loader helpers
    # --------------------------------------------------------------------- #

    def _loader(self, paths: Sequence[str]):
        """
        Robust PKL loader that:
        - supports list of PKL paths
        - skips empty and malformed files
        """
        paths = sorted(paths)
        data_list = []

        for pth in paths:
            if not pth.endswith(".pkl"):
                raise ValueError("BaseDataset loader currently expects only .pkl files.")
            try:
                if os.path.getsize(pth) == 0:
                    print(f"⚠️ Skipping empty file: {pth}")
                    continue
                with open(pth, "rb") as f:
                    data = pickle.load(f)
                if len(data) == 0:
                    print(f"⚠️ Skipping empty file: {pth}")
                    continue
                # If this is RGB-like (N,H,W,3), flatten to (N,H,W) to be consistent
                if isinstance(data, np.ndarray) and data.ndim == 4 and data.shape[-1] == 3:
                    data = data[..., 0]
                data_list.append(data)
            except Exception as e:
                print(f"❌ Unexpected error loading {pth}: {e}")
                continue

        if not data_list:
            return None

        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    "Each input data({}) should have the same length.".format(paths[idx])
                )
            if len(data) == 0:
                raise ValueError(
                    "Each input data({}) should have at least one element.".format(paths[idx])
                )
        return data_list

    def _load_pkl(self, pkl_file: str):
        with open(pkl_file, "rb") as f:
            return pickle.load(f)

    def _load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)
