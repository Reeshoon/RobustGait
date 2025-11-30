# datasets/augmented_dataset.py

import os
import os.path as osp
import random
from typing import List, Sequence

from .base_dataset import BaseDataset


class AugmentedDataset(BaseDataset):
    """
    Dataset that randomly samples sequences from either:
    - the original dataset_root
    - an augmented (perturbed) dataset root

    Controlled by:
        data_cfg["aug_ratio"]  in [0, 1]
            probability of sampling from the *clean* (original) root.
            (1 - aug_ratio) is the probability of sampling from the augmented root.

        data_cfg["augmented_root_map"][dataset_name] (optional)
            to specify where the augmented PKLs live.
            If missing, a repo-relative default is used.
    """

    def __init__(self, data_cfg: dict, training: bool):
        self.aug_ratio = float(data_cfg.get("aug_ratio", 0.5))
        super().__init__(data_cfg, training)

    def _build_seqs_info_for_labels(
        self,
        label_set: Sequence[str],
        dataset_root: str,
        data_in_use: Sequence[bool],
        data_config: dict,
        training: bool,
    ) -> List[Sequence]:
        dataset_name = data_config.get("dataset_name", "")

        # Augmented dataset root configuration
        datasets_dir = osp.dirname(__file__)
        root_map = data_config.get("augmented_root_map", {})

        if dataset_name in root_map:
            augmented_root = root_map[dataset_name]
            if not osp.isabs(augmented_root):
                augmented_root = osp.join(datasets_dir, augmented_root)
        else:
            # Generic default: datasets/<dataset_name>/augmented/
            augmented_root = osp.join(datasets_dir, dataset_name, "augmented")

        clean_ratio = self.aug_ratio
        noisy_ratio = 1.0 - clean_ratio

        seqs_info_list: List[Sequence] = []

        for lab in label_set:
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]

                    # Decide whether to sample from original or augmented root
                    dataset_choice = random.choices(
                        ["original", "augmented"],
                        weights=[clean_ratio, noisy_ratio],
                        k=1,
                    )[0]

                    if dataset_choice == "augmented":
                        cand_path = osp.join(augmented_root, *seq_info)
                        seq_path = cand_path if osp.exists(cand_path) else osp.join(
                            dataset_root, *seq_info
                        )
                    else:
                        seq_path = osp.join(dataset_root, *seq_info)

                    if not osp.isdir(seq_path):
                        self.msg_mgr.log_debug(
                            f"[AugmentedDataset] Missing dir: {seq_path}"
                        )
                        continue

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

                    # Additional robustness: filter CCPG / SUSTech1K shape if needed
                    seq_dirs_new = []
                    for item in seq_dirs:
                        if os.path.getsize(item) == 0:
                            print(f"Error: File is empty - {item}")
                            continue
                        item_pkl = self._load_pkl(item)
                        if (
                            data_config.get("dataset_name") in ["CCPG", "SUSTech1K"]
                            and getattr(item_pkl, "ndim", 0) != 3
                        ):
                            continue
                        seq_dirs_new.append(item)

                    if seq_dirs_new:
                        seqs_info_list.append([*seq_info, seq_dirs_new])
                    else:
                        self.msg_mgr.log_debug(
                            ".pkl files empty in %s-%s-%s." % (lab, typ, vie)
                        )

        return seqs_info_list
