# datasets/clean_gallery_dataset.py

import os
import os.path as osp
from typing import List, Sequence

from tqdm import tqdm

from .base_dataset import BaseDataset


class CleanGalleryDataset(BaseDataset):
    """
    Dataset where *gallery* sequences at test time are loaded from a clean
    gallery root, while all other sequences use the original dataset_root.

    Intended for:
    - Evaluating robustness when gallery is clean and probe may be noisy.
    """

    def _build_seqs_info_for_labels(
        self,
        label_set: Sequence[str],
        dataset_root: str,
        data_in_use: Sequence[bool],
        data_config: dict,
        training: bool,
    ) -> List[Sequence]:
        dataset_name = data_config.get("dataset_name", "")
        # Default gallery types (you can extend based on your paper)
        if dataset_name == "CASIA-B":
            gallery_typs = ["nm-01", "nm-02", "nm-03", "nm-04"]
        elif dataset_name == "SUSTech1K":
            gallery_typs = ["00-nm"]
        else:
            gallery_typs = []

        # Configure clean gallery root:
        # 1. Prefer explicit config key.
        # 2. Fall back to a path relative to this repo.
        clean_gallery_root = data_config.get("clean_gallery_root")
        if clean_gallery_root is None:
            # e.g. datasets/<dataset_name>/clean_gallery/
            datasets_dir = osp.dirname(__file__)
            clean_gallery_root = osp.join(
                datasets_dir, dataset_name, "clean_gallery"
            )

        seqs_info_list: List[Sequence] = []

        for lab in tqdm(label_set, desc="CleanGalleryDataset: build seqs_info"):
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]

                    # Test-time gallery sequences come from clean_gallery_root
                    if (not training) and (typ in gallery_typs):
                        seq_path = osp.join(clean_gallery_root, *seq_info)
                    else:
                        seq_path = osp.join(dataset_root, *seq_info)

                    if not osp.isdir(seq_path):
                        self.msg_mgr.log_debug(
                            f"[CleanGallery] Missing dir: {seq_path}"
                        )
                        continue

                    seq_dirs = sorted(os.listdir(seq_path))
                    if not seq_dirs:
                        self.msg_mgr.log_debug(
                            "Find no .pkl file in %s-%s-%s."
                            % (lab, typ, vie)
                        )
                        continue

                    seq_dirs = [osp.join(seq_path, d) for d in seq_dirs]
                    if data_in_use is not None:
                        seq_dirs = [
                            d for d, use_bl in zip(seq_dirs, data_in_use) if use_bl
                        ]

                    # Use BaseDataset helper to filter malformed / empty PKLs
                    seq_dirs_new = []
                    for item in seq_dirs:
                        item_pkl = self._load_pkl(item)
                        if (
                            data_config.get("dataset_name") in ["CCPG", "SUSTech1K"]
                            and getattr(item_pkl, "ndim", 0) < 3
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
