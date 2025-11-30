# datasets/perturbed_gallery_dataset.py

import os
import os.path as osp
import json
from typing import List, Sequence

from tqdm import tqdm

from .base_dataset import BaseDataset


class PerturbedGalleryDataset(BaseDataset):
    """
    Dataset where *gallery* sequences at test time are loaded from
    perturbation-specific roots based on a noise assignment JSON.

    Expects:
    - A JSON file `noise_severity_assignments.json` with entries:
        "<lab>-<typ>-<vie>": ["<perturb>", "<severity>"]
    - A root for perturbed PKLs, which can be configured via:
        data_cfg["perturbed_gallery_root_map"][dataset_name]
      or falls back to a path relative to `datasets/`.
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

        # Gallery types per dataset
        if dataset_name == "CASIA-B":
            gallery_typs = ["nm-01", "nm-02", "nm-03", "nm-04"]
        elif dataset_name == "SUSTech1K":
            gallery_typs = ["00-nm"]
        else:
            gallery_typs = []

        # Load noise assignment JSON (relative by default)
        datasets_dir = osp.dirname(__file__)
        default_noise_json = osp.join(
            datasets_dir, dataset_name, "noise_severity_assignments.json"
        )
        noise_json_path = data_config.get(
            "noise_severity_json", default_noise_json
        )

        with open(noise_json_path, "r") as f:
            noise_sev_map = json.load(f)

        # Perturbed gallery base root config
        # You can pass an explicit map in data_cfg:
        #   "perturbed_gallery_root_map": {"CASIA-B": "relative/path", ...}
        root_map = data_config.get("perturbed_gallery_root_map", {})
        if dataset_name in root_map:
            base_perturbed_root = root_map[dataset_name]
            if not osp.isabs(base_perturbed_root):
                base_perturbed_root = osp.join(datasets_dir, base_perturbed_root)
        else:
            # Generic relative default:
            # datasets/<dataset_name>/perturb/<perturb>/<severity>/
            base_perturbed_root = osp.join(datasets_dir, dataset_name, "perturb")

        seqs_info_list: List[Sequence] = []

        for lab in tqdm(label_set, desc="PerturbedGalleryDataset: build seqs_info"):
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]

                    # Decide whether to use perturbed gallery for this (lab, typ, vie)
                    if (not training) and (typ in gallery_typs):
                        video_name = f"{lab}-{typ}-{vie}"
                        if video_name not in noise_sev_map:
                            # Fallback to clean if no assignment
                            seq_path = osp.join(dataset_root, *seq_info)
                        else:
                            perturb, sev = noise_sev_map[video_name]
                            seq_path = osp.join(
                                base_perturbed_root, perturb, sev, *seq_info
                            )
                    else:
                        seq_path = osp.join(dataset_root, *seq_info)

                    if not osp.isdir(seq_path):
                        self.msg_mgr.log_debug(
                            f"[PerturbedGallery] Missing dir: {seq_path}"
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

                    # Here we keep all files; if you want additional shape filtering,
                    # you can reuse logic similar to BaseDataset.
                    if seq_dirs:
                        seqs_info_list.append([*seq_info, seq_dirs])

        return seqs_info_list
