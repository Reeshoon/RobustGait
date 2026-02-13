"""Dataset parser helpers.

This module contains *only* dataset-parsing functions (no Dataset class).
Each function populates `self.seqs_info` on the passed-in Dataset instance.

A parser's job:
- Read the partition file (TRAIN_SET / TEST_SET)
- Walk the dataset directory structure
- Build `self.seqs_info` as a list of:
    [label, type, view, paths]
  where `paths` is typically a list of frame/clip files for that sequence.

Expected `data_config` keys vary per parser; see each function docstring.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import random
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from utils import get_msg_mgr

# Optional registry (useful if you want dynamic lookup elsewhere).
# Keeping it here is harmless even if your main file doesn't import/use it.
PARSER_REGISTRY = {
    "MultiModal": "multimodal_dataset_parser",
    "MEVID": "mevid_dataset_parser",
    "CleanGallery": "clean_gallery_dataset_parser",
    "PerturbedGallery": "perturbed_gallery_dataset_parser",
    "Augmented": "augmented_dataset_parser",
}

# ------------------------------------------------------------------
# Training dataset parsers
# ------------------------------------------------------------------

def multimodal_dataset_parser(self, data_config: Dict[str, Any], training: bool) -> None:
    """Parse a *multi-modal* dataset where silhouettes and RGB are stored separately.

    It expects:
    - data_config['dataset_root'] is a dict with:
        - 'sil_root': root folder for silhouettes (OpenGait-style: pid/type/view/files)
        - 'rgb_root': root folder for RGB data with the *same* pid/type/view structure
    - data_config['dataset_partition']: JSON containing TRAIN_SET / TEST_SET
    - Optional: data_config['data_in_use'] (mask list aligned with files in each sequence)

    Output:
    - self.seqs_info = [[pid, typ, view, paths], ...]
      where `paths` contains the silhouette files plus the RGB files found in the matching folder.
    """
    sil_dataset_root = data_config['dataset_root']["sil_root"]
    img_dataset_root = data_config['dataset_root']["rgb_root"]

    try:
        data_in_use = data_config['data_in_use']  # [n], true or false
    except Exception:
        data_in_use = None

    with open(data_config['dataset_partition'], "rb") as f:
        partition = json.load(f)

    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]

    label_list = os.listdir(sil_dataset_root)
    train_set = [label for label in train_set if label in label_list]
    test_set = [label for label in test_set if label in label_list]
    miss_pids = [label for label in label_list if label not in (train_set + test_set)]

    msg_mgr = get_msg_mgr()

    def log_pid_list(pid_list: List[str]) -> None:
        if len(pid_list) >= 3:
            msg_mgr.log_info('[%s, %s, ..., %s]' % (pid_list[0], pid_list[1], pid_list[-1]))
        else:
            msg_mgr.log_info(pid_list)

    if len(miss_pids) > 0:
        msg_mgr.log_debug('-------- Miss Pid List --------')
        msg_mgr.log_debug(miss_pids)

    if training:
        msg_mgr.log_info("-------- Train Pid List --------")
        log_pid_list(train_set)
    else:
        msg_mgr.log_info("-------- Test Pid List --------")
        log_pid_list(test_set)

    def get_seqs_info_list(label_set: List[str]) -> List[List[Any]]:
        seqs_info_list: List[List[Any]] = []
        for lab in label_set:
            for typ in sorted(os.listdir(osp.join(sil_dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(sil_dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]
                    seq_path = osp.join(sil_dataset_root, *seq_info)
                    seq_dirs = sorted(os.listdir(seq_path))
                    if seq_dirs != []:
                        seq_dirs = [osp.join(seq_path, d) for d in seq_dirs]
                        if data_in_use is not None:
                            seq_dirs = [d for d, use_bl in zip(seq_dirs, data_in_use) if use_bl]

                        # add RGB paths from the matching folder
                        img_path = osp.join(img_dataset_root, *seq_info)
                        img_path_dirs = sorted(os.listdir(img_path))
                        if img_path_dirs != []:
                            seq_dirs.extend(osp.join(img_path, d) for d in img_path_dirs)
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            print("Path not found")
                            msg_mgr.log_debug(f"Found no rgb files at {img_path}")
                    else:
                        msg_mgr.log_debug('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
        return seqs_info_list

    self.seqs_info = get_seqs_info_list(train_set) if training else get_seqs_info_list(test_set)


def mevid_dataset_parser(self, data_config: Dict[str, Any], training: bool) -> None:
    """Parse MEVID when train and test roots are stored separately.

    It expects:
    - data_config['dataset_root'] for training
    - data_config['dataset_root_test'] for testing
    - data_config['dataset_partition'] with TRAIN_SET / TEST_SET

    MEVID structure note:
    The leaf folder contains multiple tracklets per (pid, type, view).
    This parser appends tracklet id to the view string: view_<tracklet>.
    """
    # ----------- If MEVID train and test sets are stored seperately -----------
    if training:
        dataset_root = data_config['dataset_root']
    else:
        dataset_root = data_config['dataset_root_test']
    # ------------------------------------------------------------------------

    try:
        data_in_use = data_config['data_in_use']  # [n], true or false
    except Exception:
        data_in_use = None

    with open(data_config['dataset_partition'], "rb") as f:
        partition = json.load(f)

    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]
    label_list = os.listdir(dataset_root)
    train_set = [label for label in train_set if label in label_list]
    test_set = [label for label in test_set if label in label_list]
    miss_pids = [label for label in label_list if label not in (train_set + test_set)]
    msg_mgr = get_msg_mgr()

    def log_pid_list(pid_list: List[str]) -> None:
        if len(pid_list) >= 3:
            msg_mgr.log_info('[%s, %s, ..., %s]' % (pid_list[0], pid_list[1], pid_list[-1]))
        else:
            msg_mgr.log_info(pid_list)

    if len(miss_pids) > 0:
        msg_mgr.log_debug('-------- Miss Pid List --------')
        msg_mgr.log_debug(miss_pids)

    if training:
        msg_mgr.log_info("-------- Train Pid List --------")
        log_pid_list(train_set)
    else:
        msg_mgr.log_info("-------- Test Pid List --------")
        log_pid_list(test_set)

    # ---------------- updated to handle MEVID dataset structure ----------------
    def get_seqs_info_list(label_set: List[str]) -> List[List[Any]]:
        seqs_info_list: List[List[Any]] = []
        for lab in label_set:
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]
                    seq_path = osp.join(dataset_root, *seq_info)
                    seq_dirs = sorted(os.listdir(seq_path))  # tracklets
                    if seq_dirs != []:
                        for seq_dir in seq_dirs:
                            tracklet_num = seq_dir.split(".")[0]
                            s_info = [lab, typ, vie + "_" + str(tracklet_num)]
                            seq_dir = [osp.join(seq_path, seq_dir)]
                            seqs_info_list.append([*s_info, seq_dir])
                    else:
                        msg_mgr.log_debug('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
        return seqs_info_list

    self.seqs_info = get_seqs_info_list(train_set) if training else get_seqs_info_list(test_set)


def augmented_dataset_parser(self, data_config: Dict[str, Any], training: bool) -> None:
    """Parse an *augmented* training set by sampling from original vs perturbed data.

    Logic (unchanged):
    - For each sequence, randomly choose:
        - 'original' with probability = aug_ratio
        - 'augmented' with probability = 1 - aug_ratio
    - If chosen augmented path doesn't exist, fall back to original.

    It expects:
    - data_config['dataset_root'] (original)
    - data_config['dataset_partition']
    - data_config['aug_ratio'] (float in [0, 1])
    - data_config['augmented_root'] 

    """
    dataset_root = data_config['dataset_root']

    # Prefer config-driven path (no hardcoding).
    perturbed_dataset_root = data_config.get("augmented_root")
    if perturbed_dataset_root is None:
        raise KeyError(
            "Missing 'perturbed_dataset_root' in data_config for Augmented Dataset."
        )

    try:
        data_in_use = data_config['data_in_use']  # [n], true or false
    except Exception:
        data_in_use = None

    with open(data_config['dataset_partition'], "rb") as f:
        partition = json.load(f)

    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]
    label_list = os.listdir(dataset_root)
    train_set = [label for label in train_set if label in label_list]
    test_set = [label for label in test_set if label in label_list]
    miss_pids = [label for label in label_list if label not in (train_set + test_set)]
    msg_mgr = get_msg_mgr()

    clean_ratio = float(data_config.get("aug_ratio", 1.0))
    noisy_ratio = float(1 - clean_ratio)

    def log_pid_list(pid_list: List[str]) -> None:
        if len(pid_list) >= 3:
            msg_mgr.log_info('[%s, %s, ..., %s]' % (pid_list[0], pid_list[1], pid_list[-1]))
        else:
            msg_mgr.log_info(pid_list)

    if len(miss_pids) > 0:
        msg_mgr.log_debug('-------- Miss Pid List --------')
        msg_mgr.log_debug(miss_pids)

    if training:
        msg_mgr.log_info("-------- Train Pid List --------")
        log_pid_list(train_set)
    else:
        msg_mgr.log_info("-------- Test Pid List --------")
        log_pid_list(test_set)

    def get_seqs_info_list(label_set: List[str]) -> List[List[Any]]:
        seqs_info_list: List[List[Any]] = []
        for lab in label_set:
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]

                    dataset_choice = random.choices(
                        ['original', 'augmented'],
                        weights=[clean_ratio, noisy_ratio],
                        k=1
                    )[0]

                    if dataset_choice == 'augmented':
                        augmented_path = osp.join(perturbed_dataset_root, *seq_info)
                        if osp.exists(augmented_path):
                            seq_path = augmented_path
                        else:
                            seq_path = osp.join(dataset_root, *seq_info)
                    else:
                        seq_path = osp.join(dataset_root, *seq_info)

                    seq_dirs = sorted(os.listdir(seq_path))
                    if seq_dirs != []:
                        seq_dirs = [osp.join(seq_path, d) for d in seq_dirs]
                        if data_in_use is not None:
                            seq_dirs = [d for d, use_bl in zip(seq_dirs, data_in_use) if use_bl]
                        seqs_info_list.append([*seq_info, seq_dirs])
                    else:
                        msg_mgr.log_debug('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
        return seqs_info_list

    self.seqs_info = get_seqs_info_list(train_set) if training else get_seqs_info_list(test_set)


# ------------------------------------------------------------------
# Evaluation gallery parsers
# ------------------------------------------------------------------

def clean_gallery_dataset_parser(self, data_config: Dict[str, Any], training: bool) -> None:
    """Parse a dataset but swap *gallery* sequences with a clean-gallery root during evaluation.

    Common use:
    - Keep probe sequences from `dataset_root`
    - For evaluation (training == False), replace gallery types with `clean_gallery_root`

    It expects:
    - data_config['dataset_root']
    - data_config['clean_gallery_root'] (required when using this parser)
    - data_config['dataset_partition']

    Dataset-specific:
    - CASIA-B gallery types are hardcoded as nm-01..nm-04 (unchanged from your code).
    """
    dataset_root = data_config['dataset_root']
    clean_gallery = True

    clean_gallery_root = data_config['clean_gallery_root']
    if clean_gallery_root is None:
        raise KeyError(
            "Missing 'clean_gallery_root' in data_config for Clean Gallery Evaluation."
        )

    if data_config['dataset_name'] == 'CASIA-B':
        gallery_typs = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    else:
        gallery_typs = []

    try:
        data_in_use = data_config['data_in_use']  # [n], true or false
    except Exception:
        data_in_use = None

    with open(data_config['dataset_partition'], "rb") as f:
        partition = json.load(f)

    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]
    label_list = os.listdir(dataset_root)
    train_set = [label for label in train_set if label in label_list]
    test_set = [label for label in test_set if label in label_list]
    miss_pids = [label for label in label_list if label not in (train_set + test_set)]
    msg_mgr = get_msg_mgr()

    def log_pid_list(pid_list: List[str]) -> None:
        if len(pid_list) >= 3:
            msg_mgr.log_info('[%s, %s, ..., %s]' % (pid_list[0], pid_list[1], pid_list[-1]))
        else:
            msg_mgr.log_info(pid_list)

    if len(miss_pids) > 0:
        msg_mgr.log_debug('-------- Miss Pid List --------')
        msg_mgr.log_debug(miss_pids)

    if training:
        msg_mgr.log_info("-------- Train Pid List --------")
        log_pid_list(train_set)
    else:
        msg_mgr.log_info("-------- Test Pid List --------")
        log_pid_list(test_set)

    def get_seqs_info_list(label_set: List[str]) -> List[List[Any]]:
        seqs_info_list: List[List[Any]] = []
        for lab in tqdm(label_set, desc='get_seqs_info_list'):
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]

                    if (clean_gallery) and (not training) and (typ in gallery_typs):
                        seq_path = osp.join(clean_gallery_root, *seq_info)
                    else:
                        seq_path = osp.join(dataset_root, *seq_info)

                    seq_dirs = sorted(os.listdir(seq_path))
                    if seq_dirs != []:
                        seq_dirs = [osp.join(seq_path, d) for d in seq_dirs]
                        if data_in_use is not None:
                            seq_dirs = [d for d, use_bl in zip(seq_dirs, data_in_use) if use_bl]
                        seqs_info_list.append([*seq_info, seq_dirs])
                    else:
                        msg_mgr.log_debug('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
        return seqs_info_list

    self.seqs_info = get_seqs_info_list(train_set) if training else get_seqs_info_list(test_set)


def perturbed_gallery_dataset_parser(self, data_config: Dict[str, Any], training: bool) -> None:
    """Parse a dataset and, during evaluation, swap gallery sequences to a *perturbed* gallery root.

    This is typically used to evaluate robustness by perturbing only the gallery.

    It expects:
    - data_config['dataset_root']
    - data_config['dataset_partition']
    - data_config['noise_map_file']: JSON mapping "pid-typ-view" -> [perturb, severity]
    - data_config['perturbed_dataset_root']:
        root folder that contains: <perturb>/<severity>/<pid>/<typ>/<view>/...

    Dataset-specific:
    - CASIA-B gallery types: nm-01..nm-04
    - SUSTech1K gallery type: 00-nm
    """
    with open(data_config["noise_map_file"], "r") as file:
        noise_sev_map = json.load(file)

    dataset_root = data_config['dataset_root']
    perturbed_gallery = True

    if data_config['dataset_name'] == 'CASIA-B':
        gallery_typs = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    elif data_config['dataset_name'] == 'SUSTech1K':
        gallery_typs = ['00-nm']
    else:
        gallery_typs = []

    try:
        data_in_use = data_config['data_in_use']
    except Exception:
        data_in_use = None

    with open(data_config['dataset_partition'], "rb") as f:
        partition = json.load(f)

    train_set = partition["TRAIN_SET"]
    test_set = partition["TEST_SET"]
    label_list = os.listdir(dataset_root)
    train_set = [label for label in train_set if label in label_list]
    test_set = [label for label in test_set if label in label_list]
    miss_pids = [label for label in label_list if label not in (train_set + test_set)]
    msg_mgr = get_msg_mgr()

    def log_pid_list(pid_list: List[str]) -> None:
        if len(pid_list) >= 3:
            msg_mgr.log_info('[%s, %s, ..., %s]' % (pid_list[0], pid_list[1], pid_list[-1]))
        else:
            msg_mgr.log_info(pid_list)

    if len(miss_pids) > 0:
        msg_mgr.log_debug('-------- Miss Pid List --------')
        msg_mgr.log_debug(miss_pids)

    if training:
        msg_mgr.log_info("-------- Train Pid List --------")
        log_pid_list(train_set)
    else:
        msg_mgr.log_info("-------- Test Pid List --------")
        log_pid_list(test_set)

    
    perturbed_root = data_config.get("perturbed_gallery_root")
    if perturbed_root is None:
        raise KeyError(
            "Missing 'perturbed_dataset_root' in data_config for PerturbedGallery."
        )

    def get_seqs_info_list(label_set: List[str]) -> List[List[Any]]:
        seqs_info_list: List[List[Any]] = []

        for lab in tqdm(label_set, desc='get_seqs_info_list'):
            for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                    seq_info = [lab, typ, vie]

                    if (perturbed_gallery) and (not training) and (typ in gallery_typs):
                        video_name = f"{lab}-{typ}-{vie}"
                        perturb, sev = noise_sev_map[video_name]
                        seq_path = osp.join(perturbed_root, perturb, str(sev), *seq_info)
                    else:
                        seq_path = osp.join(dataset_root, *seq_info)

                    seq_dirs = sorted(os.listdir(seq_path))
                    if seq_dirs != []:
                        seq_dirs = [osp.join(seq_path, d) for d in seq_dirs]
                        if data_in_use is not None:
                            seq_dirs = [d for d, use_bl in zip(seq_dirs, data_in_use) if use_bl]
                        seqs_info_list.append([*seq_info, seq_dirs])
                    else:
                        msg_mgr.log_debug('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))

        return seqs_info_list

    self.seqs_info = get_seqs_info_list(train_set) if training else get_seqs_info_list(test_set)