import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr
from tqdm import tqdm
import random

import numpy as np

# to read videos
#import decord
import io
import cv2

# NEW: import the parser functions from your separate file
# (Assumes: datasets/dataset_parsers.py is in the same package/folder)
from . import dataset_parsers as dp


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating
                            a certain gait sequence presented as [label, type, view, paths];
        """

        dataset_type = data_cfg.get("dataset_type", None)
        self.aug_ratio = data_cfg["aug_ratio"]

        parser_map = {
            "MultiModal": dp.multimodal_dataset_parser,
            "MEVID": dp.mevid_dataset_parser,
            "CleanGallery": dp.clean_gallery_dataset_parser,
            "PerturbedGallery": dp.perturbed_gallery_dataset_parser,
            "Augmented": dp.augmented_dataset_parser,
        }

        parser_fn = parser_map.get(dataset_type, None)

        if parser_fn is not None:
            parser_fn(self, data_cfg, training)
        else:
            self.__dataset_parser(data_cfg, training)

        self.cache = data_cfg["cache"]

        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)
    
    def _load_pkl(self, pth):
        with open(pth, "rb") as f:
            return pickle.load(f)


    def _load_mp4(self, pth, resize=None):
        cap = cv2.VideoCapture(pth)
        if not cap.isOpened():
            print(f"Error: Couldn't open the video {pth}")
            return None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if resize is not None:
                frame = cv2.resize(frame, resize)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

        cap.release()
        return frames


    def _load_avi(self, pth, resize=None):
        sil_vid = decord.VideoReader(pth)
        frames = [frm.asnumpy() for frm in sil_vid]

        if len(frames) == 0:
            print(f"Error: No frames found in {pth}")
            return "__EMPTY_AVI__"

        processed = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if resize is not None:
                gray = cv2.resize(gray, resize)
            processed.append(gray)

        return np.array(processed)


    def _validate_data_list(self, paths, data_list):
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    f"Each input data({paths[idx]}) should have the same length."
                )
            if len(data) == 0:
                raise ValueError(
                    f"Each input data({paths[idx]}) should have at least one element."
            )
    def __loader__(self, paths):

        paths = sorted(paths)
        data_list = []

        for pth in paths:

            if pth.endswith(".pkl"):
                try:
                    if os.path.getsize(pth) == 0:
                        print(f"⚠️ Skipping empty file: {pth}")
                        continue

                    data = self._load_pkl(pth)

                    if len(data) == 0:
                        print(f"⚠️ Skipping empty file: {pth}")
                        continue

                    if isinstance(data, np.ndarray) and data.ndim == 4 and data.shape[-1] == 3:
                        data = data[..., 0]
                    data_list.append(data)

                except Exception as e:
                    print(f"Unexpected error loading {pth}: {e}")
                    continue

            elif pth.endswith(".mp4"):

                frames = self._load_mp4(pth)
                if frames is None:
                    continue
                data_list.append(frames)

            elif pth.endswith(".avi"):
                arr = self._load_avi(pth, resize=(64, 64))
                if arr == "__EMPTY_AVI__":
                    return
                data_list.append(arr)
            else:
                raise ValueError("- Loader - just support .pkl, .mp4 and .avi !!!")

        # Validate
        self._validate_data_list(paths, data_list)

        return data_list

    def __multimodal_loader__(self, mm_paths):

        sil_paths, rgb_paths = mm_paths

        if not isinstance(sil_paths, list):
            sil_paths = [sil_paths]

        paths = sorted(sil_paths)
        data_list = []

        # ---- Load SIL ----
        for pth in paths:

            if pth.endswith(".pkl"):
                data_list.append(self._load_pkl(pth))

            elif pth.endswith(".mp4"):
                frames = self._load_mp4(pth, resize=(128, 256))
                if frames is None:
                    continue
                data_list.append(frames)

            elif pth.endswith(".avi"):
                arr = self._load_avi(pth)
                if arr == "__EMPTY_AVI__":
                    return
                data_list.append(arr)

            else:
                raise ValueError("- Loader - just support .pkl, .avi and .mp4 !!!")

        # Validate SIL
        self._validate_data_list(paths, data_list)

        # ---- Load RGB ----
        for pth in sorted([rgb_paths]):
            if pth.endswith(".pkl"):
                data_list.append(self._load_pkl(pth))

        return data_list


    def __getitem__(self, idx):
        if not self.cache:
            if self.multimodal:
                data_list = self.__multimodal_loader__(self.seqs_info[idx][-1])
            else:
                data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            if self.multimodal:
                data_list = self.__multimodal_loader__(self.seqs_info[idx][-1])
            else:
                data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]

        if data_list is None or any(len(d) == 0 for d in data_list):
            return None

        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def load_pkl(self, pkl_file):
        with open(pkl_file, "rb") as f:
            obj = pickle.load(f)
        return obj

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config["dataset_root"]
        print("Inside data loader")
        try:
            data_in_use = data_config["data_in_use"]  # [n], true or false
        except:
            data_in_use = None

        with open(data_config["dataset_partition"], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info("[%s, %s, ..., %s]" % (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug("-------- Miss Pid List --------")
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            skipped_files = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir) for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(seq_dirs, data_in_use) if use_bl]

                            seq_dirs_new = []
                            for item in seq_dirs:
                                if os.path.getsize(item) == 0:
                                    print(f"Error: File is empty - {item}")
                                    continue
                                else:
                                    item_pkl = self.load_pkl(item)
                                    if len(item_pkl) == 0:
                                        skipped_files.append([*seq_info, seq_dirs, len(item_pkl.shape)])
                                        continue
                                    if data_config["dataset_name"] in ["CCPG", "SUSTech1K"] and len(item_pkl.shape) != 3:
                                        skipped_files.append([*seq_info, seq_dirs, len(item_pkl.shape)])
                                        continue
                                    else:
                                        seq_dirs_new.append(item)

                            if seq_dirs_new != []:
                                seqs_info_list.append([*seq_info, seq_dirs_new])
                            else:
                                msg_mgr.log_debug(".pkl files empty in %s-%s-%s." % (lab, typ, vie))

                        else:
                            msg_mgr.log_debug("Find no .pkl file in %s-%s-%s." % (lab, typ, vie))
            print(f"skipped a total of {len(skipped_files)} files")
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(train_set) if training else get_seqs_info_list(test_set)
