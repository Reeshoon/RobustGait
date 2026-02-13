import os
import glob
import cv2
import numpy as np
import random
import json
from torch.utils import data

from .transforms import get_affine_transform
from .robust_transforms import ALL_TRANSFORMS
from .utils import box_to_center_scale, load_pickle, get_project_root


class SusTech1K(data.Dataset):
    """SusTech1K RGB dataset with predefined train/test split via JSON files."""

    def __init__(
        self,
        root,
        input_size=(512, 512),
        transform=None,
        robust_transform=None,
        test_set=(74, 125),  # kept for backward compatibility
        test_only=False,
        sev=3,
    ):
        self.root = root
        self.input_size = np.asarray(input_size)
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.severity = sev

        # Robust transform setup
        self.robust_name = robust_transform
        if isinstance(robust_transform, str):
            if robust_transform == "random":
                self.robust_transform = "random"
            else:
                self.robust_transform = ALL_TRANSFORMS.get(robust_transform)
        elif callable(robust_transform):
            self.robust_transform = robust_transform
        else:
            self.robust_transform = None

        # Load train/test splits from JSON files
        project_root = get_project_root()
        splits_dir = os.path.join(project_root, "splits")

        train_split_path = os.path.join(splits_dir, "sustech_train.json")
        test_split_path = os.path.join(splits_dir, "sustech_test.json")

        with open(train_split_path, "r") as f:
            self.train_subjects = json.load(f)
        with open(test_split_path, "r") as f:
            self.test_subjects = json.load(f)

        dataset_full = sorted(os.listdir(self.root))

        self.data_list = []
        self.test_list = []

        for subject in dataset_full:
            subject_dir = os.path.join(self.root, subject)
            for seq_type in os.listdir(subject_dir):
                type_dir = os.path.join(subject_dir, seq_type)
                for view in os.listdir(type_dir):
                    rgb_dir = os.path.join(type_dir, view, "RGB_raw")

                    if test_only:
                        if subject in self.test_subjects:
                            self.data_list.append(rgb_dir)
                            self.test_list.append(rgb_dir)
                    else:
                        self.data_list.append(rgb_dir)
                        if subject in self.test_subjects:
                            self.test_list.append(rgb_dir)

    def __len__(self):
        return len(self.data_list)

    def find_rgbs_pkl_files(self, directory):
        """Kept for compatibility with older pipelines."""
        file_paths = []
        for root_dir, _, _ in os.walk(directory):
            pkl_files = glob.glob(os.path.join(root_dir, "05-*.pkl"))
            pkl_files = [os.path.relpath(p, directory) for p in pkl_files]
            file_paths.extend(pkl_files)
        return file_paths

    def load_pkl(self, pkl_file):
        return load_pickle(pkl_file)

    def __getitem__(self, index):
        rgb_dir = self.data_list[index]
        video_name = rgb_dir.replace(self.root, "")

        subject_id = video_name.split("/")[0]
        is_test_subject = subject_id in self.test_subjects

        frames = []
        frame_names = []

        img_files = sorted(os.listdir(rgb_dir))
        for filename in img_files:
            frame_names.append(filename)
            frames.append(cv2.imread(os.path.join(rgb_dir, filename)))

        random_robust_name = None

        # Random transform selection
        if self.robust_name == "random" and len(frames) > 0:
            random_robust_name = random.choice(list(ALL_TRANSFORMS.keys()))
            self.robust_transform = ALL_TRANSFORMS[random_robust_name]
            self.severity = random.choices(
                [1, 2, 3],
                weights=[0.6, 0.3, 0.1],
            )[0]

        # Apply robustness transform
        if self.robust_transform is not None and len(frames) > 0:
            frames = self.robust_transform(frames, sev=self.severity)

        processed_inputs = []
        metas = []

        for frame in frames:
            height, width, _ = frame.shape

            person_center, scale = box_to_center_scale(
                [0, 0, width - 1, height - 1],
                self.aspect_ratio,
            )

            affine_matrix = get_affine_transform(
                person_center,
                scale,
                0,
                self.input_size,
            )

            warped = cv2.warpAffine(
                frame,
                affine_matrix,
                (int(self.input_size[1]), int(self.input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            transformed_input = self.transform(warped)

            meta = {
                "name": video_name,
                "frame_names": frame_names,
                "center": person_center,
                "height": height,
                "width": width,
                "scale": scale,
                "rotation": 0,
            }

            processed_inputs.append(transformed_input)
            metas.append(meta)

        return processed_inputs, metas
