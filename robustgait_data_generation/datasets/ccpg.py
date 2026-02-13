import os
import cv2
import numpy as np
import random
from torch.utils import data

from .transforms import get_affine_transform
from .robust_transforms import ALL_TRANSFORMS
from .utils import box_to_center_scale


class CCPG(data.Dataset):
    """CCPG RGB dataset with subject-wise split and optional robustness transforms."""

    def __init__(
        self,
        root,
        input_size=(512, 512),
        transform=None,
        robust_transform=None,
        test_set=(100, 199),
        test_only=False,
        sev=3,
    ):
        """
        Args:
            root (str): Root directory where CCPG sequences are stored.
            input_size (tuple): (height, width).
            transform (callable): Transform applied to each frame.
            robust_transform (str or callable or None):
                - None: no robustness transform.
                - "random": randomly sample a transform from ALL_TRANSFORMS.
                - string name: lookup in ALL_TRANSFORMS.
                - callable: custom function(frames, sev=...).
            test_set (tuple): Range of subject IDs used as test subjects.
            test_only (bool): If True, keep only test subjects.
            sev (int): Default severity level.
        """
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

        # Build test/train subjects
        self.test_subjects = list(range(*test_set))
        self.train_subjects = []

        self.test_list = []
        self.data_list = []

        dataset_full = sorted(os.listdir(self.root))

        for subject in dataset_full:
            self.train_subjects.append(subject)
            subject_dir = os.path.join(self.root, subject)

            for seq_type in os.listdir(subject_dir):
                type_dir = os.path.join(subject_dir, seq_type)

                for view in os.listdir(type_dir):
                    view_dir = os.path.join(type_dir, view)
                    subject_id = int(subject)

                    if test_only:
                        if subject_id in self.test_subjects:
                            self.data_list.append(view_dir)
                            self.test_list.append(view_dir)
                    else:
                        self.data_list.append(view_dir)
                        if subject_id in self.test_subjects:
                            self.test_list.append(view_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sequence_dir = self.data_list[index]
        relative_name = sequence_dir.replace(self.root, "")
        video_name = relative_name.replace("/", "-")

        subject_id = int(video_name.split("-")[0])
        is_test_subject = subject_id in self.test_subjects

        frames = []
        frame_names = []

        for filename in os.listdir(sequence_dir):
            frame_names.append(filename)
            frame = cv2.imread(os.path.join(sequence_dir, filename))
            frames.append(frame)

        random_robust_name = None

        # -------------------------------------------------------------- #
        # Select robust transform if in "random" mode
        # -------------------------------------------------------------- #
        if self.robust_name == "random" and len(frames) > 0:
            random_robust_name = random.choice(list(ALL_TRANSFORMS.keys()))
            self.robust_transform = ALL_TRANSFORMS[random_robust_name]
            self.severity = random.choices(
                [1, 2, 3],
                weights=[0.6, 0.3, 0.1],
            )[0]

        # -------------------------------------------------------------- #
        # Apply robustness transform (if any)
        # -------------------------------------------------------------- #
        if self.robust_transform is not None and len(frames) > 0:
            frames = self.robust_transform(frames, sev=self.severity)

        # -------------------------------------------------------------- #
        # Apply affine transform & metadata
        # -------------------------------------------------------------- #
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
