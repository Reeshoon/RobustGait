import os
import cv2
import numpy as np
import datetime
import random
from collections import defaultdict
from torch.utils import data

from .transforms import get_affine_transform
from .robust_transforms import ALL_TRANSFORMS
from .utils import box_to_center_scale, get_project_root


class CASIAB(data.Dataset):
    """CASIA-B RGB dataset with optional robustness transforms."""

    def __init__(
        self,
        root,
        input_size=(512, 512),
        transform=None,
        robust_transform=None,
        test_set=(75, 125),
        test_only=False,
        sev=3,
    ):
        self.root = root
        self.input_size = np.asarray(input_size)
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.severity = sev

        # Robust setup
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

        # Logging
        project_root = get_project_root()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.log_file = os.path.join(logs_dir, f"noise_log_{timestamp}.txt")

        self.noise_summary = defaultdict(int)

        # Video lists
        self.test_subjects = list(range(*test_set))
        self.data_list = []
        self.test_list = []

        for video_name in os.listdir(self.root):
            if "bkgrd" in video_name:
                continue
            subject_id = int(video_name.split("-")[0])

            if test_only:
                if subject_id in self.test_subjects:
                    self.data_list.append(video_name)
                    self.test_list.append(video_name)
            else:
                self.data_list.append(video_name)
                if subject_id in self.test_subjects:
                    self.test_list.append(video_name)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        video_name = self.data_list[index]
        video_path = os.path.join(self.root, video_name)
        subject_id = int(video_name.split("-")[0])

        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for _ in range(total_frames):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)

        frames = np.asarray(frames)
        random_robust_name = None

        # Random robustness
        if self.robust_name == "random" and frames.shape[0] > 0:
            random_robust_name = random.choice(list(ALL_TRANSFORMS.keys()))
            self.robust_transform = ALL_TRANSFORMS[random_robust_name]
            self.severity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

            with open(self.log_file, "a") as f:
                f.write(f"{video_name}: noise={random_robust_name}, severity={self.severity}\n")

            self.noise_summary[(random_robust_name, self.severity)] += 1

        # Apply robustness
        if self.robust_transform is not None and frames.shape[0] > 0:
            name = random_robust_name if self.robust_name == "random" else self.robust_name

            if name == "snow":
                angle = int(video_name.split("-")[3].replace(".avi", ""))
                frames = self.robust_transform(frames, sev=self.severity, angle=angle)
            else:
                frames = self.robust_transform(frames, sev=self.severity)

            if isinstance(frames, list):
                frames = np.asarray(frames, dtype=np.uint8)

        processed_inputs, metas = [], []

        for frame in frames:
            h, w, _ = frame.shape

            center, scale = box_to_center_scale([0, 0, w - 1, h - 1], self.aspect_ratio)
            affine_mat = get_affine_transform(center, scale, 0, self.input_size)

            warped = cv2.warpAffine(
                frame,
                affine_mat,
                (self.input_size[1], self.input_size[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            inp = self.transform(warped)
            meta = dict(name=video_name, center=center, height=h, width=w, scale=scale, rotation=0)

            processed_inputs.append(inp)
            metas.append(meta)

        return processed_inputs, metas
