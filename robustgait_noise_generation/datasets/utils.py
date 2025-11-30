import os
import pickle
import numpy as np


def load_pickle(pkl_file_path):
    """Load a pickle object from disk."""
    with open(pkl_file_path, "rb") as file:
        return pickle.load(file)


def get_project_root():
    """
    Return project root assuming:
        <root>/datasets/<files>
    """
    datasets_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(datasets_dir)


def _xywh_to_center_scale(x, y, w, h, aspect_ratio):
    """Convert (x, y, w, h) → (center, scale) with aspect-ratio correction."""
    center = np.zeros((2,), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w, h], dtype=np.float32)
    return center, scale


def box_to_center_scale(box, aspect_ratio):
    x, y, w, h = box[:4]
    return _xywh_to_center_scale(x, y, w, h, aspect_ratio)
