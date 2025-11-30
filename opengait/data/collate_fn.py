import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    """
    Default collate function for gait sequences.

    Assumes each batch item has the structure:
        (seqs, (label, type, view))

    - seqs: list of features, each feature is a list of frames
    - label_set: list of all labels; labels are mapped to indices via label_set.index(...)
    """

    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        self.num_modalities = sample_config.get("modalities", 1)

        # Parse sample type: e.g., "fixed_ordered", "unfixed_unordered", "all_ordered"
        sample_type = sample_config["sample_type"].split("_")
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]

        if self.num_modalities > 1:
            self.modality_config = sample_config.get("modality", None)
        else:
            self.modality_config = None

        if self.sampler not in ["fixed", "unfixed", "all"]:
            raise ValueError(f"Unknown sampler type: {self.sampler}")
        if self.ordered not in ["ordered", "unordered"]:
            raise ValueError(f"Unknown ordering: {self.ordered}")

        # Convert to bool: True if "ordered", False if "unordered"
        self.ordered = (self.ordered == "ordered")

        # Fixed-length sampling
        if self.sampler == "fixed":
            self.frames_num_fixed = sample_config["frames_num_fixed"]

        # Random-length sampling within [min, max]
        if self.sampler == "unfixed":
            self.frames_num_max = sample_config["frames_num_max"]
            self.frames_num_min = sample_config["frames_num_min"]

        # Skip size used for ordered sampling (except for "all")
        if self.sampler != "all" and self.ordered:
            self.frames_skip_num = sample_config["frames_skip_num"]

        # For "all" sampler, optionally cap the number of frames
        self.frames_all_limit = -1
        if self.sampler == "all" and "frames_all_limit" in sample_config:
            self.frames_all_limit = sample_config["frames_all_limit"]

    # --------------------------------------------------------------------- #
    # Helper: build a frame-sampling function (shared between subclasses)
    # --------------------------------------------------------------------- #
    def _make_sample_frames_fn(self, feature_num, labs_batch, typs_batch, vies_batch):
        """
        Returns a closure `sample_frames(seqs)` that samples frame indices
        according to self.sampler / self.ordered / etc., and logs empty seqs.
        """
        count = 0  # used only for debug logging

        def sample_frames(seqs):
            nonlocal count
            sampled_frames = [[] for _ in range(feature_num)]

            seq_len = len(seqs[0])
            indices = list(range(seq_len))

            if self.sampler in ["fixed", "unfixed"]:
                # Number of frames to sample
                if self.sampler == "fixed":
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max + 1))
                    )

                if self.ordered:
                    # Ordered sampling with skip
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / max(seq_len, 1))
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    window_indices = list(range(seq_len))[start:end]
                    window_indices = sorted(
                        np.random.choice(window_indices, frames_num, replace=False)
                    )
                    indices = [indices[i] for i in window_indices]
                else:
                    # Unordered random sampling
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug(
                            "Find no frames in the sequence %s-%s-%s."
                            % (
                                str(labs_batch[count]),
                                str(typs_batch[count]),
                                str(vies_batch[count]),
                            )
                        )

                    count += 1
                    indices = np.random.choice(indices, frames_num, replace=replace)

            # Apply optional cap for "all" sampler
            effective_indices = (
                indices[: self.frames_all_limit]
                if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit
                else indices
            )

            for i in range(feature_num):
                for j in effective_indices:
                    sampled_frames[i].append(seqs[i][j])

            return sampled_frames

        return sample_frames

    # --------------------------------------------------------------------- #
    # Main collate
    # --------------------------------------------------------------------- #
    def __call__(self, batch):
        # Filter out invalid / empty items
        batch = [
            item
            for item in batch
            if item is not None
            and item[0]
            and all(len(seq) > 0 for seq in item[0])
        ]

        if len(batch) == 0:
            # Gracefully skip if everything is invalid
            return None

        batch_size = len(batch)

        # Currently feature_num refers to number of input streams (e.g., silhouette, skeleton).
        # For multi-modal setups, this is effectively kept at 1 per collate call.
        if self.num_modalities > 1:
            feature_num = 1
        else:
            feature_num = len(batch[0][0])

        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for seqs, meta in batch:
            seqs_batch.append(seqs)
            labs_batch.append(self.label_set.index(meta[0]))
            typs_batch.append(meta[1])
            vies_batch.append(meta[2])

        # Shared frame-sampling function
        sample_frames = self._make_sample_frames_fn(
            feature_num, labs_batch, typs_batch, vies_batch
        )

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num

        # Sample frames for each sequence in the batch: [b, f]
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]

        # Batch structure: [features, labels, types, views, seq_lengths]
        batch_out = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            # Convert to [f, b] with numpy arrays
            fras_batch = [
                [np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                for j in range(feature_num)
            ]  # [f, b]
        else:
            # Variable-length case: concat along time, record sequence lengths
            seqL_batch = [[len(fras_batch[i][0]) for i in range(batch_size)]]  # [1, p]

            def my_cat(k):
                return np.concatenate(
                    [fras_batch[i][k] for i in range(batch_size)], axis=0
                )

            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]
            batch_out[-1] = np.asarray(seqL_batch)

        batch_out[0] = fras_batch
        return batch_out


class MultiCollateFn(CollateFn):
    """
    Collate function that supports multiple modalities.

    Assumes batch item:
        bt[0][0] -> primary modality (e.g., silhouette)
        bt[0][1] -> secondary modality (e.g., RGB)
    and bt[1] is (label, type, view) as in CollateFn.
    """

    def __init__(self, label_set, sample_config):
        super(MultiCollateFn, self).__init__(label_set, sample_config)

    def __call__(self, batch):
        # Filter out invalid / empty items (same as base collate)
        batch = [
            item
            for item in batch
            if item is not None
            and item[0]
            and all(len(seq) > 0 for seq in item[0])
        ]

        if len(batch) == 0:
            return None

        batch_size = len(batch)

        if self.num_modalities > 1:
            feature_num = 1
        else:
            feature_num = len(batch[0][0])

        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        # Only used when num_modalities > 1
        seqs_primary_batch = []
        seqs_rgb_batch = []

        for seqs, meta in batch:
            if self.num_modalities > 1:
                # Expect at least 2 modalities: [primary, rgb, ...]
                seqs_batch.append([seqs[0]])       # main modality
                seqs_primary_batch.append([seqs[1]])  # secondary (e.g., RGB)
            else:
                seqs_batch.append(seqs)

            labs_batch.append(self.label_set.index(meta[0]))
            typs_batch.append(meta[1])
            vies_batch.append(meta[2])

        # Shared sampling function for the main modality
        sample_frames = self._make_sample_frames_fn(
            feature_num, labs_batch, typs_batch, vies_batch
        )

        # Main modality sampling
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]

        batch_out = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        # Pack main modality into final format
        if self.sampler == "fixed":
            fras_batch = [
                [np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                for j in range(feature_num)
            ]  # [f, b]
        else:
            seqL_batch = [[len(fras_batch[i][0]) for i in range(batch_size)]]  # [1, p]

            def my_cat(k):
                return np.concatenate(
                    [fras_batch[i][k] for i in range(batch_size)], axis=0
                )

            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]
            batch_out[-1] = np.asarray(seqL_batch)

        # ------------------------------------------------------------------ #
        # Secondary modality (e.g., RGB) when num_modalities > 1
        # ------------------------------------------------------------------ #
        if self.num_modalities > 1 and len(seqs_primary_batch) > 0:
            # Save original sampler state
            original_sampler = self.sampler
            original_frames_num_fixed = getattr(self, "frames_num_fixed", None)
            original_frames_skip_num = getattr(self, "frames_skip_num", None)

            # If modality-specific config exists, override for RGB
            if self.modality_config is not None and "rgb" in self.modality_config:
                rgb_cfg = self.modality_config["rgb"]
                self.sampler = "fixed"
                self.frames_num_fixed = rgb_cfg["frames_num_fixed"]
                self.frames_skip_num = rgb_cfg["frames_skip_num"]

            # Build a fresh sampler closure (independent counter)
            sample_frames_rgb = self._make_sample_frames_fn(
                feature_num, labs_batch, typs_batch, vies_batch
            )

            fras_rgb_batch = [
                sample_frames_rgb(seqs) for seqs in seqs_primary_batch
            ]  # [b, f]

            if self.sampler == "fixed":
                fras_rgb_batch = [
                    [np.asarray(fras_rgb_batch[i][j]) for i in range(batch_size)]
                    for j in range(feature_num)
                ]  # [f, b]
            else:
                seqL_rgb_batch = [
                    [len(fras_rgb_batch[i][0]) for i in range(batch_size)]
                ]  # [1, p]

                def my_cat_rgb(k):
                    return np.concatenate(
                        [fras_rgb_batch[i][k] for i in range(batch_size)], axis=0
                    )

                fras_rgb_batch = [[my_cat_rgb(k)] for k in range(feature_num)]  # [f, g]

                # Append RGB sequence lengths along a new row
                batch_out[-1] = np.concatenate(
                    [batch_out[-1], np.asarray(seqL_rgb_batch)], axis=0
                )

            # Restore original sampler state
            self.sampler = original_sampler
            if original_frames_num_fixed is not None:
                self.frames_num_fixed = original_frames_num_fixed
            if original_frames_skip_num is not None:
                self.frames_skip_num = original_frames_skip_num

            # Concatenate modalities: [f_main, f_rgb]
            fras_batch = fras_batch + fras_rgb_batch

        batch_out[0] = fras_batch
        return batch_out
