#!/bin/bash

#######################################################################
# OpenGait Robustness Evaluation Script 
#
# This script demonstrates how to run different dataset variants:
#
# TRAINING DATASET VARIANTS:
#   - Standard        (default silhouettes)
#   - Augmented       (clean + noisy mix during training)
#   - MEVID           (tracklet-based dataset structure)
#   - MultiModal      (silhouette + RGB together)
#
# EVALUATION GALLERY VARIANTS:
#   - NormalGallery   (default gallery)
#   - CleanGallery    (use clean gallery only)
#   - PerturbedGallery(use noisy/perturbed gallery only)
#
#######################################################################

# ----------------------------
# Common Config
# ----------------------------

CFG=./configs/gaitset/gaitset.yaml
SAVE_NAME="GaitSet_experiment"

# Partition file
PARTITION=./datasets/CASIA-B/partition.json

# GPU selection
GPU=0

#######################################################################
# 1. STANDARD TRAINING (Baseline)
#######################################################################
# Trains normally on original silhouettes
#
# Example:
#   --train_dataset_type Standard
#######################################################################

DATA_ROOT=/path/to/casiab/sil_pkl/orig

CUDA_VISIBLE_DEVICES=$GPU python3 opengait/main.py \
  --cfgs $CFG \
  --phase train \
  --save_name ${SAVE_NAME} \
  --dataset_root $DATA_ROOT \
  --dataset_partition $PARTITION \

#######################################################################
# 2. AUGMENTED TRAINING (Robust Training)
#######################################################################
# Trains on a mixture of clean + perturbed silhouettes
#
# Requires:
#   --augmented_root
#
# Example:
#   --train_dataset_type Augmented
#######################################################################

AUGMENTED_ROOT=/path/to/casiab/sil_pkl/augmented

CUDA_VISIBLE_DEVICES=$GPU python3 opengait/main.py \
  --cfgs $CFG \
  --phase train \
  --save_name ${SAVE_NAME} \
  --dataset_root $DATA_ROOT \
  --augmented_root $AUGMENTED_ROOT \
  --dataset_partition $PARTITION \
  --dataset_type Augmented \
  --aug_ratio 0.5


#######################################################################
# 3. MULTIMODAL TRAINING (Silhouette + RGB)
#######################################################################
# Trains using both silhouette PKLs and RGB videos/images
#
# Requires:
#   --sil_root and --rgb_root
#
# Example:
#   --train_dataset_type MultiModal
#######################################################################

SIL_ROOT=/path/to/casiab/sil_pkl/orig
RGB_ROOT=/path/to/casiab/rgb_videos

CUDA_VISIBLE_DEVICES=$GPU python3 opengait/main.py \
  --cfgs $CFG \
  --phase train \
  --save_name ${SAVE_NAME}_multimodal \
  --dataset_partition $PARTITION \
  --dataset_type MultiModal \
  --sil_root $SIL_ROOT \
  --rgb_root $RGB_ROOT


#######################################################################
# 4. CLEAN GALLERY EVALUATION (Clean Probe → Clean Gallery)
#######################################################################
# Evaluates using ONLY the clean gallery sequences
#
# Requires:
#   --clean_gallery_root
#
# Example:
#   --eval_gallery_type CleanGallery
#######################################################################

CLEAN_GALLERY_ROOT=/path/to/casiab/sil_pkl/clean_gallery

CUDA_VISIBLE_DEVICES=$GPU python3 opengait/main.py \
  --cfgs $CFG \
  --phase test \
  --save_name ${SAVE_NAME} \
  --dataset_root $DATA_ROOT \
  --dataset_partition $PARTITION \
  --dataset_type CleanGallery \
  --clean_gallery_root $CLEAN_GALLERY_ROOT


#######################################################################
# 5. PERTURBED GALLERY ROBUSTNESS TEST (Clean Probe → Noisy Gallery)
#######################################################################
# Evaluates robustness by replacing gallery with perturbed silhouettes
#
# Requires:
#   --perturbed_dataset_root
#   --noise_map_file
#
# Example:
#   --eval_gallery_type PerturbedGallery
#######################################################################

PERTURBED_ROOT=/path/to/casiab/sil_pkl/perturb
NOISE_MAP=./datasets/CASIA-B/noise_severity_assignments.json

CUDA_VISIBLE_DEVICES=$GPU python3 opengait/main.py \
  --cfgs $CFG \
  --phase test \
  --save_name ${SAVE_NAME} \
  --dataset_root $DATA_ROOT \
  --dataset_partition $PARTITION \
  --dataset_type PerturbedGallery \
  --perturbed_dataset_root $PERTURBED_ROOT \
  --noise_map_file $NOISE_MAP


#######################################################################
# DONE
#######################################################################
echo "All training/evaluation variants finished."