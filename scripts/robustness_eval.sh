#!/bin/bash

###############################################################
# Script for robustness evaluation in gait recognition
#
#  This script trains on CLEAN silhouettes and tests on NOISY
#  silhouettes, using silhouettes extracted from ANY segmentation
#  / human-parsing model (SCHP is only used as an example).
#
#  This clean → noisy protocol is used to evaluate the robustness
#  of gait recognition models under appearance degradation.
###############################################################

# Choose model config: gaitset, gaitpart, gaitbase, deepgaitv2, gaitgl, etc.
CFG=./configs/gaitset/gaitset.yaml

# CLEAN silhouettes (e.g., SCHP clean output)
CLEAN_ROOT=/path/to/casiab/sil_pkl/orig/clean_sils

# NOISY silhouettes (same segmentation model, but perturbed)
NOISY_ROOT=/path/to/casiab/sil_pkl/perturb/gaussian_noise/sev1

# Checkpoint folder name
SAVE_NAME="GaitSet_clean_train"

###############################################################
# 1. TRAIN ON CLEAN DATA
###############################################################
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nproc_per_node=1 opengait/main.py \
    --cfgs $CFG \
    --dataset_root $CLEAN_ROOT \
    --save_name $SAVE_NAME \
    --phase train


###############################################################
# 2. TEST ON NOISY DATA (ROBUSTNESS EVALUATION)
###############################################################
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nproc_per_node=1 opengait/main.py \
    --cfgs $CFG \
    --dataset_root $NOISY_ROOT \
    --save_name $SAVE_NAME \
    --phase test
