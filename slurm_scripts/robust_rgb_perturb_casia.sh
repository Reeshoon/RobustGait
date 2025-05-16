#!/bin/bash

# ======== GPU Config ========
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -C 'volta|turing|ampere'
#SBATCH --cpus-per-gpu=4
#SBATCH -C gmem11

# ======== Slurm Config ========
#SBATCH --job-name=R8_CA
#SBATCH -o slurm_outputs/R8_CA_AllModels_CasiaB_Robustness_%j.out

# ======== Conda Environment ========
conda activate deepgait

# ======== Base Dataset Root Placeholder ========
BASE_DATASET_ROOT="/path/to/robust_gait/casiab/sil_pkl/perturb"

# ======== Perturbations & Severities ========
PERTURBS=(
    "gaussian_noise"
    "defocus_blur"
    "zoom_blur"
    "impulse_noise"
    "speckle_noise"
    "shot_noise"
    "zoom_in"
    "freeze"
    "sampling"
    "low_light"
    "rain"
    "snow"
    "fog"
    "occlusion"
    "impulse_noise2"
)
SEVS=(1 2 3 4 5)

# ======== DeepGaitV2 ========
echo "*********************************************************************************"
echo "** DeepGaitV2 on CASIA-B (SCHP), robustness test **"
echo "*********************************************************************************"
CONFIG_FILE="./configs/deepgaitv2/DeepGaitV2_casiab.yaml"
SAVE_NAME="DeepGaitV2_schp"
for PERTURB in "${PERTURBS[@]}"; do 
    for sev in "${SEVS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12358 --nproc_per_node=1 opengait/main.py \
            --phase test \
            --cfgs $CONFIG_FILE \
            --save_name $SAVE_NAME \
            --dataset_root "$BASE_DATASET_ROOT/$PERTURB/sev$sev"
    done
done

# ======== GaitBase ========
echo "*********************************************************************************"
echo "** GaitBase on CASIA-B (SCHP), robustness test **"
echo "*********************************************************************************"
CONFIG_FILE="./configs/gaitbase/gaitbase_da_casiab.yaml"
SAVE_NAME="GaitBase_DA_schp"
for PERTURB in "${PERTURBS[@]}"; do 
    for sev in "${SEVS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12359 --nproc_per_node=1 opengait/main.py \
            --phase test \
            --cfgs $CONFIG_FILE \
            --save_name $SAVE_NAME \
            --dataset_root "$BASE_DATASET_ROOT/$PERTURB/sev$sev"
    done
done

# ======== GaitGL ========
echo "*********************************************************************************"
echo "** GaitGL on CASIA-B (SCHP), robustness test **"
echo "*********************************************************************************"
CONFIG_FILE="./configs/gaitgl/gaitgl.yaml"
SAVE_NAME="GaitGL_schp"
for PERTURB in "${PERTURBS[@]}"; do 
    for sev in "${SEVS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12360 --nproc_per_node=1 opengait/main.py \
            --phase test \
            --cfgs $CONFIG_FILE \
            --save_name $SAVE_NAME \
            --dataset_root "$BASE_DATASET_ROOT/$PERTURB/sev$sev"
    done
done

# ======== GaitPart ========
echo "*********************************************************************************"
echo "** GaitPart on CASIA-B (SCHP), robustness test **"
echo "*********************************************************************************"
CONFIG_FILE="./configs/gaitpart/gaitpart.yaml"
SAVE_NAME="GaitPart_schp"
for PERTURB in "${PERTURBS[@]}"; do 
    for sev in "${SEVS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12361 --nproc_per_node=1 opengait/main.py \
            --phase test \
            --cfgs $CONFIG_FILE \
            --save_name $SAVE_NAME \
            --dataset_root "$BASE_DATASET_ROOT/$PERTURB/sev$sev"
    done
done

# ======== GaitSet ========
echo "*********************************************************************************"
echo "** GaitSet on CASIA-B (SCHP), robustness test **"
echo "*********************************************************************************"
CONFIG_FILE="./configs/gaitset/gaitset.yaml"
SAVE_NAME="GaitSet_schp"
for PERTURB in "${PERTURBS[@]}"; do 
    for sev in "${SEVS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12362 --nproc_per_node=1 opengait/main.py \
            --phase test \
            --cfgs $CONFIG_FILE \
            --save_name $SAVE_NAME \
            --dataset_root "$BASE_DATASET_ROOT/$PERTURB/sev$sev"
    done
done

# ======== SwinGait ========
echo "*********************************************************************************"
echo "** SwinGait on CASIA-B (SCHP), robustness test **"
echo "*********************************************************************************"
CONFIG_FILE="./configs/swingait/swingait3D_B1122C2_casiab.yaml"
SAVE_NAME="SwinGait_schp"
for PERTURB in "${PERTURBS[@]}"; do 
    for sev in "${SEVS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12363 --nproc_per_node=1 opengait/main.py \
            --phase test \
            --cfgs $CONFIG_FILE \
            --save_name $SAVE_NAME \
            --dataset_root "$BASE_DATASET_ROOT/$PERTURB/sev$sev"
    done
done
