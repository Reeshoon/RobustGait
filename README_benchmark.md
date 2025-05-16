# Robust Gait Benchmarking on CASIA-B

This benchmark evaluates the robustness of multiple gait recognition models on the perturbed versions of the CASIA-B dataset using the [OpenGait](https://github.com/ShiqiYu/OpenGait) framework.

---

## 📦 Environment Setup

To reproduce the results, first set up the environment by following the official [OpenGait installation instructions](https://github.com/ShiqiYu/OpenGait/blob/main/docs/INSTALL.md). A summary is provided below:

```bash
# Clone OpenGait
git clone https://github.com/ShiqiYu/OpenGait.git
cd OpenGait

# Create and activate the conda environment
conda create -n deepgait python=3.8 -y
conda activate deepgait

# Install dependencies
pip install -r requirements.txt

# Install mmcv and mmdet as specified by OpenGait
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.25.0

## 📂 Dataset Structure

This benchmark expects the perturbed CASIA-B dataset to be located at:

```
/path/to/robust_gait/casiab/sil_pkl/perturb/<PERTURB>/<SEVERITY>/
```

Update the placeholder `BASE_DATASET_ROOT` in the `run_robust_benchmark.sh` script accordingly:

```bash
BASE_DATASET_ROOT="/path/to/robust_gait/casiab/sil_pkl/perturb"
```

Each folder should include test sequences under different perturbation types (e.g., `gaussian_noise`) and severity levels (`sev1`–`sev5`).

---

## 🚀 Running the Benchmark

This project includes a SLURM-compatible script that evaluates the following models:

- DeepGaitV2
- GaitBase
- GaitGL
- GaitPart
- GaitSet
- SwinGait

### 1. Make the script executable

```bash
chmod +x run_robust_benchmark.sh
```

### 2. Submit the SLURM job

```bash
sbatch run_robust_benchmark.sh
```

All outputs will be logged under `slurm_outputs/`, and models will be evaluated across all corruptions and severity levels.

---

## 📌 Notes

- Make sure the configuration files under `./configs/` exist and match the model names.
- Edit the `BASE_DATASET_ROOT` path in `run_robust_benchmark.sh` to point to your local dataset root.
- Different `--master_port` values are set for each model to avoid networking conflicts.
- You can selectively enable specific models or perturbations by modifying the script.

---

## 📝 Citation

```bibtex
@inproceedings{OpenGait,
  title     = {OpenGait: An Open-source Toolkit for Gait Recognition},
  author    = {Yu, Shiqi and others},
  year      = {2022},
  booktitle = {Proceedings of the XXX}
}
```

---

