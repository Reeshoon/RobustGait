# Robust Gait Noise Generation

This repository provides a clean and minimal pipeline for generating **noisy or perturbed silhouettes** for gait-recognition experiments.  
The main script (`run.py`) handles dataset loading, noise application, transformation, and saving outputs in a format compatible with OpenGait models.

---

## Features
- Supports **CASIA-B**, **CCPG**, and **SUSTech1K** datasets  
- Noise/robustness transformations (Gaussian noise, blur, dropout, occlusion, etc.)  
- Saves silhouettes in **64×64 PKL format**  

---

## Installation
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

Run the main generation script:

```
python run.py \
    --dataset casia \
    --input_path /path/to/input \
    --output_path /path/to/save \
    --noise_type gaussian \
    --severity 3
```
