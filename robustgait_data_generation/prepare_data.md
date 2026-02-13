# Robust Gait Noise Generation

This directory provides the pipeline for generating **noisy or perturbed silhouettes** for gait-recognition experiments.  
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

### Download the COCO Annotation JSON File

The noise generation pipeline requires the COCO-style annotation file:

- `instances_train2017.json`

📌 Download it from the official COCO website:

https://cocodataset.org/#download

After downloading, place the file in the same directory as `run.py`.

### Pretrained Parsing Model
Download the SCHP pretrained human parsing checkpoint (e.g., `resnet101-imagenet.pth`) from the official Google Drive folder:  
https://drive.google.com/drive/folders/1uOaQCpNtosIjEL2phQKEdiYd0Td18jNo  
Place the downloaded `.pth` file inside `robustgait_noise_generation/checkpoints/` and provide its path via `--model_restore`.


## Running the Pipeline

Run the main generation script:

```
python run.py \
    --dataset casia \
    --input_path /path/to/input \
    --output_path /path/to/save \
    --noise_type gaussian \
    --model_restore /path/to/model \
    --severity 3
```
