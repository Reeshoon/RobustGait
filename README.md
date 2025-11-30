# RobustGait

Official repository of WACV 2026 paper "RobustGait: Robustness Evaluation for Appearance-Based Gait Recognition". \
This repository provides a complete framework to evaluate how appearance-based gait recognition models behave under realistic degradations.  
Traditional robustness tests apply noise directly to silhouettes, which bypasses the effects of segmentation errors.  
RobustGait instead applies noise at the RGB level, allowing distortions to naturally propagate through segmentation/parsing and into downstream gait models—better reflecting real-world deployment.


---

## Data Generation: Noise Pipeline and Silhouette Extraction

A modular pipeline to:
- Generate **digital**, **environmental**, **temporal**, and **occlusion-based** perturbations  
- Apply 15 noise types × 5 severity levels to RGB videos  
- Extract silhouettes after corruption (any segmentation model can be used; SCHP is provided as an example)

This produces **clean**, **noisy**, and **augmented** gait datasets used in our benchmark.

### Running the Pipeline

Run the main generation script:

```
python robustgait_noise_generation/run.py \
    --dataset casia \
    --input_path /path/to/input \
    --output_path /path/to/save \
    --noise_type gaussian \
    --severity 3
```
---

## RobustGait Evaluation

OpenGait extension for robustness evaluation of gait models.  
This includes:
- Training gait models (e.g., GaitSet, GaitGL, GaitBase, DeepGaitV2) 
- Testing them on noisy, perturbed, or augmented silhouettes  
- Cross-parser, cross-gallery, and cross-dataset robustness evaluation  
- Improvement methods such as noise-aware training and robustness-oriented distillation  
- Evaluating performance degradation and robustness curves across corruption types and severities

### Installation

```
git clone https://github.com/<your-username>/RobustGait.git
cd RobustGait
pip install -e .
```

### Robustness Evaluation
```
bash scripts/robustness_eval.sh
```


---

## Acknowledgements

This repository borrows code from the following open-source projects:

- [ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)



