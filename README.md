# VisTA-SR: Improving the Accuracy and Resolution of Low-Cost Thermal Imaging Cameras for Agriculture

[![CVPRW 2024](https://img.shields.io/badge/CVPRW-2024-blue.svg)](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Yun_VisTA-SR_Improving_the_Accuracy_and_Resolution_of_Low-Cost_Thermal_Imaging_CVPRW_2024_paper.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/heesup/VisTA-SR)

Official PyTorch implementation of **VisTA-SR** (Visual & Thermal Alignment and Super Resolution Enhancement), published at the **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2024**.

> **Authors:** Heesup Yun, Sassoum Lo, Christine H. Diepenbrock, Brian N. Bailey, J. Mason Earles  
> *University of California, Davis*  
> [[Paper PDF]](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Yun_VisTA-SR_Improving_the_Accuracy_and_Resolution_of_Low-Cost_Thermal_Imaging_CVPRW_2024_paper.pdf) | [[Hugging Face Dataset]](https://huggingface.co/datasets/heesup/VisTA-SR)

---

## 🌟 Overview

Thermal cameras are essential in agricultural research for non-invasive plant temperature measurement, crop water stress indexing, and biophysical modeling. However, high-resolution industrial thermal cameras cost over $10,000, limiting accessibility.

**VisTA-SR** solves this limitation by combining low-cost consumer thermal imagery (e.g. FLIR One Pro) with visible light RGB images. The framework consists of two main stages:
1. **Stage 1: Domain Transfer and Alignment**: Translates RGB images into the thermal domain using CycleGAN and aligns the images via normalized cross-correlation template matching.
2. **Stage 2: Super-Resolution Fusion Network**: Fuses aligned RGB, domain-translated thermal, and low-resolution thermal inputs through a ResNet generator to enhance thermal resolution and sharpness.

![VisTA-SR Architecture](https://raw.githubusercontent.com/heesup/VisTA-SR/main/assets/vista_sr_architecture.png)

---

## 🛠️ Software Environment & Setup

### Requirements
- Python >= 3.9
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV, NumPy, SciPy, Pillow, tqdm, PyYAML

### Setup Options

#### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate vista-sr
```

#### Option B: Using Pip
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 📊 Dataset

The paired RGB and thermal dataset is hosted on Hugging Face: **[`heesup/VisTA-SR`](https://huggingface.co/datasets/heesup/VisTA-SR)**.

### Download Dataset
```bash
# Uploading / downloading dataset via provided script:
python scripts/upload_hf_dataset.py --dataset-dir /home/lion397/data/datasets/GEMINI/Training_All_221201 --repo-id heesup/VisTA-SR
```

---

## 🚀 Quickstart

### 1. Training VisTA-SR
```bash
python scripts/train.py \
  --data-dir /home/lion397/data/datasets/GEMINI/Training_All_221201 \
  --exp-name vista_sr_run \
  --epochs 200 \
  --batch-size 8 \
  --lr 1e-4
```

### 2. Evaluation & Metric Benchmark
```bash
python scripts/test.py \
  --data-dir /home/lion397/data/datasets/GEMINI/Training_All_221201 \
  --weights results/vista_sr_run/g-best.pth
```

### 3. Qualitative Visualization
```bash
python scripts/visualize.py \
  --data-dir /home/lion397/data/datasets/GEMINI/Training_All_221201 \
  --weights results/vista_sr_run/g-best.pth \
  --output-dir results/visualizations
```

---

## 📈 Quantitative Results

### Low-Cost Thermal Camera Temperature Calibration (Table 3 from paper)

| Parameter Mode | All Data RMSE (°C) | All Data R² | 15°C - 30°C RMSE (°C) | 15°C - 30°C R² |
|---|---|---|---|---|
| Factory Parameters | 1.52 | 0.86 | 1.52 | 0.83 |
| **Calibrated Parameters (Ours)** | **1.40** | **0.89** | **1.39** | **0.86** |

### Super-Resolution Performance Comparison (Table 4 from paper)

| Technique | RMSE (°C) | SSIM | PSNR |
|---|---|---|---|
| Bilinear Interpolation | 2.84 | 0.74 | 23.84 |
| SRGAN [31] | 2.74 | 0.63 | 24.26 |
| **VisTA-SR (Ours)** | **2.75** | **0.63** | **23.67** |

---

## 📝 Citation

If you find this work or repository useful for your research, please cite our paper:

```bibtex
@inproceedings{yun2024vista,
  title={VisTA-SR: Improving the Accuracy and Resolution of Low-Cost Thermal Imaging Cameras for Agriculture},
  author={Yun, Heesup and Lo, Sassoum and Diepenbrock, Christine H and Bailey, Brian N and Earles, J Mason},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={5470--5479},
  year={2024}
}
```

---

## 📄 License

This repository is released under the [Apache 2.0 License](LICENSE).