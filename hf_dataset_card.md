---
annotations_creators:
- expert-generated
language_creators:
- found
language:
- en
license: apache-2.0
multimonolingual: false
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- image-to-image
- super-resolution
task_ids:
- image-super-resolution
- thermal-imaging
pretty_name: VisTA-SR Paired Thermal-RGB Agricultural Dataset
tags:
- agriculture
- thermal-imaging
- super-resolution
- cvpr-2024
---

# VisTA-SR: Paired Low/High-Resolution Thermal & RGB Agricultural Dataset (`Training_T4_1_2_3`)

Official dataset repository for the CVPR 2024 Workshop paper:
**"VisTA-SR: Improving the Accuracy and Resolution of Low-Cost Thermal Imaging Cameras for Agriculture"**

[[Paper HTML]](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Yun_VisTA-SR_Improving_the_Accuracy_and_Resolution_of_Low-Cost_Thermal_Imaging_CVPRW_2024_paper.html) | [[PDF]](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Yun_VisTA-SR_Improving_the_Accuracy_and_Resolution_of_Low-Cost_Thermal_Imaging_CVPRW_2024_paper.pdf) | [[GitHub Repo]](https://github.com/heesup/VisTA-SR)

## Dataset Description

This dataset consists of aligned multi-modal image triplets captured in field conditions (University of California, Davis) during the 2022 growing season across warm-season grain legume fields (Cowpea *Vigna unguiculata* and Common Bean *Phaseolus vulgaris*).

### Image Modalities & Camera Hardware
- **Low-Resolution Thermal (`IR_LOW`)**: FLIR One Pro (160x120 radiometric thermal sensor, 8-14 µm spectral range).
- **High-Resolution Ground Truth Thermal (`IR_HIGH`)**: FLIR Boson / VarioCAM HD (640x512 / 1024x768 industrial radiometric thermal sensor).
- **Visible RGB (`RGB`)**: Integrated FLIR One Pro visible camera (1440x1080 resolution).

## Dataset Structure (`Training_T4_1_2_3`)

```
Training_T4_1_2_3/
├── train/
│   ├── IR_LOW/      # Low-resolution 160x120 thermal images
│   ├── IR_HIGH/     # High-resolution ground truth thermal images
│   └── RGB/         # Paired visible RGB images
└── val/
    ├── IR_LOW/
    ├── IR_HIGH/
    └── RGB/
```

## Quickstart & Usage

### Downloading via Hugging Face `datasets`

```python
from datasets import load_dataset

dataset = load_dataset("heesup/VisTA-SR")
print(dataset)
```

### Citation

```bibtex
@inproceedings{yun2024vista,
  title={VisTA-SR: Improving the Accuracy and Resolution of Low-Cost Thermal Imaging Cameras for Agriculture},
  author={Yun, Heesup and Lo, Sassoum and Diepenbrock, Christine H and Bailey, Brian N and Earles, J Mason},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={5470--5479},
  year={2024}
}
```
