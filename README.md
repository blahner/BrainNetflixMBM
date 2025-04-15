# BrainNetflixMBM
Masked Brain Modeling code for "Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals". 

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03816.pdf) | [Project Page](https://blahner.github.io/BrainNetflixECCV/)

## Usage

### 🔧 Inference with a Pretrained Model

This script runs inference on the BMD dataset using a pretrained Masked Autoencoder (MAE) model for fMRI.

#### 1. Prepare your environment

Make sure you have the required dependencies installed:

```bash
pip install torch numpy tqdm
```

Ensure your project directory includes:

- `dataset.py`
- `sc_mbm/mae_for_fmri.py`
- `roi_list/roi_list_reduced41.txt` (or your own ROI file)
- Pretrained model file (e.g., `sub-03.pth`)

#### 2. Prepare the Dataset

To prepare the data, download the BMD dataset (if you have access) and organize it in the following structure:
  ```
  /path/to/bmd_dataset/
  ├── sub-01/
  │   └── Group41_betas-GLMsingle_type-typed_z=1.pkl
  ├── sub-02/
  │   └── ...
  └── ...
  ```
  Each subject folder should contain preprocessed ROI data saved as `.pkl` files.

TODO: @blahner Provide the access to the dataset.

#### 3. Run inference

Use the following command to run the inference script:

```bash
python mbm_inference.py \
--pretrain_model ./pretrained_model/sub-03.pth \
--dataset_path /path/to/bmd_dataset/
```

#### Optional Arguments

| Argument           | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `--pretrain_model` | Path to the pretrained `.pth` model file. *(Required)*                     |
| `--dataset_path`   | Path to the BMD dataset. *(Default: `/path/to/bmd_dataset/`)*                    |
| `--roi_list`       | Path to the ROI list file. *(Default: `./roi_list/roi_list_reduced41.txt`)* |

This code is adapted from the excellent [mind-vis](https://github.com/zjc062/mind-vis)  repository. Please check out their work for further details on masked brain modeling.

## Pretrained models
You can download pretrained models in this Google drive [link](https://drive.google.com/drive/folders/1yt7JqVm5tv13JEx--FRfFMPYV9ENF8Y3?usp=sharing). Each model is pre-trained on 41 RoI groups of one subject's training split.

## Citation
```
@inproceedings{fosco2024brain,
  title={Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals},
  author={Fosco, Camilo and Lahner, Benjamin and Pan, Bowen and Andonian, Alex and Josephs, Emilie and Lascelles, Alex and Oliva, Aude},
  booktitle={European Conference on Computer Vision},
  pages={457--474},
  year={2024},
  organization={Springer}
}
```