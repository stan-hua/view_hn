# Supervised Contrastive Learning for Improved View Labeling of Ultrasound Videos

## Context

This project aims to use self-supervised learning (SSL) to learn features that may be beneficial for the task of automated view labeling of frames in a renal ultrasound video.

Repository contains code for:
 a) pre-training with SSL methods (SimCLR, MoCo, TCLR) with custom losses,
 b) fine-tuning of SSL-pretrained models,
 c) exploratory data analysis and data visualization,
 d) model evaluation, and
 e) clinical model evaluation with a downstream task (surgery prediction).

---

## Quick Setup

#### Clone directory
```
git clone https://github.com/stan-hua/view_hn.git
cd view_hn
```

#### 1. Install dependencies
```
# Conda
conda create --name view python=3.9.19
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -f environment.yml

# Pip and Venv
python -m venv view
source view/bin/activate
pip install -r requirements.txt
```

#### 2. Create symbolic link to data directory
```
# (In a console with admin priviliges)
# 1. In Windows
mklink /d ".\src\data" "PATH\TO\DATA\DIRECTORY"

# 2. In Linux
ln -s /PATH/TO/DATA/DIRECTORY data
```

#### 3. Update `config/constants.py` with new image directory paths (if any)
```
# -- update config/constants.py

# Update metadata paths
python -m src.data_prep.scripts.create_metadata update_img_dirs
```

#### 4. Set up Comet ML for Online Logging
```
comet init                  # create account, if not already

# Export env. variable COMET_API_KEY during shell init (~/.bashrc)
# echo "export COMET_API_KEY=insert_here" >> ~/.bashrc
```

---

## Package Layout

**TIP**: For each file, see bottom-most functions to look at more high-level code.

**EXTRA TIP**: Modules under `...` are not required for the current training/evaluation flow.

```
.
└── src
    ├── scripts                     (scripts for training/evaluation/embedding)
    │   ├── model_training.py       (to train models)
    │   ├── load_model.py           (to load models)
    │   ├── model_eval.py           (to evaluate models)
    │   ├── ssl_model_eval.py       (to evaluate SSL-pretrained models; wraps over `model_eval.py`)
    │   ├── ...
    │   ├── embed.py                (to extract embeddings for UMAP)
    │   └── grid_search.py          (to do grid search on hyperparams)
    ├── data_prep
    │   ├── dataset.py              (base class for all US data loading/splitting)
    │   ├── moco_dataset.py         (class to load data for MoCo-pretraining)
    │   ├── ssl_collate_fn.py       (to pair images for the contrastive pretraining tasks)
    │   ├── utils.py                (helper functions for metadata loading/preprocessing/splitting, image preprocessing)
    │   └── ...
    ├── models
    │   ├── moco.py                 (implementation for MoCo)
    │   ├── efficientnet_lstm_pl.py (for fully supervised or ImageNet-finetuned models)
    │   ├── lstm_linear_eval.py     (for SSL-finetuned models)
    │   ├── tclr.py                 (implementation for TCLR)
    │   └── ...
    ----------------------------------------------------------------------------------------------------------------------
    ├── data_viz                    (for eda and data visualization)
    │   └── ...
    ├── loss                        (for experimental contrastive losses)
    │   └── ...
    └── utils                       (for utilities)
        └── ...
```
