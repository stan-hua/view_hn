# Supervised Contrastive Learning for Improved View Labeling of Ultrasound Videos

## Quick Setup

### 0. Go to project directory
```
# Go to `view_hn` repo directory
cd PATH/TO/DIRECTORY
```

### 0. (Optional) Create virtual environmeent
```
# Create environment
python -m venv VIEW_HN

# Activate environment
# 1. In Windows
VIEW_HN\Scripts\activate
# 2. In Linux
source VIEW_HN/bin/activate
```

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Create symbolic link to data directory
```
# 1. In Windows
mklink /d ".\src\data" "PATH\TO\DATA\DIRECTORY"

# 2. In Linux
ln -s /PATH/TO/DATA/DIRECTORY ./src/data
```

---

## Package Layout

**TIP**: For each file, see bottom-most functions to look at more high-level code.

**EXTRA TIP**: Modules under `...` are not required for the current training/evaluation flow.

```
.
└── src
    ├── drivers                     (scripts for training/evaluation/embedding)
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
    └── utilities                   (for easier logging)
        └── ...
```
