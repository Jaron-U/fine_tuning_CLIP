# Fine Tuning CLIP (Not Finished Yet)
## Introduction
This repository contains the code for fine-tuning OpenAI's CLIP model on custom datasets.  
Using [CLIP-adapted](https://github.com/gaopengcuhk/CLIP-Adapter) and [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) as the base, this repository provides a simple and easy-to-use interface for fine-tuning CLIP on custom datasets.

## Dataset
The dataset include 5 classes of images: `doughnut`, `glass cup`, `lemon`, `chinese noodle`, `chinese flute`. which can be found in [my hunggingface](https://huggingface.co/datasets/JaronU/CLIP_train_dataset)

where I separated the training set from the test set. The training set has a total of 2500 images, 500 for each class. And the test set contains 2000 images, 200 per class, which also contains 1000 obfuscated images. So the final evaluation metrics are precision and recall.

## Installation
```bash
# Clone this repo
git clone git@github.com:Jaron-U/fine_tuning_CLIP.git
cd fine_tuning_CLIP/

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

## Usage
### Download the dataset
```bash
# goto the ./datasets/my_dataset.py and modeify the path of the dataset you want to save
python ./datasets/my_dataset.py
```

### Fine-tuning
```bash
chmod +x ./scripts/run.sh
./scripts/run.sh
```

### Evaluation
I am still working on the evaluation part, so it is not available yet...😇

