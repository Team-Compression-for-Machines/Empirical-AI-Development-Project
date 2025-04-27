#!/bin/bash

set -e

git clone https://github.com/Sense-X/Co-DETR.git

pip install --no-cache-dir --upgrade pip wheel setuptools
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-cache-dir mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install openmim timm
mim install mmdet==2.25.3

pip install fairscale==0.4.13 scipy==1.10.1 scikit-learn
pip install -U Cython

pip install yapf==0.40.1
pip install matplotlib numpy pycocotools six terminaltables fvcore tensorboard einops

