
# CO-DETR 환경 세팅 및 demo code 실행

> MMDetection 기반의 대용량 이미지 객체 탐지 demo code를 실행하기 위한 환경 세팅 및 실행 절차


##  권장 환경

| 항목        | 권장 버전             |
|-------------|------------------------|
| OS          | Ubuntu 20.04           |
| Python      | 3.8 ~ 3.10             |
| CUDA        | ≥ 9.2        |
| PyTorch     | 1.10 ~ 2.0.x            |

> ⚠ 현재 실행 환경
> - CUDA: 11.8
> - PyTorch: 1.13.1
> - MMDetection: 2.25.3
> - MMCV: mmcv-full==1.5.0



## Docker 파일 및 환경 스크립트 작성

### 1. Dockerfile

```bash
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt -y update && apt-get -y install python3-pip python3-dev python3-setuptools
RUN pip3 install --upgrade pip
RUN apt-get install -y python3-dev python3-venv python3-tk

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-cache policy python3.8
RUN apt install -y python3.8
RUN apt install -y python3.8-dev python3.8-venv python3.8-tk 

COPY set_env.sh /workspace/set_env.sh

WORKDIR /workspace

```
### 2. set_env.sh

```bash
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
```

## Docker 이미지 빌드
```bash
docker buildx build -t co-detr .
```
> ⚠️ buildx가 없다면 일반 docker build -t co-detr .로 대체 가능

## Docker 컨테이너 실행
```bash
docker run --gpus all -it --name co-detr-container co-detr bash
```

## 컨테이너 내에서 환경 세팅 실행
```bash
cd /workspace
chmod 777 set_env.sh
./set_env.sh
```

## demo code 실행

### 필요한 파일 다운로드
```bash

# 데모 코드
cd Co-DETR
wget https://raw.githubusercontent.com/Sense-X/Co-DETR/main/video_demo.py

# 테스트 이미지
wget https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4 -O demo/demo.mp4

# Config 파일
wget https://raw.githubusercontent.com/Sense-X/Co-DETR/main/projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py -P projects/configs/co_dino/

# 학습된 모델 가중치(check point)
wget https://github.com/Sense-X/Co-DETR/releases/download/v1.0/co_dino_5scale_r50_1x_coco.pth

```

 
## 데모 실행
```
python3 video_demo.py demo/demo.mp4 projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py co_dino_5scale_r50_1x_coco.pth --out outputimage.mp4 --show
```
> 정상적으로 실행되면 입력 영상에 대해 객체 탐지 결과가 시각화된 영상이 출력

> --show 옵션을 통해 실시간으로 결과를 확인

