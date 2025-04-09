
# Co-DETR Demo Code Analysis

1. [실행 목적](#실행-목적)  
2. [주요 구성 요소 및 기능](#주요-구성-요소-및-기능)  
3. [실행 방법](#실행-방법)  
4. [데이터셋 구조](#데이터셋-구조)  
5. [최소 사양](#최소-사양)  
6. [참고 사항](#참고-사항)  

<br>

## 실행 목적
- **고해상도 이미지(예: 위성 사진)** 에 대해 Faster R-CNN과 같은 객체 탐지 모델을 사용하여 슬라이스 단위로 추론을 수행하고, NMS로 병합하여 최종 결과를 생성.

- 큰 이미지를 작은 패치로 나누고 추론 결과를 합치는 방식이라 VRAM 부족 문제를 회피하면서도 정밀한 탐지가 가능.

<br>

## 주요 구성 요소 및 기능

1. **이미지 슬라이싱**
   - **SAHI의 `slice_image` 함수 사용**
   - `patch_size`만큼 자르고, `patch_overlap_ratio`로 겹침 비율 설정
   - 예: `patch_size=640`, `patch_overlap_ratio=0.25`

2. **모델 초기화 및 추론**
   - **`init_detector(config, checkpoint)`** 사용
   - **슬라이스 단위로 추론** 수행 (batch로 진행 가능)

3. **결과 병합**
   - **`shift_predictions`** 로 패치 위치에 맞게 좌표 보정
   - **`merge_results_by_nms`** 로 전체 결과 병합

4. **시각화 및 저장**
   - **`VISUALIZERS`** 사용해 결과 저장 및 시각화

<br>

## 실행 방법

### 설치

1. **필요한 라이브러리 설치**:
   - PyTorch ≥ 1.8
   - CUDA ≥ 9.2 (GPU 사용 시)
   - MMCV 및 MMEngine
   - SAHI: `pip install sahi`

2. **MMDetection 설치**:
   - MMDetection과 해당 버전에 맞는 MMCV 및 MMEngine을 설치
  
| MMDetection 버전 |      MMCV 버전       |     MMEngine 버전     |
| :--------------: | :------------------: | :-------------------: |
| main, 3.3.0, 3.2.0 | mmcv>=2.0.0, <2.2.0 | mmengine>=0.7.1, <1.0.0 |
| 3.1.0, 3.0.0     | mmcv>=2.0.0, <2.1.0 | mmengine>=0.7.1, <1.0.0 |
| 3.0.0rcx         | mmcv>=2.0.0rc1, <2.1.0 | mmengine>=0.1.0, <1.0.0 |

### 실행


```bash
python demo/large_image_demo.py {input image 경로} {configs file 경로} {checkpoint file 경로} --out {output image 이름}
```


**예시**:


```bash
python demo/large_image_demo.py demo/demo.mp4 configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py faster_rcnn_fpn_2x_coco_bbox_map-0.398_20200504_210455-1d2da9c.pth --out output_SFU.png
```


### 입력 및 출력

- **입력**

| 항목 | 설명 |
|------|------|
| 이미지 / 영상 경로 | `demo/demo.jpg`, `demo/demo.mp4` 등 |
| Config 파일 | `configs/xxx/xxx.py` (모델 구성 파일) |
| Checkpoint 파일 | 사전 학습된 `.pth` (예: `faster_rcnn_xx.pth`) |
| 옵션 | `patch_size`, `batch_size`, `score_thr`, `merge_iou_thr`, `show`, `debug`, `save_patch` 등 |

- **출력**

| 항목 | 설명 |
|------|------|
| 결과 이미지 | 추론 결과가 그려진 이미지가 `--out-dir` 경로에 저장됨 |
| 디버그 이미지 | 각 패치에 대한 탐지 결과 (`--debug`, `--save-patch` 설정 시) |
| 병합 결과 이미지 | 전체 이미지에서 추론된 bbox 및 클래스 표시 이미지 |

<br>

## 데이터셋 구조

```plaintext
Co-DETR
└── data
    ├── coco
    │   ├── annotations
    │   │      ├── instances_train2017.json
    │   │      └── instances_val2017.json
    │   ├── train2017
    │   └── val2017
    │
    └── lvis_v1
        ├── annotations
        │      ├── lvis_v1_train.json
        │      └── lvis_v1_val.json
        ├── train2017
        └── val2017
```

<br>

## 최소 사양

> 고해상도 이미지 슬라이싱 및 병합 과정이 있기 때문에, VRAM과 CPU 메모리 사용량이 비교적 높은 편

### 라이브러리 버전

- **Python**: ≥ 3.7
- **PyTorch**: ≥ 1.8
- **CUDA**: ≥ 9.2

### 시스템 사양

- **GPU**: ≥ 8GB VRAM (권장)
- **CPU**: ≥ Intel i5
- **RAM**: ≥ 8GB
- **운영체제**: ≥ Ubuntu 18.04 또는 Windows 10


<br>

## 참고 사항 

- **SAHI 설치 필요**: 슬라이싱 기반 탐지를 위해 필요 (`pip install sahi`)
- **슬라이스 단위로 병렬 처리 가능**: `--batch-size` 증가로 추론 속도 향상 가능
- **추론 결과 품질 향상**을 위해 `--patch-overlap-ratio` 조정 필요 (기본: 0.25)
- **TTA(Test Time Augmentation)** 옵션 사용 시 config에 `tta_model`, `tta_pipeline` 정의 필요
- **디버그 모드**를 활용하면 추론 전후 과정 시각화를 통해 성능을 점검할 수 있음

