# End-to-End Learnable Multi-Scale Feature Compression for VCM
 
[1. Abstract](#1-abstract)  
[2. Introduction](#2-introduction)  
[3. Related Works](#3-related-works)  
[4. Proposed Methods](#4-proposed-methods)  
[5. Experiments](#5-experiments)  
[6. Conclusion](#6-conclusion) 

## 1. Abstract

딥러닝 기반의 machine vision 응용이 증가하면서, 새로운 압축 방식인 VCM(Video Coding for Machine)이 등장했다. 

본 논문에서는 기존 방식의 비효율성과 인코딩 속도 문제를 해결하기 위해 새로운 압축 방식을 제안한다. 학습 가능한 압축기(compressor)와 융합 네트워크(fusion network)를 결합하여 feature 간 중복을 제거한다. 

결과적으로, 기존 방법보다 최소 52% 더 높은 압축 성능과 최대 27배 빠른 인코딩 속도를 달성했으며, 최소한의 데이터로도 성능을 유지할 수 있음을 입증했다.
<br><br>

## 2. Introduction

### 연구 배경
 - 기존 영상 압축 기술은 HVS(Human Visual System)에 최적화되어 있었음
 - 최근에는 기계가 영상을 분석하는 경우가 많아지며 VCM(Video Coding for Machine)이 도입됨

### MPEG-VCM 연구 트랙
  **1) Track 1 (Feature Compression)** : 기계 분석을 위해 이미지에서 추출된 feature을 직접 압축하는 방식 <br>
  → 인코더에서 미리 feature을 추출하면 디코더에서의 연산 부담이 줄어듦. <br>
  → 사람이 원본 이미지를 직접 볼 수 없으므로 개인 정보 보호 가능.  <br><br>
  **2) Track 2 (Image/Video Compression)** : 기존 방식처럼 전체 영상 압축 후 분석하는 방식 <br>
  → Track 1과 비교했을때, 연산 부담이 큼.<br><br>
  **→ 본 논문은 Track 1 개선을 목표로 함.**

### 기존 연구 한계점
  - 기존 VVC(Versatile Video Coding) 기반 압축 방식은 영상 압축에 최적화되어 있으며, feature 간 중복 제거가 비효율적.
  - VVC의 복잡한 구조로 인해 경량화된 인코더를 설계하기 어려움.

### 제안 방법 및 기여 (L-MSFC)
  - 기존 연구의 한계점을 해결하기 위해, L-MSFC(Learnable Multi-Scale Feature Compression)을 제안함
  1) Multi-Scale feature 융합과 압축기(Compressor)를 하나의 네트워크로 통합하여 최적화.
  2) 대규모 feature을 먼저 압축한 후, 소규모 feature와 반복적으로 융합하여 중복 제거.
  3) 실험 결과, 기존 방법 대비 최대 98.22% BD-rate 감소, 0.002-0.003% 크기의 데이터만으로도 성능 유지.
<br><br>

## 3. Related Works
### A. MPEG-VCM Anchors
![image](https://github.com/user-attachments/assets/009b74ef-7cc3-4769-8ff5-9e830cdf7424)<br>

  **1) Track 1 anchor (Feature Anchor)** : 먼저 이미지에서 feature을 추출한 후, 이를 압축하여 저장/전송하고, 모델에 활용<br>
  **2) Track 2 anchor (Image Anchor)** : 전체 입력 영상을 VVC(Versatile Video Coding) 코덱으로 압축한 후 모델에 적용<br><br>
  **3) 적용 모델 및 구조**<br>
   - FPN (Feature Pyramid Network) 기반 Faster/Mask R-CNN 사용<br>
   - RoI Pooling/Align 기법을 적용하여 Object Detection 및 Instance Segmentation 작업을 수행<br>

### B. Feature Compression Models
![image](https://github.com/user-attachments/assets/16bbe9bb-73ac-4d24-8549-117b8131c2b2)<br>

  **1) MSFC (Multi-Scale Feature Compression) [Zhang et al.]** 
  - MSFF (Multi-Scale Feature Fusion): Multi-Scale feature을 하나의 feature map으로 결합
  - SSFC (Single-Stream Feature Codec): 단일 스트림으로 압축 수행
  - MSFR (Multi-Scale Feature Reconstruction): 압축된 특징을 복원하는 과정에서 업샘플링 
  
  **2) S-MSFC (Standard codec-based MSFC) [Kim et al., Han et al.]** 
  - 기존 MSFC에 VTM(VVC Test Model) 추가하여 압축 성능 향상시킨 모델
  - Han et al: SSFC의 활성화 함수를 PReLU로 변경하고 SSFC의 인코더와 디코더 사이에 VTM 추가
  - Kim et al: SSFC를 완전히 VTM으로 교체하고 MSFF에 Bottom-up Pathway 추가
  - 결과적으로 Object Detection에서 BD-rate 최대 96% 감소, Instance Segmentation에서 93% 감소.
  
  **3) end-to-end trainable model [Zhang et al.]**
  - 딥러닝을 이용한 학습 가능한 압축기(compressor)를 사용
  - feature을 0~1 범위로 정규화 후 입력, 각 Multi-Scale 특징 맵을 개별적으로 압축
  - VTM이나 특징 결합(Fusion) 과정 없이도 92% BD-rate 감소 달성

  **4) Other approaches for feature compression [Zhang et al.]**
  - PCA 기반 압축(Lee et al.): feature을 주요 성분으로 변환 후 압축 (DeepCABAC 사용)
  - Super-Resolution 기반 압축(Kang et al.): 특징 맵을 낮은 해상도로 압축한 후, 디코딩 시 SR 모델을 사용해 원래 해상도로 복원.

### C. Learned Image Compression
  **1) Hyperprior Model [Ballé et al.]**
  - 오토인코더(Autoencoder) 기반 이미지 압축 기법을 제안.
  - entropy 코딩으로 압축 성능을 개선,  이후 연구의 핵심 기반이 됨

  **2) Autoregressive Context Model [Minnen et al.]**
  - 이미 복원된 데이터로부터 추가적인 정보를 추출하여 더 효율적으로 압축.
  - 기존 HEVC(High Efficiency Video Coding)보다 우수한 성능을 보임

  **3) Transformer-based Compression [Lu et al., Zhu et al.]**
  - Attention 메커니즘과 Gaussian mixture model (GMM)을 적용하여 VVC 내부 코딩과 유사한 성능을 보임
<br><br>

## 4. Proposed Methods
### A. Overview
![image](https://github.com/user-attachments/assets/ae39a9fd-ba81-4e09-80e4-5aa2452bb5a0)<br>
본 논문에서는 기존 방식의 한계를 극복하기 위해 Learnable Multi-Scale Feature Compression (L-MSFC)기법을 제안함. <br>
기존 방식과 비교했을 때, 엔드투엔드 학습이 가능하며, Multi-Scale Feature 간 중복을 효과적으로 제거할 수 있는 방식임. <br>
제안된 모델의 주요 구성요소로는 FENet (Fusion and Encoding Network)과 DRNet (Decoding and Reconstruction Network)이 있음

### B. Feature Fusion and Encoding Network (FENet)
![image](https://github.com/user-attachments/assets/ea2db3bc-7284-45ea-9e01-414045a1f3ed)<br>

  **1) 기존 방식과의 차별점**
   - 기존 방식은 Multi-Scale 특징을 하나로 융합 후 압축하지만, 계산량이 많고 비효율적
   - FENet은 적은 수의 계산량으로 두 기능(fusion, encoding)을 수행할 수 있는 모델

  **2) 구조 및 동작 방식**
   - 하위 계층의 특징 맵(latent representation)을 상위 계층과 순차적으로 융합하며 인코딩 수행<br><br>
    1) 최하위 계층 특징 맵 p2 → 첫 번째 인코딩 블록에서 잠재 표현 y2 생성<br>
    2) y2와 p3를 채널별 연결(channel-wise concatenation)→ 두 번째 인코딩 블록에서 y23생성<br>
    3) 같은 방식으로 y2345까지 반복 수행하여 최종 인코딩된 잠재 표현 y획득<br>

  **3) 주요 구성 요소**
   - **Residual Blocks (Resblock)** <br>
     -> 첫 3개 인코딩 블록은 Residual Block을 사용<br>
     -> 3×3 커널, 스트라이드 1, LeakyReLU 활성화 함수, Residual 연결 적용<br>
  
   - **Down-sampling Residual Block (Resblock 2↓)** <br>
     -> 첫 번째 합성곱 레이어의 Stride=2로 설정 → 다운샘플링 수행<br>
     -> 마지막 LeakyReLU 대신 GDN (Generalized Divisive Normalization) 적용<br>
     -> Residual 연결에 1×1 합성곱 레이어 (Stride=2) 적용(Identity function 대체).<br>

  **4) GDN(Generalized Divisive Normalization)의 활용**
   - GDN은 자연 이미지 데이터의 분포를 정규화하는 비선형 변환 기법
   - 학습 기반 이미지 및 특징 압축 모델([8])에서 널리 사용됨
   - FENet의 입력 특징 맵 역시 자연 이미지에서 추출된 것이므로, GDN 적용이 적절함<br><br>


### C. Feature Decoding and Reconstruction Network (DRNet)
![image](https://github.com/user-attachments/assets/feb358e3-0dd8-4244-8846-2a9c3b7fa2c9)<br>

  **1) 주요 구조**
  - FENet에서 압축된 잠재 표현(bitstream)을 공유된 엔트로피 모델 기반의 AD로 복원<br>
  - 복원된 y에 간소화된 Attention Module을 적용.<br>
  - 이후 계층별 브랜치를 통해 다중 해상도(feature) 맵 p2,p3,p4,p5 복원<br>

  **2) 주요 구성 요소**
  - Up-sampling Residual Block (Resblock 2↑)<br>
   -> 기존 Residual Block의 첫 번째 레이어를 3×3 서브픽셀 컨볼루션 (stride=2)으로 대체<br>
   -> 마지막 활성화 함수를 IGDN (Inverse GDN)으로 변경<br>
   -> Residual 연결에도 3×3 서브픽셀 컨볼루션(stride=2) 적용<br>
  - Feature Mixing Block <br>
   -> 고해상도 특징 맵 pH과 저해상도 특징 맵 pL을 결합하여 상위 계층 특징 재구성<br>
   -> pH → 5×5 컨볼루션(stride=2) → 저해상도 특징 맵 pL과 연결<br>
   -> 연결된 특징 맵을 3×3 컨볼루션(stride=1)으로 변환후, pL과 Residual 연결하여 최종 출력 생성<br><br>


### D. Entropy Model
  - 본 논문의 Entropy Model은 딥러닝 기반으로 학습 가능하게 설계된 확률 모델
  - 기존 연구 중 Autoregressive Context Model과 Hyperprior Model을 결합
  - 실험 결과, 기존 방법 대비 최소 52% BD-rate 감소, 0.002-0.003% 크기의 데이터만으로도 성능 유지.
### E. Training Loss
  - Task-specific loss(작업별 손실) : 기계 학습 성능 극대화 ->훈련 시간이 4배이상 걸림
  - reconstruction loss(재구성 손실) : 일반적인 특징 압축에 적용 -> 본 논문에서 선택
  - 모델 학습을 위한 Rate-Distortion Loss 적용 <br>
![image](https://github.com/user-attachments/assets/1b36767f-0f79-4506-b98d-e56a76fca142)<br>

  - R : Rate(압축률), Entropy 모델을 사용해 압축된 데이터의 비트레이트를 최소화
  - Dtotal : Distortion(재구성), 원본 특징과 압축된 특징 간의 차이를 최소화(MSE 사용) 
  - λ : 비율 조정 파라미터, 높은 값이면 재구성 우선, 낮은 값이면 압축률 우선<br>
  
  **-> 실험 결과, 작업에 따른 손실 없이 충분히 높은 속도와 성능을 달성함**
<br><br>

## 5. Experiments
### A. Training and Implementation Details 
  **1) Datasets** : OpenImages V6의 90000개 이미지(512x512 이상)를 랜덤 샘플링하여 학습<br>
  
  **2) Models**<br>
   - 기존 MPEG-VCM 앵커 모델 (Feature Anchor, Image Anchor)
   - 기존 VVC 기반 MSFC모델 [kim et al, Han et al]
   - 기존 학습 기반 압축 모델 [Zhang et al]<br>
   
  **3) Training settings**<br>
   - 손실 가중치 설정: 다양한 비율-왜곡(trade-off) 결과를 얻기 위해 λ 값을 6개 설정
   - 학습 환경 : Adam optimizer 사용 (PyTorch 기본 설정)
   - 배치 크기 : 4 
   - 초기 학습률 : 1e-4 (최소 학습률 5e-6 이하가 되면 중단)
   - 파인 튜닝 : 배치크기 1, 학습률 1e-5로 200,000번 반복학습
### B. Experimental Results
  **1) Evaluation Conditions**
   - BD-rate(압축 성능 평가) : 낮을수록 좋음
   - bpp(bits per pixel) : 비트당 픽셀수
   - mAP(Mean Average Precision): 객체 탐지 성능 
   - RNL(Near-lossless bitrate) : 원본 성능과 1% 이하 손실을 허용하는 최소 비트레이트
   - CRNL(Compression Ratio at Near-Lossless bitrate) : 원본 대비 압축률
<br>

  **2) Compression Performance** <br>
![image](https://github.com/user-attachments/assets/ff8ca359-0513-4869-b956-c1a1ff62fd0e)<br>
   - Object Detection <br>
    -> Image Anchor 대비 98.22% BD-rate 절감.<br>
    -> Feature Anchor 대비 93.95% BD-rate 절감.<br>
    
   - Instance Segmentation<br>
    -> Image Anchor 대비 98.85% BD-rate 절감.<br>
    -> Feature Anchor 대비 95.79% BD-rate 절감.<br>
   **- 제안 모델이 기존 방법 대비 최소 52% 더 높은 BD-rate 감소율을 기록**
<br>

  **3) Near-Lossless Performance** <br>
![image](https://github.com/user-attachments/assets/4c1a99c1-d35e-4bb7-9140-8ac253b71a80)<br>
   - Object Detection<br>
    -> CM 사용 모델 → 27,555배 압축 (원본 대비).<br>
    -> CM 미사용 모델 → 31,505배 압축.<br>
   - Instance Segmentation<br>
    -> CM 사용 모델 → 44,426배 압축.<br>
    -> CM 미사용 모델 → 38,680배 압축.<br>
   - 제안 모델이 기존 방법 대비 27,555배 압축하면서도 성능 손실이 거의 없음
<br>

  **4) Complexity**<br>
    - 인코딩 및 디코딩 시간 비교 (제안된 모델 vs. 기존 연구 [Kim et al.])<br>
    - GPU : NVIDIA GeForce RTX 3090, CPU : AMD EPYC 7313<br>
     **-> 제안된 모델은 기존 모델보다 인코딩 시간이 크게 단축됨** <br>
     **-> 경량형 인코더 설계가 가능하여 저성능 엣지 디바이스(CCTV 등)에서도 활용 가능.**
     
### C. Additional Analyses 
  **1) Feature Mixing Pathways in DRNet**
![image](https://github.com/user-attachments/assets/e12029f6-6206-4ad9-8ea1-7723515fc212)<br>

   - 제안된 DRNet은 기본적으로 하위 계층 특징을 상위 계층으로 전달(bottom-up) 하는 방식 사용.<br>
     -> 실험을 통해 top-down 방식과 비교<br>
   **- 결과 : Bottom-up 방식이 Top-down 방식보다 BD-rate 17.50% 더 우수**
<br>

  **2) Relationship between Distortion and Task Performance**
   - 실험 결과, 왜곡(Dtotal)이 증가할수록 객체 탐지 성능(mAP)이 점진적으로 감소.<br>
   -> 제안된 모델의 Rate-Distortion Optimization가 효과적으로 작동함을 입증.
<br>

  **3) Layer-wise Distortion Analysis**
  ![image](https://github.com/user-attachments/assets/fd5d3f8a-5043-4a81-b624-a5711fe9bf5f)<br>
   - 각 feature map layer의 왜곡이 객체 탐지 성능에 미치는 영향을 분석.
   - 실험 방식 <br>
    -> 특정 계층(pi)만 압축하여 왜곡을 추가한 후, 객체 탐지 성능(mAP) 비교.<br>
    -> MS-COCO 데이터셋(5,000개 이미지) 사용, 다양한 크기의 객체 포함.<br>
    -> IoU 기준: 0.5~0.95 적용.<br>
   - 결과<br>
    -> Small object detection : 고해상도 특징(p2)이 가장 중요한 역할을 함<br>
    -> Large object detection : 저해상도 특징(p4,p5)이 성능에 큰 영향을 줌<br>
    -> 특정 object 크기에 따라 다른 계층의 중요도가 다름을 확인
<br><br>

## 6. Conclusion 
제안된 모델은 가장 높은 압축 성능과 빠른 인코딩 속도를 제공하며, Near-lossless 성능을 유지하면서도 최소 20,000배의 압축률을 달성했다. 이를 통해 객체 탐지 및 인스턴스 분할 작업에서 기존 방법 대비 우수한 성능을 입증했다. 향후에는 객체 추적 등 비디오 작업으로 확장하고, 시간적 중복 제거 기술을 연구할 예정이다.
<br><br>
