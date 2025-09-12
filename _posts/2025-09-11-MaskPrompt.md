---
layout: post
title: "🎭 MaskPrompt: 오픈 보캐뷸러리 Affordance Segmentation을 위한 객체 마스크 프롬프트"
author: [DrFirst]
date: 2025-09-11 07:00:00 +0900
categories: [AI, Research]
tags: [Computer Vision, Affordance, Segmentation, Open-Vocabulary, Robotics, VLM, AAAI, AAAI 2025]
sitemap:
  changefreq: monthly
  priority: 0.8
---

### 🎭 (한국어) MaskPrompt: 객체 Shape Mask 프롬프트로 Open-Vocabulary Affordance Segmentation 달성!  

![Image](https://github.com/user-attachments/assets/cbe289c4-dada-435e-b2ee-2fca297c2166)  

* **제목**: [MaskPrompt: Open-Vocabulary Affordance Segmentation with Object Shape Mask Prompts](https://ojs.aaai.org/index.php/AAAI/article/view/32200)   
* **학회**: AAAI 2025  
* **저자**: Dongpan Chen, Dehui Kong, Jinghua Li, Baocai Yin (Beijing Univ. of Tech)  
* **핵심 키워드**: `Affordance`, `Segmentation`, `Open-Vocabulary`, `Mask Prompt`, `Vision-Language`  
* **요약**: MaskPrompt는 **객체의 기능 단위(affordance)**를 복잡한 장면과 열린 어휘 상황에서 정확히 분할하기 위해, **객체 마스크 기반 프롬프트**를 활용하는 새로운 방법을 제안. OVAS-25 벤치마크를 새롭게 구축하고, 기존 SOTA 대비 성능을 크게 개선! 🚀


---

### 🚀 연구 핵심 요약

> 한 줄 요약: **“MaskPrompt = 객체 마스크 + 텍스트 프롬프트로 open-world affordance segmentation 해결!”**

1) **새 과제 정의 (OVAS)**  
- **Open-Vocabulary Affordance Segmentation (OVAS)** 제안  
- 훈련 데이터에 없는 affordance까지 일반화  

2) **MaskPrompt 방법론**  
- **Mask Prompt Generation (MPGM)**: DETR + SAM으로 객체 마스크 생성, Alpha-CLIP으로 마스크 영역 캡션 생성  
- **Mask Prompt Feature Enhancement (MPFEM)**: 배경 제거 후 객체 인스턴스 feature 강화  
- **Affordance Prediction Module (APM)**: 시각 feature + 텍스트 프롬프트를 융합해 세밀한 affordance 분할

3) **벤치마크 & 실험 성능**  
- 신규 **OVAS-25 데이터셋** 구축 (28개 객체, 25개 affordance, 1.9만 이미지)  
- IIT-AFF, UMD 등 기존 데이터셋에서도 우수한 성능 달성

---

### 🔍 기존 연구의 한계와 차별점  

- 기존 Affordance Segmention 방법:  
  - Attention segmentation 을 시도했지만, 데이터가 부족했다.  
  - 그래서 Weakly supervised 로 연구도 됬다.  
  - 최근에는 3D 데이터를 기반으로한 affordance Segmentation 도 있었다.
  - 다만!! 이런 기존 연구들은 전역 feature만 활용 → 배경/인접 객체 간섭에 취약했다.

- Open-Vocabulary Image Segmentation에서는??  
  - 학습때 보지 못한 카테고리고 segmentation 하고자한다!  
  - 기존 존재하는 연구들은 이미지와 단어의 임베딩을 연결시키거나, CLIP같은 VLM을 써서 이미지-단어의 연결된 지식을 활용했다.
  - 또한 프롬포트 러닝방법 등 다양한 방법을 통해서 Segmentation 성능을 향상시킨 모델도 있었다.


---

### 🧱 MaskPrompt 구조 (Architecture)

![Image](https://github.com/user-attachments/assets/ea6f1ecc-1b08-4fc3-9a21-d01cd728d74f)  

1) **MPGM**(mask prompt generation module): 객체 마스크 + 마스크 캡션 생성  
  a. Object Shape Mask(M_os)를 만듬 : DETR(Bbox Detection) + SAM(segmentation) 으로!  
  b. Mask Caption(w_mask) 생성 : 원래 이미지 + Mask 를 Alpha-CLIP(BLIP2의 확장판)에 넣어서 마스크에 대한 캡션 만든다  

2) **MPFEM**(mask prompt feature enhancement module): 배경 제거 + 객체 중심 feature 강화  
  a. 원래 이미지를 ViT에 넣어서 Global Feature 만들고  
  b. MPGM에서 나온 Object mask(M_os)로 객채별로의 instance feature를 만든다음  
  c. 이들을 모두 Concat해서 CNN에 넣어 차원울 줄여서 Enhanced visual Feature (F_v)를 만든다.  

3) **APM**(affordance prediction module): 텍스트 프롬프트와 융합해 최종 affordance segmentation map 출력  
![Image](https://github.com/user-attachments/assets/9a828b15-9f5e-4db9-8afd-b0b4eebd48c1)

  a. 첫번째로 클래스에 매칭되는 mask를 만들고(Mask proposals)
    - 객채 이름(w_obj), Affordance 명칭(w_aff) 이랑 1-b의 w_mask를 각각 클립으로 토크나이즈로 토큰만들고,  
    - 토큰을 합쳐서 CLIP으로 임베딩한 F_t를 만든다!!  
    - 2-c에서 만든 Visual Feature F_v랑 텍스트 임베딩 F_t를 **Pixel Decoder**에 넣는다.  
    - Pixel Decoder는 F_v는 self-attention block 및 L2정규화를 지나 cross-attention block에 F_t랑 같이 들어가고,  
    - 그들은 그 다음 FFN 블록을 지나서 (L번 반복해서) F_vt라는 Feature로 만들어진다.  
  b. 두번쨰로 그 mask에 대한 affordance class를 예측한다.(Mask Class Embedding)  
    - 마지막으로 F_vt는 MLP를 지나서 클클래스에 매칭되는 mask(M_ca)랑 mask class embedding(F_cls)를 생성한다.    
    - 그리고 F_cls 랑 F_t를 dot product 해서 open set of affordance classes에 대한 점수(s_cls)를 구한다.  

  - 이때의 Loss function은 Class 구분에 대한 정확성 + mask의 정확석 을 가지고 구함!
      `L = L_cls(ˆs_cls; s_cls) + λ*L_mask( ˆm; m)`


---

### 🧪 실험 결과  

#### 실험 데이터셋  

1. OVAS-25 (본 연구에서 제안한 데이터셋)
- **구성**: IIT-AFF + Pascal-Part-108 재주석 (객체, 사람, 동물의 affordance 기준으로 라벨링)
- **클래스**: 28개 엔티티 클래스, 25개 affordance 클래스
- **규모**: 총 18,938장 (IIT-AFF 8,835 + Pascal 10,103)
  - 학습: 11,363장
  - 테스트: 7,575장

---

2. IIT-AFF (Nguyen et al. 2017)
- **클래스**: 10개 객체 카테고리, 9개 affordance 카테고리
- **규모**: 총 8,835장
  - ImageNet에서 6,496장
  - 로봇 카메라로 수집된 복잡한 장면 비디오 프레임 2,339장

---

3. Pascal-Part-108 (Michieli et al. 2020)
- **클래스**: 20개 객체 카테고리, 108개 객체 파트 카테고리
- **규모**: 총 10,103장
- 본 연구에서는 **annotation을 affordance 기준으로 변경**하여 OVAS-25 구축에 활용

---

4. UMD (Myers et al. 2015) & 기타 파트 데이터셋
- **UMD affordance dataset**
- **추가 평가 데이터셋**:
  - Pascal-Part-58 (Chen et al. 2014)
  - Pascal-Part-116 (Wei et al. 2024)
  - Pascal-Part-201 (Singh et al. 2022)
  - ADE20K-Part-234 (Wei et al. 2024)

---

#### 실험 설계 및 평가지표  


- **Object Detector**: Pre-trained DETR 사용  
  - Threshold \(T = 0.7\)  
  - DETR, SAM, Alpha-CLIP → **모두 freeze**  

- **학습 세팅**  
  - Iterations: **120K**  
  - Learning Rate: **1e-4**,  
    - 60K, 100K에서 **10배 감소**  
  - Optimizer: **AdamW**  
  - Weight Decay: **1e-4**  
  - Batch Size: **32**  

- **Pixel Decoder**  
  - Layer 수 \(L\): **6**  
  - Embedding Dimension: **768**  
  - Multi-head Attention Head 수: **12**  
  - Hidden Dimension (FFN): **3072**  
  - Feature Dimension:  
    - \(d = 512\)  
    - \(d_t, d_v, d_{vt}, d_{cls} = 512\)  

- **실험 환경**  
  - **NVIDIA A800 80GB GPU**

- 평가 지표
  - **mIoU (mean Intersection over Union)**
  - **mAvg (mean Average)**
  - **F1-Score**

#### 실험 결과 및 분석 

A 실험결과지표  
![Image](https://github.com/user-attachments/assets/39b2293d-d9bf-4e2c-a7da-0d4d1fec83ab)

1.🎯 OVAS-25 (본 논문 제안 벤치마크)  
- MaskPrompt (ResNet-101): **mIoU 71.26, F1 81.58** → 기존 SOTA 대비 **+5.27% 향상**

2. 🎯 기존 데이터셋 (IIT-AFF, UMD)  
- IIT-AFF: F1 89.46  
- UMD: F1 93.83 (기존 최고 성능 모델과 경쟁적)

3. 🎯 Part Segmentation 확장성  
- Pascal-Part-58, 108, 201, ADE20K-Part-234에서도 강력한 일반화 성능 입증  

---

B. 👀 정성 비교  

![Image](https://github.com/user-attachments/assets/2d46fde6-7ba3-427e-92f9-aaa88eb7a65b)

a. **복잡한 배경**: 기존 모델 대비 간섭 억제 성능 우수  
b. **작은 객체 부품 탐지**: 예) 병뚜껑의 “contain” affordance까지 정확히 탐지  
c. **인접 객체 처리**: 경계가 섞이는 경우에도 정밀하게 분리  

---

C. 🧪 Ablation 분석  

- 2헹: **MPFEM 을 추가** → mIoU 6.9% 향상  
- 3행: **MPGM추가** + Pixel Decoder가 텍스트를 받을 수 있도록 변형됨 (cross-attention 추가) → 추가로 mIoU +2.24%  
- 4행 : **Pixel Decoder 추가** → 최고 성능   

D. 또한 Computing power 도 적게썻다!!  
---

## ✅ 결론  

- MaskPrompt는 **open-vocabulary affordance segmentation**을 위한 새로운 접근법  
- 주요 기여:  
  1. OVAS 과제 및 **OVAS-25 데이터셋** 최초 제안  
  2. 객체 마스크 기반 **MaskPrompt 프레임워크** 개발  
  3. 다양한 데이터셋에서 SOTA 수준 성능 달성  
- → 로봇, HOI, AR/VR 등 **실세계 응용**에 중요한 기여를 할 수 있음 🎯

---
