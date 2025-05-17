---
layout: post
title: "Grounding DINO 논문 공부!"
author: [DrFirst]
date: 2025-05-15 15:00:00 +0900
categories: [AI, Research]
tags: [grounding, grounding dino, grounded sam, DINO, computer vision, AI ,ECCV, ECCV 2024, DETR]
sitemap :
  changefreq : monthly
  priority : 0.8
---


---

## (한국어) 📝 Grounding DINO 알아보기!!
_『Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection』(ECCV, 2024) 공부_

📖 **논문 제목**: Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection  
✍️ **저자**: Xinyu Chen, Xueyan Zou, Ze Liu, et al.  
🌟 **한줄 요약**: 제시된 텍스트 프롬프트 기반 객체 탐지기!

- 오늘은 [실습을 먼저 진행해보았던](https://drfirstlee.github.io/posts/groundingDINO_Detection_usage/) Grounding DINO 모델의 논문을 공부해보고자합니다!!  

---

### 🧠 핵심 아이디어 요약

#### 1️⃣ DINO 기반 구조와 모달 융합 강화

![detector_structure]()

- Grounding DINO는 **Transformer 기반 객체 탐지기인 DINO**를 기반으로 설계됨.
- 기존 Faster R-CNN 구조와 달리, DINO는 **텍스트와 이미지 간 layer-level 융합**이 자연스럽게 가능한 구조를 가짐.
- **Neck(phase A), Query Initialization(phase B), Head(phase C)** 단계 모두에서 **cross-modality fusion**이 이루어지도록 설계하여, 텍스트 기반 객체 탐지 성능을 향상시킴.

#### 2️⃣ Grounded Pretraining을 통한 Open-Set 일반화
- CLIP은 이미지 전체 수준에서는 뛰어나지만, 영역(region) 수준 텍스트 대응에는 한계가 존재.  
- 이런 CLIP 기반 zero-shot 방식의 한계를 극복하기 위해, **region-text 쌍에 대한 contrastive pretraining**을 도입.
- GLIP의 phrase grounding 방식을 개선하여, **sub-sentence 단위 텍스트 처리**를 통해 클래스 간 간섭을 줄임.
- 이로써 Grounding DINO는 **“텍스트 → 탐지”가 가능한 open-set object detector**로서, COCO 및 ODinW 등에서 **zero-shot 성능의 새로운 기준**을 제시함.


---

### 🔍 Grounding DINO 연구의 배경

Grounding DINO는 기존의 객체 탐지(Object Detection) 모델들이 가진 **고정된 클래스 제한**을 뛰어넘기 위해 제안된 모델입니다.  
이전까지의 흐름은 다음과 같습니다:

---

#### 🧩 DETR 이후 DINO, 하지만 여전히 클래스는 고정

- **[DETR](https://drfirstlee.github.io/posts/DETR/) (2020, Facebook AI)**  
  Transformer 기반으로 객체 탐지를 수행한 최초의 end-to-end 모델  
  → 하지만 클래스는 COCO처럼 **사전 정의된 클래스셋에 한정**됨

- **[DINO](https://drfirstlee.github.io/posts/DINO_Detection/) (ICLR 2023)**  
  DETR 구조를 개선해 학습 안정성과 정확도를 높인 모델  
  → 뛰어난 성능을 보였지만 **여전히 고정된 클래스(class token)**만 탐지 가능

즉, DINO는 **탐지는 잘하지만 '무엇을 탐지할지'는 이미 정해져 있어야** 했습니다.

---

#### 🧩 Open-Set Object Detection, 즉 고정된 객체 한계를 넘어서는 연구들  

##### 🔍 GLIP, OV-DETR* 등 소개

기존 객체 탐지는 사전에 정의된 클래스(bounding box 어노테이션)에만 반응하는  
**고정 클래스 기반(closed-set)** 탐지 방식에 한정되어 있었습니다.  

이에 대해 **GLIP**(Grounded Language-Image Pre-training, Microsoft)은 다음과 같은 방향을 제시합니다:

- **오픈셋 객체 탐지 (Open-Set Object Detection)**  
- **임의의 클래스 (arbitrary class)**에 대한 탐지 수행  
- **자연어 기반 일반화 (language generalization)**를 통해 새로운 객체를 이해하고 탐지

즉, 정해진 라벨 없이도 **텍스트 프롬프트 기반으로 객체를 탐지할 수 있는 능력**을 목표로 합니다.

한편, **OV-DETR**은 Transformer 구조 기반의 객체 탐지기로,  
언어 정보가 포함된 쿼리(query)를 디코더에 직접 주입하여 open-vocabulary 탐지를 수행합니다.

---

##### ⚠️ 기존 연구들의 한계점

이러한 모델들은 모두 **이미지와 언어라는 멀티모달 정보**를  
**일부 모듈에만 국한하여 융합(fusion)**함에 따라,  
**언어 기반 일반화 성능이 최적보다 낮게(sub-optimal) 작동할 가능성**이 존재합니다.

---

##### 📊 예시: 멀티모달 결합 위치 비교

| 모델        | 멀티모달 결합 위치              | 설명                                          | 한계점 |
|-------------|----------------------------------|-----------------------------------------------|--------|
| **GLIP**    | Phase A (Feature Enhancement)    | 백본 이후 neck 단계에서 이미지-텍스트 특징 융합 | 이후 디코더와의 연결성 부족 |
| **OV-DETR** | Phase B (Decoder Input)          | 디코더에 언어 쿼리(query)를 직접 삽입           | 초기 시각 정보와의 깊은 융합 부족 |

---

➡️ 이러한 구조적 제약은,  
**텍스트와 이미지 간의 깊이 있는 정렬(alignment)이 요구되는 open-vocabulary 탐지**에서  
**성능 저하** 또는 **일반화 한계**로 이어질 수 있습니다.


---

#### 🗣️ SAM의 가능성: 텍스트 프롬프트 기반 분할

- **SAM (Segment Anything Model, 2023)**  
  포인트, 박스, 마스크 기반의 **범용 세그멘테이션 모델**  
  → *Segment Anything*이라는 이름에 걸맞게 어떤 객체든 잘라낼 수 있음

- 그러나 SAM은 **텍스트를 직접 입력해 segmentation을 수행할 수는 없었음**  
  (텍스트는 개념적으로 제시되었지만, 실제 텍스트 인식을 하지 않음)

---

### 💡 그래서 등장한 Grounding DINO!

Grounding DINO는 이러한 두 흐름을 **자연스럽게 연결**합니다:

- **DINO의 객체 탐지 능력** + **텍스트 프롬프트 해석 능력(CLIP 기반)**  
- → 결국 **"말로 탐지하는(open-vocabulary) 객체 탐지기"**가 된 것!!  

이후 SAM과 결합하여 **Grounded SAM**으로 확장되며,  
"텍스트 → 탐지 → 분할"이라는 전체 파이프라인이 완성됩니다.  

---

### 🧪 Grounding DINO의 구성


#### 📐 아키텍처 개요

Grounding DINO는 **dual-encoder + single-decoder 구조**를 채택합니다.  

구성 요소는 다음과 같습니다:

1. **Image Backbone**: 이미지 특징 추출
2. **Text Backbone**: 텍스트 특징 추출
3. **Feature Enhancer**: 이미지-텍스트 특징 융합 (Sec. 3.1)
4. **Language-Guided Query Selection**: 쿼리 초기화 (Sec. 3.2)
5. **Cross-Modality Decoder**: 박스 refinement 수행 (Sec. 3.3)

---

##### 3.1 🔧 Feature Extraction and Enhancer

- **이미지 특징**: Swin Transformer와 같은 백본을 통해 다중 스케일 특징 추출
- **텍스트 특징**: BERT 기반의 백본으로 추출
- **융합 방식**:
  - 이미지: **Deformable self-attention**
  - 텍스트: **Vanilla self-attention**
  - 크로스모달 융합: 
    - **Image-to-Text Cross-Attention**
    - **Text-to-Image Cross-Attention**
  - 다수의 Feature Enhancer Layer로 구성

👉 서로 다른 모달리티의 특징 정렬(alignment)을 위한 핵심 모듈

---

##### 3.2 🎯 Language-Guided Query Selection

Grounding DINO는 **입력 텍스트에 따라 탐지 쿼리를 동적으로 선택**합니다.

- 이미지 특징: $X_I \in \mathbb{R}^{N_I \times d}$
- 텍스트 특징: $X_T \in \mathbb{R}^{N_T \times d}$  
  (보통 $N_I > 10,000$, $N_T < 256$, $d = 256$)

- 목표: 이미지 특징에서 $N_q = 900$개의 쿼리를 선택하여 디코더에 입력
- 선택 식:

```
I_{N_q} = Top_{N_q}(Max^{(-1)}(X_I X_T^T))
```

- 선택된 인덱스로부터 쿼리 특징을 추출해 디코더에 입력
- 쿼리는 다음 두 구성으로 구성됨:
- **Positional Part**: encoder 출력으로부터 anchor box 초기화
- **Content Part**: 학습 가능한 쿼리 벡터

---

##### 3.3 🔄 Cross-Modality Decoder

- 각 디코더 레이어는 다음 블록으로 구성됨:
1. **Self-Attention**
2. **Image Cross-Attention**
3. **Text Cross-Attention**
4. **Feed-Forward Network (FFN)**

- DINO의 디코더 구조에 비해 **Text Cross-Attention 블록이 추가**됨  
→ 텍스트 정보가 쿼리 업데이트에 더 강하게 반영됨

---

##### 3.4 ✂️ Sub-Sentence Level Text Feature

- 기존 텍스트 인코딩 방식:
- **Sentence-level**: 문장을 하나의 벡터로 처리 → 정밀도 손실
- **Word-level**: 여러 단어를 한 번에 인코딩 → 단어 간 불필요한 상호작용 발생

- 문제: 텍스트가 여러 클래스명을 포함할 경우, **무관한 단어 간 상호작용(attention)이 생김**

- 해결:  
**Sub-sentence level representation** 도입  
→ 서로 다른 클래스명 사이의 attention을 **mask**하여 불필요한 상호작용 제거  
→ 단어 단위 정밀 표현 유지 + 상호 간섭 방지


---
#### 🎯 Loss의 구성

---

##### 🔧 3.5 Loss Function

Grounding DINO는 기존의 DETR 계열 모델들과 유사하게,  
다음 세 가지 주요 손실 함수를 조합하여 학습합니다:


##### 📦 1. Bounding Box Regression
- **L1 Loss**  
- **GIoU Loss** (Generalized Intersection over Union)  
- → 박스 위치 예측 정밀도 향상에 사용  
- 참고: DETR, Deformable DETR 등에서 사용된 방식과 동일

---

##### 🏷️ 2. Classification (텍스트 기반 분류)
- **Contrastive Loss** (GLIP 방식 채택)  
  - 예측된 객체와 텍스트 토큰 간의 대응 관계 학습
- **방식**:
  - 각 쿼리와 텍스트 특징 간의 **dot product** → logits 계산
  - 각 텍스트 토큰별로 **Focal Loss** 적용하여 분류 학습

---

##### 🔄 3. 매칭 및 총합 계산

- 예측값과 정답 간 **이중 이분 매칭 (bipartite matching)** 수행  
  → 박스 regression cost + classification cost 기반
- 매칭 후 최종 손실은 다음을 합산하여 계산:
  - **Bounding Box Loss (L1 + GIoU)**  
  - **Classification Loss (Focal + Contrastive)**

---

##### 🧱 4. Auxiliary Loss

- **DETR 계열 구조를 따르기 때문에**, 다음 두 위치에 보조 손실(auxiliary loss)을 추가합니다:
  - **각 디코더 레이어 출력**
  - **인코더 출력 (encoder outputs)**

➡️ 이 보조 손실은 학습 초기 안정성과 수렴 가속에 기여합니다.

---

### 📊 성능

- 다양한 Open-Vocabulary 데이터셋에서 **기존 SOTA 대비 우수한 성능**
- 특히 **RefCOCO, LVIS, COCO** 등에서 탁월한 zero-shot 성능
- 이미지 내 존재하지 않는 클래스에 대해서도 견고한 탐지 가능

---

### 🛠️ 실습 참고

- GitHub: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- Hugging Face: 사전 학습 모델 및 demo 제공
- Segment Anything 등과 조합하여 **Grounded SAM**으로도 활용 가능

---

### 💡 느낀점

Grounding DINO는 단순히 탐지를 잘하는 것을 넘어,  
**텍스트와 시각 정보를 효과적으로 연결하는 방식**을 잘 보여주는 논문입니다.  
CLIP 이후 등장한 다양한 Vision-Language 모델들과 비교해도 **유연성과 확장성 면에서 강력**한 인상을 받았습니다.

---

### 📚 참고 문헌

1. Grounding DINO paper: https://arxiv.org/abs/2303.05499  
2. Grounding DINO GitHub: https://github.com/IDEA-Research/GroundingDINO  
3. CLIP: Learning Transferable Visual Models From Natural Language Supervision

---

👉 다음 포스팅에서는 Grounding DINO를 활용한 실습 코드도 함께 정리해볼 예정입니다.  
궁금한 점이나 요청사항은 댓글로 남겨주세요!