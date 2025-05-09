---
layout: post
title: " 🦖 DINO: DETR의 진화형 객체 탐지 모델 DINO!! (ICLR 2023)"
author: [DrFirst]
date: 2025-05-09 09:00:00 +0900
categories: [AI, Research]
tags: [DETR, DINO, 객체 탐지, Object Detection, Transformer, 딥러닝, CV, ICLR, ICLR 2023]
lastmod : 2025-05-09 09:00:00
sitemap :
  changefreq : monthly
  priority : 0.9
---

## 🦖 DINO: DETR의 진화형 객체 탐지 모델 DINO!! 
_🔍 DETR 계열 모델의 느린 학습과 작은 객체 탐지 문제를 해결한 강력한 대안!_

> 논문: [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605)  
> 발표: ICLR 2023 (by IDEA Research)  
> 코드: [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)

---

### ✅ DINO란?

> DINO는 **DETR 계열의 한계를 극복**한 객체 탐지 모델  
> 특히 **학습 속도 향상**과 **소형 객체 성능 개선**에 중점을 둔 구조로 설계  

- DINO = **DETR with Improved DeNoising Anchors**
- 기본 구조는 DETR 기반이지만, 다양한 전략으로 성능을 강화한 모델
- **One-stage** 구조지만 **Two-stage 수준의 성능**을 달성!

---

### 🚨 DINO 등장의 배경 - DETR의 주요 한계  
1. ❌ **학습이 너무 느리다** (수십만 스텝)
   - DETR은 학습 초기 단계에서 object query들이 **무작위한 위치에 박스를 예측**  
   - 이로 인해 query와 GT 간의 효과적인 매칭이 어렵고 학습 신호가 희박함  
   - → 결국 **수렴 속도가 매우 느리고**, 일반적인 모델보다 수십 배 더 많은 epoch 필요(500 epock!?)  

2. ❌ **작은 객체 탐지가 약하다**
   - DETR은 CNN backbone의 마지막 feature map만 사용하기 때문에 **해상도가 낮음**
     - (예: ResNet의 C5 레벨 feature 사용 → 해상도 축소)
   - 작은 객체는 이 coarse feature map에서 **존재 정보가 거의 사라지거나 희미하게 표현됨**
   - 또한, **Transformer는 전역적 attention에 집중**하기 때문에 로컬 디테일이 약해짐
   - → 결과적으로 **작은 물체에 대한 box 예측이 정확하지 않음**

3. ❌ **Object Query 학습 초기에 성능이 낮다**  
   - DETR의 object query는 초기에는 **random하게 초기화**되어 있음  
   - 학습 초기에 어떤 query가 어떤 객체를 예측할지 **역할이 정해져 있지 않음**  
   - Hungarian Matching이 강제로 1:1 매칭을 수행하지만, 이 매칭이 **일관성이 없음**  
   - → 학습 초기에 query들이 **서로 중복되거나 엉뚱한 위치를 예측**하는 경우가 많아 성능이 낮음  

---


### 💡 DINO의 핵심 아이디어

| 주요 구성 요소              | 설명 |
|----------------------------|------|
| 🔧 **DeNoising Training** (+CDN)   | 학습 시, GT 주위에 노이즈 박스를 일부러 생성하여 Query를 빠르게 수렴시킴 <br> DINO에서는 이를 Contrastive하게 확장하여 정답 vs 오답을 구분하는 학습(CDN)도 수행|
| 🧲 **Matching Queries**     | GT에 가까운 위치에 고정된 Query Anchor를 배치해 안정적인 학습 유도 |
| 🧠 **Two-stage 구조 추가**  | Encoder에서 coarse object 후보를 뽑고, Decoder에서 refinement 수행 |
| **Look Forward Twice**   | Decoder에서 한 번이 아니라 두 번 attention을 주는 방식으로 정확도 향상 |

---


#### 💡 해결책1: DeNoising Training (+ CDN)

DINO에서는 학습 초기에 object query들이 **정답(GT) 주변 정보를 빠르게 인식하고 학습**할 수 있도록 돕기 위해  
**“의도적으로 노이즈를 주입한 학습 샘플”을 사용**합니다.

---

##### 🔧 작동 방식

1. **Ground Truth 복제**  
   - Ground Truth box와 label을 복제하여 query target으로 사용합니다.

2. **의도적으로 노이즈 추가**  
   - 복제된 box에 위치 노이즈 (좌표 jittering)와 class 노이즈 (잘못된 label)를 추가합니다.
   - 예:
     - box 좌표를 살짝 이동시킴 (e.g., 5~10% jitter)
     - class label을 다른 label로 바꿈 (e.g., person → dog)

3. **Query 분리 학습**  
   - 전체 object query 중 일부는 denoising query로 지정되고,
   - 이 query는 원래 GT가 아닌, 노이즈가 섞인 box를 예측하도록 유도됩니다.

4. **Loss 계산에 사용**  
   - GT에 대한 matching loss 외에도, 노이즈된 query에 대해 예측 정확성을 측정하는 loss가 함께 사용됩니다.

---

##### 🧠 📌 CDN(Contrastive DeNoising) 확장

DINO에서는 이 DeNoising 전략을 더욱 확장하여, **positive와 negative query를 동시에 구성하는 Contrastive DeNoising (CDN)**을 도입합니다.

- **Positive query**:  
  - GT에서 생성된 노이즈 박스 (위치/클래스만 약간 변경된 진짜에 가까운 것)

- **Negative query**:  
  - 완전히 무관한 박스나 클래스 정보로 생성된 "틀린 예측" 후보

- 이 두 종류의 query를 모두 decoder에 넣어 학습함으로써,
  - 모델이 정답을 맞추는 것뿐 아니라,
  - **"정답과 유사한 오답을 구분하는 능력까지 학습"**하게 됩니다.

💡 즉, CDN은 단순히 빠른 수렴을 넘어서,  
**모델의 표현력과 구분 능력 자체를 강화하는 contrastive 학습 요소**입니다.

---


##### ⚙️ 구성 요소

| 요소          | 설명 |
|---------------|------|
| 🎯 Positive query | Ground truth box에 노이즈를 추가한 DeNoising 샘플 |
| ❌ Negative query | 완전히 잘못된 위치나 클래스 정보를 주입한 샘플 |
| 🧲 Matching Head | 각각에 대해 분리된 디코더에서 예측값을 얻고 학습 |
| 🧪 Loss           | Positive에는 정확히 예측하도록, Negative에는 확실히 틀리게 예측하도록 유도 |

---

##### 💡 작동 방식

1. **GT box 복제 → Positive Query**
   - 약간의 노이즈를 추가하여 GT 근처에서 시작
2. **랜덤 박스 생성 → Negative Query**
   - 클래스 오류, 위치 오류 등 의도적 혼란 삽입
3. **두 query를 같은 디코더에 넣어 예측**
4. **Loss 계산 시 Positive는 ground truth와 정렬되도록, Negative는 no-object로 분류되도록 유도**

---

##### 🧠 Contrastive 효과

- 모델이 **"이건 진짜 객체야!"** vs **"이건 헷갈리지만 가짜야!"** 를 명확히 판단하게 됨
- 특히 비슷한 배경, 작은 객체, overlap 상황에서 **오탐지 줄이는 데 기여**

---

##### ✅ 요약

| 항목        | 설명 |
|-------------|------|
| CDN 목적     | 정답과 유사한 오답을 구분하는 능력 강화 |
| Positive 샘플 | GT 주변 노이즈 추가된 query |
| Negative 샘플 | 랜덤하거나 잘못된 box/class를 가진 query |
| 학습 효과    | false positive 감소, 초기 수렴 가속화, 더 견고한 탐지 |

---

> 📌 CDN은 DeNoising Training을 **contrastive 학습 형태로 확장한 기법**이며,  
> DINO가 기존 DETR보다 더 빠르고 정확하게 수렴할 수 있게 만들어주는 핵심 기술 중 하나입니다.


#### 📈 시각적으로 표현하면:

| Query Type       | Input                     | 목표                          |
|------------------|---------------------------|-------------------------------|
| Matching Query   | GT box                    | 정확한 객체 예측              |
| Denoising Query  | GT + noise (jittered box) | 노이즈에 강인한 예측 학습     |


---

##### 🎯 효과

- Query가 **GT 근처에서 학습되도록 유도**
- “정답 근처지만 정확하지 않은 예측”을 처리하는 능력 향상
- 초기에 의미 없는 예측을 하던 query들이 **빠르게 정답과 관련된 위치로 수렴**
- 전체 학습 속도 향상 + 성능 안정화


---

#### 💡 해결책2: Matching Queries (고정 Anchor 기반)

DINO는 DETR와 달리, object query가 **완전히 랜덤하게 위치를 찾는 방식**이 아니라  
**초기부터 GT 위치 근처에 정해진 query anchor를 배치**합니다.

---

##### 🧲 작동 방식

1. **GT 중심 Anchor 생성**
   - 학습 시 GT 위치를 기준으로 일정 수의 고정된 query anchor를 생성

2. **각 anchor에 query 지정**
   - 이 anchor는 특정 GT를 예측해야 할 책임 있는 query로 할당됨

3. **Matching 과정 안정화**
   - Hungarian Matching이 이 anchor query와 GT를 1:1 매칭하기 쉬워짐

---

##### 🎯 효과

- query가 **GT 근처에서 시작하므로 빠르게 수렴**
- 초기에 발생하던 **매칭 불안정 문제를 줄임**
- GT마다 명확히 대응되는 query가 있어 **성능과 수렴 속도 향상**

---

#### 💡 해결책3: Two-stage 구조

DINO는 기존 DETR의 one-stage 구조를 확장하여  
**Encoder → Decoder로 이어지는 두 단계 구조**를 적용합니다.

---

##### 🧠 작동 방식

1. **1단계 (Encoder)**
   - CNN + Transformer encoder를 통해 **dense한 object 후보 (anchors)** 추출
   - Top-K scoring anchor들 선택

2. **2단계 (Decoder)**
   - Encoder에서 선택된 anchor들을 기반으로 **refined prediction 수행**
   - 클래스 및 정확한 box 조정

---

##### 🎯 효과

- 첫 단계에서 coarse하게 위치를 파악하고,  
- 두 번째 단계에서 정확히 조정 → **정밀도 향상**
- 작은 객체나 복잡한 배경에서의 **탐지 안정성 증가**

---

#### 💡 해결책4: Look Forward Twice

기존 DETR 계열은 decoder에서 object query가 encoder feature에 attention을 한 번 수행합니다.  
DINO는 **이 attention 연산을 두 번 반복(Look Twice)** 하여 더 깊은 상호작용을 유도합니다.

---

##### 🔁 작동 방식

1. **첫 번째 attention**
   - object query가 encoder output과 기본 attention 수행

2. **두 번째 attention**
   - 첫 attention 결과를 다시 encoder feature에 attention  
   - 즉, **query → encoder → query → encoder**

---

##### 🎯 효과

- 더 깊은 context 정보 활용
- 복잡한 장면에서도 **정확한 클래스 및 위치 예측 가능**
- 특히 overlapping 객체, 작은 물체에 대해 **강한 표현력 확보**



---

### 🧱 DINO 아키텍처 요약

```
Input Image
 → CNN Backbone (e.g., ResNet or Swin)
   → Transformer Encoder
     → Candidate Object Proposals (Two-stage)
       → Transformer Decoder
         → Predictions {Class, Bounding Box}₁~ₙ
```



---

### ✅ 요약

| 항목                   | 설명 |
|------------------------|------|
| 목적                   | Object query 학습 초기 수렴 가속 |
| 방법                   | GT box에 노이즈를 추가해 query에 학습 유도 |
| 효과                   | 학습 안정화, 작은 객체에도 민감한 예측 가능 |
| 최종 성능 기여         | 학습 속도 향상 + AP 성능 향상 |

---

> DeNoising Training은 DINO를 DETR보다 훨씬 **실용적이고 빠른 객체 탐지기**로 만들어주는 핵심 기술입니다.


---

### 📊 성능 비교 (COCO 기준)

| 모델     | AP (val) | FPS | Backbone  |
|----------|----------|-----|-----------|
| DETR     | 42.0     | 10  | ResNet-50 |
| DAB-DETR | 44.9     | 11  | ResNet-50 |
| DINO     | **49.0+**| 12  | ResNet-50 |
| DINO     | **~54.0**| --  | Swin-L    |

---

### 🧠 DINO vs DETR

| 항목                 | DETR                     | DINO (Improved)              |
|----------------------|--------------------------|------------------------------|
| 학습 수렴 속도        | 느림                     | ✅ 빠름 (DeNoising)          |
| 작은 객체 탐지 성능   | 낮음                     | ✅ 향상됨                    |
| Object Query 구조    | 단순                     | ✅ GT 기반 Matching 추가     |
| Stage 구조           | One-stage                | ✅ Two-stage 구조 포함       |

---

### 📌 요약

- DINO는 DETR의 구조를 유지하면서, **실제 사용에 적합하도록 빠르고 정확하게 개선**한 모델
- 다양한 후속 연구(Grounding DINO, DINgfO-DETR, DINOv2)의 기반이 되는 핵심 모델
- 🔥 **open-vocabulary detection**, **grounding**, **segment anything** 같은 최신 비전 연구와도 잘 결합됨

---

### 💬 개인 정리

> DINO는 **DETR의 학습 효율성과 성능 문제**를 해결한 훌륭한 개선안이다.  
> 특히 작은 객체, 빠른 학습 수렴, ViT 백본 호환 등 실무 활용도가 매우 높음!  
> Grounding DINO나 DINOv2 등으로 확장할 때도 핵심 개념을 그대로 공유하므로  
> **DETR 계열 Transformer 탐지 모델을 이해하려면 반드시 알아야 할 모델!**

