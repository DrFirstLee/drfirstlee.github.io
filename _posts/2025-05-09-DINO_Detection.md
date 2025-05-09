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

> DINO는 **DETR 계열의 한계를 극복**한 최신 객체 탐지 모델입니다.  
> 특히 **학습 속도 향상**과 **소형 객체 성능 개선**에 중점을 둔 구조로 설계되었습니다.

- DINO = **DETR with Improved DeNoising Anchors**
- 기본 구조는 DETR 기반이지만, 다양한 전략으로 성능을 강화한 모델
- **One-stage** 구조지만 **Two-stage 수준의 성능**을 달성!

---

### 🚨 DINO가 등장한 이유

#### DETR의 주요 한계
1. ❌ **학습이 너무 느리다** (수십만 스텝)
2. ❌ **작은 객체 탐지가 약하다**
3. ❌ **Object Query 학습 초기에 성능이 낮다**

---

### 💡 DINO의 핵심 아이디어

| 주요 구성 요소              | 설명 |
|----------------------------|------|
| 🔧 **DeNoising Training**   | 학습 시, GT 주위에 노이즈 박스를 일부러 생성하여 Query를 빠르게 수렴시킴 |
| 🧲 **Matching Queries**     | GT에 가까운 위치에 고정된 Query Anchor를 배치해 안정적인 학습 유도 |
| 🧠 **Two-stage 구조 추가**  | Encoder에서 coarse object 후보를 뽑고, Decoder에서 refinement 수행 |
| **Look Forward Twice**   | Decoder에서 한 번이 아니라 두 번 attention을 주는 방식으로 정확도 향상 |

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

### 🔍 What is DeNoising Training in DINO?

> DeNoising Training (DN Training)은 DINO에서 **Object Query 학습 초기의 비효율성**을 해결하기 위한 핵심 전략입니다.

---

#### 🧩 배경: DETR의 문제점

- DETR은 학습 초기 단계에서 object query들이 **무작위한 예측을 하며 의미 없는 위치에 box를 출력**합니다.
- 이로 인해 학습이 매우 느리게 수렴하고, 수십만 스텝이 필요합니다.
- 왜냐하면 query가 "어떤 GT를 예측해야 하는지" 전혀 감이 없는 상태에서 시작되기 때문입니다.

---

#### 💡 해결책: DeNoising Training

DINO에서는 학습 초기에 object query들이 **정답(GT) 주변 정보를 빠르게 인식하고 학습**할 수 있도록 돕기 위해  
**“의도적으로 노이즈를 주입한 학습 샘플”을 사용**합니다.

---

#### 🔧 작동 방식

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

#### 🎯 효과

- Query가 **GT 근처에서 학습되도록 유도**
- “정답 근처지만 정확하지 않은 예측”을 처리하는 능력 향상
- 초기에 의미 없는 예측을 하던 query들이 **빠르게 정답과 관련된 위치로 수렴**
- 전체 학습 속도 향상 + 성능 안정화

---

#### 📈 시각적으로 표현하면:

| Query Type       | Input                     | 목표                          |
|------------------|---------------------------|-------------------------------|
| Matching Query   | GT box                    | 정확한 객체 예측              |
| Denoising Query  | GT + noise (jittered box) | 노이즈에 강인한 예측 학습     |

---

#### ✅ 요약

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

