---
layout: post
title: "Grounding DINO 논문 공부!"
author: [DrFirst]
date: 2025-05-15 15:00:00 +0900
categories: [AI, Research]
tags: [grounding, grounding dino, grounded sam, affordance grounding, computer vision, AI]
sitemap :
  changefreq : monthly
  priority : 0.8
---


---

## (한국어) 📝 Grounding DINO 알아보기!!
_『Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection』(ECCV, 2024) 공부_

📖 **논문 제목**: Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection  
✍️ **저자**: Xinyu Chen, Xueyan Zou, Ze Liu, et al.  
🌟 **한줄 요약**: 텍스트로 뭐든 찾아주는 프롬프트 기반 객체 탐지기!

---

### 🧠 핵심 아이디어

#### ✅ DINO의 구조 활용

- DINO (DETR-like 모델)의 **query-based detection** 구조를 활용  
- **비지도 학습 기반 객체 탐지 능력**을 바탕으로 발전  

#### ✅ Grounded Pretraining 결합

- **CLIP** 기반 텍스트-이미지 매핑 능력 사용  
- **"텍스트 박스 매칭"** 기반 학습 → 모델이 프롬프트에 따라 box를 예측 가능  

#### ✅ Open-Set 대응

- 정해진 클래스 없이 **임의의 텍스트 입력**으로 객체 탐지 가능  
- 예: "a person holding a red umbrella" → 실제 이미지에서 해당 객체 위치 예측  

---

### 🔍 Grounding DINO 연구의 배경

Grounding DINO는 기존의 객체 탐지(Object Detection) 모델들이 가진 **고정된 클래스 제한**을 뛰어넘기 위해 제안된 모델입니다.  
이전까지의 흐름은 다음과 같습니다:

---

#### 🧩 DETR 이후 DINO, 하지만 여전히 클래스는 고정

- **DETR (2020, Facebook AI)**  
  Transformer 기반으로 객체 탐지를 수행한 최초의 end-to-end 모델  
  → 하지만 클래스는 COCO처럼 **사전 정의된 클래스셋에 한정**됨

- **DINO (2022)**  
  DETR 구조를 개선해 학습 안정성과 정확도를 높인 모델  
  → 뛰어난 성능을 보였지만 **여전히 고정된 클래스(class token)**만 탐지 가능

즉, DINO는 **탐지는 잘하지만 '무엇을 탐지할지'는 이미 정해져 있어야** 했습니다.

---

#### 🧩 객채탐지, 고정된 객체 제한을 탈출하고자 등장한 GLIP (Open-Set Object Detection)

오픈셋 객체 탐지란, 기존의 bounding box 어노테이션으로 학습하면서도,
**언어 기반 일반화(language generalization)**를 활용해 **임의의 클래스(arbitrary class)**를 탐지하는 것을 목표로 한다.

⚠️ 기존 연구의 한계
하지만 이전 연구들은 대부분 **멀티모달 정보(이미지 + 언어)를 일부 단계에서만 결합(fusion)**하는 데 그쳤으며,
이로 인해 언어 일반화(language generalization) 능력이 **최적 이하(sub-optimal)**일 수 있다.

예:

GLIP은 **feature enhancement 단계(phase A)**에서만 멀티모달 결합을 고려

OV-DETR는 **decoder 입력 단계(phase B)**에만 언어 정보를 삽입


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
- → 결국 **"말로 탐지하는(open-vocabulary) 객체 탐지기"**가 된 것이죠.

이후 SAM과 결합하여 **Grounded SAM**으로 확장되며,  
"텍스트 → 탐지 → 분할"이라는 전체 파이프라인이 완성되었습니다.


---

### 🧪 모델 구성

- **Backbone**: Swin Transformer
- **Language Encoder**: BERT 또는 CLIP Text Encoder
- **Object Query**: Multi-head attention 구조
- **Matching Loss**: Grounding Loss (Text ↔ Box alignment)

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