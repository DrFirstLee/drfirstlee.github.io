---
layout: post
title: "🧩 PartCLIPSeg: Open-Vocabulary Part-level Segmentation with CLIP Guidance"
author: [DrFirst]
date: 2025-09-05 07:00:00 +0900
categories: [AI, Research]
tags: [Open-Vocabulary, Segmentation, CLIP, Part-level, NeurIPS 2024, Recognition, NeurIPS]
sitemap:
  changefreq: monthly
  priority: 0.8
---

---
### 🧩 (한국어) PartCLIPSeg: CLIP으로 파트 단위까지 인식하는 Open-Vocabulary 세분화!  

![Image](https://github.com/user-attachments/assets/fda00fce-3252-44bb-a18b-4d91a653ef45)

* **제목**: [PartCLIPSeg: Understanding Multi-Granularity for Open-Vocabulary Part Segmentation](https://arxiv.org/pdf/2406.11384)    
* **학회**: NeurIPS 2024    
* **코드/체크포인트**: [GitHub – PartCLIPSeg](https://github.com/kaist-cvml/part-clipseg/)  
* **핵심 키워드**: `Open-Vocabulary`, `Part-level Segmentation`, `CLIP`, `Recognition`, `Fine-grained`  
* **요약**: CLIP의 텍스트-비전 정렬 능력을 이용해, **객체 수준을 넘어 ‘파트 단위(part-level)’까지 오픈 보카 인식·분할**을 실현!  

---

### 🚀 PartCLIPSeg 핵심 요약

> 한 줄 요약: **“객체 전체가 아니라, ‘핸들’, ‘바퀴’, ‘날개’ 같은 파트까지 잘라서 이름 붙인다!”**

1) **CLIP 기반 파트 인식**  
- CLIP의 비전–텍스트 임베딩 공간을 활용해 **세부 파트 단위**까지 라벨링  
- “car”뿐 아니라 “car-wheel”, “car-door” 같은 파트별 분할이 가능  

2) **Open-Vocabulary 확장 🎯**  
- 기존 객체 단위 OV segmentation → **세분화된 파트 레벨 확장**  
- 사전 정의 라벨이 없어도 “wing”, “leaf”, “handle” 등 **텍스트 프롬프트** 기반 인식  

3) **상호작용적 프롬프트 입력 🛠️**  
- 포인트, 바운딩 박스, 마스크 프롬프트 지원  
- 텍스트 프롬프트와 조합하여 **실시간 파트 인식+분할**  

4) **세밀한 비전 이해 ⚡**  
- 로보틱스, AR/VR, 3D 이해, 의료 영상 등 **정밀 분석**이 필요한 분야에 바로 활용 가능  

---

### 🔍 기존 연구의 흐름  

- CLIP: contrastive vision-language pre-training 덕분에 zero-shot 성능 우수  
- Open-Vocabulary: 객체 감지·분할까지 확장됨 (OV detection/segmentation)  
- SAM: segmentation-anything 모델의 등장이 다양한 downstream task의 기반이 됨  
- 하지만 기존 연구는 **객체 단위**에 머물렀고, **세밀한 파트 인식**까지 다루지 못했음  
- → PartCLIPSeg는 CLIP과 segmentation을 융합해 **part-level open-vocabulary segmentation**을 제안!  

---

### 🧱 PartCLIPSeg 구조 (Architecture)

![Image](https://github.com/user-attachments/assets/b94f3050-433b-4c57-89ae-cb3b7e8916a5)

#### 1) Backbone: CLIP Visual Encoder  
- 멀티스케일 feature 추출, 파트 단위 구분에 적합한 해상도 유지  

#### 2) Part-level Adapter (CLIP2PartSeg)  
- CLIP feature → segmentation decoder 호환 feature로 변환  
- FPN 기반 멀티스케일 정렬  

#### 3) Prompt Encoder  
- SAM 방식 point/box/mask 프롬프트  
- 텍스트 프롬프트(예: “car wheel”)와 결합  

#### 4) Mask Decoder  
- CLIP2PartSeg feature + 프롬프트 융합  
- 파트 단위 segmentation mask 생성  

#### 5) Recognition Head (Part2CLIP)  
- segmentation된 마스크를 CLIP 임베딩 공간으로 재투영  
- 텍스트 임베딩과 코사인 유사도 매칭 → 파트 레이블 결정  

---

### 🔧 학습법(Training Recipe)

1) **Pre-training 단계**  
- COCO, LVIS, PartImageNet 등에서 CLIP2PartSeg 학습  
- Mask supervision + Text alignment joint loss  

2) **Fine-grained Distillation**  
- 기존 object-level SAM feature와 CLIP feature를 정렬  
- 파트 단위 annotation에 대해 CLIP 기반 텍스트 유사도 증류  

3) **Zero-shot 확장**  
- ImageNet-22K 텍스트 임베딩 활용 → 2만+ 클래스 파트 인식  

---

### 🧪 실험 결과  

#### 🎯 Open-Vocabulary Part Segmentation  

- **PartImageNet**: mIoU 76.4 / novel class IoU 72.1  
- **Pascal-Part**: mIoU 74.9 / novel class IoU 70.8  
- 기존 object-level OVS 모델 대비 **세밀한 파트 구분 성능 우수**  

#### 🎯 Efficiency  

- FLOPs와 파라미터 수 모두 기존 OV baselines 대비 절반 수준  
- 실시간 인터랙션 성능 유지  

---

### 👀 정성 비교  

![Image](https://github.com/user-attachments/assets/f6da88fb-4b61-4518-b2c7-76c0a23b2e5c)

- 자동차 이미지 → “door”, “wheel”, “window”까지 분리하고 이름 붙임  
- 동물 이미지 → “wing”, “tail”, “head” 세부 파트 분할 가능  
- 의료 영상 → 장기 내 “sub-part”까지 인식 확장  

---

### 🧪 Ablation 분석  

- **Part2CLIP 정렬**이 없으면 파트별 인식 정확도 급락  
- **Text similarity loss** 추가 시 novel 파트 인식에서 큰 성능 향상  
- 클래스 확장성: 1천 → 2만 파트 클래스 확장해도 선형 추론 비용 증가  

---

## ✅ 결론  

- **PartCLIPSeg**는 기존 open-vocabulary segmentation을 **객체 단위 → 파트 단위**로 확장  
- CLIP 기반 zero-shot 인식과 SAM 스타일 segmentation을 결합해 **세밀한 비전 이해**를 제공  
- 로보틱스, AR/VR, 산업/의료 영상 등 **세부 구조 이해**가 중요한 응용 분야에서 차세대 표준으로 자리 잡을 모델!  
