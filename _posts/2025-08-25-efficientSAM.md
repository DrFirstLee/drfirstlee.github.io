---
layout: post
title: "🧠 EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything — 실전형 SAM의 표준"
author: [DrFirst]
date: 2025-08-25 10:00:00 +0900
categories: [AI, Research]
tags: [EfficientSAM, SAM, Segmentation, MAE, Foundation-Model、 CVPR 2024, CVPR]
sitemap:
  changefreq: monthly
  priority: 0.8
---

---

### 🧠 (English) EfficientSAM: A ‘Light & Fast’ Segment Anything via Leveraged Masked Image Pretraining

![Image](https://github.com/user-attachments/assets/87091298-8c04-4d89-a0f1-9f4a47459569)

* **Title**: [EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything](https://arxiv.org/abs/2312.00863)  
* **Venue**: CVPR 2024 (OpenAccess) – [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiong_EfficientSAM_Leveraged_Masked_Image_Pretraining_for_Efficient_Segment_Anything_CVPR_2024_paper.pdf)  
* **Code/CKPT**: [GitHub – yformer/EfficientSAM](https://github.com/yformer/EfficientSAM)  
* **Keywords**: `Segment Anything`, `Masked Image Pretraining`, `Lightweight ViT`, `Promptable Segmentation`  
* **TL;DR**: EfficientSAM keeps SAM’s strengths while being **faster and lighter** for practical use.

---

### 🚀 EfficientSAM — Key Points

> One-liner: **“Retain SAM’s capability, optimize weight and speed for deployment.”**

1) **Efficient architecture 🧠**  
- **Lightweight image encoder**: Replace SAM’s heavy ViT-H with **ViT-Tiny/Small** backbones. **Prompt encoder & mask decoder stay compatible** with SAM, preserving the pipeline.

2) **Smarter pretraining 🎯**  
- **SAMI (SAM-Leveraged Masked Image Pretraining)**: Train the lightweight encoder to **reconstruct features from SAM’s ViT-H** with a masked pretext task → **transfers SAM’s representation power** into a compact backbone.

3) **Practical extensibility 🛠️**  
- Keeps SAM’s **interactive prompts** (points/boxes/“segment everything”) and can be fine-tuned for **classification, detection, segmentation** downstream.

4) **Better efficiency–accuracy trade-off ⚡**  
- Aims to **retain segmentation quality** while **cutting params/FLOPs**, ideal for **edge/mobile/real-time** scenarios.

---

### 🔍 Prior Work

- **Making SAM & ViT efficient**  
  - SAM is widely used; many works reduce its compute cost.  
  - **FastSAM** uses a **CNN (e.g., YOLOv8-seg)** to segment all objects efficiently.  
  - **MobileSAM** distills a **light image encoder** via decoupled distillation.  
  - Efficient ViT variants continue to emerge: **ViT/DeiT Tiny/Small**, **MobileViT, LeViT, EfficientViT**, etc.

- **Knowledge Distillation (KD)**
  - **KD** transfers knowledge from a large **teacher** to a small **student** **without changing architecture**, supervised by **hard + soft labels**.  
    - **Hard labels**: one-hot targets (e.g., `[cat=1, fox=0, car=0]`); typically trained with CE; lack inter-class similarity.  
    - **Soft labels**: teacher’s probability distribution (e.g., `[cat=0.60, fox=0.35, car=0.05]`), often with **temperature** to reveal **dark knowledge** (class relations), improving generalization/calibration.  
  - Recent trends: stronger **soft-label KD**, **decoupling** (separate feature learning vs. classification), and **Decoupled KD** (split KD loss into **target/non-target** parts) so the student learns both confidence for the true class and relations among the rest.  
  - Another line matches **intermediate features** directly—e.g., **FitNet**, SSTA for ViT students, or aligning features between **MAE teacher/student**.

- **MIM (Masked Image Modeling)**
  - Self-supervised pretraining: **mask patches** and **reconstruct** the missing parts.  
  - **BEiT** predicts **visual tokens**; **SimMIM** reconstructs pixels; **MaskFeat** reconstructs HOG features.  
  - **MAE (Masked Autoencoder)**: high mask ratio (~75%), **asymmetric** encoder–decoder; encoder sees only visible patches, decoder reconstructs the full image (usually pixels).

---

### 🧱 EfficientSAM Architecture

![Image](https://github.com/user-attachments/assets/f94e0292-d4e5-44eb-8dde-c9e7ff6681b9)

- **Image Encoder**  
  - **ViT-Tiny/Small** backbones.  
  - **SAMI pretraining** teaches them to **reconstruct SAM ViT-H features**, so the compact encoder **inherits SAM-like representations**.  
  - Instead of vanilla KD, masking improves **local/occluded-region awareness** and robustness.

- **Prompt Encoder (same as SAM)**  
  - Lightweight transformer that embeds **points/boxes** into a unified prompt embedding.

- **Mask Decoder (same as SAM)**  
  - Combines image & prompt embeddings with **dual cross-attention**, outputs **masks (+ IoU prediction)**.  
  - Full compatibility with existing SAM tooling/interfaces.

---

### 🔧 Training Recipe & Results

- **1) SAMI Pretraining**  
  - **Teacher**: SAM’s ViT-H image encoder features.  
  - **Student**: lightweight ViT-T/S.  
  - **Goal**: via **masked reconstruction**, reproduce SAM features → student learns **promptable-segmentation-friendly** representations.

- **2) SA-1B Finetuning**  
  - **SAMI-initialized encoder + SAM decoder** are finetuned on **SA-1B** for points/boxes/“everything”.

- **3) Downstream transfer**  
  - Use the SAMI encoder for **classification/detection/segmentation** to show broad applicability.

![Image](https://github.com/user-attachments/assets/17184218-d11b-4d30-ac3f-ffa5fdf1b058)

- Shows solid performance on **Image Classification, Object Detection & Instance Segmentation, Semantic Segmentation**.

---

### 🧪 Segmentation Results & Ablations

1) **Benefit of SAMI**  
   - Compared to vanilla MAE-like pretraining, **SAMI** (reconstructing SAM features) learns representations **more suitable for promptable segmentation**.

2) **Effectiveness of lightweight backbones**  
   - With **ViT-T/S** + **SAMI + finetune**, EfficientSAM **keeps quality while boosting efficiency**, reducing reliance on ViT-H.

3) **Practical compatibility**  
   - Maintains **points/boxes/everything** prompts and **SAM mask decoder**, minimizing replacement cost (checkpoints/examples provided).

#### 🎯 Zero-shot single-point valid mask evaluation (1-click / 1-box)

- **Protocol**: Random foreground point within GT mask; tight GT bbox as prompt; among multiple predictions, evaluate the **highest-confidence** mask.

![Image](https://github.com/user-attachments/assets/d47f8842-c867-483f-9077-5ec92fdc4cc4)

- **Highlights**
  - **EfficientSAM-Ti**: vs **MobileSAM**, **+1.9 mIoU (1-click)**, **+1.5 mIoU (1-box)** at similar complexity.  
  - **SAMI > MAE**: SAMI-pretrained weights outperform MAE-pretrained on COCO/LVIS interactive.  
  - **EfficientSAM-S**: COCO(box) **−1.5 mIoU** vs SAM; LVIS(box) **−3.5 mIoU** (**~20× fewer params**).  
  - Competitive on multi-click as well.

---

#### 📦 Zero-shot instance segmentation

![Image](https://github.com/user-attachments/assets/5f8fbdcd-e317-44d6-b231-000b670fcde6)

- **Protocol**: Use **ViTDet**-generated **bbox prompts**; pick the mask with **max IoU** to the bbox.  
  - Thus **ViTDet-H** serves as a **strong upper baseline** for comparison.
- **Results**
  - **EfficientSAM-S**: vs FastSAM **COCO +6.5 AP**, **LVIS +7.8 AP**.  
  - **EfficientSAM-Ti**: vs FastSAM **COCO +4.1 AP**, **LVIS +5.3 AP**; vs MobileSAM **COCO +3.6 AP**, **LVIS +5.5 AP**.  
  - **Model size**: **Ti 9.8M** vs **FastSAM 68M** → **much lighter**.  
  - **S model** narrows the gap to full **SAM (0.6G params)** to about **~2 AP**.  
- **Summary**: Beats other **lightweight** models; **slightly below** the **very large** ViTDet-H+SAM pipeline.

---

#### 👀 Qualitative & Salient Instance Segmentation

![Image](https://github.com/user-attachments/assets/ee90727f-09e1-4a65-aeeb-2279180b631d)

- **Qualitative**: For points/boxes/“segment everything,” EfficientSAM’s boundaries & occlusion reasoning are **close to SAM**.  
- **Salient Instance Seg.**: Generate a **saliency map** with **U²-Net**, then sample **3 points (3-click)** inside the map to segment with EfficientSAM.  
  → Promising for **accessibility** (e.g., users with limited hand mobility).

---

### 🧪 Core Ablations

- **Reconstruction loss in SAMI**: **MSE > Cosine** → directly reconstructing SAM **feature values** works better.  
- **Cross-attention decoder**: **Query only masked tokens** (encoder outputs act like anchors) → **+3% Top-1** vs decoding all tokens (MAE-style) on ImageNet-1K (SAMI-Ti).  
- **Mask ratio**: **High ratio (~75%)** remains consistently strong (50/75/85% tested).  
- **Reconstruction target**: Using **CLIP encoder features** as target still yields **+0.8%p over MAE** (ViT-Tiny, IN-1K) → validates **Guided MIM** with strong teacher features.  
- **Finetuning steps**: Good results even at **0.1 epoch**; **+2.5 mIoU** by **1 epoch**.  
  - **EfficientSAM-S** final **76.9 mIoU**, only **−1.5 mIoU** vs SAM.

---

## ✅ Conclusion

- **EfficientSAM** transfers **SAM’s representational power** into a **lightweight encoder** via **SAMI pretraining**, achieving **similar accuracy with much better efficiency**.  
- With **prompt compatibility** (points/boxes/everything) and **open checkpoints**, it’s highly suitable for **edge, real-time, and large-scale deployment**.



---

### 🧠 (한국어) EfficientSAM : Leveraged Masked Image Pretraining로 ‘가볍고 빠른’ SAM!   

![Image](https://github.com/user-attachments/assets/87091298-8c04-4d89-a0f1-9f4a47459569)

* **제목**: [EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything](https://arxiv.org/abs/2312.00863)  
* **학회**: CVPR 2024 (OpenAccess) – [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiong_EfficientSAM_Leveraged_Masked_Image_Pretraining_for_Efficient_Segment_Anything_CVPR_2024_paper.pdf)  
* **코드/체크포인트**: [GitHub – yformer/EfficientSAM](https://github.com/yformer/EfficientSAM)  
* **핵심 키워드**: `Segment Anything`, `Masked Image Pretraining`, `Lightweight ViT`, `Promptable Segmentation`  
* **요약**: EfficientSAM은 빠르면서도 정확한 SAM!! 효율적인 SAM!    

---

### 🚀 EfficientSAM 핵심 요약

> 한 줄 요약: **“SAM의 강점은 유지, 무게와 속도는 실전에 맞게 최적화.”**

1) **효율적인 모델 구조 🧠**  
- **이미지 인코더 경량화**: SAM의 고가용량 ViT-H 대신 **ViT-Tiny/Small 백본**으로 교체. **프롬프트 인코더/마스크 디코더는 SAM과 호환**해 파이프라인은 그대로 유지합니다.  

2) **더 똑똑해진 사전학습 🎯**  
- **SAMI(SAM-Leveraged Masked Image Pretraining)**: **SAM ViT-H에서 나온 특징을 ‘재구성’하도록** 경량 인코더를 마스킹 프리텍스트로 학습 → **SAM의 표현력을 경량 백본에 이식**합니다.   

3) **실전 확장성 🛠️**  
- 포인트/박스/에브리싱 프롬프트 등 **SAM의 상호작용 방식**을 유지하고, 다양한 비전 태스크(분류·검출·분할)로도 확장 가능!!  

4) **효율–정확도 트레이드오프 향상 ⚡**  
- **파라미터·연산 감소** 대비 **Segmentation 품질 유지**를 목표로 설계되어, **엣지·모바일·실시간** 활용에 유리합니다.  

---

### 🔍 기존 연구들!!

- **SAM&ViT의 경량화!**  
  - 기존 SAM은 다양한분야에서 환영받으며, 그 연산비용을 줄이는데 연구들이 이어져옴!  
  - `FastSAM`은 효율 향상을 위해 이미지 내 모든 객체를 분할하는 **CNN 기반 아키텍처(예: YOLOv8-seg[30])**를 개발  
  - `MobileSAM`은 경량 이미지 인코더를 얻기 위한  decoupled distillation 방법을 제시  
  - ViT(Vision Transformer)또한 `ViTSmall/Deit-Small and ViT-Tiny/DeiT-Tiny` 등이 공개됨
  - 이어서 `MobileViT`, `LeViT`, `EfficientViT` 등의 연구가 공개되며 지속적으로 발전!  
  
- **지식 증류(KD)**
  - **지식증류(Knowledge Distillation)**은 모델 구조를 바꾸지 않고, 큰 **교사 모델**의 지식을 작은 **학생 모델**로 옮겨 성능을 높이는 기법으로 하드 라벨 + 소프트 라벨 감독으로 구분!!  
    - **하드 라벨 (Hard label)**  
      - **원-핫(one-hot)** 정답: 정답 클래스만 1, 나머지는 0 (예: `[cat=1, fox=0, car=0]`).  
      - 학습 손실은 보통 **크로스 엔트로피(CE)** 를 사용.  
      - 단점: **클래스 간 유사도 정보가 없음** → “고양이와 여우가 비슷하다” 같은 **미묘한 관계**를 학생이 배우기 어려움.
    - **소프트 라벨 (Soft label)**  
      - **교사 모델의 확률 분포**를 그대로 사용(예: `[cat=0.60, fox=0.35, car=0.05]`).  
      - **온도(Temperature) \(T>1\)** 를 적용한 **소프트맥스**로 분포를 **더 부드럽게** 만들어 ‘**암묵지(dark knowledge)**’(유사도·경계 정보)를 드러냄.  
      - 학생은 이 분포를 따라가며 **클래스 간 관계/난이도** 를 학습 → **일반화·캘리브레이션** 개선.  
  - 최근 연구는 **소프트 라벨 활용** 위주의 지식증류 + **디커플링**(표현 학습과 분류를 별도 학습습)  + **Decoupled KD**(KD 손실을 **타깃/비타깃**으로 분리)로 진행됨!    
    - **디커플링** : **Feature extractor**와 **분류기**를 **분리**, 간섭을 줄이고 자유도를 높임!  
    - **Decoupled KD**: 손실을 타깃 vs 비타깃으로 분리 → 정답 자신감과 오답들 간 관계를 둘 다 제대로 배우게 함!  
  - 또 다른 흐름은 **중간 특징**을 직접 맞추는 방식으로, **FitNet**이 대표적이며(**[47]**), **SSTA**로 ViT 학생을 보조 지도하거나(**[60]**), **MAE 사전학습 교사–학생**의 **중간 특징 정렬**을 통해 지식을 전이한다(**[2]**).


- **MIM (Masked Image Modeling)**
  - 이미지를 **패치 단위로 가리고(mask)**, **가려진 부분을 복원**하도록 학습하는 **Self-supervised pretraining** 방법  
  - **BEiT**는 ViT에 **초기 MIM을 대표**하는 방법 중 하나(토크나이저로 만든 **비주얼 토큰** 복원)  
  - 이후 **SimMIM**(픽셀 복원), **MaskFeat**(HOG 특징 복원) 등 다양한 타깃으로 확장.  
  - **MAE (Masked Autoencoder)**
    - MIM의 한 변형으로, **높은 마스크 비율(~75%)**, **비대칭 인코더–디코더**를 사용.
    - **인코더는 보이는 패치만** 처리해 효율적이고, **디코더가 전체 이미지를 복원**(주로 **픽셀 값** 복원).

---

#＃# 🧱 EfficientSAM 구조(Architecture)

![Image](https://github.com/user-attachments/assets/f94e0292-d4e5-44eb-8dde-c9e7ff6681b9)

- **Image Encoder**:  
  - **ViT-Tiny / ViT-Small** 등 **경량 백본** 기반    
  - **SAMI 프리트레이닝**으로 **SAM ViT-H 특징을 재구성**하게 학습 → 경량 인코더가 **SAM의 표현력**을 습득.  
  - SAM의 특징을 보다 잘 배우기 위해, SAM의 이미지 임베딩을, 동일하게 KD하는것이 아니라, masked 된 이미지로 학습헤서 부위부위별, 혹은 보이지 않는 부분도 추론하여여 특징을 잘 파악할수 있게함   

- **Prompt Encoder (SAM과 동일)**:  
  - **포인트/박스 프롬프트**를 동일한 임베딩으로 변환하는 **경량 트랜스포머 인코더**를 그대로 사용  

- **Mask Decoder (SAM과 동일)**:  
  - **이미지·프롬프트 임베딩**을 **듀얼 크로스어텐션**으로 결합해 **마스크(및 IoU 예측)를 출력**.  
  - 구조 호환을 통해 **기존 SAM 툴링/인터페이스**를 대부분 재사용  

---

#＃# 🔧 학습법(Training Recipe) 및 학습 결과(Results)  

- **1) SAMI 프리트레이닝 (사전학습)**  
  - **교사**: SAM의 ViT-H 인코더로부터 얻은 **고품질 특징**.  
  - **학생**: 경량 ViT-T/S 인코더.  
  - **목표**: **마스킹 복원**을 통해 **SAM 특징을 재현** → 경량 인코더가 **프롬프트 분할에 적합한 표현** 습득  

- **2) SA-1B 파인튜닝 (세그멘트 애니싱 태스크 적합화)**  
  - **SAMI로 초기화된 인코더 + SAM 디코더**를 **SA-1B**로 파인튜닝해 **포인트/박스/Everything** 설정에서의 성능을 맞춤  

- **3) 다운스트림 전이**: SAMI에서 나온 인코더를 바탕으로 분류·검출·분할 등 **다양한 과제**에 파인튜닝 해보며 다양한 과제에 사용 가능함을 테스트

![Image](https://github.com/user-attachments/assets/17184218-d11b-4d30-ac3f-ffa5fdf1b058)

- Image Classification. Object Detection and Instance Segmentation. Semantic Segmentation 에서 모두 좋은 성능을 보임!  
- 

---

### 🧪 Segmentation 결과 분석 & Ablation 테스트  

1) **SAMI의 이득**  
   - 일반 MAE류 대비, **SAM 특징 복원**에 목표를 둔 **SAMI**가 **프롬프트 분할**에 더 적합한 표현을 학습하는 것으로 보고됩니다. :contentReference[oaicite:16]{index=16}

2) **경량 백본의 실효성**  
   - **ViT-T/S**로 교체해도, **SAMI+파인튜닝**으로 **품질을 유지하면서 효율성**을 확보. **대규모 ViT-H 의존도**를 낮춥니다. :contentReference[oaicite:17]{index=17}

3) **실전 호환성**  
   - **포인트/박스/Everything** 프롬프트와 **SAM 마스크 디코더**를 유지해, **기존 파이프라인 대체 비용**이 낮습니다(체크포인트·예제 제공). :contentReference[oaicite:18]{index=18}



#### 🎯 Zero-shot single point valid mask evaluation results (1-click / 1-box)  

- **프로토콜**: GT 마스크 내부 랜덤 포인트, GT 마스크에 대한 tight bbox를 프롬프트로 사용. 다중 마스크 중 **최고 신뢰도** 하나만 평가.

![Image](https://github.com/user-attachments/assets/d47f8842-c867-483f-9077-5ec92fdc4cc4)

- **결과**
  - **EfficientSAM-Ti**: MobileSAM 대비 **+1.9 mIoU(1-click)**, **+1.5 mIoU(1-box)** (복잡도 유사)
  - **SAMI > MAE**: COCO/LVIS 인터랙티브에서 **SAMI 사전학습 가중치**가 **MAE 사전학습**보다 우수
  - **EfficientSAM-S**: COCO(box) 기준 **SAM 대비 −1.5 mIoU**, LVIS(box) **−3.5 mIoU** (파라미터는 **~20× 적음**)
  - **다중 클릭**에서도 MobileSAM, SAM-MAE-Ti와 **경쟁적 성능**

---

#### 📦 Zero-shot instance segmentation  

![Image](https://github.com/user-attachments/assets/5f8fbdcd-e317-44d6-b231-000b670fcde6)  

- **프로토콜**: ViTDet이 생성한 **bbox 프롬프트** 사용, bbox와 **IoU 최대** 마스크를 선택  
  - 그래서 ViTDet-H 가 상한선으로 보면됨!!  
- **결과**
  - **EfficientSAM-S**: FastSAM 대비 **COCO +6.5 AP**, **LVIS +7.8 AP**
  - **EfficientSAM-Ti**: FastSAM 대비 **COCO +4.1 AP**, **LVIS +5.3 AP** / MobileSAM 대비 **COCO +3.6 AP**, **LVIS +5.5 AP**
  - **모델 크기**: **Ti 9.8M** vs **FastSAM 68M** → **훨씬 경량**
  - **S 모델**: **0.6G 파라미터 SAM** 대비 **AP ~2** 차이까지 **격차 축소**
- 요약 : 다른 경량모델들에 비해서는 성능이 좋고 큰모델(ViTDet-H)에 는 조금 떨어지는 성능  

---

#### 👀 정성 비교 & 주목 객체(Salient) 세그멘테이션  

![Image](https://github.com/user-attachments/assets/ee90727f-09e1-4a65-aeeb-2279180b631d)  

- **정성 결과**: 포인트/박스/“segment everything” 시나리오에서 **SAM에 근접한 경계·가림 추론 품질**
- **Salient Instance Seg.**: **U²-Net**으로 **Saliency map** 생성 → 맵 내부 **3점(3-click)**만으로 관심 객체 분할  
  → **손 사용이 어려운 사용자**를 돕는 **접근성** 시나리오 가능성

---

### 🧪 Ablation 핵심

- SAMI에서의 **Reconstsuction lOss의 설계**: **MSE > Cosine** → **SAM 피처의 ‘값’**을 직접 재구성하는 편이 좋다!!  
- **크로스-어텐션 디코더**: **Masked 토큰만 디코더에서 쿼리**(인코더 출력 토큰은 앵커처럼 사용)  
  → **모든 토큰 디코딩(MAE식)** 대비 **Top-1 +3%p**(ImageNet-1K, SAMI-Ti)
- **마스크 비율**: **50/75/85%** 실험에서 **높은 비율(≈75%)**이 **일관되게 우수**
- **재구성 타깃**: **CLIP 인코더 피처**를 타깃으로 해도 **MAE 대비 +0.8%p**(ViT-Tiny, IN-1K)  
  → **강력한 교사 피처를 타깃**으로 하는 **Guided MIM**의 효과
- **파인튜닝 스텝**: **0.1 epoch**에서도 준수, **1 epoch**에 **+2.5 mIoU** 상승  
  - **EfficientSAM-S 최종 76.9 mIoU**, **SAM 대비 −1.5 mIoU**

---

## ✅ 결론

- **EfficientSAM**은 **SAM의 표현 능력**을 **경량 인코더**에 이식하는 **SAMI 사전학습**으로, **정확도 유지 + 추론 효율 개선**을 이룸!!    
- **프롬프트 호환성(포인트/박스/Everything)**과 **오픈된 체크포인트** 덕분에, **엣지·실시간·대규모 배포**에 활용 가능!!  

