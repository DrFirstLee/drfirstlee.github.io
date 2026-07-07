---
layout: post
title: "⚙️ Vision Transformers Need Registers 논문 공부 (ICLR 2025)"
author: [DrFirst]
date: 2025-12-23 09:00:00 +0900
categories: [AI, Research]
tags: [Computer Vision, Vision Transformer, Register Tokens, Dense Prediction, ICLR, ICLR 2025]
sitemap:
  changefreq: monthly
  priority: 0.8
---

### ⚙️ Vision Transformers Need Registers — 핵심 논문 리포트

![vit-registers](https://github.com/user-attachments/assets/aaf1eeda-e64e-4b9f-9226-bd295ef89674)

> **논문**: Vision Transformers Need Registers  
> **저자**: Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski  
> **학회**: ICLR 2024  
> **핵심 요약**: Vision Transformer (ViT)가 생각을 정리할수 있는 연습장(Reg)를 제공해서, 이미지 내에 저장하던 글로벌정보로 튀던부분(Outlier)를 제거함!  

---

### 🧠 문제 정의: ViT Feature Artifact

Vision Transformer 계열 모델 (DeiT-III, OpenCLIP, DINOv2 등)에서는 아래와 같은 이상 현상이 발견!!

> 배경같은 부위에 뜬금없이 집중된 어텐션이 보임 (노랑색부분)
![error](https://github.com/user-attachments/assets/e0e52988-b2c0-441b-83c4-2c783ce7c8ea)

- 모델 내부 **feature map에서 일부 patch token의 L2 norm이 매우 크게 튀는 artifact 존재**
- 이 token들은 **대부분 배경 영역**에서 나타남  
- 전체 토큰 중 약 **2% 정도만이 이상치로 나타남  
- high-norm 토큰이 attention map에서 **spur/spike 형태의 noise**를 유발함  

---

### 🧐 왜 이런 artifact가 생길까? - 🔍 Artifact 토큰의 특성

![characteristic](https://github.com/user-attachments/assets/724ccf0a-6866-4fdb-ba10-c70d4c59e8c7)

![char-2](https://github.com/user-attachments/assets/5ff3fdd8-4626-48a2-b98b-7f51b99431e4)

![global](https://github.com/user-attachments/assets/4f67c2fa-35e8-4d09-b26e-8a473845697e)

1. **높은 norm 값 (High-norm)**  : 다른 token 대비 norm이 비정상적으로 큼  
2. **낮은 로컬 & 높은 글로벌 정보**  : 토큰 자체는 **patch 위치나 픽셀 정보는 적음**(5-a), But **global feature 표현에 매우 효과적**(이미지 분류 성능이 훨씬 높음)   
3. **대형 모델에서(4-c), 중간층(Middle Layer)에서 생김**(4-a)  
4. **학습의 1/3 이 지난 후 발생 (4-b)   

> 즉, **대형 모델**이 **상당히 학습된 단계** 에서, **로컬 정보가 중복되거나 중요하지 않은 patch token을 ‘쓸모없는 공간’으로 인식**하고, 해당 token을 **내부 계산 저장 공간처럼 변형해 global feature 처리에 사용**하고 있음을 관찰!!

---

### 🛠 해결책: Register 토큰 도입  
> 쓸모없는 공간에 저장하던 patch token을 **register로 분리**해보면 잘될까? 하는 가설로 시작!!  
![reg](https://github.com/user-attachments/assets/a98fef0a-0999-4316-ae96-6e8fb10da0fc)  

- 논문에서는 **artifact 용도로 patch token을 재활용하는 대신**, 모델 입력에 **추가적인 trainable 토큰(*)을 미리 제공**하는 방식으로 접근  
- 이 토큰들을 **register (레지스터)** 라고 부르며, 입력 시퀀스에 patch token과 함께 추가됨

#### ✨ Register 토큰의 역할

✔ artifact 생성 용도였던 token 역할을 **register로 분리**  
✔ patch token은 순수하게 **이미지 정보 표현에 집중**  
✔ register 토큰은 **내부 계산/aggregator 역할 담당**

후처리에서 이 register token은 결과에 포함시키지 않으며, 모델이 자동으로 이 토큰을 활용해 global 정보/계산 처리에 쓰도록 학습함  

---

### 📊 실험결과

1. 아웃라이어가 없어짐!!  
![reg](https://github.com/user-attachments/assets/5f94997b-9461-41f7-ad0f-a67e3b582c84)  

2. 이미지의 문제를 푸는 성능이 조금씩 좋아짐
![reg2](https://github.com/user-attachments/assets/637a7214-91fc-41b7-8512-4491106eeceb)
- ImageNet 분류 (linear probe)
- ADE20K segmentation
- NYU Depth estimation
- Object discovery (LOST)  

![dinov2](https://github.com/user-attachments/assets/be450132-466b-49d8-b0b4-c3acdcb0273f)  
→ 특히 object discovery task에서는 **DINOv2 기준 크게 향상됨**


3. Register 개수 및 효율성
![reg_num](https://github.com/user-attachments/assets/0080bde5-77df-492d-a8f8-7c604e79746d)  
- **1개 register**만 있어도 artifact 제거 효과 매우 큼  
- Dense task 성능은 보통 **4개 정도가 최적**  
- FLOPs 증가율은 **2% 이내로 매우 작음**

![reg_num2](https://github.com/user-attachments/assets/8f925182-d1b0-4470-97a4-8722c255fc15)  
- 각 Reg별로 담는 정보가 다른것 같아 흥미로웠다! 더 연구해볼만한 주제야!


### 결론!! 왜 register가 이렇게 효과적이었을까??

1. ViT 모델은 입력 시퀀스 길이에 있는 여분의 토큰을  
→ *추론/내부 계산을 저장하는 공간으로 무의식적으로 사용*했음  

2. 이 저장공간이 본래 patch token 역할을 하면서  
→ L2 norm 이상치가 생기고 attention map에 artifact를 만들었음  

3. register만 따로 제공하면  
→ “계산 메모리”는 register가 담당하고  
→ **patch token은 로컬 디테일/feature 표현에 집중**하게 됨

---

### 🧠 나의 코멘트!!  

![my](https://github.com/user-attachments/assets/c3d2203f-cc9c-4376-b6b5-ddabff9605a0)  

실제로 직접 Attention 결과를 그려보면서 왜 이런 노란 부위가 뜨는지 궁금했었는데!!  
이 논문의 분석을 통해서 ViT 내부의 원인을 이해할수 있어 좋았다!!
논문에서 언급한데로 Register에는 어떤 정보를 담고 있는지 더 알아보고싶다!!  