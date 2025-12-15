---
layout: post
title: "🔍양방향 학습을 통한 Affordance Grounding 문제해결! (ICCV 2025)"
author: [DrFirst]
date: 2025-12-12 09:00:00 +0900
categories: [AI, Research]
tags: [Computer Vision, Affordance, Weakly-Supervised, Cloased-Loop, ICCV 2025, ICCV]
sitemap:
  changefreq: monthly
  priority: 0.8
---

### 🔍 Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding 논문 읽기!  

![wow](https://github.com/user-attachments/assets/392d75be-41d6-4c4a-85a6-b5b48cb9041a)
> 결과이미지 감상하기!

* **제목**: [Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding](https://arxiv.org/pdf/2503.06287)  
* **학회 및 저자**: Kang et al., CVPR 2025  
* **요약**: Training-Free 방식으로 LVLM의 일부 head만 사용하여 Segmentation을 할 수 있다! 🚀    
* **Code/Checkpoints**: [GitHub – LocalizationHeads](https://github.com/seilk/LocalizationHeads)
 

---

### 🚀 연구 핵심 요약

> **한 줄 요약:**  
> **LVLM 내부에는 'Localization Head'라는 강력한 grounding 전용 head가 숨어 있으며, 단 2–3개만 추출해도 Fine-tuning 없이 bounding box·mask 생성이 가능하다! 🔥**  
 
- 기존: LVLM attention 평균은 너무 noisy → grounding 불가  
- 제안: 특정 head는 **일관적으로 referring object를 정확히 가리킴**  
- 결과: **Training-free임에도 LISA·Shikra 수준의 성능 달성**

![image](https://github.com/user-attachments/assets/0c2aad80-df03-40cd-b1ef-2b275aa882c0)
> 위의 Fig. 1 예시로보면 Llava-1.5에서는 14번째 레이어 24번째 헤드(L14H24)과  14번째 레이어 13번째 헤드(L14H13)가 localization에 의미있었다!! 



---

### 🔍 기존의 관련 연구들!    

1. LVLM의 attention은 ‘텍스트 생성’에 최적화  
- 전체 attention map 평균은 sparse + noisy: 즉 잡음이 많음!!    
- grounding에 직접 사용 불가

2. 기존 training-free grounding은 CLIP 또는 Diffusion 기반  
- LVLM의 강력한 vision-language reasoning을 활용 못함

3. LVLM 기반 grounding 연구는 모두 Fine-tuning 필수  
- LISA, Shikra, Ferret 등은 별도 decoder 및 segmentation head 필요  
- 비용 ↑, 확장성 ↓

> 결론: **LVLM 내부 구조 자체에서 grounding signal을 찾아야 한다!**


### 🔍 본 연구의 핵심 아이디어: Localization Heads 발견!  

기존 평균 attention 대신 **특정 head만 보면 매우 선명한 grounding map이 등장함**  
→ LVLM은 내부적으로 이미 grounding 기능을 학습하고 있었음!

이 head들은 다음 특성을 가짐:

- 이미지 토큰에 높은 attention  
- 특정 객체 주변에 집중된 attention (low entropy)  
- 다양한 샘플에서도 일관적으로 동일 head 활성화

![models](https://github.com/user-attachments/assets/ea6656ca-b18f-4fce-aae9-69e614accb42)
> 모델별로 Segmentation이 잘되는 head들이 있다!!


#### 모델 Architecture  

![Image](https://github.com/user-attachments/assets/1cb78a2a-ea4e-4355-9a52-8e4ee320d7b4)

0. 필요한 지시문 프롬포트를 LVLM에 입력!  
  - input: 이미지와 (PXP, P 는 이미지 패치수) + L (텍스트 토큰수)
1. 마지막 문장의 토큰을 query(q_txt)로 하며, 이미지를 key(k_img)로 하는 모든 head에 대해 attention weight 계산  
  - 마지막문장을 선정한 이유는 Auto-regressive 디코더 구조로 마지막 토큰이 전체 문맥(context)을 가장 많이 포함
2. S_img가 높은 head만 남김  
  - S_img가 높다는 것은 이미지 토큰에 대해 적극적으로 보고있다는 뜻
  - 따라서 S_img가 낮은것은 이미지보다는 텍스트에 집중하는것일수 있다  
3. spatial entropy가 가장 낮은 head 10개 선택  
  - Spatial Entropy가 낮다는 것은, attention이 이미지의 특정 한 지역(=객체)에 집중되어 있다는 뜻  
  - "가장 낮은 head 10개"라는것은 이미지 내의 특정 부분을 집중적으로 찍은 10개라는것을 의미!  
4. 이를 1000개 sample에 반복  
5. **선택 횟수(frequency)가 가장 높은 상위 2~3개 head = Localization Heads**

놀라운 점:
- 다양한 LVLM(LLaVA·DeepSeek·InternVL 등)에서 항상 2~3개의 head만 반복적으로 선정됨  
- 즉, grounding은 소수의 head가 “전담 처리”한다는것을 알 수 있다!  


#### 이후 단계는 GT와의 비교를 위한 BBOX 만들기 혹은 Segmentation!  

![post](https://github.com/user-attachments/assets/59dc9912-029c-4d72-a239-a74bf5b34517)
1. Localization head attention map 추출  
2. Gaussian smoothing 적용  
3. attention maps sum  
4. thresholding → pseudo-mask 생성  
5. convex hull → bounding box  
6. segment-anything(SAM)과 결합 가능

> **추가 학습 없음, Decoder 없음, Fine-tuning 없음!**

### 🧪 실험 결과 및 Ablation   
 
#### 기존 연구와의 비교

![result](https://github.com/user-attachments/assets/9ddeb95b-cd7b-461d-94cc-c8c75b251619)

→ Training-free 모델로는 성능이 최고이다!! bbox. segmentation 에서 모두!!

#### Ablation Test  

![abla1](https://github.com/user-attachments/assets/d41f9bf4-8955-4cdb-b28e-af423a816f5f)

→ 3개정도의 헤드를 찾았을때 제일 성능이 좋았다!

![abla2](https://github.com/user-attachments/assets/99b822b4-5fb8-4e1f-b89f-9b362e2a30ee)  

- S_img : 이미지 전체에 대하여 많이 보는지 여부  
- H_A_l_h : 이미지의 일부 스팟에 집중해서 보는지여부   
- Fixed = 글로벌(best-of-best) head 집합  
- Greedy = 로컬(best-per-sample) head 선택  

> 결국 이미지 전체를 많이보며, 그 중 일부 스팟에 집중한 것에 대하여, 글로벌로 Layer-Head를 선정한것에 대하여 가장 성능이 좋았다!  

---

## ✅ 결론  

**LVLM 내부에는 이미 ‘grounding 전담 head’가 존재하며,  
이를 정확히 찾아 조합하면 Fine-tuning 없이도 SOTA 수준 grounding이 가능하다.**

