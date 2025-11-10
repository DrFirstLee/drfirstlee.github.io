---
layout: post
title: "🔍 Contrastive Learning을 통한 Affordance Grounding 문제해결! (ICCV 2025)"
author: [DrFirst]
date: 2025-11-1009:00:00 +0900
categories: [AI, Research]
tags: [Computer Vision, Affordance, Weakly-Supervised, Cloased-Loop, ICCV 2025, ICCV]
sitemap:
  changefreq: monthly
  priority: 0.8
---

### 🔍 Closed-Loop Transfer for Weakly-supervised Affordance Grounding 논문 읽기!  

![manhwa]()

* **제목**: [Closed-Loop Transfer for Weakly-supervised Affordance Grounding](https://arxiv.org/pdf/2510.17384)  
* **학회 및 저자**: Tang et al., ICCV 2025  
* **요약**: 기존 연구인 LOCATE의 일방향 transfer를 넘어, 쌍방향 Transfer를 통해 고도화하자!! 🚀    
 

---

### 🚀 연구 핵심 요약

> 한 줄 요약: **기존 연구인 LOCATE의 일방향 transfer를 넘어, 쌍방향 Transfer를 통해 고도화하자!! 🚀**   
> - 일방향 : Activation -> Localization    
> - 양방향 : Activation -> Localization  + Localization -> Activation      

![Image](https://github.com/user-attachments/assets/38862db4-7e68-4154-b73a-a908da2684ad)

1) **Open Problems!**  
- 기존 연구인 LOCATE는 exo에서 ego로으 knowledge transfer만 있다!!  
  - 그래서 `exocentric (interaction-centered)` 관점과 `egocentric (object-centered)` 관점의 차이가 생기는데, 반영을 못하고!  
  - exo의 가림도 반영이 안되기에 진짜 affordance 영역을 잘 못찾는다!  

2) **Their solutions**  
① 양방향의 학습법(LoopTrans)를 통해서 exo와 ego domain을 모두 학습한다!  

---

### 🔍 기존의 관련 연구들!    

1. [LOCATE!!](https://drfirstlee.github.io/posts/LOCATE/)    
  - CAM(Class Activation Map)을 기반, Action Label을 바탕으로하는 Weakly Supervisedlearning 방법론을 제시  
  - Exo 이미지의 CAM 모델을 만들어 ego 이미지에 적용,   
  - 결국, exo 의 추출 내용을 ego 이미지에 반영하는 일방향의 연구!  
  - **LOCATE** : Li et al., "Locate: Localize and transfer object parts for weakly supervised affordance grounding." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.  

2. [WSMA!!](https://drfirstlee.github.io/posts/WSMA/)    
  - exo 이미징서 상호작용 affordance 지식을 추출하고,  
  - ego 이미지와 텍스트 특징을 결합한것과 align 한다.  
  - 결국, exo 의 추출 내용을 ego 이미지에 반영하는 일방향의 연구!  
  - Xu et al., "Weakly Supervised Multimodal Affordance Grounding for Egocentric Images" AAAI 2024  

### 🔍 본 연구의 방법론!!!    

#### 4 모델 Architecture  

![Image](https://github.com/user-attachments/assets/15d0fc26-2dfd-4cbe-a671-a0b7df7ee04c)

A. 𝜣_pixel : 최종 inference에 사용되는 모델.  
  - 추론단계 : I_ego => DINO-ViT => F_ego => 𝜣_pixel 을 통해 최종 heatmap(P) 가 나온다.  
B. 𝜣_scam : ego 이미지와 exo 이미지를 동시에 활용하여 학습된, 공통 CAM 모델 (shared CAM)으로 𝜣_pixel 학습에 활용됨    
C. Loss는?  3가지 Loss로 구성  
  - 1번.  Interaction → Activation 로,  CAM 모델을 학습시키는 Loss. `L_cls`  
  - 2번. Activation → Localization 으로써, ego 이미지의 activation 결과를 모델 locatalization에 적용시키는 `L_pixel`  
  - 3번. Localization -> Activation 로써, Localized 된 결과를 Activation 과 align 하는 `L_dill`   
  - 4번. exo와 ego 의 align을 담당하는 `L_corr`   


#### 4.1 Unified Exo-to-Ego Activation  
> 𝜣_scam 을 학습시킨다!!  

![Image4.1](https://private-user-images.githubusercontent.com/43365171/512231866-9a3c6263-3ac2-45c2-94f2-ab27d5ad48bc.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjI3ODc3NjIsIm5iZiI6MTc2Mjc4NzQ2MiwicGF0aCI6Ii80MzM2NTE3MS81MTIyMzE4NjYtOWEzYzYyNjMtM2FjMi00NWMyLTk0ZjItYWIyN2Q1YWQ0OGJjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTExMTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMTEwVDE1MTEwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJhNWM4ZGUzNjhlNDczZjgyY2M5ODk0NWRmMTU4Njg0MDQ5MDEwZjAyNDljNzQ4ZWFiMWMxNDg0MzYxMGUxYTUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.sPuL_WCsnnnLOQQzLcDn0gLZvhVv4oEcv2ervZRo4aE)

- exo 와 exo 이미지를 통합, action 을 label로하는 shared CAM 모델을 만든다!
- `L_cls`를 통하여 학숩됨  

#### 4.2. Region Activation to Pixel Localization  
> inference의 메인 모델인 𝜣_pixel 을 학습시킨다!!  

![Image4.2](https://private-user-images.githubusercontent.com/43365171/512233616-0c3829a6-cb22-4654-aa10-106d4c5ca5dc.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjI3ODc5NzUsIm5iZiI6MTc2Mjc4NzY3NSwicGF0aCI6Ii80MzM2NTE3MS81MTIyMzM2MTYtMGMzODI5YTYtY2IyMi00NjU0LWFhMTAtMTA2ZDRjNWNhNWRjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTExMTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMTEwVDE1MTQzNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI2MWQzZGY0ZDliNDFjODZmMjIwOTZjMDhlMzc1NWRiZTAxMTQ5MzUyZDgzMmE0ZGQ4NWNkOGExYTE0YjBkNDImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.3PS_Z_fM3fjLRUqiSYx6cnLN2TgxcKbfI413QtJCjKY)

- DINO 결과를 통해 clusted 된 Localization 결과물(M_ego)에 우리의 activation 결과를 일치시킨다.  
- 즉! Activation → Localization  
- `L_pixel`를 통하여 학습됨  


#### 4.3. Ego-to-Exo Denoising Distillation
> dino의 activation 결과에 우리의 localization 결과를 align 시킨다!  

![iamge4.3](https://github.com/user-attachments/assets/e17df4c2-2c8c-4a1b-8dd5-c9b537c8c439)  

1. 우리의 메인모델 𝜣_pixel 를 통해 나온 결과물(activation) 이 exo이미지의 ViT의 결과(localization) 와 align되며!  
2. 추가로, noise head를 통해 추출된 `f_noise_m`이 xo이미지의 ViT의 결과(localization)와 negative align 된다!!  

이를 통해서 Loop의 마지막인 Localization -> Activation  진행!!  


### 🧪 실험 결과 및 Ablation   
 
#### Ablation Test  

![ablation](https://github.com/user-attachments/assets/0ade267a-fc8a-47c2-a79c-b7daec39cdd7)

- Base line은 LOCATE에서 part selection 뺸 것!! L_corr 만 있음  
- Activation → Localization 안되고, Localization -> Activation만 된 1번 부분은 별로 안좋네!?  
- 모두 합쳐진 Closed-Loop에서 역시 제일 좋구나!! 

![Image](https://github.com/user-attachments/assets/1f2d1c61-0b4b-4a91-a88c-f1dbafec5c88)

최종 성능도 좋았다!!!

---

## ✅ 결론  

- 양방향 학습을 통한 새로운 방법 정의!  

---
