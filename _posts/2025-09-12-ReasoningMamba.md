---
layout: post
title: "🐍 Reasoning Mamba: Hypergraph 기반 추론으로 Weakly Supervised Affordance Grounding 강화!"
author: [DrFirst]
date: 2025-09-12 07:00:00 +0900
categories: [AI, Research]
tags: [Computer Vision, Affordance, Weakly-Supervised, Hypergraph, Mamba, CVPR 2025, CVPR, Robotics]
sitemap:
  changefreq: monthly
  priority: 0.8
---

### 🐍 (한국어) Reasoning Mamba: Hypergraph + Mamba로 Affordance Grounding 문제 해결!  

![Image](https://github.com/user-attachments/assets/rmamba-cvpr2024)

* **제목**: [Reasoning Mamba: Hypergraph-Guided Region Relation Calculating for Weakly Supervised Affordance Grounding](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Reasoning_Mamba_Hypergraph-Guided_Region_Relation_Calculating_for_Weakly_Supervised_Affordance_CVPR_2025_paper.pdf)  
* **학회**: CVPR 2024  
* **저자**: Yuxuan Wang, Aming Wu, Muli Yang, Yukuan Min, Yihang Zhu, Cheng Deng (Xidian Univ. & A*STAR)  
* **핵심 키워드**: `Affordance`, `Weakly-Supervised`, `Hypergraph`, `State-Space Model`, `Mamba`, `Robotics`  
* **요약**: R-Mamba는 **사물의 기능 단위(affordance)**를 더 정확히 찾기 위해, 사물 부위 간 관계를 **Hypergraph**로 모델링하고, 이를 **Mamba 기반 State-Space 모델**로 전역적으로 재조직하는 새로운 접근을 제안. AGD20K, HICO-IIF 등에서 SOTA 성능 달성! 🚀  

---

### 🚀 연구 핵심 요약

> 한 줄 요약: **“R-Mamba = Hypergraph로 지역 관계 포착 + Mamba로 전역 추론 → Affordance Grounding 성능 업그레이드!”**

1) **새 과제 배경 (WSAG)**  
- **Weakly Supervised Affordance Grounding (WSAG)**: 픽셀 단위 라벨 없이, 이미지 수준 라벨만으로 affordance 영역(localization) 찾기  
- 기존 방법: 단순 CAM 기반 → 사물의 복합적 부위 관계(예: 컵 손잡이 + 몸통 = 따르기)를 무시  

2) **R-Mamba 방법론**  
- **Hypergraph Construction**: 사물 부위(feature)를 vertex로 두고, 여러 부위를 동시에 연결하는 hyperedge 구성 → 다대다 관계 표현  
- **Hypergraph Evolution**: 불필요한 연결 제거 + affordance 관련 부위 강화 (K-means, Gumbel Softmax)  
- **Hypergraph-guided State Space (HSS) Block**: Hypergraph 특징을 Mamba 기반 selective scan으로 전역적 관계로 재조직  
- **Ego-HSS / Exo-HSS 모듈**: egocentric / exocentric 이미지에서 추출한 affordance 관련 관계를 교차 학습  

3) **최종 출력**  
- affordance heatmap을 통해 객체에서 어떤 부위가 "잡기, 따르기, 앉기" 등의 기능과 대응되는지 정확히 지역화  

---

### 🔍 기존 연구의 한계와 차별점  

- 기존 WSAG 연구:  
  - CAM 기반 활성화 맵 활용 → 단순 부분 강조  
  - Graph Neural Network(GNN) 기반 → 점-점 관계만 처리, 다대다 관계 표현 부족  
- R-Mamba의 차별점:  
  - **Hypergraph**로 **복수 부위 간 관계** 포착  
  - **Mamba 기반 State-Space 모델**로 **전역 시퀀스 스캔**  
  - Egocentric–Exocentric 양방향 학습으로 일반화 능력 강화  

---

### 🧱 R-Mamba 구조 (Architecture)

![Image](https://github.com/user-attachments/assets/rmamba-arch-2024)

1) **Hypergraph Construction**  
  - DINO-ViT로 feature 추출 후, feature point → vertex, 인접 feature 묶음 → hyperedge  
  - vertex–hyperedge 변환으로 지역 관계 강화  

2) **Hypergraph Evolution**  
  - K-means로 cluster center 기반 hyperedge 확장  
  - egocentric feature로 affordance 관련 vertex/edge 선택  
  - Gumbel Softmax로 불필요한 edge 제거  

3) **Hypergraph-guided State Space (HSS) Block**  
  - Evo-hypergraph를 입력 받아 Mamba 기반 selective scan 적용  
  - 지역 관계(local)를 전역(global) 맥락에서 재조직  
  - Ego-HSS와 Exo-HSS로 양방향 학습  

4) **출력 단계**  
  - affordance heatmap + classification score 산출  
  - Loss: cross-entropy + cosine similarity + geometric concentration  

---

### 🧪 실험 결과  

#### 데이터셋 & 지표  
- **AGD20K (seen/unseen split)**  
- **HICO-IIF**  
- 평가 지표: **KLD ↓, SIM ↑, NSS ↑**  

#### 결과  

- AGD20K-seen: **KLD 1.173, SIM 0.414, NSS 1.247** (기존 LOCATE 대비 성능 향상)  
- AGD20K-unseen: **KLD 1.372, SIM 0.380, NSS 1.190**  
- HICO-IIF: 기존 모델 대비 성능 우수  

#### 정성적 비교 (Qualitative)  
- 컵 손잡이, 칫솔 끝 등 작은 affordance 부위 정확히 탐지  
- 배경 간섭 억제 및 unseen 객체에서도 일반화 잘 수행  

---

### 🧪 Ablation 분석  

- **Distance Threshold (ε)**: 적절한 값(3)일 때 최고 성능, 너무 크면 배경 간섭 ↑  
- **Cluster Number (k)**: 5~7 범위에서 안정적 성능  
- **Loss Function**: cosine similarity loss (L_sim) + geometric concentration loss (L_gc) 조합이 가장 효과적  
- **Component Ablation**: Hypergraph, HSS, Evolution 제거 시 모두 성능 하락 → 각 모듈 기여 확인  

---

## ✅ 결론  

- R-Mamba는 **Hypergraph + Mamba 결합**으로 affordance localization을 크게 향상  
- 주요 기여:  
  1. Hypergraph로 다대다 부위 관계 포착  
  2. HSS block으로 지역 관계를 전역적 관점에서 재구성  
  3. 다양한 데이터셋에서 SOTA 수준 성능 달성  
- → 로봇 지각, 인간-로봇 상호작용(HOI), AR/VR 등 **실세계 응용**에 중요한 기여 🎯  

---
