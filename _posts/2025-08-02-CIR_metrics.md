---
layout: post
title: "📊 Evaluation Metrics in CIRR - CIRR분야의 Metrics 알아보기"
author: [DrFirst]
date: 2025-08-02 09:00:00 +0900
categories: [AI, Research]
tags: [CIR, Metrics, Evaluation, Image Retrieval, Recall, TIFA, mAP, CIRR, CIRCO]
sitemap :
  changefreq : monthly
  priority : 0.8
---

### 🧠 Understanding Metrics in the CIRR Domain  

![Image](https://github.com/user-attachments/assets/5e3b4dac-28b6-4012-b460-13b01bd21c4c)

- While exploring **OSrCIR** and **CIReVL**, several evaluation metrics commonly used in the CIRR domain appeared.  
- Since these metrics also frequently appear in other research, I studied their definitions more carefully!  

#### 1. Recall@K (R@K)  
- **Definition**: A metric that measures the proportion of queries where the ground-truth image is within the Top-K retrieved results.  

$$
Recall@K = \frac{\#\{\text{쿼리에서 Top-K 안에 정답 존재}\}}{\#\{\text{전체 쿼리}\}}
$$
 
- **Strengths**: Simple to compute and highly intuitive; widely adopted as the standard in most CIR benchmarks.  
- **Limitations**: Fails to fully capture cases with multiple correct answers, and may **underestimate performance in datasets with many false negatives** (e.g., CIRR).  
- **Common Usage**: CIRR (ICCV 2021), FashionIQ.  

---

#### 2. Subset Recall@K  
- **Definition**: Similar to Recall@K, but instead of the whole database, evaluation is restricted to a **smaller predefined subset of related images** (e.g., the same scene group).  
- The subset is **not chosen arbitrarily by the researcher**, but is **predefined within the dataset** itself.  
- **Strengths**: Allows for fine-grained comparison of model performance within a small candidate pool.  
- **Limitations**: Does not directly reflect retrieval performance across the entire DB, and results may vary depending on how subsets are defined.  
- **Common Usage**: CIRR Subset Evaluation.  

---

#### 3. mAP@K (mean Average Precision at K)  
- **Definition**: Considers the **ranking positions of all correct answers** within the Top-K results and computes the average precision.  
$$
mAP@K = \frac{1}{N}\sum_{i=1}^N \frac{1}{|G_i|}\sum_{j=1}^{|G_i|} Precision@r_{ij}
$$

  (where \(r_{ij}\) is the rank of the j-th ground-truth image for the i-th query)  
- **Strengths**: Fair in **multi-ground-truth** scenarios; reflects retrieval quality more precisely than Recall@K.  
- **Limitations**: More complex to compute and less intuitive than Recall.  
- **Common Usage**: CIRCO (ICCV 2023), GeneCIS.  

---

#### 4. TIFA (Text-to-Image Faithfulness Assessment, Hu et al. 2023)  
> I will write a separate post on TIFA in more detail soon!!  

- **Definition**: A metric for evaluating the **faithfulness of generated images** to the input text prompt.  
  - Uses a **VQA (Visual Question Answering) model** to automatically generate questions like *“Is there a dog?”*, *“Is the dog brown?”* and compares them with the image.  
- **Strengths**: Goes beyond Recall by measuring **textual faithfulness**, and shows strong correlation with human evaluation.  
- **Limitations**: Highly dependent on the VQA model’s performance; incurs additional computational cost; not yet a fully standardized evaluation metric in CIR.  
- **Common Usage**: Hu et al. (ICCV 2023), Vision-by-Language (CIReVL, ICLR 2024).  

---

👉 **Summary**:  
- **Recall@K** → the basic standard metric in CIR.  
- **Subset Recall** → used in CIRR for fine-grained model comparison.  
- **mAP@K** → ensures fairness in multi-ground-truth settings (CIRCO, GeneCIS).  
- **TIFA** → evaluates textual faithfulness, providing a more human-aligned interpretation.  


---


### 🧠 (한국어) CIRR분야의 Metrics 알아보기 

![Image](https://github.com/user-attachments/assets/5e3b4dac-28b6-4012-b460-13b01bd21c4c)

- OSrCIR, CIReVL을 진행하며 나왔던 CIRR 분야의 모델 평가 Metrics!  
- 다른연구에서도 많이 나오기에 정의에 대하여 공부해보았습니다!! 

#### 1. Recall@K (R@K)  
- **정의**: 검색된 Top-K 결과 안에 정답 이미지가 존재하는 비율을 측정하는 지표.  

$$
Recall@K = \frac{\#\{\text{쿼리에서 Top-K 안에 정답 존재}\}}{\#\{\text{전체 쿼리}\}}
$$

- **장점**: 계산이 단순하고 직관적이며, 대부분의 CIR 벤치마크에서 표준적으로 사용됨.  
- **한계**: 정답 이미지가 여러 개인 경우 이를 충분히 반영하지 못하며, False Negative가 많은 데이터셋(CIRR)에서는 성능을 과소평가할 수 있음.  
- **주 사용처**: CIRR(ICCV 2021), FashionIQ.  

---

#### 2. Subset Recall@K  
- **정의**: Recall@K와 유사하지만, 전체 DB가 아닌 쿼리와 관련된 **작은 서브셋(예: 동일한 장면의 이미지 그룹)**에서만 평가.  
- Subset은 연구자가 임의로 뽑는 게 아니라, 데이터셋 자체에 미리 정의되어 있는 관련 이미지 그룹을 의미  
- **장점**: 작은 후보군 내에서 모델의 세밀한 성능 차이를 평가할 수 있음.  
- **한계**: 전체 DB 검색 성능을 직접적으로 반영하지 못하고, Subset 정의 방식에 따라 편향 발생 가능.  
- **주 사용처**: CIRR Subset Evaluation.  

---

#### 3. mAP@K (mean Average Precision at K)  
- **정의**: Top-K 결과에서 **모든 정답 이미지의 위치(rank)**를 고려해 평균 정밀도를 계산.  
$$
mAP@K = \frac{1}{N}\sum_{i=1}^N \frac{1}{|G_i|}\sum_{j=1}^{|G_i|} Precision@r_{ij}
$$

  (여기서 \(r_{ij}\)는 i번째 쿼리의 j번째 정답 이미지의 순위)  
- **장점**: 다중 정답(Multi-Ground Truth) 상황에서도 공정하게 평가 가능, Recall@K보다 세밀하게 검색 품질을 반영.  
- **한계**: 계산이 복잡하며, 직관적으로 이해하기는 Recall보다 어려움.  
- **주 사용처**: CIRCO(ICCV 2023), GeneCIS.  

---

#### 4. TIFA (Text-to-Image Faithfulness Assessment, Hu et al. 2023)  
> TIFA에 대하여는 다시한번 포스팅해보겠쓰므니다!!  

- **정의**: 텍스트 조건(prompt)과 결과 이미지의 충실도를 평가하는 지표.  
  - VQA(Visual Question Answering) 모델을 이용해 "강아지가 있나요?", "강아지는 갈색인가요?" 같은 질문을 자동 생성하고, 이미지 결과와 비교.  
- **장점**: Recall처럼 단순 정답 여부가 아니라, **텍스트 조건 충실성(faithfulness)**을 정밀하게 평가 가능.  
  - 사람 평가와 높은 상관성을 보임.  
- **한계**: VQA 모델의 성능에 크게 의존하며, 연산 비용이 추가됨. 아직 CIR에서 완전한 표준은 아님.  
- **주 사용처**: Hu et al. (ICCV 2023), Vision-by-Language (CIReVL, ICLR 2024).  

---

👉 **정리**:  
- CIR 분야는 **Recall@K**를 기본 지표로 활용.  
- **Subset Recall** → CIRR에서 세밀한 비교용.  
- **mAP@K** → CIRCO, GeneCIS처럼 다중 정답 환경에서 공정성 확보.  
- **TIFA** → 텍스트 조건 충실성까지 평가하여 인간 친화적 해석 가능.  


