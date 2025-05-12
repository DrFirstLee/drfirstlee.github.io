---
layout: post
title: " 🦖 DINO: The Evolutionary Object Detection Model of DETR!! - DINO: DETR의 진화형 객체 탐지 모델!! (ICLR 2023)"
author: [DrFirst]
date: 2025-05-09 09:00:00 +0900
categories: [AI, Research]
tags: [DETR, DINO, 객체 탐지, Object Detection, Transformer, 딥러닝, CV, ICLR, ICLR 2023]
lastmod : 2025-05-09 09:00:00
sitemap :
  changefreq : monthlyg
  priority : 0.9
---


## 🦖 DINO: The Evolutionary Object Detection Model of DETR!!
_🔍 A powerful alternative that solves the slow training and small object detection issues of DETR-based models!_

> Paper: [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605)  
> Presentation: ICLR 2023 (by IDEA Research)  
> Code: [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)  
> Comment: After DETR was released, DAB-DETR/ DN-DETR / Deformable DETR, etc., were continuously released, and this model combines their concepts with DINO's own concepts. It's difficult to understand for someone who has only studied DETR!  

---

### ✅ What is DINO?

![manwha](https://github.com/user-attachments/assets/7cada129-804b-45da-a99e-0bfdd91d42eb)

> DINO is an object detection model that **overcomes the limitations of the DETR family**  
> Designed with a focus on **improving training speed** and **small object performance**  

- DINO = **DETR with Improved DeNoising Anchors**
- Basic structure is DETR-based, but performance is enhanced through various strategies
- Achieves **performance comparable to Two-stage** with a **One-stage** structure!

---

### 🚨 Background of DINO's Emergence - Major Limitations of DETR
1. ❌ **Training is too slow** (hundreds of thousands of steps)
   - In the early stages of training, DETR's object queries **predict boxes at random locations**
   - This makes effective matching between queries and GT difficult, resulting in sparse learning signals
   - → Consequently, the **convergence speed is very slow**, requiring dozens of times more epochs than typical models (500 epochs!?)

2. ❌ **Weak at detecting small objects**
   - DETR uses only the final feature map of the CNN backbone, resulting in **low resolution**
     - (e.g., using C5 level features of ResNet → resolution reduction)
   - Information about small objects almost disappears or is faintly represented in this coarse feature map
   - Also, **Transformer focuses on global attention**, making it weak in local details
   - → As a result, **box predictions for small objects are not accurate**

3. ❌ **Low performance of Object Query in the early stages of learning**
   - DETR's object queries are **randomly initialized** in the beginning
   - The **role of which query will predict which object is not determined** in the early stages of learning
   - Hungarian Matching forcibly performs 1:1 matching, but this matching is **inconsistent**
   - → In the early stages of learning, queries often **overlap or predict irrelevant locations**, leading to low performance

---

### Briefly Looking at Additional Research in the DETR Family Before DINO

> Here's a brief summary of the major DETR family research before DINO!!  
> We should study each of these researches as well!!  

The following studies have attempted to improve various aspects such as convergence speed, learning stability, and positional accuracy while maintaining the basic framework of DETR.

---

#### 🔹 **Deformable DETR (2021, Zhu et al.)**
- Core Idea: **Deformable Attention**
  - Performs attention only on a **few significant locations** instead of the entire image.
- Advantages:
  - Significantly improved training speed (more than 10 times)
  - Introduction of a two-stage structure enables coarse-to-fine detection

---

#### 🔹 **Anchor DETR (2021, Wang et al.)**
- Redefined Query in an **Anchor-based manner**.
- Enables **better local search** by having Query possess location information.

---

#### 🔹 **DAB-DETR (2022, Liu et al.)**
- Initializes Query as a **Dynamic Anchor Box** and refines it progressively in the decoder.
- Improves convergence by providing stable location information from the early stages of learning.

---

#### 🔹 **DN-DETR (2022, Zhang et al.)**
- Introduced **DeNoising Training** for learning stabilization.
- By including **fake queries with added noise to the ground truth (GT) boxes** in the training,
  Contributes to **resolving the instability of Bipartite Matching**.

---

### 💡 Core Ideas of DINO

> The reason why understanding DAB-DETR/ DN-DETR / Deformable DETR is necessary!!  
> This research successfully combines DINO's own additional ideas (CDN, Mixed Query Selection) with successful cases from previous DETR research!  

| Main Components                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Introduced Paper (Source)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  **DeNoising Training** (+CDN) | Intentionally generates noise boxes around GT during training to quickly converge Queries. <br> DINO extends this contrastively to perform Contrastive DeNoising (CDN) to distinguish between correct and incorrect predictions.                                                                                                                                                                                                                                                                                                                                                                                                                                                   | **DN-DETR** [G. Zhang et al., 2022] + **DINO** [Zhang et al., 2022]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|  **Matching Queries** | Places fixed Query Anchors at locations close to GT to induce stable learning.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | **DAB-DETR** [Liu et al., 2022]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|  **Adding Two-stage Structure** | The Encoder extracts coarse object candidates, and the Decoder performs refinement.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | **Deformable DETR** [Zhu et al., 2021]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|  **Look Forward Twice** | Improves accuracy by giving attention twice in the Decoder instead of once.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | **DINO** [Zhang et al., 2022]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|  **Mixed Query Selection** | Uses only the top-K locations selected from the Encoder as Anchors, and the Content remains static to balance stability and expressive power.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | **DINO** [Zhang et al., 2022]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

---

####  Idea 1: DeNoising Training (+ CDN)

DINO additionally uses **intentionally noisy training samples (denoising query)** to help object queries quickly recognize information around the **ground truth (GT)** in the early stages of training. This strategy alleviates the existing unstable bipartite matching issue and leads to DINO's unique extension, **CDN (Contrastive DeNoising)**.

---

#####  Basic DeNoising Training Method

1. **GT Replication & Noise Addition**
   - Replicates the ground truth box and label
   - Adds **position noise** (e.g., coordinate jitter 5~10%) and **class noise** (e.g., person → dog)

2. **Denoising Query Generation**
   - Designates some object queries as denoising queries
   - Induces learning to predict the noisy boxes

3. **Loss Calculation**
   - Calculates the prediction error for noise queries separately from normal matching queries and includes it in the training

---

#####  CDN (Contrastive DeNoising): DINO's Extension

Extending the existing denoising technique, DINO introduces a contrastive strategy that simultaneously trains **positive / negative query pairs**.

| Query Type       | Generation Method                         | Learning Objective                      |
|------------------|-----------------------------------------|---------------------------------------|
|  Positive Query | Slight noise added to GT (position/class) | Induce accurate prediction              |
| ❌ Negative Query | Insert random location or incorrect class | Induce definite 'incorrect' prediction |

- Both types are put into the same decoder, and a different learning objective (loss) is assigned to each.

---

##### ⚙️ Main Components

| Component         | Description                              |
|------------------|------------------------------------------|
| Positive Query   | Slight noise added to GT box             |
| Negative Query   | Incorrect box/class unrelated to GT      |
| Matching Head    | Generates prediction results for each    |
| Loss             | Induces Positive to match GT, Negative to no-object |

---

#####  Summary of CDN Effects

- **Reduced false positives**
  → Prevents false detections in similar backgrounds/small objects/overlap situations

- **Induced faster convergence**
  → Queries that were random in the early stages quickly move closer to the correct answer

- **Improved model's discrimination ability**
  → Strengthens the ability to distinguish between correct answers and similar incorrect answers

---

#####  Key Summary

| Item           | Description                                 |
|----------------|---------------------------------------------|
|  Purpose     | Enhance the ability to distinguish correct answers from similar incorrect answers |
|  Strategy    | Extend DeNoising query to positive/negative |
| ✅ Learning Effect | Fast convergence + high accuracy + robust detection |

---

> CDN is not just a simple learning stabilization technique; it is a core technology that makes **DINO the fastest and most robust DETR-based model** to train.

---

####  Idea 2: Matching Queries (Fixed Anchor Based)

Unlike DETR, DINO's object queries do not find locations completely randomly but rather **place pre-defined query anchors near GT locations from the beginning**.

---

#####  How it Works

1. **GT Center Anchor Generation**
   - Generates a fixed number of query anchors based on GT locations during training

2. **Query Assignment to Each Anchor**
   - These anchors are assigned as responsible queries to predict specific GTs

3. **Matching Process Stabilization**
   - Hungarian Matching makes it easier to match these anchor queries and GTs in a 1:1 manner

---

#####  Effects

- Queries **start near GT, leading to faster convergence**
- **Reduces the matching instability issues** that occurred in the early stages
- Improved **performance and convergence speed** due to each GT having a clearly corresponding query

---

####  Idea 3: Two-stage Structure

DINO extends the existing one-stage structure of DETR by applying a **two-stage structure** consisting of **Encoder → Decoder**.

---

#####  How it Works

1. **Stage 1 (Encoder)**
   - Extracts **dense object candidates (anchors)** through a CNN + Transformer encoder
   - Selects Top-K scoring anchors

2. **Stage 2 (Decoder)**
   - Performs **refined prediction** based on the anchors selected from the Encoder
   - Adjusts class and accurate box

---

#####  Effects

- Coarsely identifies locations in the first stage and accurately adjusts them in the second stage → **Improved precision**
- Increased **detection stability** in small objects or complex backgrounds

---

####  Idea 4: Look Forward Twice (LFT)

![LFT](https://github.com/user-attachments/assets/774eb274-d61a-48e6-9115-a5d27254acc9)

Existing DETR-based models perform attention once on the encoder feature by the object query in the decoder. DINO **repeats this attention operation twice (Look Twice)** to induce deeper interaction.

---

#####  How it Works

1. **First Attention**
   - Object query performs basic attention with the encoder output

2. **Second Attention**
   - Performs attention on the encoder feature again with the first attention result
   - That is, **query → encoder → query → encoder**

---

#####  Effects

- Utilizes deeper context information
- Enables **accurate class and location prediction** even in complex scenes
- Secures **strong representation power**, especially for overlapping objects and small objects

---

####  Idea 5: Mixed Query Selection (MQS)

Existing DETR-based queries mostly used the **same static queries for all images**, and while there was a method like Deformable DETR that used dynamic queries, changing the content as well could cause confusion. DINO introduces a **Mixed Query Selection** strategy that compromises the advantages of both.

---

#####  How it Works

![MQS](https://github.com/user-attachments/assets/49f1db5e-22b4-4ed5-95cf-f17a3834bfd3)

1. **Select Top-K Important Encoder Features**
   - Selects the **features with high objectness scores** from the encoder output

2. **Anchor (Location Information) is Dynamically Set**
   - Sets the **initial anchor box** of each query based on the selected Top-K locations

3. **Content Remains Static**
   - The content information of the query remains the **learned fixed vector** as is

> In other words, a structure where **"where to look changes depending on the image"** and **"what to look for remains as the model has learned."**

---

#####  Effects

- Starts searching from more accurate locations (anchors) suited for each image
- Prevents confusion caused by ambiguous encoder features by maintaining content information
- Achieves **fast convergence + high precision** simultaneously

---

##### ✅ Summary

| Component        | Method                                                         |
|-----------------|----------------------------------------------------------------|
| Anchor (Location)| Initialized with the location of the Top-K features extracted from the Encoder |
| Content (Meaning)| Maintains a static learned vector                              |
| Expected Effect  | Adapts to the location of each image + maintains stable search content |

---

###  DINO Architecture

![archi](https://github.com/user-attachments/assets/8dcc79ba-981a-4a29-b67d-c460e87ff535)

```
Input Image
 → CNN Backbone (e.g., ResNet or Swin)
   → Transformer Encoder
     → Candidate Object Proposals (Two-stage)
       → Transformer Decoder
         → Predictions {Class, Bounding Box}₁~ₙ
```


---

####  Explanation of Main Architecture Stages

> DINO maintains the simplicity of the existing DETR while also being one of the **definitive DETR models** that enhances **training speed, accuracy, and stability**.


##### 1. ️ Input Image
- The input image is typically entered into the model in 3-channel RGB format.

##### 2.  CNN Backbone
- e.g., **ResNet-50**, **Swin Transformer**, etc.
- Role of extracting **low-level feature maps** from the image

##### 3.  Transformer Encoder
- Receives features extracted from the CNN and learns **global context information**
- Enables each position to relate to other parts of the entire image

##### 4.  Candidate Object Proposals (Two-stage)
- Selects the **Top-K locations with high objectness** from the Encoder output
- Configures the **initial anchor** of the query based on this (including Mixed Query Selection)

##### 5.  Transformer Decoder
- Queries perform attention twice on the encoder feature (**Look Forward Twice**)
- Denoising queries are also processed together to induce stable learning (including CDN)

##### 6.  Predictions
- Finally predicts the **object class and box location** for each query
  → Result: N `{class, bounding box}` pairs are output



---

###  Final Summary: DINO vs DETR

| Item                   | DETR                     | DINO (Improved)              |
|------------------------|--------------------------|------------------------------|
| Training Convergence Speed | Slow                     | ✅ Fast (DeNoising)           |
| Small Object Detection | Low                      | ✅ Improved                   |
| Object Query Structure | Simple                   | ✅ Added GT-based Matching   |
| Stage Structure        | One-stage                | ✅ Includes Two-stage Structure |

---

###  Summary

- DINO maintains the structure of DETR while being a model **quickly and accurately improved for practical use**.
- A core model that forms the basis of various subsequent studies (Grounding DINO, DINgfO-DETR, DINOv2)
-  A highly scalable model that combines well with the latest vision research such as **open-vocabulary detection** and **segment anything**!! :)

---

###  Personal Thoughts

DINO seems to be an excellent improvement research that solved the **learning efficiency and performance issues of DETR** by well combining various researches and merging them with their own new results! As the core concepts are shared when extending to Grounding DINO or DINOv2, **it is a model that must be remembered to understand DETR-based Transformer detection models!**

---


## 🦖 (한국어) DINO: DETR의 진화형 객체 탐지 모델 DINO!! 
_🔍 DETR 계열 모델의 느린 학습과 작은 객체 탐지 문제를 해결한 강력한 대안!_

> 논문: [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605)  
> 발표: ICLR 2023 (by IDEA Research)  
> 코드: [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)  
> 코멘트 : DETR 공개 이후, DAB-DETR/ DN-DETR / Deformable DETR 등 연속적으로 공개되었고 이들의 개념과 DINO 자체 개념을 조합하여 제안한 모델로,., DETR만 공부하고 넘어온 입장에서는 이해하기가 어렵다!   

---

### ✅ DINO란?

![manwha](https://github.com/user-attachments/assets/7cada129-804b-45da-a99e-0bfdd91d42eb)  

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

### 간단하게 살펴보는 DINO 전의 DETR 계열의 추가 연구

> DINO 이전의 주요 DETR 계열 연구들을 간략히 정리해보았습니다!!  
> 각각의 연구도 모두 공부해봐야하겠습니다!!  

아래 연구들은 DETR의 기본 골격을 유지하면서도, 수렴 속도, 학습 안정성, 위치 정밀도 등 다양한 측면에서 개선을 시도해왔습니다.  

---

#### 🔹 **Deformable DETR (2021, Zhu et al.)**
- 핵심 아이디어: **Deformable Attention**
  - 이미지 전체가 아닌 **몇 개의 유의미한 위치에만 attention**을 수행.
- 장점:
  - 학습 속도 대폭 향상 (10배 이상)
  - 두 단계(two-stage) 구조 도입으로 coarse-to-fine 탐지 가능

---

#### 🔹 **Anchor DETR (2021, Wang et al.)**
- Query를 **Anchor-based 방식**으로 재정의.
- Query가 위치 정보를 갖도록 하여 **더 나은 지역 탐색** 가능.

---

#### 🔹 **DAB-DETR (2022, Liu et al.)**
- Query를 **Dynamic Anchor Box**로 초기화하고 decoder에서 점진적으로 refine.
- 학습 초기부터 안정적인 위치 정보를 제공함으로써 수렴성 향상.

---

#### 🔹 **DN-DETR (2022, Zhang et al.)**
- 학습 안정화를 위한 **DeNoising Training** 도입.
- 정답 GT 박스에 **노이즈를 추가한 가짜 query**를 함께 학습에 포함시켜,
  **Bipartite Matching의 불안정성 해소**에 기여.

---


### 💡 DINO의 핵심 아이디어

> DAB-DETR/ DN-DETR / Deformable DETR 등의 이해가 필요한이유!!  
> DINO 자체의 추가 아이디어(CDN, Mixed Query Selection)과 기존 DETR 분야의 연구의 성공적 사례를 잘 조합한 연구입니다!  

| 주요 구성 요소               | 설명 | 도입한 논문 (출처) |
|----------------------------|------|---------------------|
| 🔧 **DeNoising Training** (+CDN) | 학습 시, GT 주위에 노이즈 박스를 일부러 생성하여 Query를 빠르게 수렴시킴 <br> DINO에서는 이를 Contrastive하게 확장하여 정답 vs 오답을 구분하는 학습(CDN)도 수행 | **DN-DETR** [G. Zhang et al., 2022] + **DINO** [Zhang et al., 2022] |
| 🧲 **Matching Queries**     | GT에 가까운 위치에 고정된 Query Anchor를 배치해 안정적인 학습 유도 | **DAB-DETR** [Liu et al., 2022] |
| 🧠 **Two-stage 구조 추가**  | Encoder에서 coarse object 후보를 뽑고, Decoder에서 refinement 수행 | **Deformable DETR** [Zhu et al., 2021] |
| 🔁 **Look Forward Twice**   | Decoder에서 한 번이 아니라 두 번 attention을 주는 방식으로 정확도 향상 | **DINO** [Zhang et al., 2022] |
| 🧩 **Mixed Query Selection** | Encoder에서 선택된 top-K 위치만 Anchor로 사용하고, Content는 static하게 유지하여 안정성과 표현력 균형 확보 | **DINO** [Zhang et al., 2022] |

---


#### 💡 아이디어 1: DeNoising Training (+ CDN)

DINO는 학습 초기에 object query들이 **정답(GT) 주변 정보를 빠르게 인식**하도록 돕기 위해,  
**의도적으로 노이즈가 섞인 학습 샘플(denoising query)**을 추가로 사용합니다.  
이 전략은 기존의 불안정한 bipartite matching 문제를 완화하고,  
DINO만의 확장 기법인 **CDN (Contrastive DeNoising)**으로 이어집니다.

---

##### 🔧 기본 DeNoising Training 방식

1. **GT 복제 & 노이즈 추가**
   - Ground truth box와 label을 복제
   - **위치 노이즈** (e.g., 좌표 jitter 5~10%)와 **클래스 노이즈** (e.g., person → dog) 추가

2. **Denoising Query 생성**
   - 일부 object query를 denoising query로 지정
   - 노이즈된 box를 예측하게 학습 유도

3. **Loss 계산**
   - 일반 matching query와 별도로, 노이즈 query에 대한 예측 오차도 함께 학습

---

##### 🧠 CDN (Contrastive DeNoising): DINO의 확장

기존 denoising 기법을 확장하여,  
**positive / negative query 쌍을 동시에 학습**하는 contrastive 전략을 도입합니다.

| Query 종류      | 생성 방식                            | 학습 목적                  |
|------------------|-------------------------------------|----------------------------|
| 🎯 Positive Query | GT에 약간의 노이즈 추가 (위치/클래스) | 정확한 예측 유도            |
| ❌ Negative Query | 무작위 위치나 잘못된 클래스 삽입     | 확실히 '오답'으로 예측 유도 |

- 두 종류를 동일한 decoder에 넣고,  
  각각 다른 방식으로 학습 목표(loss)를 부여

---

##### ⚙️ 주요 구성 요소

| 구성 요소         | 설명 |
|------------------|------|
| Positive Query   | GT box에 약간의 노이즈 추가 |
| Negative Query   | GT와 무관한 잘못된 box/class |
| Matching Head    | 각각에 대해 예측 결과 생성 |
| Loss             | Positive는 정답과 일치하게, Negative는 no-object로 유도 |

---

##### 🧠 CDN의 효과 요약

- **false positive 감소**  
  → 유사한 배경/작은 객체/overlap 상황에서 오탐 방지

- **빠른 수렴 유도**  
  → 초기에 무작위였던 query들이 빠르게 정답 근처로 이동

- **모델의 구분 능력 향상**  
  → 정답과 유사한 오답을 판별하는 표현력 강화

---

##### 📌 핵심 요약

| 항목           | 설명 |
|----------------|------|
| 🎯 목적         | 정답과 유사한 오답을 구분하는 능력 강화 |
| 💡 전략         | DeNoising query를 positive/negative로 확장 |
| ✅ 학습 효과     | 빠른 수렴 + 높은 정확도 + robust detection |

---

> CDN은 단순한 학습 안정화 기법을 넘어,  
> **DINO가 DETR 계열 중 가장 빠르고 강건하게 학습**될 수 있도록 만든 핵심 기술입니다.


---

#### 💡 아이디어2 : Matching Queries (고정 Anchor 기반)

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

#### 💡 아이디어3: Two-stage 구조

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

#### 💡 아이디어4: Look Forward Twice (LFT)

![LFT](https://github.com/user-attachments/assets/774eb274-d61a-48e6-9115-a5d27254acc9)

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

#### 💡 아이디어5: Mixed Query Selection (MQS)

기존 DETR 계열의 query는 대부분 **모든 이미지에서 동일한 static query**를 사용했으며,  
Deformable DETR처럼 dynamic query를 사용하는 방식도 있었지만 content까지 바꾸면서 오히려 혼란을 줄 수 있음    
DINO는 이 둘의 장점을 절충한 **Mixed Query Selection** 전략을 도입  

---

##### 🧲 작동 방식

![MQS](https://github.com/user-attachments/assets/49f1db5e-22b4-4ed5-95cf-f17a3834bfd3)

1. **Top-K 중요한 encoder feature 선택**
   - encoder 출력 중 **objectness 점수가 높은 feature들**을 골라냄

2. **Anchor (위치 정보)는 동적으로 설정**
   - 선택된 Top-K 위치를 기반으로 각 query의 **초기 anchor box를 설정**

3. **Content는 static하게 유지**
   - query의 내용 정보는 **학습된 고정된 vector** 그대로 사용

> 즉, **"어디를 볼지는 이미지에 따라 다르게"**,  
> **"무엇을 찾을지는 모델이 배운 대로 유지"**하는 구조

---

##### 🎯 효과

- 각 이미지에 맞는 더 정확한 위치(anchor)에서 탐색 시작
- content 정보를 유지함으로써 **모호한 encoder feature로 인한 혼란 방지**
- **빠른 수렴 + 높은 정탐률**을 동시에 달성

---

##### ✅ 요약

| 구성 요소      | 방식 |
|----------------|------|
| Anchor (위치)   | Encoder에서 뽑은 Top-K feature의 위치로 초기화 |
| Content (내용)  | Static한 학습 vector 유지 |
| 기대 효과      | 이미지별 위치 적응 + 안정적인 탐색 내용 유지 |


---

### 🧱 DINO 아키텍처 

![archi](https://github.com/user-attachments/assets/8dcc79ba-981a-4a29-b67d-c460e87ff535)

```
Input Image
 → CNN Backbone (e.g., ResNet or Swin)
   → Transformer Encoder
     → Candidate Object Proposals (Two-stage)
       → Transformer Decoder
         → Predictions {Class, Bounding Box}₁~ₙ
```

---

#### 🔍 주요 구성 단계 설명

> DINO는 기존 DETR의 심플함은 유지하면서도,  
> **학습 속도, 정확도, 안정성**을 모두 강화한 **DETR의 결정판 모델 중 하나**입니다.


##### 1. 🖼️ Input Image
- 입력 이미지는 보통 3채널 RGB 형태로 모델에 입력됩니다.

##### 2. 🧠 CNN Backbone
- 예: **ResNet-50**, **Swin Transformer** 등
- 이미지로부터 **저수준 특징(feature map)**을 추출하는 역할

##### 3. 🔁 Transformer Encoder
- CNN에서 추출한 feature를 받아 **글로벌 context 정보**를 학습
- 각 위치가 전체 이미지의 다른 부분과 관계를 맺도록 함

##### 4. 🎯 Candidate Object Proposals (Two-stage)
- Encoder 출력에서 **objectness가 높은 위치 Top-K를 선택**
- 이를 기반으로 **query의 초기 anchor**를 구성 (Mixed Query Selection 포함)

##### 5. 🧩 Transformer Decoder
- query들이 encoder feature에 attention을 두 번 수행 (**Look Forward Twice**)  
- denoising query들도 함께 처리되어 안정적 학습 유도 (CDN 포함)

##### 6. 📦 Predictions
- 각 query에 대해 최종적으로 **물체 클래스와 박스 위치**를 예측  
  → 결과: `{class, bounding box}` 쌍이 N개 출력됨





---

### 🧠 최종 정리 : DINO vs DETR

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
- 🔥 **open-vocabulary detection**, **segment anything** 같은 최신 비전 연구와도 잘 결합되는, 확장가능성이 큰 모델!! :) 

---

### 💬 개인 정리

DINO는 여러 연구들을 잘 조합, 본인들만의 새로운 결과하 합쳐서 **DETR의 학습 효율성과 성능 문제**를 해결한 훌륭한 개선연구인 것 같다!  
Grounding DINO나 DINOv2 등으로 확장할 때도 핵심 개념을 그대로 공유하므로  
**DETR 계열 Transformer 탐지 모델을 이해하려면 반드시 기억해야 할 모델!**  

