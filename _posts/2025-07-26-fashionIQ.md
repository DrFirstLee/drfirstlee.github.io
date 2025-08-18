---
layout: post
title: "🧠 FashionIQ - Fashion Image Retrieval with Natural Language: 패션 이미지 검색의 새로운 표준"
author: [DrFirst]
date: 2025-07-26 10:00:00 +0900
categories: [AI, Research]
tags: [VLM, Fashion, CIR, Image Retrieval, CVPR, CVPR 2021]
sitemap:
  changefreq: monthly
  priority: 0.8

---

---

### 🧠 Fashion Image Retrieval with Natural Language: A New Standard

![fashionIQ_main](https://github.com/user-attachments/assets/ce3e8deb-4b4f-429c-8a62-d09314c0d68f)

* **Title**: [Fashion IQ: A Novel Dataset for Fashion Image Retrieval with Natural Language](https://arxiv.org/abs/1905.12794)
* **Conference**: CVPR 2021 (WU et al.)
* **Code**: [FashionIQ (GitHub)](https://github.com/XiaoxiaoGuo/fashion-iq)
* **Keywords**: `Image Retrieval`, `Natural Language`, `FashionIQ Dataset`, `Composed Image Retrieval`

---

### 🔍 Research Background

Traditional **Image Retrieval** has been a research topic for a long time. The core goal is to find the most similar image to a given query.

However, the new field of **Composed Image Retrieval (CIR)**, which the FashionIQ paper addresses, has distinct characteristics. Instead of a single modality (a single piece of information), it uses **two modalities simultaneously**: a reference image and a text modification. The goal isn't just to "find this image," but to **"find an image like this, but modified in this way!"**

> "Find a dress similar to this, but with longer sleeves."
> → **Reference Image + Text Modification** = Composed Fashion Image Retrieval

This type of **natural language-based fashion search** was a challenging task for existing methods.

---

### 🧠 Key Contributions
> A pioneering work in Composed Image Retrieval!

1.  #### Establishing the Fashion-specific CIR Task
    * The paper clearly defined the concept of **Fashion Composed Image Retrieval**.
    * The goal is to capture subtle stylistic changes requested by users.

2.  #### Introducing the FashionIQ Dataset
    * This is a **large-scale benchmark for fashion image retrieval**, including over 209,000 images and 77,000 queries.
    * The quality of the queries is high because the natural language text was written by **professional fashion designers**.
    * It includes a variety of clothing categories, such as `dresses`, `shirts`, `tops`, and `pants`.

3.  #### A Custom Query Structure for Fashion
    * The text queries include specific modifications related to fashion, such as **color, material, pattern, and design**.
    * Example: "Add a belt," "Change from stripes to polka dots."

4.  #### Comparing Performance with Existing Methods
    * The authors evaluated the performance of various state-of-the-art models like CLIP, TIRG, and FiLM on the FashionIQ dataset.
    * They demonstrated that due to the specific nature of the fashion domain, existing models still have much room for improvement, thus establishing a **new direction for fashion AI research**.

---

### 📘 The FashionIQ Dataset: 'Reference Image + Text' Fashion Search

The **FashionIQ** dataset is a large-scale fashion dataset specifically for **Composed Image Retrieval (CIR)**. Unlike traditional image retrieval, its goal is to understand a user's complex intent through a combination of a **'reference image' and 'natural language text'** to find the desired fashion item.

---

#### 🧩 Key Features
* **Scale**: Consists of around 209,000 high-quality fashion images and 77,000 queries.
* **Query Composition**: Each query is made up of a **reference image** and **two natural language texts** that describe the visual difference between the reference and the target image.
* **Professionalism**: The text queries were written by professional fashion designers, reflecting granular and realistic fashion attributes like "shorten the sleeves" or "change the color to black."
* **Categories**: It includes a wide range of clothing categories, such as women's dresses, tops, and men's shirts.

> This dataset was a major contribution, as it pushed the development of AI models to go beyond simple image search and **understand user intent**. This is why it's frequently used as a reference not just in fashion, but in the broader Composed Image Retrieval field.

---

#### 📷 FashionIQ Query Example

* **Reference Image**: A long-sleeved red dress
* **Text Modification**: "Change color to black and make sleeves shorter"
* **Target Image**: A short-sleeved black dress

---

### 🧠 FashionIQ Model Learning Structure Summary

The **FashionIQ** paper uses the following learning structure for **Composed Image Retrieval**:

* **Step 1: Encoding with Pre-trained Models**
    * **Images**: It uses the **EfficientNet-b7** model, pre-trained on the `DeepFashion` dataset, to extract image features.
    * **Text**: It uses **GloVe embeddings**, pre-trained on a large external text corpus, to convert words into vectors.

* **Step 2: Integrated Learning with a Custom Transformer**
    * The image features and text embeddings from Step 1 are fed into a **newly designed 6-layer transformer** that combines the two modalities.
    * This **multimodal transformer** learns the relationship between the image and text to generate the final query vector used for retrieval.

In conclusion, this model is a hybrid structure that **leverages pre-trained encoders for each modality (image, text)**, while **its core role of integrating the two is handled by a custom-built transformer**.

---

### 🧩 Conclusion

> **FashionIQ (CVPR 2021)** is a pioneering study that **established the CIR task specific to the fashion domain** and built a **large-scale benchmark with professional natural language queries**. The emergence of this dataset was a crucial catalyst, accelerating research in **image retrieval that understands human intent** within the field of fashion AI.

---

### 🧠 (한국어) 자연어로 패션 이미지를 찾아낸다!

![fashionIQ_main](https://github.com/user-attachments/assets/ce3e8deb-4b4f-429c-8a62-d09314c0d68f)

* **제목**: [Fashion IQ: A Novel Dataset for Fashion Image Retrieval with Natural Language](https://arxiv.org/abs/1905.12794)
* **학회**: CVPR 2021 (WU et al.)
* **코드**: [FashionIQ (GitHub)](https://github.com/XiaoxiaoGuo/fashion-iq)
* **핵심 키워드**: `Image Retrieval`, `Natural Language`, `FashionIQ Dataset`, `Composed Image Retrieval`

---

---

### 🔍 연구 배경

일반적인 이미지 검색(Image Retrieval)은 사실 꽤 오래전부터 연구되어 왔습니다.
핵심은 '주어진 쿼리(질문)'와 '가장 유사한 이미지'를 찾는 것이죠.

하지만 FashionIQ가 다룬 CIR이라는 새로운 분야는 다음과 같은 특징을 가집니다.
하나의 모달리티(단일 정보)가 아닌, 두 가지 모달리티(참조 이미지 + 수정 텍스트)를 동시에 사용하는 것입니다.
'이 이미지를 찾아줘'가 아닌, **'이 이미지를 바탕으로 이렇게 바꿔서 찾아줘'라는 사용자 의도를 파악하는 것!!**

> "이 드레스와 비슷한데, 소매가 긴 걸로 찾아줘."
> → **참조 이미지 + 수정 텍스트** = 조합적 패션 이미지 검색

이러한 **자연어 기반의 패션 검색**은 기존 방법론으로는 해결하기 어려운 과제였습니다.

---

### 🧠 주요 기여
> Composed Image Retrieval의 원조격!?!!

1.  #### 패션 특화 CIR 과제 정립
    * **Fashion Composed Image Retrieval**이라는 개념을 명확히 정의했습니다.
    * 사용자의 미묘한 스타일 변경 요구사항을 반영하는 것이 목표입니다.

2.  #### FashionIQ 데이터셋 제안!!  
    * **대규모 패션 이미지 검색 벤치마크**로, 209,000개 이상의 이미지와 77,000개 이상의 쿼리를 포함합니다.
    * **전문 패션 디자이너**들이 직접 작성한 자연어 텍스트를 사용하여 쿼리의 품질을 높였습니다.
    * `드레스`, `셔츠`, `탑`, `바지` 등 다양한 의류 카테고리를 포함합니다.

3.  #### 패션 맞춤형 쿼리 구조
    * **색상, 재질, 패턴, 디자인** 등 패션에 특화된 수정 사항을 텍스트 쿼리에 담았습니다.
    * 예) "Add a belt", "Change from stripes to polka dots"

4.  #### 기존 방식 성능 비교
    * CLIP, TIRG, FiLM 등 다양한 최신 모델들의 성능을 FashionIQ 데이터셋에서 평가했습니다.
    * 패션 도메인의 특수성 때문에 기존 모델들이 여전히 개선의 여지가 많다는 것을 보여주며, **패션 AI 연구의 새로운 방향**을 제시했습니다.

---

### 📘 FashionIQ 데이터셋: '참조 이미지 + 텍스트' 패션 검색
FashionIQ는 **조합적 이미지 검색(Composed Image Retrieval, CIR)**에 특화된 대규모 패션 데이터셋!!  
**'참조 이미지'와 '자연어 텍스트'**를 조합한 쿼리를 통해,  
사용자의 복잡한 의도를 파악하고 원하는 패션 아이템을 찾아내는 것을 목표로 함!!

---

#### 🧩 주요 특징
- 규모: 약 209,000개의 고품질 패션 이미지와 77,000개의 쿼리로 구성  
- 쿼리 구성: 각 쿼리는 참조 이미지와 정답 이미지의 차이를 설명하는 두 개의 자연어 텍스트로 구성 
- 전문성: 패션 전문 디자이너가 직접 작성한 텍스트 쿼리를 포함하여, 소매를 짧게, 색상을 검은색으로와 같이 세밀하고 현실적인 패션 속성을 반영  
- 카테고리: 여성용 드레스, 상의, 남성용 셔츠 등 다양한 의류 카테고리 포함  

> 이 데이터셋은 기존의 단순한 이미지 검색을 넘어 사용자의 의도를 이해하는 AI 모델 개발에 중요한 기여를 함!!  
> 그래서 꼭 패션이 아니라 Composed Image Retrieval 쪽에서 레퍼런스로 많이 쓰임임

---

#### 📷 FashionIQ의 쿼리 예시

* **Reference Image**: 빨간색 긴팔 드레스
* **Text Modification**: “Change color to black and make sleeves shorter”
* **Target Image**: 검은색 반팔 드레스

---
### 🧠 FashionIQ 모델 학습 구조 요약

**FashionIQ** 논문은 **조합적 이미지 검색(Composed Image Retrieval)**을 위해 다음과 같은 학습 구조를 사용

* **1단계: 사전 학습된 모델을 사용한 인코딩**
    * **이미지**: `DeepFashion` 데이터셋으로 미리 학습된 **EfficientNet-b7** 모델을 활용하여 이미지의 특징 추출
    * **텍스트**: 외부 대량의 텍스트로 미리 학습된 **GloVe 임베딩**을 사용  

* **2단계: 자체 트랜스포머를 통한 통합 학습**
    * 위 1단계에서 얻은 이미지와 텍스트의 벡터를 입력으로 받아, **논문에서 새롭게 설계한 6개 레이어의 트랜스포머**를 통해 두 정보를 결합  
    * 이 **멀티모달 트랜스포머**는 이미지와 텍스트의 관계를 학습하여 최종적으로 검색에 사용될 쿼리 벡터를 생성

결론적으로, 이 모델은 **각각의 모달리티(이미지, 텍스트)를 위한 사전 학습된 인코더를 활용**하면서, **이 둘을 통합하는 핵심적인 역할은 자체 제작한 트랜스포머가 담당**하는 하이브리드 구조로 구성  
---

### 🧩 결론

> **FashionIQ (CVPR 2021)**는 **패션 도메인에 특화된 CIR 과제를 정립**하고, **전문적인 자연어 쿼리를 포함한 대규모 벤치마크**를 구축한 선구적인 연구입니다. 이 데이터셋의 등장은 패션 AI 분야에서 **사람의 의도를 이해하는 이미지 검색** 연구를 가속화하는 중요한 계기가 되었습니다.