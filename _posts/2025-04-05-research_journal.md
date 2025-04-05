---
layout: post
title: "Understanding Vision Transformers (ViT) - 비전 트랜스포머 이해하기"
author: [DrFirst]
date: 2025-03-24 09:00:00 +0900
categories: [Computer Vision, Research]
tags: [ViT, Vision Transformer, AI]
lastmod : 2025-03-24 09:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---
# Hello, everyone! 👋

Today, let’s dive into the world of **Vision Transformers (ViT)** — a groundbreaking approach in computer vision  
that applies the **Transformer architecture**, originally designed for natural language processing, to image understanding.

Unlike traditional CNNs that rely on convolutions to extract spatial features,  
**ViT splits an image into patches**, embeds them, and processes them through a Transformer encoder —  
treating the image more like a sequence of tokens than a grid of pixels.

This paradigm shift has opened new doors in computer vision research and has shown competitive or even superior results  
on large-scale image recognition tasks, especially when trained on massive datasets.

I'll be sharing more about ViT’s core ideas, its architecture, and practical implications in future posts —  
including comparisons with CNNs and hybrid models.

Stay tuned for more deep dives into the intersection of **AI** and **vision**. 👁️🤖

— *DrFirst*

---

## 🧪 Sample Code: Using ViT in Python

Here’s a simple example of how to use a pre-trained Vision Transformer model from Hugging Face to classify an image:

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import torch

# Load image from URL
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image = Image.open(requests.get(url, stream=True).raw)

# Load feature extractor and model
extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Preprocess the image
inputs = extractor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Print result
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

Make sure to install the required packages:

```bash
pip install transformers torch pillow
```

---
# AI와 관련된 주요 저널 알아보기 (feat. h-index)

안녕하세요!! 👋

오늘은 **인공지능**, 즉 **AI와 관련된 주요 저널**들을 알아보고자 합니다.

알아보기에 앞서!! 저널들, 그리고 연구자들의 **연구 역량을 정량적으로 판단할 수 있는 지표**에 대해 먼저 알아보겠습니다!

먼저 다양한 저널들을 살펴볼 수 있는 **Google Scholar**를 방문해볼까요?  
🔗 [Google Scholar 바로가기](https://scholar.google.com/citations?view_op=metrics_intro&hl=en)

📸 ![Google Scholar](https://github.com/user-attachments/assets/e2b1716a-259d-440f-882a-340a06e65207)


위 이미지를 보면 알 수 있듯, 구글에서는 **h5-index를 저널의 평가지표**로 삼아 순위를 매기고 있습니다.  
뿐만 아니라 연구자 개개인의 연구 성과도 정량적으로 판단할 수 있답니다!

📸 ![einstein](https://github.com/user-attachments/assets/564b6e3d-304a-40d7-86d7-6b0fcc295659)

---

## 🔍 먼저 h-index부터 알아봅시다!

**h-index**는 2005년 물리학자 Jorge Hirsch가 제안한 지표로,  
연구자의 **생산성과 영향력을 동시에 측정**할 수 있습니다.

### ✅ 정의  
> “한 연구자가 발표한 논문 중에서 **h번 이상 인용된 논문이 h편 이상** 있을 때, h-index는 h이다.”

📸 ![h-index](https://github.com/user-attachments/assets/fbaae532-1a1f-4d1a-ad27-e8397d498ab3)


### 🔎 예시  
- A 연구자가 10편의 논문을 냈고, 그 중 5편이 각각 5회 이상 인용되었다면 → **h-index = 5**  
- 논문 100편을 썼더라도 인용이 골고루 분포되지 않았다면 h-index는 낮을 수 있음

### 🧠 장점  
- 인용 수만으로 평가할 때보다, **대박 논문에 의한 왜곡을 줄임**  
- **꾸준한 영향력**을 가늠할 수 있음

### ⚠️ 단점  
- **시간에 민감**: 젊은 연구자는 불리  
- **분야별 인용 문화 차이**를 반영하지 못함  
  (예: 인용이 활발한 생명과학 vs 상대적으로 적은 수학/철학)

---

## 📊 그렇다면 h5-index는?

**h5-index**는 Google Scholar에서 사용하는 지표로, **최근 연구 성과를 반영**하기 위한 버전입니다.

### ✅ 정의  
> "최근 5년간 발표된 논문 중에서, **h번 이상 인용된 논문이 h편 이상**일 때의 h 값"

즉, **h-index의 범위를 최근 5년으로 제한**한 것입니다! 🤓  
AI와 같이 빠르게 발전하는 분야에서, **저널이나 학회의 최신 영향력을 파악**할 때 유용합니다.

---

## 🧠 GPT가 알려준 연구자 h-index 단계 구분!

| 단계 | h-index 범위 | 설명 |
|------|---------------|------|
| 🐣 초기 연구자 | 0–10 | 석사 졸업~박사과정/초기 포닥 수준 |
| 🌱 신진 연구자 | 10–20 | 박사 졸업 후, 몇 년간 활동 |
| 🌿 중견 연구자 | 20–40 | 조교수/부교수급에서 활발히 연구 |
| 🌳 저명 연구자 | 40–60+ | 정교수급, 학회장, 연구센터장급 |
| 🧠 세계적 석학 | 60–100+ | 분야 리더, 주요 이론/모델 제안자 |

> 📌 단, 분야와 인용 특성에 따라 다르기 때문에 **절대 기준은 아니며 참고용**입니다.

---

## 🔁 다시 Google Scholar로 돌아가볼까요?

🔗 [Google Scholar Metrics 바로가기](https://scholar.google.com/citations?view_op=metrics_intro&hl=en)

📸 *[이미지: 구글스코랄]*

`Rank 1`에는 모두가 아는 **Nature**, `Rank 4`에도 유명한 **Science**가 위치해 있네요!  
너무 유명한 저널이라 별도 설명은 생략할게요 😄

---

## 🎯 이제 AI & Computer Vision 분야로 이동해봅시다!

🔗 [AI 분야 저널 랭킹 바로가기](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng)

📸 *[이미지: computerscience]*

### 제가 자주 참고하는 주요 학회들을 정리해보면 다음과 같습니다:

| 순위 | 학회명 | h5-index |
|------|--------|----------|
| 1 | IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**) | 440 |
| 2 | Neural Information Processing Systems (**NeurIPS**) | 337 |
| 4 | International Conference on Learning Representations (**ICLR**) | 304 |
| 5 | IEEE/CVF International Conference on Computer Vision (**ICCV**) | 291 |
| 7 | International Conference on Machine Learning (**ICML**) | 268 |
| 13 | AAAI Conference on Artificial Intelligence (**AAAI**) | 220 |
| 18 | European Conference on Computer Vision (**ECCV**) | 206 |

---

## 🗓️ 각 학회별 요약 정리표

| 랭킹 | 학회 | 분야 | 개최 주기 | 개최 월 | 강점 요약 |
|------|------|------|------------|---------|-----------|
| 1 | CVPR | 컴퓨터 비전 | 매년 | 6월 | 산업/비전 실용 연구의 중심 |
| 2 | NeurIPS | 머신러닝 | 매년 | 11–12월 | 딥러닝 이론·응용 모두 포함 |
| 4 | ICLR | 딥러닝 | 매년 | 4–5월 | 표현 학습, 최신 트렌드에 민감 |
| 5 | ICCV | 컴퓨터 비전 | 격년 (홀수 해) | 10월 | CVPR보다 이론적 비중 높음 |
| 7 | ICML | 머신러닝 | 매년 | 7월 | 수학적 기초 연구 강세 |
| 13 | AAAI | 전통 AI + 딥러닝 | 매년 | 1–2월 | 추론, 계획 등 고전 AI와 최신 기술의 조화 |
| 18 | ECCV | 컴퓨터 비전 | 격년 (짝수 해) | 10월 | 유럽 중심, 비전 분야 핵심 연구 소개 |

---

## 💡 흥미로운 점!

**ICCV**와 **ECCV**는 격년제임에도 불구하고 상위권에 포진해 있다는 사실!  
(즉, 다른 학회에 비해 절반만 열리는데도 h5-index가 높음)

📸 *[이미지: scholar2]*

그리고 놀랍게도…  
**전 세계 학술지 상위 14위 안에 AI 관련 학회가 무려 4개**나 포함되어 있습니다!!

- CVPR는 **Science보다 높고**, **Nature 바로 아래**에 위치한 **2위**!

---

## 🎉 마무리하며…

이처럼 **컴퓨터 과학**, 특히 **AI와 Computer Vision** 분야는  
전 세계적으로 **엄청난 연구 경쟁이 펼쳐지고 있는 분야**입니다.

우리가 논문에서 자주 접하는 **CVPR, NeurIPS, ICLR, ICML, AAAI, ICCV, ECCV**는  
모두 **세계 최고 수준의 학회**이며,  
그 속에서 발표되는 논문들은 **수많은 심사를 거친 정제된 연구 결과**라는 점을 기억해 주세요 😊

---

## 📎 참고1: 학회별 세부 소개

<details>
<summary>🧠 자세히 보기</summary>

**1. CVPR**  
- 정식 명칭: *IEEE/CVF Conference on Computer Vision and Pattern Recognition*  
- 개최: 매년 (6~7월)  
- 특징: 세계 최대 컴퓨터 비전 학회. Vision Transformer, 3D Vision 등 실용 중심 논문이 많음  

**2. NeurIPS**  
- 정식 명칭: *Conference on Neural Information Processing Systems*  
- 개최: 매년 (11–12월)  
- 특징: 머신러닝 이론, 신경망, LLM 등 최첨단 딥러닝 연구 발표  

**3. ICLR**  
- 정식 명칭: *International Conference on Learning Representations*  
- 개최: 매년 (4–5월)  
- 특징: OpenReview 방식, representation learning, diffusion model 중심  

**4. ICCV**  
- 정식 명칭: *IEEE/CVF International Conference on Computer Vision*  
- 개최: 격년 (홀수 해, 10월)  
- 특징: 비전 이론 연구 비중이 크며, 글로벌 연구자 참여 활발  

**5. ICML**  
- 정식 명칭: *International Conference on Machine Learning*  
- 개최: 매년 (6~7월)  
- 특징: 머신러닝 수학 이론, 최적화 중심  

**6. AAAI**  
- 정식 명칭: *AAAI Conference on Artificial Intelligence*  
- 개최: 매년 (1–2월)  
- 특징: 전통 AI(추론, 계획)과 최신 딥러닝을 함께 다룸  

**7. ECCV**  
- 정식 명칭: *European Conference on Computer Vision*  
- 개최: 격년 (짝수 해, 10월)  
- 특징: 유럽 중심이지만 국제적 영향력 큼. 비전 기술 혁신 소개

</details>

---

감사합니다! 다음 포스트에서는 이들 학회에서 나온 **대표 논문들**을 소개해볼게요 😄
