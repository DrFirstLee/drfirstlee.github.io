---
layout: post
title: "👁️ MLLMs Know Where to Look: Training-free Visual Detail Perception"
author: [DrFirst]
date: 2025-07-25 09:00:00 +0900
categories: [AI, Research]
tags: [MLLM, Attention, Cropping, Visual Reasoning, ICLR, ICLR 2025 ]
sitemap :
  changefreq : monthly
  priority : 0.8

---

---

### 👁️ MLLMs Know Where to Look: Training-free Perception of Visual Details

- **Title**: [MLLMs know where to look: Training-free perception of small visual details with multimodal LLMs](https://arxiv.org/abs/2502.17422)
- **Conference**: ICLR 2025 (Zhang, Jiarui et al.)
- **Code**: [saccharomycetes/mllms_know](https://github.com/saccharomycetes/mllms_know)
- **Keywords**: `Multimodal LLM`, `Small Visual Details`, `Attention Map`, `Cropping`, `Gradient`, `Inference`

---

### 🧠 TL;DR in 3 Lines

1. MLLMs are generally good at **knowing where to look**,  
   but often **fail to understand what they’re seeing**.

2. Simply **cropping the relevant part of the image** and feeding it back  
   significantly improves detail-level recognition.

3. If the image is too large, it is **split and reprocessed to ensure accurate attention**.

---

### ⚠️ Problem Background

![prob1](https://github.com/user-attachments/assets/d959e40b-4cda-40b5-8d29-2184607d97e5)  

- MLLMs often fail on questions about small objects in an image,  
  but they succeed if we crop and provide only the relevant region.

---

### 📚 Datasets Used

The authors validate their method on the following 6 datasets:

| Dataset     | Purpose                                         | Image Type             | Question Focus                          | External Knowledge | Example Models                |
|-------------|--------------------------------------------------|-------------------------|------------------------------------------|---------------------|-------------------------------|
| **DocVQA**   | Document-level question answering                | Document images (PDFs) | Text extraction + layout understanding   | ❌                  | LayoutLM, Donut, DocFormer   |
| **TextVQA**  | Scene text-based VQA                             | Natural images w/ text | Text in context of visual scene          | ❌                  | M4C, GRILL, LLaVA            |
| **POPE**     | Evaluating model bias and hallucination          | Mixed image types       | Robustness to misleading contexts        | ❌                  | BLIP2, Pythia                |
| **A-OKVQA**  | Knowledge-based multiple-choice VQA              | Natural images          | External knowledge + choice selection    | ✅                  | ReGAT, RAVQA, NoteMR         |
| **GQA**      | Relation reasoning and scene understanding       | Complex scenes          | Logic and spatial reasoning              | ❌                  | MAC, NS-VQA, GraftNet        |
| **VQAv2**    | General-purpose VQA benchmark                    | Natural images          | Object, attribute, and general questions | ❌                  | UpDn, Pythia, LXMERT         |

---

### 🔧 Three Key Investigations

0. **Can humans solve these problems better just by cropping?**  
   → Manually cropping the region significantly improved model performance!

1. **Do LLMs fail because they don’t know where to look, or because they can’t understand even when looking correctly?**  
   → It’s the latter: they look in the right place but misinterpret it.

2. **Then what if we just show them the right region only?**  
   → That works very well!

#### 0. Human cropping improves accuracy

![crop_Effect](https://github.com/user-attachments/assets/a7a2ab90-5998-41bd-8bea-31fdb64acd76)  
- When humans crop only the relevant region of the image,
- MLLMs answer detail-based questions much more accurately.

#### 🔍 1. Do MLLMs attend to the right place?

![looking](https://github.com/user-attachments/assets/e73e8306-f992-43cd-b948-8723e1884ca2)  
- By visualizing attention layers,  
- It turns out the model **does look in the right area** even when it gives a wrong answer.

#### ✂️ 2. Just give the right region → better performance!

![cropping](https://github.com/user-attachments/assets/74214622-05ed-4dc8-9b43-ce3f2ff8857d)

- As seen above, **cropping and reinserting alone greatly boosts performance**
- So, how to crop effectively?
- The authors propose **3 attention-based cropping strategies**:

| Method      | Description |
|-------------|-------------|
| **Rel-Att** (Relative Attention) | Compares attention maps between the true question and a generic one to highlight the difference |
| **Grad-Att** (Gradient-weighted Attention) | Uses gradients to find regions most sensitive to the model's confidence |
| **Pure-Grad** (Input Gradient) | Uses input image gradients to locate visually salient pixels |

**Cropping pipeline**:

- **Input**: image + question  
- **Process**: compute attention map via one of the above → derive ROI crop  
- **Output**: crop image → reinsert to MLLM → generate answer

The paper also compares cropping methods using **external tools like YOLO, CLIP, and SAM**:

> Surprisingly, even against SOTA external methods, their proposed internal methods held up well.

![crop_res](https://github.com/user-attachments/assets/3d5510f0-4e4d-4e1a-a054-2a213158e9a2)

| Method         | One-line Summary |
|----------------|------------------|
| **CLIP ViCrop** | Uses CLIP similarity to iteratively crop toward the most semantically aligned region |
| **YOLO ViCrop** | Selects bounding boxes from YOLO with highest CLIP similarity to the question |
| **SAM ViCrop**  | Converts segmentation masks from SAM into bounding boxes, then selects the one with best CLIP match |

---

### 🧪 Experiment Results

- The system performs **inference-only cropping**—no retraining required
- Large images are pre-cropped to better guide attention
- Evaluation covers multiple datasets and question types

---

### 📈 Key Results

![res1](https://github.com/user-attachments/assets/ebb016dc-7ef6-448a-859c-d8a1bc8284d2)  
- Attention-based crops like Rel-Att and Grad-Att outperform other approaches—especially for small-object questions.

![res2](https://github.com/user-attachments/assets/9a546460-7318-4318-bf91-6a2e1b4f988d)  
- Cropping greatly helps when image resolution is high.

**Summary of Effects**:

| Setup                      | Performance Impact |
|---------------------------|--------------------|
| Full image only           | Poor on detail-based questions |
| Crop via attention-guided methods | Much higher accuracy |
| No retraining needed      | Zero-shot + Inference-time only |

Overall, this approach **greatly improves fine-grained perception**,  
even without scaling up the model size.

---

## ✅ Conclusion & Impact

- The paper shows MLLMs **already know where to look**,  
  but need help **seeing better** via focused cropping.
- Significant performance gains are possible **without any retraining**—just with attention-based inference.
- Has strong applicability in domains like **OCR**, **tiny-object detection**, or **interactive AI tutors**.

> "**MLLMs know where to look. Let’s help them see better.**"


---

### 👁️ (한국어) MLLMs Know Where to Look: Training-free 시각 디테일 인식

- **제목**: [MLLMs know where to look: Training-free perception of small visual details with multimodal llms](https://arxiv.org/abs/2502.17422)
- **학회**: ICLR 2025 (Zhang, Jiarui et al.)
- Code: [saccharomycetes/mllms_know](https://github.com/saccharomycetes/mllms_know)
- **핵심 키워드**: `Multimodal LLM`, `Small Visual Details`, `Attention Map`, `Cropping`, `Gradient`, `Inference`

---

### 🧠 3줄 요약

1. MLLM은 이미지 내 **‘어디를 보는지’는 잘 파악**하지만,  
   **‘무엇을 보는지’는 정확히 인식하지 못하는 한계**가 있음.

2. 이미지의 중요한 부분을 **crop해서 다시 입력**하면,  
   모델이 시각적 디테일을 훨씬 정확히 인식함.

3. 이미지가 너무 큰 경우에는 정확한 attention을 위해 잘라서 사용하고 붙임!   


---

### ⚠️ 배경: 기존 문제점 요약

![prob1](https://github.com/user-attachments/assets/d959e40b-4cda-40b5-8d29-2184607d97e5)  

- 이미지 내에서 작은 객체에 대한 질문을 헀을때 답을 틀리지만, 해당 부분만을 crop 해서 보여주면 답을 잘함  

---

### 참고. 사용한 데이터셋  
- 여기서는 로직검증을 위해 아래 6가지 데이터셋을 사용했습니다!!  

| 데이터셋     | 주요 목적                                      | 이미지 유형           | 질문 초점                            | 외부 지식 필요 | 대표 모델 예시               |
|--------------|-----------------------------------------------|------------------------|---------------------------------------|----------------|------------------------------|
| **DocVQA**   | 문서 기반 질의응답 (인보이스, 보고서 등)      | 문서 이미지 (PDF 등)  | 텍스트 정보 추출 + 문서 구조 이해     | ❌             | LayoutLM, Donut, DocFormer  |
| **TextVQA**  | 장면 내 글자를 포함한 질의응답                | 자연 이미지 + 텍스트  | 시각 문맥 속 텍스트 이해               | ❌             | M4C, GRILL, LLaVA           |
| **POPE**     | VQA 모델의 편향(Bias)과 환각(Hallucination) 평가 | 다양한 (혼합형) 이미지 | 모델의 bias robustness 평가           | ❌             | BLIP2, Pythia               |
| **A-OKVQA**  | 외부 지식 기반 VQA + 정량 평가                | 자연 이미지           | 지식 기반 질의 + 선택지 기반 응답      | ✅             | ReGAT, RAVQA, NoteMR        |
| **GQA**      | 관계 추론, 객체 간 의미적 연결                | 복잡한 장면 이미지     | 장면 이해 + 관계 기반 질의응답         | ❌             | MAC, NS-VQA, GraftNet       |
| **VQAv2**     | 일반 VQA 벤치마크, 다양한 질문 유형 포함       | 자연 이미지           | 객체, 속성, 장면 등 전반적 질의응답    | ❌             | UpDn, Pythia, LXMERT        |



---

### 🔧 3가지 단계로 나누어서 해결 방법을 찾음

0. 정말 작은 부위를 crop해서 보여주면 문제를 잘 맞출까?  
 - 사람이 크롭해서 테스트해봄!!  

1. LLM은 어디를 볼지도 몰라서 틀린걸까? 혹은 부위는 잘 찾았는데 잘못 인지한걸까?  
 - 결론은 후자, 부위는 잘 찾았지만 잘못 인지한것임!  

2.그럼! 해당 부위만을 제시하만 잘 작동할까??
 - 그렇다!!

####  0. 정말 작은 부위를 crop해서 보여주면 문제를 잘 맞출까?  
![crop_Effect](https://github.com/user-attachments/assets/a7a2ab90-5998-41bd-8bea-31fdb64acd76)  
- 이미지 내의 작은 부분을 맞추는 질문에서,  
- 사람이 정답부분만 crop해서 제시할 경우 확실히 잘 대답해!!  


#### 🔍 1. LLM은 어디를 볼지도 몰라서 틀린걸까? 혹은 부위는 잘 찾았는데 잘못 인지한걸까?  
![looking](https://github.com/user-attachments/assets/e73e8306-f992-43cd-b948-8723e1884ca2)  
- MLLM 레이어에서 어텐션을 추출해서 시작화해보면!!  
- 비록 정답은 틀렸지만 어디를 봐야하는지는 잘 알고 있다는것을 알수 있지!!    



#### ✂️ 2.그럼! 해당 부위만을 제시하만 잘 작동할까??

![cropping](https://github.com/user-attachments/assets/74214622-05ed-4dc8-9b43-ce3f2ff8857d)

- 0에서 확인했듯, **이미지를 잘라 다시 넣기만 해도 성능이 급상승!**
- 그럼, 어떻게 crop 하지?!  
-  3가지 Attention 기반 Cropping 전략  

| 방법 | 설명 |
|------|------|
| **Rel-Att** (Relative Attention) | 정답 질문 vs 일반 질문의 attention map을 비교해, **차이점을 강조**하여 crop 영역 도출 |
| **Grad-Att** (Gradient-weighted Attention) | 정답 확률에 대한 **gradient를 통해 민감 영역**을 강조함 |
| **Pure-Grad** (Input Gradient) | 이미지 자체의 gradient를 통해, **픽셀 단위로 중요한 영역**을 추출함 |

- crop 방법은?  

  - **입력**: 이미지 + 질문  
  - **처리**: 위 3가지 방법 중 하나로 attention map 계산 → crop 영역 설정  
  - **출력**: crop된 이미지를 MLLM에 다시 넣어 **답을 생성**

- 추가로 이번 연구에서는 YOLO, CLIP, SAM등을 사용한 crop 방법과 성능을 비교했고!  

> 기존 SOTA 연구를 활용한 crop 과 비교해도 나쁘지 않았다!! 
![crop_res](https://github.com/user-attachments/assets/3d5510f0-4e4d-4e1a-a054-2a213158e9a2)  

| 방법              | 한 줄 요약                                                              |
| --------------- | ------------------------------------------------------------------- |
| **CLIP ViCrop** | CLIP을 사용해 질문과 의미적으로 가장 관련 있는 영역을 **점진적으로 잘라가며 반복 선택**하는 방식.         |
| **YOLO ViCrop** | YOLO로 탐지된 객체 영역 중, 질문과 **CLIP 유사도가 가장 높은 바운딩 박스를 선택**하는 방식.         |
| **SAM ViCrop**  | SAM이 제공하는 **세그멘트 마스크를 바운딩 박스로 변환한 후**, CLIP 유사도가 가장 높은 영역을 선택하는 방식. |


---

### 🧪 실험 분석 결과!!   

- 실험은 **training 없이**, inference 시 attention 기반 crop을 수행하는 구조  
- 큰 이미지는 **사전 crop**하여 attention이 더 잘 잡히도록 설계함  
- 다양한 질문 유형에 대해 crop 후 답변을 생성하고 성능 비교  

---

### 📈 주요 성과

![res1](https://github.com/user-attachments/assets/ebb016dc-7ef6-448a-859c-d8a1bc8284d2)  
- Rel-att 이나 grad-att 방식으로 크롭한것이 가장 결과가 좋다!! 특히 작은 객체에 대한 질문에서!!

![res2](https://github.com/user-attachments/assets/9a546460-7318-4318-bf91-6a2e1b4f988d)  
- 해상도가 큰 이미지는, 잘라서 작업하는게 효과가 좋았다!!


- 성과 요약!!  
| 조건 | 성능 |
|------|------|
| Full image 입력 | 작은 디테일 질문에 취약 |
| Attention-guided crop → 재입력 | 디테일 질문 정확도 **상당 향상** |
| No retraining | **Zero-shot + Inference-time only** 방식 |

- 실험 결과, 작은 디테일이 중요한 task에서 성능이 **확연히 향상**
- 특히 기존 MLLM 대비, **고성능 대형모델 없이도 개선** 가능

---

## ✅ 결론 및 의의

- 이 논문은 MLLM이 정확히 "어디를 보아야 하는지"는 잘 아는데,  
  "보는 방식"이 부족하다는 점을 **Attention-based Cropping**으로 해결함
- **Training 없이 inference만으로 성능 향상** 가능하다는 점에서  
  **경량화, 응용성, 해석력 측면**에서 매우 실용적인 접근
- 다양한 downstream task (e.g. OCR, 세밀한 물체 인식, 튜터링 시스템)에 응용 가능

> "MLLMs know where to look. Let’s help them see better."