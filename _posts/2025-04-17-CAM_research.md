---
layout: post
title: "Peeking into the Mind of AI: Understanding CAM! - AI의 속마음을 들여다본다!! CAM 알아보기 "
author: [DrFirst]
date: 2025-04-17 09:00:00 +0900
categories: [AI, Research]
tags: [CAM, Vision AI, GAP, CVPR, CVPR2016, XAI, Class Activation Map]
lastmod : 2025-04-17 09:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---

## (English) Peeking into the Mind of AI: Understanding CAM!

![cartoon](https://github.com/user-attachments/assets/b283ac2e-21d0-4d64-9405-e4a1cd15f39c)

> A groundbreaking study that allowed us to peek into a computer's decision-making process: CAM!

## Learn About the Groundbreaking Research of CAM! (CVPR 2016, 13,000+ Citations)

![paper](https://github.com/user-attachments/assets/ba4f09c8-f0cd-4aa9-b035-4ef6c70cb3c1)

This paper was presented at CVPR 2016 and has been cited more than **13,046 times**!  
In image analysis, not knowing CAM might be a crime!  
Even if you’re unfamiliar with the research, you've probably seen the image below:

![cam_sample](https://github.com/user-attachments/assets/0e660a0e-ac17-4c35-a091-7c67f8b951eb)

In today’s post, we’ll take a deep dive into this game-changing research!

---

## 🤔 Why Was CAM Introduced?

Back in the days when AlphaGo made headlines (around 2015), image classification models like ResNet dramatically improved accuracy.  
But regardless of whether predictions were right or wrong,  
**"Why did the model make that prediction?"** was still a tough question.

![confuse](https://github.com/user-attachments/assets/701acf02-c18a-4123-942c-2821b66ed2af)

> **Why did it predict this image is a dog?  
> Is the model really looking at the right part?**

This curiosity led to the birth of **CAM (Class Activation Map)** research.

---

## 🔍 What is CAM (Class Activation Map)?

CAM is a technique that **visually shows which part of the image the model focused on to make a prediction.**  
In other words, it highlights the **decisive areas** in the form of a heatmap.

As shown in the **cartoon thumbnail**, CAM allows AI to say:  
> "I predicted this as a dog because I focused on the eyes and ears!"

---

## 🧠 How Does CAM Work? + Example

![cam_structure](https://github.com/user-attachments/assets/32074284-e8be-432c-bf76-18886967af70)

1. Extract the **feature map from the last convolutional layer** of a CNN.
2. Instead of using the **fully connected layer’s class weights** (as in traditional models),  
   apply **Global Average Pooling (GAP)** to the feature map to get a feature vector.
3. Multiply this feature vector by the class-specific weights in the softmax layer  
   to calculate a CAM that shows **which spatial locations contributed most to the prediction**.

If that sounds a bit complicated, let’s compare it side by side to understand it better!

### Before We Start: Compare Traditional [Image Classifier] vs. [CAM-based Structure]
- Traditional: conv → flatten → FC → softmax  
- CAM: conv → GAP → FC → softmax  

---

### Traditional Classification Model: Feeding a Single Image (224×224) into a CNN with FC Layer

| Step | Data Shape         | Description |
|------|--------------------|-------------|
| 📷 Input Image            | `[3, 224, 224]` | RGB image |
| 🔄 Last Conv Output       | `[512, 7, 7]`   | 512 feature maps of size 7×7 |
| 🔃 Flatten                | `[512 × 7 × 7]` = `[25088]` | Flattened into a vector |
| 🧮 Fully Connected Layer  | `[N_classes]`   | Generates class scores (`weight shape = [N_classes, 512]`)  
⚠️ The weights `[512]` for a specific class are used as **class_weight** for CAM |
| 🎯 Softmax                | `[N_classes]`   | Converts scores to probabilities |
| 🚫 CAM Not Available      | ❌ Not possible | Spatial information is lost during flattening |

- As shown above, CAM is not possible in this structure.  
- Only the final probabilities `[N_classes]` are available.  
- `N_classes` is the number of classes to distinguish (e.g., dog, cat, etc.).  
- The weights used in the Fully Connected Layer serve as **class_weight** in CAM.

---

### CAM Flow: Feeding a Single Image (224×224) into ResNet18 to Generate CAM

| Step | Data Shape         | Description |
|------|--------------------|-------------|
| 📷 Input Image            | `[3, 224, 224]` | RGB image |
| 🔄 Last Conv Output       | `[512, 7, 7]`   | 512 feature maps of size 7×7 |
| 🔥 CAM Calculation        | `[7, 7]`        | Weighted sum of feature maps × **class_weight** |
| 🔼 Final CAM Image (Upsample) | `[224, 224]`   | Upsampled to overlay on the original image |
| 📉 GAP (Global Average Pooling) | `[512]`     | Channel-wise average of the `[512, 7, 7]` feature map |
| 🧮 FC Layer               | `[N_classes]`   | Converts GAP result to class scores |
| 🎯 Softmax                | `[N_classes]`   | Outputs prediction probabilities |

- CAM generates an interpretable heatmap  
- The class prediction from GAP → FC → softmax may differ from traditional CNNs!

---

## 📸 Real Example: How CAM is Actually Used

- If AI predicts an image as ‘dog’,  
  CAM highlights **the face, ears, and tail** regions that contributed most.
- This allows us to verify if the model made a **reasonable prediction**.
- Below is a CAM result for a golden retriever. It seems the model focused on the ears!

![golden](https://github.com/user-attachments/assets/d3fd65b1-11bd-44dc-9516-3118fd586bf0)

| In the next post, we’ll dive into the code behind this and explore its structure in detail!

---

## 🧰 CAM’s Transformative Impact

- CAM was one of the **first methods to visually explain CNN predictions**.
- It had a massive influence on **weakly supervised object localization (WSOL)**.
- Inspired follow-up methods like **Grad-CAM**, which removed architectural constraints.

---

## 💡 Why CAM is Amazing

- ✅ Helps visually confirm why a model made a prediction  
- ✅ Makes it easier to find errors, bias, and reasoning mistakes  
- ✅ Improves model transparency and trustworthiness

---

CAM was the beginning of the end for the "black box" era in AI.  
Even today, researchers are extending CAM into the broader domain of explainable AI (XAI).  
Try implementing CAM yourself to take a peek inside your AI's mind!


---

## (한국어)  AI의 속마음을 들여다본다!! CAM 알아보기 

![cartoon](https://github.com/user-attachments/assets/b283ac2e-21d0-4d64-9405-e4a1cd15f39c)

> 이미지 분류에있어, 컴퓨터의 속마음을 알게 해준 연구 CAM!!!

## 엄청난 연구, CAM을 알아보자!! (CVPR 2016, 인용 13,000회+)

![paper](https://github.com/user-attachments/assets/ba4f09c8-f0cd-4aa9-b035-4ef6c70cb3c1)  

2016년 CVPR에서 발표된 이 논문, **인용수가 무려 13,046회**에 달합니다!  
이미지 분석에서 ‘CAM’ 을 모르면 간첩!?
비록 연구는 모를수 있지만 아래의 이미지는 많이 보셨을듯 하네요~~!

![cam_sample](https://github.com/user-attachments/assets/0e660a0e-ac17-4c35-a091-7c67f8b951eb)

오늘의 포스팅은 이 획기적인 연구를 알아보겠습니다!!!

---

## 🤔 왜 CAM이 등장했을까?

알파고가 화제가 되었던 시절(2015년 즈음), 이미지 분류 분야에서는 ResNet 등 뛰어난 모델이 나와  
정확도가 눈에 띄게 향상되었지만,  
틀리는지, 맞는지 결과를 떠나서,  
**"모델이 왜 그런 예측을 했는지"**에 대한 해석은 여전히 어려운 숙제였습니다.

![confuse](https://github.com/user-attachments/assets/701acf02-c18a-4123-942c-2821b66ed2af)

> **왜 이 이미지를 강아지라고 생각했지?  
> 정말 모델이 제대로 보고 있는 걸까?**

이런 궁금증에서 CAM(Class Activation Map) 연구가 탄생하게 됩니다.

---

## 🔍 CAM(Class Activation Map)이란?

CAM은 이미지 분류 모델이 **어떤 부분을 근거로 예측했는지 시각적으로 보여주는** 기법입니다.  
즉, 이미지의 **결정적 부위**를 heatmap으로 표시해주죠!

**썸네일 만화**에서처럼, AI가  
> “이 강아지의 눈과 귀를 보고 ‘강아지’라고 했어요!”  
라고 설명할 수 있게 해줍니다.

---

## 🧠 CAM의 작동 원리 + 예시

![cam_structure](https://github.com/user-attachments/assets/32074284-e8be-432c-bf76-18886967af70)  

1. CNN의 **마지막 합성곱 층에서 나온 feature map**을 뽑아냅니다.
2. **Fully Connected Layer의 클래스별 가중치**를 가져져오는 대신!! (기존 이미지 분류에서는 이렇게 했었더레요!!),  
   **Global Average Pooling** (GAP) 을 통해 각 feature map을 하나의 값으로 평균 내어 feature 벡터를 생성합니다.
3. 그 feature vector에 대해 Softmax에 연결된 클래스별 가중치를 곱해서 CAM을 계산할 수 있습니다.
   **어느 위치가 해당 클래스 예측에 기여했는지 heatmap 형태로 시각화**합니다.

| 위 내용이 요약이지만,, 조금 어려울수도!?  그래서 아래와 같이 비교해보며 알아봅시다!!

### 시작전, 기존 [이미지 분류 모델]의 구조와 [CAM 구조]의 차이 구분!!  
- 기존 구조: conv → flatten → FC → softmax  
- CAM 구조: conv → GAP → FC → softmax  


### 기존의 분류모델 원리!! : 한 장의 이미지 (224×224) 를 일반적인 CNN (FC layer 포함) 분류 모델넣을떄!!
| 단계 | 데이터 형태        | 설명                                      |
|------|--------------------|-------------------------------------------|
| 📷 입력 이미지             | `[3, 224, 224]`     | RGB 이미지                                 |
| 🔄 CNN 마지막 conv 출력     | `[512, 7, 7]`       | 512개의 7×7 feature map                     |
| 🔃 Flatten                 | `[512 × 7 × 7]` = `[25088]` | 공간 정보를 펼쳐서 1차원 벡터로 만듦       |
| 🧮 Fully Connected Layer   | `[N_classes]`       | FC Layer에서 예측 score 생성 (weight shape: `[N_classes, 512]`)  
⚠️ 여기서 특정 클래스의 가중치 `[512]`가 CAM의 **class_weight**로 사용됨 |
| 🎯 Softmax                 | `[N_classes]`       | 확률화된 예측 결과             |
| 🚫 CAM 불가능             | ❌ 없음             | 공간 정보가 flatten으로 사라졌기 때문에 CAM 생성 불가 |

- 위와 같이 CAM은 불가하며, [N_classes] 들에 대한 확률값만 나오게됩니다!!
- N_classes란 객채로 구분하고자하는 대상의 갯수 (ex. 강아지,고양이 등 개체의 갯수)
- 이때 Fully Connected Layer 에 사용되는 weight!! 그 weight가 **class_weight**로서 CAM에 활용되어요!! 

### 한 장의 이미지 (224×224) 를 ResNet18에 넣어 CAM 이미지 만드는 과정!!

| 단계 | 데이터 형태        | 설명                                      |
|------|--------------------|-------------------------------------------|
| 📷 입력 이미지       | `[3, 224, 224]`   | RGB 이미지                                 |
| 🔄 CNN(resnet) 마지막 conv 출력 | `[512, 7, 7]`     | 512개의 7×7 feature map                     |
| 🔥 CAM 계산 : CNN(resnet) 마지막 conv 출력 과  **class_weight**의 weighted sum | `[7, 7]`     | 7×7 feature map                     |
| 🔼 최종 CAM 이미지 만들기 (Upsample)       | `[224, 224]`      | 원본 이미지 위에 히트맵 overlay 가능        |
| 📉 GAP(Global Average Pooling) | `[512]`             | feature map[512, 7, 7]의 채널별 평균 벡터               |
| 🧮 FC Layer               | `[N_classes]`       | GAP 결과를 클래스별 score로 변환              |
| 🎯 Softmax               | `[N_classes]`       | 예측 클래스 확률값 출력                        |

- CAM이미지를 만들고 여기서도 최종 Class구분을 할 수 있는데, 기존의 분류모델과 다른 결과가 나올수도 있습니다!!  

---

## 📸 CAM의 실제 활용 예시

- 예를 들어, AI가 강아지 이미지를 ‘dog’로 분류했다면  
  CAM은 **얼굴, 귀, 꼬리** 등 강아지의 주요 특징 부위를 밝게 표시해줍니다.
- 사용자는 “AI가 정말 합리적으로 분류했는가?”를 직관적으로 파악할 수 있습니다.
- 아래는 골든 리트리버를 CAM으로 분류해보았어요!! 귀부분을 바탕으로 분류했다고하네요~~^^
| 다음 포스팅에서 이 코드를 분석해보며 구조에 대하여 더 자세히 알아보겠습니다!!

![golden](https://github.com/user-attachments/assets/d3fd65b1-11bd-44dc-9516-3118fd586bf0)

---

## 🧰 CAM이 놀라운 영향력

- CAM은 **CNN 예측 결과를 시각적으로 설명**한 최초 연구 중 하나입니다.
- 약지도 학습 기반 객체 지역화(Weakly-Supervised Object Localization) 분야의 발전에 큰 영향을 주었고,  
  이후 Grad-CAM 등 더 다양한 해석 가능 방법이 개발되는 계기가 되었습니다.

---

## 💡 결론 : CAM이 놀라운 이유!!

- ✅ 모델의 예측 근거를 시각적으로 확인할 수 있다
- ✅ 잘못된 판단, 편향, 오류의 원인을 쉽게 진단할 수 있다
- ✅ 모델의 신뢰성과 투명성이 대폭 향상된다

---

CAM은 "AI 블랙박스"의 벽을 허무는 시작점이었습니다.  
지금도 많은 연구자들이 다양한 해석 가능 인공지능(XAI) 분야로 확장해 나가고 있습니다.  
여러분도 직접 CAM을 실습해보며 AI의 속마음을 들여다보는 경험을 꼭 해보세요!
