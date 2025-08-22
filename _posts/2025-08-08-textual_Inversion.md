---
layout: post
title: "🎨 An Image is Worth One Word: Textual Inversion - 이미지를 `거시기` 화 해보리기!!"
author: [DrFirst]
date: 2025-08-08 09:00:00 +0900
categories: [AI, Research]
tags: [Text-to-Image, Diffusion, Personalization, Textual Inversion, Generative AI]
sitemap :
  changefreq : monthly
  priority : 0.8
---

---

## 🧠 (한국어) Textual Inversion 알아보기!  
_🔍 아따 거시기와 같은 모양으로 그리랑깨!!_

![Image](https://github.com/user-attachments/assets/8acd5852-f556-433c-8641-833cabe58b7b)

- **제목**: [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)  
- **학회**: ICLR 2023 (Rinon Gal et al.)  
- **핵심 키워드**: `Text-to-Image`, `Diffusion Models`, `Personalization`, `Textual Inversion`  

---

### 🧠 문제의식  

Text-to-Image 생성 모델들 (Stable Diffusion, DALL·E, Imagen 등)은 뛰어나지만,  
**개인화(personalization)** 측면에서 한계가 있음.  

- 내가 키우는 강아지 🐶 같은 구체적인 객체를 그리게 하고 싶어도, 모델은 모름.  
- 매번 새로운 객체/사람을 학습시키려면 **모델 전체를 재학습해야 하는데 이는 비효율적**.  

---

### 🌱 핵심 아이디어: Textual Inversion  

> **단 몇 장의 이미지를 통해 새로운 개념을 표현하는 “단어 임베딩”을 만들자!!!**

> 첫번째이미지의 거시기는 "머리가 없는 가부좌 동상" 이고 둘째이미지의 거시기는 "고양이 모양의 수제품" 이다!
![Image](https://github.com/user-attachments/assets/b039b904-cc71-4c83-9a9a-40deb7036f39)

- 전체 Diffusion 모델을 학습시키는 대신,  
- **하나의 토큰 임베딩 벡터**만 학습 → *가짜 단어(pseudo-word)*처럼 사용.  
- 이 단어를 프롬프트에 삽입하면, 해당 객체를 다양한 장면·스타일에서 생성 가능.  

**예시**:  
- 입력: 강아지 사진 3~5장  
- 학습: `<dog123>` 이라는 토큰 벡터 생성  
- 프롬프트: `"A painting of <dog123> in the style of Van Gogh"`  
- 결과: 고흐풍으로 변환된 나의 강아지 이미지 ✨  

---

### 🔧 방법론 

![Image](https://github.com/user-attachments/assets/beddb3ba-5c7c-4816-884b-78cbb307999f)

1. **입력 데이터 준비**
   - 대상 객체/사람의 대표 이미지 **3~5장** 수집 (정면/측면, 다양한 배경/조명 포함 권장).
   - 과도한 배경 노이즈·강한 필터·낮은 해상도는 지양.
   - 3~5 장이있어야하는 이유는 거시기를 파악할때 알맞게 특징을 뽑을수 있도록하기위해서!!  

2. **모델 고정 (Freeze Pretrained Diffusion)**
   - Stable Diffusion 등 **사전학습된 Diffusion 모델과 텍스트 인코더는 동결**.
   - 단 하나의 **새 토큰 임베딩 S\*** (예: 768차원)만 학습 대상으로 등록.

3. **최적화 루프 (Token Vector Training)**
   - 프롬프트 구성: 예) `"a photo of S* dog"`, `"S* plush toy on a table"`.
   - **Forward**: 프롬프트 → 텍스트 인코더(동결) → U-Net(동결) → 노이즈 예측 / 샘플링.
   - **손실 계산**:
     - (필수) **Diffusion 노이즈 예측 MSE**:  

       $$
       \mathcal{L}_{\text{diff}} = \|\hat{\epsilon} - \epsilon\|^2
       $$

     - (선택) **정체성 유지/유사도 보조손실** (CLIP/Image encoder 특징 유사도):  

       $$
       \mathcal{L}_{\text{clip}}
       $$

     - 총손실:  

       $$
       \mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{clip}}
       $$

   - **역전파 & 갱신**: 파라미터 업데이트는 **S\*** 임베딩에만 적용 (Adam 등).
   - 여러 이미지·여러 프롬프트 변형에 대해 수십~수백 step 반복 → **S\***가 타깃 개념을 가리키도록 수렴.

4. **활용 (Generation with Learned Token)**
   - 학습 완료된 **S\***를 일반 단어처럼 프롬프트에 삽입:
     - `"a portrait of S* in the style of Van Gogh"`
     - `"S* sitting on a sandy beach at sunset"`
   - **모델 가중치는 그대로**, **토큰 임베딩만** 로드하여 다양한 장면·스타일로 생성.

---

#### 핵심 포인트
- **모델은 고정**, **새 토큰 임베딩 S\***만 학습.
- S\*는 타깃 객체를 **저장**하지 않고, **잠재 공간(latent space)에서 그 객체를 가리키는 좌표/주소** 역할.
- 학습 시 Diffusion 샘플링 반복으로 **리소스 소모↑**, 활용 시에는 **가볍고 유연**.

#### 추가 사항  
- 초기 S\*는 랜덤 또는 **유사 의미 단어 임베딩 평균**으로 초기화하면 수렴이 빠름.
- 프롬프트에 **클래스 단서**(e.g., dog, figurine, backpack)를 함께 넣으면 안정적.
- 과적합 방지: 이미지 증강(크롭/색감/좌우반전), 프롬프트 다양화, 손실 가중치 \(\lambda\) 조절.

---

### 🧪 실험  

> 꽤 잘만들쥬!?  
![Image](https://github.com/user-attachments/assets/e66503b2-1adc-41b6-9d0e-52a35190bd29)

- 단 **3~5장만으로**도 개념 학습 가능.  
- 다양한 스타일, 장소, 맥락에서 **일관된 재현** 성공.  
- 모델 전체 학습보다 **빠르고 효율적**.  

---

### 📊 결과  

- **정체성 유지(Fidelity)**: 객체의 고유한 특징 유지.  
- **일반화(Generality)**: 임의의 텍스트 프롬프트와 조합 가능.  
- **효율성(Efficiency)**: 객체마다 하나의 임베딩 벡터만 필요.  


---

### ✅ 기여  

- **Textual Inversion** 기법 제안 → 초경량 개인화 방법.  
- **Diffusion 모델**에 적용 가능함을 보임.  
- 다양한 응용 가능성 제시:  
  - 개인 사진 편집 ✨  
  - 맞춤형 캐릭터 생성 🎭  
  - 디자인·패션 👜  

---

### ⚠️ 한계 (Limitations)

- **정확한 형태(Shape) 학습 어려움** → 개념의 “의미적 본질(semantic essence)”을 주로 포착  
- **최적화 시간 길음** → 개념 하나 학습에 약 2시간 소요  
  - 개선 방안: 인코더 학습을 통해 이미지를 바로 텍스트 임베딩으로 매핑  
- 정밀도가 요구되는 작업에는 아직 한계 존재  

---
### 🚀 결론 (Conclusions)

- **Textual Inversion 기법 제안**  
  - 사전학습된 텍스트-투-이미지 모델의 임베딩 공간에 **새로운 pseudo-word** 삽입  
  - 자연어 프롬프트에 삽입하여 직관적이고 단순한 방식으로 개념을 새로운 장면·스타일에 적용 가능  

- **특징**  
  - LDM(Rombach et al., 2021) 기반으로 구현했으나 **특정 아키텍처에 종속되지 않음**  
  - **다른 대규모 모델에도 적용 가능!!** → 정합성, 형태 보존, 품질 향상 가능성  

- **의의**  
  - **개인화된 생성 AI 연구의 토대** 마련  
  - 예술적 영감, 제품 디자인 등 다양한 **후속 응용 가능성** 제시  

### 🚀 의의  

---

👉 **정리**:  
*"An Image is Worth One Word"*는 개인화를 단 하나의 **새로운 단어 학습**으로 가능케 함.  
거대한 모델을 다시 학습할 필요 없이, 작은 토큰만 학습해도 원하는 객체를 자유롭게 생성 가능.  

---
