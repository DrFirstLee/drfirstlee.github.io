---
layout: post
title: "📝Understanding CLIP-Adapter - CLIP-Adapter 알아보기?!!"
author: [DrFirst]
date: 2025-06-17 07:00:00 +0900
categories: [AI, Research]
tags: [Feature Adapter, Vision-Language Model, Few-shot learning, Open-Vocabulary, IJCV, IJCV 2024]
sitemap :
  changefreq : monthly
  priority : 0.8
---


### 🧠 Understanding CLIP-Adapter!  
_🔍 Easy Fine-Tuning for CLIP with Just One Adapter!_  

![manhwa](https://github.com/user-attachments/assets/dd150996-b141-4656-a4d8-76b4b94aeaf9)

> Paper: [CLIP-Adapter: Better Vision-Language Models with Feature Adapters](https://arxiv.org/pdf/2110.04544)  
> Published: IJCV 2024 (Gao, Peng, et al.)  
> Code: [gaopengcuhk/CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter)  

---

#### 📌 Why CLIP-Adapter?  

With the rise of large-scale vision-language models like [**CLIP**](https://drfirstlee.github.io/posts/CLIP/),  
our ability to understand images and text together has significantly improved.  
One of the most exciting aspects is **zero-shot classification** — prediction without labeled data!

However, these models also have some **serious limitations**.

---

##### ❓ Problem 1: Prompt Sensitivity  

CLIP heavily depends on **natural language prompts** like `"a photo of a {label}"`.  
Changing a few words (e.g., `"this is a dog"` vs `"a photo of a dog"`) can affect performance.  
This requires **manual prompt engineering**, which becomes useless in domain-specific tasks.

> 📌 In short: CLIP is too sensitive to small changes in prompts.

To address this, **CoOp (Context Optimization)** was introduced!

> CoOp demonstrated that prompt tuning alone can fine-tune CLIP effectively!

- 🧠 CoOp replaces natural language prompts with **learnable continuous vectors**.  
  - For example, instead of `this is a dog`, we input `[V1] [V2] [V3] dog`.  
  - Here, `[V1], [V2], [V3]` are **learned vectors**, and the user only inputs the class name like `dog`.
- No more manual prompt crafting — the model **learns the prompt by itself**!

---

##### ❗ Problem 2: Tuning Only the Text Side

But CoOp **only tunes the prompt** — that is, the **text side** of CLIP.  
The **image encoder remains fixed**.

> "We're adapting the language, but still trusting the same image representation?"

This imbalance limits performance, especially in **few-shot** or **domain-specific** scenarios.

> As shown below, CoOp learns only the `[V1], [V2]` tokens in the text.  
> CLIP-Adapter, in contrast, introduces **adapters on both the image and text branches**!

![compareCOOP](https://github.com/user-attachments/assets/fd6c145d-7576-46da-a0e2-c1c534856ead)

---

##### 💡 CLIP-Adapter Architecture!!!  

![structure](https://github.com/user-attachments/assets/8b56436c-8e37-494a-9232-5fa84ae2e9a1)

CLIP-Adapter performs **fine-tuning at the feature level for both image and text**.

```
            ┌────────────────────┐                ┌────────────────────┐
            │   Image Input (x)  │                │   Text Input (t)   │
            └────────┬───────────┘                └────────┬───────────┘
                     ↓                                     ↓
       ┌────────────────────────────┐         ┌────────────────────────────┐
       │   CLIP Image Encoder       │         │   CLIP Text Encoder        │
       │       (frozen)             │         │        (frozen)            │
       └─────────┬──────────────────┘         └────────┬───────────────────┘
                 ↓                                     ↓
     ┌─────────────────────┐               ┌─────────────────────┐
     │  Image Adapter MLP  │               │  Text Adapter MLP   │
     │     (trainable)     │               │     (trainable)     │
     └────────┬────────────┘               └──────────┬──────────┘
              ↓                                       ↓
     ┌────────────────────────────┐       ┌────────────────────────────┐
     │ Residual: image + adapted  │       │ Residual: text + adapted   │
     └────────────────────────────┘       └────────────────────────────┘
              ↓                                       ↓
     ┌────────────────────┐              ┌────────────────────┐
     │  Image Embedding   │              │  Text Embedding    │
     └────────────────────┘              └────────────────────┘
                     └───────────────┬───────────────┘
                                     ↓
                      Cosine Similarity / Classification
```

---

##### 🔧 Adapter MLPs (for image and text)

![adapter](https://github.com/user-attachments/assets/8af17eed-1b5a-4069-9836-d974b27f7bea)

The adapter is a **2-layer MLP** with ReLU, also called a bottleneck MLP:

- Structure: `Linear → ReLU → Linear`
- It reduces the feature dimension and then expands it back.

---

##### 🖇️ Residual Connection

![residual](https://github.com/user-attachments/assets/884a49f8-f76c-4dea-850e-394d93599fee)

In few-shot learning, models tend to **overfit** due to limited data.  
To solve this, CLIP-Adapter uses **residual blending**:

> "Blend new knowledge (adapter output) with original CLIP features."

The final feature becomes:

- `α × Adapter Output + (1 - α) × CLIP Feature`

This mixing helps retain the **robustness of CLIP** while injecting **task-specific knowledge**.

---

#### 🔬 Performance Experiments

##### 🧪 CLIP-Adapter Experimental Setup

**Datasets**:

- ImageNet, StanfordCars, UCF101, Caltech101, Flowers102  
- SUN397, DTD, EuroSAT, FGVCAircraft, OxfordPets, Food101

**Settings**:
- Few-shot setups: 1, 2, 4, 8, 16-shot
- Evaluation: average over 3 runs, single A100 GPU

**Implementation**:
- Visual adapter only; text frozen
- Batch size: 32, learning rate: 1e-5
- α, β tuned via grid search
- Visual backbone: ResNet-50
- Text encoder: 12-layer Transformer
- Adapter dim: 256 (¼ of original)
- Prompt: Fixed natural text ("a photo of a {class}")

---

##### 📈 CLIP-Adapter Results

**Baselines**:
- Zero-shot CLIP: frozen model + prompt only
- Linear Probe CLIP: frozen encoder + trainable linear classifier
- CoOp: learns `[V1] [V2] ...` tokens in prompt

![res_compare](https://github.com/user-attachments/assets/7418df5c-fb3e-42f7-aa99-1127700bd362)

CLIP-Adapter outperforms all baselines in accuracy, training speed, parameter efficiency —  
especially in **few-shot learning**.

---

##### 🔍 Where to Put the Adapter?

- Visual adapter: image only, text only, both
  → Best: **image-only**

![adaptersto](https://github.com/user-attachments/assets/c58fa9d4-9704-46fa-8f97-574c20601cd9)

- Insertion layer: ViT-B has 12 layers  
  → Best: insert adapter after **layer 12 (last layer)**

![where](https://github.com/user-attachments/assets/7930d693-3340-4df0-bfbf-a6af0399dd97)

---

##### 🔧 What about Residual Ratio α?

- Fine-grained datasets (e.g. EuroSAT, DTD):  
  → Best α ≈ 0.6–0.8  
- Generic datasets (e.g. Caltech101, ImageNet):  
  → Best α ≈ 0.2

---

#### 🧠 Final Thoughts

This was my second PEFT (Parameter Efficient Fine-Tuning) after studying LoRA —  
and I found CLIP-Adapter both innovative and effective.

> I used to think of "adapter" as just a power plug —  
> Now, I'll always remember CLIP-Adapter! 😄


---

### 🧠 (한국어) CLIP-Adapter 알아보기!  
_🔍 어댑터 하나로 CLIP을 쉽게 Fine tuning 하기!!_  

![manhwa](https://github.com/user-attachments/assets/dd150996-b141-4656-a4d8-76b4b94aeaf9)

> 논문: [CLIP-Adapter: Better Vision-Language Models with Feature Adapters](https://arxiv.org/pdf/2110.04544)  
> 발표: IJCV 2024 (Gao, Peng, et al.)  
> 코드: [gaopengcuhk/CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter)  

---


#### 📌 배경: CLIP-Adapter 등장의 이유!?  

[**CLIP**](https://drfirstlee.github.io/posts/CLIP/)과 같은 대규모 비전-언어 모델이 등장하면서  
이미지와 텍스트를 함께 이해하는 능력이 비약적으로 향상되었습니다.  
그중에서도 **"zero-shot classification"**은 레이블 없이도 추론이 가능하다는 점에서 혁신적이었죠.  

하지만 이런 모델에도 **중대한 한계**가 있었습니다.  

---

##### ❓ 문제 1: 프롬프트 의존성 (Prompt Sensitivity)  

CLIP은 `"a photo of a {label}"` 같은 **프롬프트에 의존**합니다.  
예를 들어 `"a photo of a dog"`과 `"this is a dog"`은 서로 다른 결과를 낼 수 있습니다.  
이에 어떤 프롬프트가 가장 좋은 성능을 내는지 사람이 직접 설계(prompt engineering) 해야 했습니다!!  
또한 특수 도메인에서는 이런 프롬포트 엔지니어링도 의미가 없었지요!!  

> 📌 이건 마치 CLIP이 단어 하나 바뀐 문장에도 민감하게 반응한다는 뜻입니다.  

그래서 등장한 것이 바로 **CoOp (Context Optimization)**!

> 이 CoOp연구를 통해 프롬포트 튜닝을 바탕으로 CLIP을 fine-tuning할수 있다는 것을 알게되었습니다!  

- 🧠 CoOp은 프롬프트 문장을 **학습 가능한 연속 벡터로 대체**합니다.  
  - 예를 들면 `this is a dog`라고 헀던것을 `[V1] [V2] [V3] dog` 라고 입력하는것입니다!  
  - 이때  `[V1] [V2] [V3]` 은 Fine-tuning하면서 학습되는 벡터로 결국 사람은 dog만 입력하면 되는거죠!  
- 사람이 직접 프롬프트를 디자인할 필요 없이(prompt-free tuning)으로써,  
- 모델이 **프롬프트 자체를 학습**하게 만든 것이죠.  


---

##### ❗ 문제 2: 텍스트만 튜닝하는 방식의 한계


하지만 CoOp은 **텍스트 프롬포트 부분만 미세조정(Fine-Tuning)**합니다.  
이미지의 feature는 그대로 둔다는 뜻이죠.

> "텍스트는 학습했는데, 이미지 표현은 여전히 고정되어 있다?"

이런 불균형은 특히 **특수 도메인**이나 **few-shot 학습**에서 성능 저하로 이어질 수 있습니다.

> CoOp의 구조!! 텍스트 프롬포트 앞의 V1,V2 등등만 학습합니다!!  
> 오늘의 Clip Adapter는 이와 다르게 텍스트, Image 에 대하여 모두 adapter가 있죠!?  
![compareCOOP](https://github.com/user-attachments/assets/fd6c145d-7576-46da-a0e2-c1c534856ead)


---

#### 💡 CLIP-Adapter 구조!!!   

![structure](https://github.com/user-attachments/assets/8b56436c-8e37-494a-9232-5fa84ae2e9a1)  

 CLIP-Adapter는 **이미지와 텍스트의 feature level에서 직접 조정**을 수행합니다.

```text
            ┌────────────────────┐                ┌────────────────────┐
            │   Image Input (x)  │                │   Text Input (t)   │
            └────────┬───────────┘                └────────┬───────────┘
                     ↓                                     ↓
       ┌────────────────────────────┐         ┌────────────────────────────┐
       │   CLIP Image Encoder       │         │   CLIP Text Encoder        │
       │       (frozen)             │         │        (frozen)            │
       └─────────┬──────────────────┘         └────────┬───────────────────┘
                 ↓                                     ↓
     ┌─────────────────────┐               ┌─────────────────────┐
     │  Image Adapter MLP  │               │  Text Adapter MLP   │
     │     (trainable)     │               │     (trainable)     │
     └────────┬────────────┘               └──────────┬──────────┘
              ↓                                       ↓
     ┌────────────────────────────┐       ┌────────────────────────────┐
     │ Residual: image + adapted  │       │ Residual: text + adapted   │
     └────────────────────────────┘       └────────────────────────────┘
              ↓                                       ↓
     ┌────────────────────┐              ┌────────────────────┐
     │  Image Embedding   │              │  Text Embedding    │
     └────────────────────┘              └────────────────────┘
                     └───────────────┬───────────────┘
                                     ↓
                      Cosine Similarity / Classification

```

결국 위의 구조에서 `Adapter MLP` 와 `Residual Connection` 부분이 이번 연구의 핵심인데요!!  

##### 🔧 Adapter MLP (Image, Text에 각각!!)

![adapter](https://github.com/user-attachments/assets/8af17eed-1b5a-4069-9836-d974b27f7bea)  

Adapter 부분의 MLP는!! 
- 두 개의 선형 계층 + ReLU 비선형 함수구조로서,  
- 구조: `Linear → ReLU → Linear`
- Bottleneck 구조로 중간 차원으로 축소했다가 다시 확장하게 됩니다!! 


##### 🖇️ Residual Connection

![residual](https://github.com/user-attachments/assets/884a49f8-f76c-4dea-850e-394d93599fee)

few-shot 으로 학습하게 된다면!!  
학습 데이터가 극히 적기 때문에, 모델이 데이터에 지나치게 맞춰지는(overfitting) 경향이 있습니다!  
이런 오버피팅에 대한 해결 방법으로   Residual Connection (잔차 연결)을 적용했습니다!  

핵심 아이디어는 `"새롭게 학습한 표현과, 기존에 잘 학습된 CLIP 표현을 비율을 조절해 섞자."` 로서  

1. (이미지와 텍스트의 CLIP 임베딩 결과를 adapter 에 통과시킨 결과) X α  
2. (이미지와 텍스트의 기존 CLIP 임베딩 결과) X (1-α)  

로 하여 학습 결과및 CLIP의 기존 결과를 알맞게 섞어 줍니다!  


#### 🔬 성능 실험!!   

##### CLIP-Adapter 실험 세팅!  

1. 📊 사용한 데이터셋

  CLIP-Adapter는 총 11개의 이미지 분류 데이터셋에서 성능을 평가했습니다:

  - **ImageNet**
  - **StanfordCars**
  - **UCF101**
  - **Caltech101**
  - **Flowers102**
  - **SUN397**
  - **DTD**
  - **EuroSAT**
  - **FGVCAircraft**
  - **OxfordPets**
  - **Food101**

  각 데이터셋에 대해 **1, 2, 4, 8, 16-shot** 설정으로 fine-tuning을 수행하고,  
  **전체 테스트 세트**에서 성능을 측정합니다.  
  모든 실험은 **NVIDIA A100 GPU 단일 장비**에서 수행되며,  
  **각 실험은 3회 반복하여 평균 정확도**를 산출합니다!!  

2. ⚙️ 구현 세부 설정

- **기본 구조**: 이미지 특성만 fine-tune (visual adapter), 텍스트(branch)는 고정  

- **하이퍼파라미터**:
  - 배치 사이즈: `32`
  - 학습률: `1e-5`
  - **잔차 비율 α, β**는 각 데이터셋마다 탐색을 통해 선택 (grid search)

- **백본(backbone)**:
  - Visual encoder: `ResNet-50`
  - Text encoder: `12-layer Transformer`

- **어댑터 hidden embedding**: 시각/텍스트 어댑터 모두 `256` (기존 임베딩의 1/4)

- **프롬프트 입력**:
  - CoOp과 달리, **고정 텍스트 프롬프트** 사용  
    예: `"a photo of a {class}"`
  - 세밀한 분류에는 도메인 키워드를 포함  
    예: `"a centered satellite photo of {class}"`


##### CLIP-Adapter 실험 결과 분석!!  

1. 기본 실험
 - CLIP-Adapter는 성능을 비교하기 위해 다음 3가지 주요 베이스라인과 비교 실험을 진행했습니다!  

- Zero-shot CLIP : CLIP 모델 그대로, `a photo of {class}` 로 프롬포트사용
- Linear probe CLIP : CLIP의 이미지 인코더는 고정시키고, 그 위에 **얕은 선형 분류기(linear classifier)**만 학습.
- CoOp (Context Optimization) : 텍스트 프롬포트에 대하여 V1 V2를 추가하여 학습  

![res_compare](https://github.com/user-attachments/assets/7418df5c-fb3e-42f7-aa99-1127700bd362)

CLIP-Adapter 결곡 좋은 성능을 보여주었습니다!!  
위 이미지에서 보듯, 짧은 학습, 적은 parameter및 GPU메모리 빠른 속도에 높은 정확도를 보여줬는데요!  
뿐만아니라 적은 데이터셋 학습 (few shot) 에서도 좋았어요!!

2. 어뎁터는 어디에!?  

추가로 어뎁터를 `이미지만`, `텍스트만`, `이미지랑 텍스트 모두` 에 붙이는 비교도 해보았고!!

![adaptersto](https://github.com/user-attachments/assets/c58fa9d4-9704-46fa-8f97-574c20601cd9)

결국 이미지만 하는게 제일 좋았다고합니다!!  

![where](https://github.com/user-attachments/assets/7930d693-3340-4df0-bfbf-a6af0399dd97)

또한 12개 Transformer레이어로 구성된 CLIP 의 앞부분, 중간부분 등에 붙이는것도 테스트해보았고,  
지금까지 이해한것 처럼 CLIP의 맨 뒷부분,  
즉 12번쨰 레이어(CLIP이 12개 Layer로 구성) 뒤에 붙이는 것이 가장 효율이 좋았습니다!!


3. 잔차 학습의 계수는?!  
- 오버피팅을 막기위한 `Residual Connection`의 계수 평가를 진행했고!!

  a. 세밀한 도메인의 fine-grained 데이터셋의 경우는 최적의 α 값이 보통 0.6 ~ 0.8 수준에!,  

  b. Caltech-101이나 ImageNet처럼 포괄적이고 일반적인 이미지 데이터셋에서는 최적 α 값이 약 0.2 수준이었다고 합니다!  


---

#### 🧠 마무리 생각

LORA에 이어 두번째로 공부해본 PEFT (Parameter Efficient Fine Tuning) 기법!!  
시도도 참신할 뿐만아니라 성능도 인상적이서!  
앞으로 이 방식을 기억해서 여러곳에 사용해봐야겠습니다!!  

\+ 어뎁터하면 전기콘센트 어뎁터만 떠올랐는데, 앞으로는 이 CLIP-Adapter가 기억에 남을것 같네요! :)

---
