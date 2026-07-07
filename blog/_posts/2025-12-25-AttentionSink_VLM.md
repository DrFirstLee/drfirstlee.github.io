---
layout: post
title: "👀 Visual Attention Sink & VAR 논문 공부 (ICLR 2025)"
author: [DrFirst]
date: 2025-12-25 09:00:00 +0900
categories: [AI, Research]
tags: [LMM, Attention Sink, VAR, Multimodal, ICLR, ICLR 2025]
sitemap:
  changefreq: monthly
  priority: 0.8
---

### 👀 SEE WHAT YOU ARE TOLD — Visual Attention Sink in LMMs

> **논문**: [SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS](https://arxiv.org/pdf/2503.03321)  
> **저자**: Seil Kang, Jinyeong Kim, Junhyeok Kim, Seong Jae Hwang  
> **학회**: ICLR 2025  
> **키워드**: Visual Attention Sink, LMM, LLaVA, VAR (Visual Attention Redistribution)  
> **한 줄 요약**:  
> LMM은 이미지를 “제대로 보지” 못하고, **이미지 안의 쓰레기통 토큰(visual sink)** 에 의미 없이 attention 을 쏟는다.  
> 이 토큰들을 분석해서 **쓸모없는 attention을 걷어다가 진짜 중요한 이미지 패치에 재분배**해 주면,  
> **추가 학습 없이도 다양한 비전-언어 벤치마크 성능이 오른다!**

---

### 🧩 문제 정의: LMM, 정말 이미지를 `잘` 보고 있나?

LLaVA, Qwen2-VL, InternVL 같은 Large Multimodal Models(LMMs)는  
이미지 인코더 + LLM 디코더 구조로 동작한다.

텍스트 토큰이 시각 정보를 읽어올 때 사용하는 통로가 바로 **Text–Image Attention**.  
이론적으로는:

> `"bird"` 토큰 → 새가 있는 패치에만 강하게 attention

이어야 하지만, 논문에서 LLaVA-1.5-7B의 attention map 을 시각화해 보니

![err](https://github.com/user-attachments/assets/bd7e8658-805e-434e-9082-76985395f249)  

- `"bird"`, `"banana"`, `"knife"` 등 **어떤 텍스트 토큰이든 상관없이**
- **항상 같은 몇 개의 이미지 토큰(패치)에 높은 attention**  
- 심지어 그 패치들은 **질문과 전혀 상관없는 배경 영역**

즉,

> LMM도 언어 모델처럼 **“특정 토큰에 attention을 버리는”** 현상을 가진다.  
> 다만 이번에는 **언어가 아니라 이미지 토큰 쪽에 생긴 sink**라는 점이 포인트!!  

논문은 이 현상을 **Visual Attention Sink**라고 부른다.

---

### 🔍 핵심 관찰 1: Visual Sink Token의 정체

논문의 첫 번째 핵심 기여는

> *“이 이상한 토큰들이 도대체 뭐냐?”* 를  
> **hidden state level**에서 해부한 것.

#### 1) Massive Activation in Sink Dimensions

언어 모델에서 알려진 현상:

- BOS, `.` , `\n` 등 **의미가 거의 없는 토큰**에 대해
- 특정 hidden 차원들만 **비정상적으로 큰 값(“massive activation”)** 을 갖는다.
- 이 차원 index 집합을 \( D_{sink} \)라고 부른다.

저자들은 LLaVA-1.5-7B, LLaVA-1.5-13B, Qwen2-VL-7B, InternVL2-8B 등 여러 LMM에서

- **이미지 토큰 중 일부가 BOS와 동일한 sink dimension에서 massive activation** 을 보인다는 것을 발견.  
- 즉, **“시각적 sink 토큰”도 언어 sink 토큰과 똑같은 패턴**을 가진다 (논문 Fig.2, Fig.7).

그래서 **Visual Sink Token**을 다음처럼 정의:

- hidden state \( x \)에서
- sink dimension value \( \phi(x) \)가 threshold \( \tau \) 이상인 토큰  
  → sink token 으로 분류 (논문에서는 \( \tau = 20 \) 사용)

#### 2) Background에 몰려 있는 토큰들

Segmentation 데이터셋(PASCAL VOC, COCO) 위에서 위치를 비교해보면 (Table 6):

- Visual Sink Token의 **90% 이상**이  
  **객체가 아닌 배경 영역**에 위치
- 의미 있는 객체와 관련된 토큰들은 대부분 **non-sink**

→ ViT에서의 "background sink / register" 현상과 완전히 평행한 그림.

---

### 🔧 핵심 관찰 2: Sink Token은 “거의 아무것도 하지 않는다”

“attention 은 높은데, 진짜로 유용한가?”를 확인하기 위해 두 가지 실험을 한다.

#### 1) Token Masking 실험 (Attention Knockout)

- Visual Sink Token → Text로 가는 attention을 **완전히 차단**  
  (해당 토큰에서 텍스트 토큰으로의 α를 0으로 만드는 방식)
- 비교군: 같은 개수의 이미지 토큰을 **랜덤으로 마스킹**

결과 (Fig.3(b)):

- **Sink Token 마스킹 → 성능 거의 변화 없음**
- **랜덤 토큰 마스킹 → 성능 뚝 떨어짐**

→ Sink Token은 **attention은 많이 받지만, 정보 기여도는 거의 0**.

#### 2) Residual Stream Contribution 분석

각 visual token이 residual stream에 주는 기여량을

\[
\|\alpha_{i,j} \, x_j W_{OV}\|
\]

으로 정의하고, sink vs non-sink 평균값을 비교 (Fig.3(c)):

- Visual Sink Token의 residual contribution은  
  다른 토큰들에 비해 **훨씬 작다**.

즉,

> Visual Attention Sink = **“이미지 버퍼/쓰레기통 역할을 하는 토큰들”**  
> 정보는 없는데 attention과 hidden activation만 크다.

---

### 💡 아이디어: 쓸모없는 Attention, 예산으로 재활용하자

여기까지의 관찰을 한 줄로 정리하면:

> "시각적 sink 토큰에 쏟아지는 attention은 **거의 낭비**다."

동시에, 다른 연구들(Chen 2024, Liu 2024 등)에서는

- LMM이 텍스트에 비해 **이미지에 너무 적은 attention**을 주고
- 그 결과 object hallucination, spatial reasoning 실패 등이 발생한다고 보고했다.

그래서 논문은 다음과 같이 생각한다:

> **“Sink 토큰으로 흘러가는 attention을 걷어다가  
> 진짜 이미지 정보(visual non-sink)에 다시 뿌리면 어떨까?”**

그리고 이 아이디어를 구체화한 것이 바로

> **VAR: Visual Attention Redistribution**

---

### 🚀 방법: VAR (Visual Attention Redistribution)

VAR는 **완전히 training-free**한 방법이다.

1. **이미지에 집중하는 head(이미지-중심 head, Image-Centric Head)를 선택하고**
2. **그 head에서만 attention을 재분배**한다.

#### 1단계: Image-Centric Head 선택

모든 head에 손대면 모델 구조를 깨버릴 수 있으니,  
**“이미지 정보를 잘 보고 있는 head만 골라서 수정”** 하는 게 중요하다.

논문은 **Visual Non-Sink Ratio**라는 지표를 정의한다 (Eq.3):

\[
r_{i}^{\ell,h} =
\frac{\sum_{j \in I_{vis} \setminus I^{\ell}_{q,vis}} \alpha_{i,j}^{\ell,h}}
     {\sum_{j \in I_{vis}} \alpha_{i,j}^{\ell,h}}
\]

- 분자: **visual non-sink 토큰에 가는 attention 합**
- 분모: **전체 visual 토큰(attention to all visual) 합**

이 비율이 높은 head일수록

> “sink garbage 말고 실제 이미지 패치에 더 집중하는 head”

라고 볼 수 있다. 논문 Fig.4를 보면:

- **non-sink ratio가 높은 head → 질문과 관련된 객체 부분에 집중**
- ratio가 낮은 head → 이미지 전반에 흐리게 뿌려진 attention

그래서

- \( r_{i}^{\ell,h} \ge \rho \) 인 head를  
  **Image-Centric Head (ICH)** 로 선택 (hyperparam \( \rho \))

#### 2단계: Sink → Non-Sink로 Attention 재분배

선택된 ICH에 한해서만, 각 text token i에 대해:

1. sink token들에 가는 attention 중 **일부 비율 p**를 걷어서  
   **attention budget \(\Omega\)** 로 모은다.

   - sink 쪽: \(\alpha_{i,j}^{q} = (1-p)\alpha_{i,j}\)
   - budget: \(\Omega = p \sum_{j \in I_q} \alpha_{i,j}\)

2. 이 budget을 **visual non-sink 토큰들에 비례해서 재분배** (Eq.4):

   \[
   \alpha^{q}_{i,j} = \alpha_{i,j} + \Omega \cdot
   \frac{\alpha_{i,j}}{\sum_{k \in I_{vis} \setminus I_{q,vis}} \alpha_{i,k}}
   \quad (j \in I_{vis} \setminus I_{q,vis})
   \]

3. 전체 attention 합은 여전히 1로 유지됨  
   → 확률 분포 성질은 보존

이 과정을 **모든 텍스트 토큰**(질문, 시스템 프롬프트, 생성된 답변 토큰)에 적용하되,

- 마지막 레이어는 건드리지 않는다 (마지막 레이어는 특수한 역할을 가진다는 선행연구를 따름).

---

### 📊 실험 결과: “이미지만 더 잘 봤을 뿐인데…”

논문은 LLaVA-1.5, VILA, Qwen2-VL, InternVL2 등 다양한 LMM에 VAR를 붙여서 평가한다.

#### 1) General Vision-Language Benchmarks (Table 1)

예: LLaVA-1.5-7B

- **VQAv2**: 78.5 → 78.6  
- **GQA**: 62.0 → 63.5  
- **VizWiz**: 50.0 → 53.7  
- **MM-Vet**: 31.1 → 33.7

재밌는 포인트:

> LLaVA-1.5-7B + VAR 가  
> **베이스 LLaVA-1.5-13B**를 일부 벤치마크에서 이김  
> → “모델 크기” 대신 “내부 attention 조정”만으로도 꽤 큰 이득을 얻을 수 있다는 시사점.

#### 2) Hallucination Benchmark (Table 2)

CHAIR, POPE, MMHal-Bench 평가에서

- **Hallucination 관련 지표↓**
- **정확도/신뢰도 관련 지표↑**

→ “이미지를 더 잘 보는 것만으로도 헛것을 덜 본다” 를 정량적으로 보여줌.

#### 3) Vision-Centric Benchmark (Table 3)

- MMVP, CV-Bench 2D/3D 같은 공간 이해·3D 관계 중심 벤치에서도  
  일관된 개선.

---

### 🔬 Ablation: 왜 “이미지 토큰만” 손대야 할까?

논문에서는 다양한 ablation도 해본다:

1. **모든 head에 VAR 적용**  
   → 아예 모델이 망가져서 점수 0 (Table 4의 `w/o Head selection`)  
   → head selection(ICH만 수정)이 **필수**

2. Attention budget을  
   - Text + Visual 모두에 재분배  
   - Text에만 재분배  
   - Visual에만 재분배(본 방법)

   결과 (Table 5):

   - Text-only: 오히려 성능 악화  
   - Text+Visual: 약간 이득  
   - **Visual-only(본 방법)가 가장 크고 안정적인 이득**

   → LMM은 이미 **텍스트에는 충분히 집중**하고 있었고,  
     진짜 부족했던 건 **이미지 쪽 attention**이라는 의미.

---

### 🧠 나의 코멘트!

이 논문을 StreamingLLM의 Attention Sink와 연결해서 보면 진짜 재미있다.

- **StreamingLLM**:  
  - “초기 몇 토큰이 사실상 **attention sink/레지스터** 역할을 한다.  
    → 그 토큰들만 유지해도 무한 스트리밍 가능”
- **이번 Visual Attention Sink 논문**:  
  - “이미지에서도 **비의미적인 sink 패치**가 존재한다.  
    → 여기에 쏟아지는 attention을 회수하면 이미지 이해가 좋아진다.”

둘의 공통점:

> 모델은 언어든 비전이든 **내부 계산을 위한 ‘쓰레기통/레지스터 공간’을 자발적으로 만들어 쓴다.**  
> 이게 학습의 부산물처럼 생겼는데,  
> **나중에 해석 관점에서 보면 꽤 일관된 구조적 패턴**이라는 점이 흥미롭다.

또 하나의 인사이트:

- ViT의 *register token* (Darcet 2023)  
- 언어 LLM의 *attention sink token*  
- LMM의 *visual attention sink token*  

이 세 가지가 **“모델 안에서 정보를 저장·고정하는 역할”**이라는  
하나의 큰 패턴 위에 놓여 있다는 느낌을 준다.

> 앞으로는 “이 sink/레지스터 공간을 어떻게 설계·제어하느냐”가  
> 단순 효율을 넘어서 **해석 가능한 제어(steering)** 의 핵심 축이 될 수도 있을 것 같다.

---

### ✅ 정리

- LMM은 **이미지 안의 의미 없는 패치(visual sink)**에  
  **과도한 attention**을 주는 경향이 있다.
- 이 토큰들은 언어 모델의 BOS sink와 같이  
  **특정 hidden dimension에서 massive activation**을 보이며,
  실제 예측에는 거의 기여하지 않는다.
- 논문은 이 낭비되는 attention을 **attention budget**으로 보고,
  **이미지-중심 head에서만 visual non-sink 토큰으로 재분배(VAR)** 하자고 제안한다.
- 추가 학습 없이도
  - 일반 VL 벤치마크
  - hallucination 감소
  - vision-centric tasks  
  모두에서 안정적인 성능 향상을 달성한다.

---

필요하다면 다음 포스트로

- VAR 수식/알고리즘을 코드 수준으로 풀어 쓰기 (PyTorch pseudo-code)  
- StreamingLLM의 attention sink와 이 논문을 **공통 프레임워크**로 묶어 보는 리뷰  
- Vision Transformer register token / LMM visual sink / LLM text sink를  
  하나의 “hidden workspace” 관점에서 비교 분석

같은 것들을 이어서 정리해봐도 재미있을 것 같다 🙂
