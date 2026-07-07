---
layout: post
title: "⚡ StreamingLLM & Attention Sink 논문 공부 (ICLR 2024)"
author: [DrFirst]
date: 2025-12-24 09:00:00 +0900
categories: [AI, Research]
tags: [LLM, StreamingLLM, Attention Sink, Transformer, ICLR, ICLR 2024]
sitemap:
  changefreq: monthly
  priority: 0.8
---

### ⚡ StreamingLLM & Attention Sink — 핵심 논문 리포트

![streamingllm](https://github.com/user-attachments/assets/3b04b7bb-8c8f-4cf2-9301-31b773afbb83)

> **논문**: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)  
> **저자**: Guangxuan Xiao et al.  
> **학회**: ICLR 2024  
> **Repo** : [GitHub - mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm?tab=readme-ov-file)
> **핵심 요약**: LLM이 긴 입력에서 성능이 붕괴되는 이유는 ‘초기 토큰이 가지는 쓰레기통(sink) 역할’ 때문!  
> 이를 유지하는 것만으로도 **무한 길이 스트리밍**이 가능해진다!

---

### 🧩 문제 정의: LLM은 긴 대화를 하면 갑자기 망가지는데 해결법은 없을까??


LLM과 대화할때 
그 길이보다 대화가 길어지면 다음문장 생성이 완전히 망가진다.  
기존 연구와 이번 연구를 타입별로 구분하면 아래와 같다!

![sink](https://github.com/user-attachments/assets/9ec1d267-8c7e-45a1-b1ac-6fc6990ad601)

(a) Dense Attention: 학습문장의 길이가 늘어나도 모든 KV를 저장해두며 Token의 Positional ID를 계속 저장한다 -> 효율이 엄청 안좋다! 뿐만아니라 positional overflow로 인해 문장이 Max Token을 넘어가면 성능이 매우 떨어진다.  
(b) Window Attention. 각 Token 기준 앞의 일정(L) window만 보고 있음 -> max length 넘어도 효율좋계 계산이 가능하지만! KV가 없어짐에 모델이 붕괴되버려!  
(c) Sliding Window + Recompute : 앞의 일부 KV만 새로 계산하며  Token의 Positional ID를 계속 저장한다 -> 매번 새로 계산하기에 효율도 안좋고 positional overflow로 인해 문장이 Max Token을 넘어가면 성능이 매우 떨어진다.  
(d) 이번연구, StreamingLLM : Attention Sink를 인정하고, 초기 토큰 몇 개만 KV 캐시에 계속 유지하며  Token의 Positional ID를 재정렬해서 사용하기에 max token넘어가도 성능이랑 효율면에서 모두 우수!  


#### 문제 1 — KV 캐시 폭발 - (a)
LLM은 **모든 과거 토큰의 Key/Value(KV)**를 저장한다.  
입력이 길어질수록 캐시는 기하급수적으로 커지고,  
메모리는 터지고, 디코딩 속도는 느려진다.

#### 문제 2 — 학습된 길이를 넘어서면 성능 붕괴 -(a)(c)  
Llama 계열은 보통 **4096 토큰까지만** 학습됨.  그 이상의 Positional ID는 계산이 불가하다!! 
이를 넘어가면 *갑자기 이상한 말을 하기 시작한다.*

그래서 등장한 해결책이 바로 **Window Attention**  
→ 최근 몇 개(L개) 토큰만 유지하는 방식!

하지만…


### 문제3 : Window Attention의 치명적 문제: 초기 토큰을 지우는 순간 붕괴한다

일반적으로 **최근 토큰만 보관**하니까 좋아 보이는데,  
문제는 **초기 토큰의 KV가 사라지는 바로 그 순간** perplexity가 폭발한다는 것!!


---

### 🧠 문제의 핵심 발견: Attention Sink 현상

본 연구의 핵심인! 놀라운 패턴이 발견되었다

> **LLM은 의미와 상관없이 '처음 몇 개의 토큰'에 굉장히 높은 attention을 준다.**

![sink](https://github.com/user-attachments/assets/37e000f3-2210-4831-bf27-4f62db29e361)

이들은 **내용이 중요해서** 집중되는 것이 아니다.  
단지 Transformer의 Softmax가 attention의 합을 1로 만들어야 하고,  
필요한 정보가 없을 때 **"어차피 항상 보이는 초기 토큰에 attention을 버린다"**.

즉, 초기 토큰은 일종의 **attention 쓰레기통**처럼 작동한다.

---


### 🚀 해결책: StreamingLLM

Attention Sink의 역할을 정식으로 인정하고,  
아예 **초기 토큰 몇 개(보통 4개)만 KV 캐시에 계속 유지**하는 방식을 제안했다.


✔ 작동 방식

![structure](https://github.com/user-attachments/assets/4415af00-6687-4efe-998e-ddbf2552d0ad)  

- 최근 토큰 L개 → 그대로 유지  
- 초기 토큰 4개 → **영구 유지 (attention sink 용도)**  
- 나머지 중간 토큰 → 과감하게 날려버려(evict)!!
- 마지막 Rolling 토큰만 사용!!


---

### 📊 실험 결과

![ppl](https://github.com/user-attachments/assets/100224d4-a4c0-4226-a6d8-3acd8da25001)

- Sliding Window with Re-computation 은 Input 이 커질수록 latency랑 사용 메모리 계속 증가  
- **StreamingLLM은 둘의 장점만 결합 → 메모리도 덜쓰고+ 빠름**

![acc](https://github.com/user-attachments/assets/6b23d8ef-8c04-4386-8b03-66663ca7082e)
- OOM도 안나면서 정확도도 엄청 높았다!

---

### 🧠 나의 코멘트!

현대 LLM이 **attention sink 같은 구조적 안정 장치** 없이는  
정상적으로 길이 일반화를 못 한다는 사실이 정말 흥미로웠다.

또한 Vision Transformer의 **Register Token** 개념과  
Transformer LLM의 **Attention Sink**가  
서로 **아주 유사한 메커니즘**이라는 점도 인사이트가 크다.

> ViT는 “여분 패치를 register로 사용”  
> LLM은 “초기 토큰을 attention sink로 사용”

둘 다 **모델 내부 계산을 위한 숨겨진 공간**이 필요하다는 사실이 동일하는것이 재밋다!

