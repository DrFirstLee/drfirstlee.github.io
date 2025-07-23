---
layout: post
title: "🧠Lost in the Middle - 긴 문맥에서 언어모델이 진짜 정보를 기억할까?"
author: [DrFirst]
date: 2025-07-16 07:00:00 +0900
categories: [AI, Experiment]
tags: [LLM, Context, Prompt Engineering, Language Models, Long Context, TACL]
sitemap :
  changefreq : monthly
  priority : 0.8
---

---

### 🧠 Reading the Paper `Lost in the Middle`
_🔍 LLMs struggle to remember information located in the middle of long documents!_

> Paper: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)  
> Venue: TACL 2023 (Liu, Nelson F., et al.)

---

### ❓ Core Question from the Paper

> “Can language models **utilize information equally** regardless of its position in a long context?”

**Short answer: No.**  
- Most LLMs are **least effective at recalling information located in the middle** of long documents.
- Even with large context windows, **position bias** still persists.

> As shown below, the performance follows a `U-shape` curve:  
> Models perform best when the answer is at the **beginning (primacy)** or **end (recency)**,  
> but significantly worse when it is **in the middle**.

![u_shape](https://github.com/user-attachments/assets/e7e2996d-25d9-49aa-9c21-8eb91ca19db7)

---

### 🧪 Experiment: Needle-in-a-Haystack

**Setup:**
- Insert a single key fact ("needle") into a long passage
- Place it at the **beginning / middle / end** of the input
- Ask the model to **extract that specific information**

```text
Example:
Document length = 8,000 tokens
[... lengthy text ...]
→ Insert target sentence in the middle
→ Ask: "What was the number mentioned in the document above?"
```

> 👇 The target sentence is hidden in the middle of the input like this:

![prompt_sample](https://github.com/user-attachments/assets/a870a97d-c1d7-44ad-87eb-ad4b9d74cea9)

---

### 📉 Summary of Results (Figure 5)

When the answer is in the **middle** of the document, model accuracy **drops significantly**.  
→ Most models, including GPT-3.5, show a **U-shaped performance curve**.

| Position  | Recall (GPT-3.5) | Recall (Claude-1.3) |
|-----------|------------------|---------------------|
| Beginning | High             | High                |
| Middle ⚠️ | **Lowest**       | Slightly lower      |
| End       | High             | High                |

![result](https://github.com/user-attachments/assets/68954152-f35b-4321-bf6d-601ff5f19404)

> 🔍 GPT-4 also shows similar patterns in a subset of experiments,  
> but was excluded from full-scale experiments due to high cost (see Appendix D).

---

### 📌 Why does this happen? (Section §2.3, §3.2)

- 🔒 **Limitations of absolute positional encoding**
- 🔄 **Self-attention's inherent position bias**  
  → Stronger focus on **early (primacy)** and **late (recency)** positions  
  → **Middle positions receive less attention**
- 📏 **The longer the input, the more the performance degrades**,  
  with over 20% drop in 30-document settings (GPT-3.5)

---

### 🧠 Why does this matter?

> Most real-world tasks like **RAG**, multi-document QA, and summarization rely on **long input contexts**.  
> But what if the model **ignores the middle**?

- 📉 The **position of retrieved documents** directly impacts answer accuracy
- 🔀 Effective chunking and **ordering of key information** is critical
- ❗ Simply increasing the context window size is **not enough**

---

### 💡 Takeaways: Position Bias Matters

> "**LLMs can remember context—but mainly the beginning and the end.**"

**Strategies to mitigate position bias:**

- ✅ **Query-aware contextualization**  
  → Place the query **before** the documents for decoder-only models
- ✅ **Chunk ordering optimization**  
  → Put more relevant content **earlier** in the input
- ✅ **Improved attention architectures**  
  → Encoder-decoder models (e.g., T5, BART) perform better with long input
- ✅ **Position-free architectures**  
  → Hyena, RWKV, and other models aim to remove positional dependence

---

### 🔍 Retrieval-Based QA Setup (Section §2.2)

- **Task**: Multi-document question answering
- **Retriever**: Contriever (fine-tuned on MS-MARCO)
- **Reader input**: Top-k retrieved documents + query
- **Number of docs (k)**: 10, 20, 30
- **Document type**: Only paragraphs (no tables or lists)

---

### 📈 Impact of Increasing Retrieved Docs (Figure 5)

![retriever](https://github.com/user-attachments/assets/e7a81572-0558-4c0d-93cd-fb49d731898c)

- ✅ k = 10 or 20 → improved accuracy
- ⚠️ k = 30 → performance **plateaus or drops**
  - When the relevant document appears **in the middle**, accuracy drops
  - Some models even perform worse than **closed-book** setting

---

### ❗ Retrieval Alone Is Not Enough

- Even if retrieval **includes the correct document**,  
  models may **fail to use it effectively**, especially if it's **in the middle**.

> Retrieval ≠ success  
> → Prompt design must account for **position bias**

**Practical strategies:**

- ✅ Move relevant docs closer to the top
- ✅ Use query-aware formatting
- ✅ Minimize irrelevant context

---

### ✅ TL;DR

> **“LLMs remember long contexts — but often forget what’s in the middle.”**

---

### 🧠 (한국어) `Lost in the Middle` 논문 읽기 
_🔍 LLM은 긴 문서 중간에 있는 정보는 잘 기억하지 못함!!_  

> 논문: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)  
> 발표: TACL 2023 (Liu, Nelson F., et al.)

---

### ❓ 논문이 던진 핵심 질문

> “긴 context 안에서, 모델은 **모든 위치의 정보를 균등하게 활용할 수 있을까?”  

**결론: No.**  
- 대부분의 LLM은 긴 문서에서 **중간 정보**를 가장 잘 놓칩니다.
- context window가 아무리 길어도 **위치 편향(position bias)**이 존재합니다.

> 아래 이미지처럼 `U-shape` 성능 곡선이 나타나며,  
> **앞(primacy)**과 **뒤(recency)** 정보는 잘 기억하지만,  
> **중간 정보는 기억력이 급락**합니다.

![u_shape](https://github.com/user-attachments/assets/e7e2996d-25d9-49aa-9c21-8eb91ca19db7)

---

### 🧪 실험: Needle-in-a-Haystack

**실험 구성:**
- 긴 문서에 단 하나의 핵심 정보 ("needle")를 삽입
- 정보를 문서의 **앞 / 중간 / 끝**에 위치시키고 비교
- 모델에게 해당 정보를 **정확히 추출하도록 질문**

```text
예시:
문서 길이 = 8,000 tokens
[... 긴 텍스트 ...]
→ 중간에 "정답 문장" 삽입
→ 모델에게: "위 문서에 나온 숫자는 몇이었지?" 질문
```

> 👇 아래와 같이 **중간 위치에 핵심 정보**가 있는 프롬프트를 활용하여 성능을 측정합니다:

![prompt_sample](https://github.com/user-attachments/assets/a870a97d-c1d7-44ad-87eb-ad4b9d74cea9)

---

### 📉 실험 결과 요약  

문서 **중간에 정보가 있을 경우**, **모델 정확도가 급락**  
→ GPT-3.5 및 대부분의 모델에서 **U자형 성능 곡선**이 나타남

| 위치     | 회상률 (GPT-3.5) | 회상률 (Claude-1.3) |
|----------|------------------|----------------------|
| 앞부분   | 높음             | 높음                 |
| 중간 ⚠️ | **최저 성능**    | 소폭 저하            |
| 끝부분   | 높음             | 높음                 |


![result](https://github.com/user-attachments/assets/68954152-f35b-4321-bf6d-601ff5f19404)
> 🔍 GPT-4도 일부 실험에서 유사한 성능 패턴을 보였으나,  
> 전체 실험에는 포함되지 않았으며 Appendix D에 제한적으로 보고됨.

---

### 📌 이 현상의 원인 (논문 §2.3, §3.2)

- 🔒 **Absolute positional encoding**의 구조적 한계
- 🔄 **Self-attention의 위치 편향(position bias)**  
  → **앞(primacy)**과 **뒤(recency)**에 주의를 집중, **중간은 희석**
- 📏 **문서 길이가 길수록 성능 하락 폭이 더 커짐**  
  → GPT-3.5 기준 30-document 설정에서 20% 이상 성능 하락

---

### 🧠 왜 중요한가?

> 대부분의 **RAG (Retrieval-Augmented Generation)**, multi-document QA, long-context summarization 시스템은  
> **긴 문맥**을 활용합니다. 그런데 **중간 정보를 모델이 무시한다면?**

- 📉 검색 결과의 **위치**가 QA 성능에 직접 영향
- 🔀 **중요 문서나 핵심 정보는 앞쪽에 배치해야 효과적**
- ❗ 단순히 context window를 늘리는 것만으로는 **문제 해결 ❌**

---

### 💡 시사점: 위치 편향을 고려한 활용 전략

> "**LLM은 context를 기억한다. 하지만, 그건 시작과 끝일 뿐...**"

**위치 편향을 줄이기 위한 전략:**

- ✅ **Query-aware contextualization**  
  → 디코더-온리 모델에서는 질문을 문서 앞에 먼저 제시
- ✅ **Chunk ordering optimization**  
  → 중요한 정보를 앞쪽에, 덜 중요한 건 뒤로 재배치
- ✅ **Attention 구조 개선**  
  → 양방향 인코더가 있는 모델 (T5, BART 등)이 더 유리
- ✅ **Position-free architectures**  
  → Hyena, RWKV 등 새로운 구조는 위치 독립성을 추구함

---

### 🔍 Retrieval 기반 실험 구성 (논문 §2.2)

- **Task**: Multi-document QA
- **Retriever**: Contriever (MS-MARCO fine-tuned)
- **Reader 입력**: 검색된 k개의 문서 + 질문
- **문서 수(k)**: 10, 20, 30개
- **문서 형태**: paragraph 기반 (표, 목록은 제외)

---

### 📈 Retrieval 수 증가 vs 성능 변화

![retriever](https://github.com/user-attachments/assets/e7a81572-0558-4c0d-93cd-fb49d731898c)

- ✅ **k=10, 20 → 성능 향상**
- ⚠️ **k=30 → 성능 하락 또는 포화**
  - 정답 문서가 **중간에 위치**할 경우 정확도 급락
  - 일부 모델은 **closed-book 성능보다도 낮아짐**

---

### ❗ Retrieval은 잘 되어도, 활용은 어려움

- LLM은 **정답 문서를 받아도**,  
  그 정보가 **중간에 있으면 잘 사용하지 못함**

> Retrieval만 잘해도 성능이 보장되지 않음!  
> → **LLM의 위치 편향을 고려한 prompt 구조 설계 필수**

**해결 전략 예시:**

- ✅ 정답 문서를 프롬프트 앞에 배치
- ✅ Query-aware 구조 사용
- ✅ Noise 문서 수를 줄이기 (문서 선택 압축)

---

### ✅ 한 줄 요약

> **“LLM은 긴 context를 기억하지만 context 내의 중간부분은 잘 망각한다!!!**
