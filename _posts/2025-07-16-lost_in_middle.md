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

### 📚 (한국어) 긴 문맥에서 LLM은 얼마나 잘 기억할까?

_📎 요약: LLM이 긴 문서에서 중간에 있는 정보를 잘 기억하지 못한다는 놀라운 실험 결과!_  
> LLM은 긴 문서를 처리할 수 있지만,  
> **문서 중간의 정보는 ‘잃어버리는’ 경향이 있다**는 사실, 알고 계셨나요?  
> 바로 이 문제를 분석한 논문이  
> **"Lost in the Middle: How Language Models Use Long Contexts"**입니다.

> 🔗 논문 원문: [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)

---

### 🧠 논문이 제기한 핵심 질문

> “긴 context 안에서, 모델은 **모든 위치의 정보를 균등하게 사용할 수 있을까?”  

**결론부터 말하면: No.**  
- 대부분의 LLM은 긴 문서의 **중간 정보**를 가장 잘 놓칩니다!
- 이는 context window가 아무리 길어도 **위치 편향(position bias)**이 존재한다는 뜻입니다.

---

### 🔍 실험 방법: Needle-in-a-Haystack

**실험 구성:**
- 긴 문서에 단 하나의 핵심 정보 ("needle")를 넣음
- 이 정보를 문서의 **시작**, **중간**, **끝**에 위치시켜 각각 테스트
- 모델에게 해당 정보를 **추출하라고 질문**함

```text
예시:
문서 길이 = 8,000 tokens
[... 긴 텍스트 ...]
→ 중간에 "정답 문장" 삽입
→ 모델에게: "위 문서에 나온 숫자는 몇이었지?" 질문
```


### 📉 주요 결과 요약

문서 **중간에 정보가 있을 경우**, 모델 정확도가 **급락**함!  
**GPT-4조차도** 중간 정보의 회상률이 매우 낮았음

| 위치     | 회상률 (GPT-3.5) | 회상률 (GPT-4) |
|----------|------------------|----------------|
| 앞부분   | 높음             | 매우 높음      |
| 중간 ⚠️ | 낮음             | 중간 이하      |
| 끝부분   | 높음             | 높음           |

👉 **LLM은 문서의 중간을 가장 취약하게 처리합니다!**

---

### 📌 이 현상의 원인

- **포지셔널 인코딩(absolute positional encoding)**의 한계
- **Self-attention 메커니즘의 bias**  
  → 앞과 뒤에 더 집중하고, **중간은 덜 중요하게 처리**
- 문서가 길어질수록 **중간 손실(middle drop)** 현상은 더 심해짐

---

### 🛠️ 왜 중요한가?

> 대부분의 **RAG (Retrieval-Augmented Generation)**, long QA, 문서 요약 시스템은 **긴 context**를 활용합니다.  
> 그런데 모델이 중간 정보를 무시한다면...?

- 🔎 검색 결과의 **위치가 성능에 직접적 영향**
- 📄 문서 **chunking 전략** 설계 시 **중간 정보 보완** 필요
- 🧠 단순히 context window를 늘리는 것만으로는 **불충분**

---

### 💡 시사점 & 다음 단계

> LLM을 사용할 때는 "**길이가 되니까 다 기억하겠지?**"는 **금물!**

**위치 편향을 줄이기 위한 방향:**

- ✅ **Chunk overlap**
- ✅ **Sliding window attention**
- ✅ **RAG 정렬/재배치 전략**
- ✅ **Position-free architectures** (e.g., *Hyena*, *RWKV*)

---

### 🔖 논문 정보 정리

- **논문 제목**: *Lost in the Middle: How Language Models Use Long Contexts*  
- **저자**: Stanford, Allen AI, Meta AI 등 공동 연구  
- **링크**: [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)  
- **발표일**: 2023년 7월  

---

### ✅ 한 줄 요약

> **“LLM은 context를 기억한다. 하지만, 그건 시작과 끝일 뿐... 중간은 길을 잃는다.”**
