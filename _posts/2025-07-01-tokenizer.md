---
layout: post
title: "🔤Understanding Tokenizers - Tokenizer 알아보기?!!"
author: [DrFirst]
date: 2025-07-01 07:00:00 +0900
categories: [AI, Experiment]
tags: [Tokenizer, NLP, BPE, WordPiece, SentencePiece, Subword]
sitemap :
  changefreq : monthly
  priority : 0.8
---

---

### 🧠 (한국어) Tokenizer 알아보기?!!  
_🔍 텍스트를 AI가 이해할 수 있는 의미 있는 단위로 나누기!!!_

> 우리가 문장을 단어로 나누어 이해하듯이,  
> AI 모델도 **토크나이저**를 통해 텍스트를 처리 가능한 단위로 변환해야 합니다!

> 주요 논문들: 
> - [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE - 2016)
> - [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) (SentencePiece - 2018)

---

### 💡 Tokenizer의 특징 요약!!

1. **텍스트와 AI 사이의 다리 역할**  
   - 인간이 읽는 텍스트를 모델이 처리할 수 있는 숫자 토큰으로 변환
2. **어휘 문제 해결**  
   - 서브워드 토크나이제이션으로 미등록 단어(OOV) 문제 해결
3. **언어 무관성**  
   - 현대 토크나이저는 다양한 언어와 문자 체계에서 작동

---

### 🧠 Tokenizer 등장의 배경

> 사람은 단어를 읽지만, 컴퓨터는 숫자가 필요해!  
> 토크나이저는 텍스트를 AI가 소화할 수 있는 형태로 변환하는 필수 다리입니다.

**텍스트 처리의 진화:**
- **1990년대**: 단어 수준 토크나이제이션 → 간단하지만 거대한 어휘
- **2000년대**: 문자 수준 → 작은 어휘이지만 의미 상실
- **2010년대**: Sub-word 토크나이제이션 → **두 세계의 장점을 융합!**

#### 🚨 **기존 방법들의 한계점**

#### **1️⃣ 단어 수준 토크나이제이션의 문제** 📚
- **어휘 폭발**: "달리다", "달리고", "달렸다" → 각각 다른 토큰
- **OOV 문제**: 학습 어휘에 없는 새로운 단어 = `<UNK>`
- **메모리 집약적**: 어휘가 10만 개 이상 도달 가능
- **언어 의존성**: 각 언어별로 다른 규칙 필요

**예시:**
```
입력: "달리는 사람이 빠르게 달린다"
단어 토큰: ["달리는", "사람이", "빠르게", "달린다"]
문제: "달려가는" → `<UNK>` (어휘에 없으면)
```

#### **2️⃣ 문자 수준의 한계** 🔤
- **의미 상실**: "고양이" → ["고", "양", "이"] 단어 의미 손실
- **긴 시퀀스**: 더 긴 시퀀스 = 더 많은 연산
- **맥락 어려움**: 모델이 단어 수준 패턴 학습하기 어려움

**예시:**
```
입력: "안녕 세상"
문자 토큰: ["안", "녕", " ", "세", "상"]
문제: 2개 단어에 5개 토큰!
```

> 💡 **해결책**: **Sub-word 토크나이제이션**이 어휘 크기와 의미적 의미의 균형을 맞춥니다!

---

### 🔧 현대 토크나이제이션 알고리즘

#### **🏗️ 1. BPE (Byte Pair Encoding)** 🧩

**핵심 아이디어**: 문자에서 시작해서 가장 빈번한 쌍을 반복적으로 병합

**알고리즘 단계:**
1. **초기화**: 텍스트를 문자로 분할 (문자수준 토크나이제이션)
2. **쌍 계수**: 가장 빈번한 인접 문자 쌍 찾기
3. **병합**: 가장 빈번한 쌍을 새 토큰으로 교체
4. **반복**: 원하는 어휘 크기까지

**예시 과정:**
```python
# 간단한 BPE(Byte Pair Encoding) 미니 실습 코드
from collections import Counter, defaultdict
import pandas as pd

# 샘플 단어 리스트
corpus = ["low", "lower", "newest", "widest"]

# Step 1: 각 단어를 문자 단위로 분해하고 끝에 특수문자 추가
def split_word(word):
    return list(word) + ["</w>"]  # 단어의 끝 표시

# 초기 vocabulary
tokens = [split_word(word) for word in corpus]

# helper 함수: 토큰 리스트를 문자열로 합침 (공백 단위로)
def get_vocab(tokens):
    vocab = Counter()
    for token in tokens:
        vocab[" ".join(token)] += 1
    return vocab

# 가장 자주 등장하는 pair 찾기
def get_most_common_pair(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq
    return max(pairs, key=pairs.get), pairs

# 병합 수행
def merge_pair(pair, tokens):
    new_tokens = []
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for token in tokens:
        token_str = " ".join(token)
        token_str = token_str.replace(bigram, replacement)
        new_tokens.append(token_str.split())
    return new_tokens

# BPE 학습 수행 (5회만 반복)
merge_history = []
for i in range(5):
    vocab = get_vocab(tokens)
    pair, all_pairs = get_most_common_pair(vocab)
    merge_history.append((pair, all_pairs[pair]))
    tokens = merge_pair(pair, tokens)

# 결과 정리
merge_df = pd.DataFrame(merge_history, columns=["Merged Pair", "Frequency"])
tokens_final = [" ".join(token) for token in tokens]
print("Final tokens:", tokens_final)
# Output: ['low </w>', 'lower </w>', 'newest </w>', 'widest </w>']

# 단계별 병합 과정 보기
print("\n단계별 병합 과정:")
for i, (pair, freq) in enumerate(merge_history):
    print(f"단계 {i+1}: {pair[0]} + {pair[1]} → {pair[0]+pair[1]} (빈도: {freq})")
# Output:
# 단계 1: e + s → es (빈도: 2)
# 단계 2: es + t → est (빈도: 2)  
# 단계 3: l + o → lo (빈도: 2)
# 단계 4: lo + w → low (빈도: 2)
# 단계 5: n + e → ne (빈도: 1)
```

**BPE 장점:**
- ✅ **희귀 단어 처리** 서브워드 분해를 통해
- ✅ **일관된 토크나이제이션** 유사한 단어들에 대해
- ✅ **언어 무관** 알고리즘

#### **🏗️ 2. WordPiece** 🧩

**핵심 아이디어**: BPE와 유사하지만 **우도 최대화** 사용

**BPE와의 주요 차이점:**
- 서브워드 연속에 `##` 접두사 사용
- 빈도가 아닌 **우도 증가**를 기반으로 병합
- **BERT**, **DistilBERT**에서 사용

**예시:**
```python
입력: "불행한"

WordPiece 토큰: ["불", "##행한"]
# "##"는 이전 토큰의 연속임을 나타냄
```

#### **🏗️ 3. SentencePiece** 🌐

**핵심 아이디어**: 사전 토크나이제이션 없는 **언어 독립적** 토크나이제이션

**주요 장점:**
- ✅ **사전 토크나이제이션 불필요** (단어 경계 없음)
- ✅ **모든 언어 처리** 중국어, 일본어, 아랍어 포함
- ✅ **가역적**: 원본 텍스트 완벽 재구성 가능
- ✅ **T5, mT5, ALBERT**에서 사용

**SentencePiece 특징:**
```python
# 공백을 일반 문자로 처리
입력: "안녕 세상"
토큰: ["▁안녕", "▁세상"]  # ▁는 공백을 나타냄

# 공백이 없는 언어 처리
입력: "こんにちは世界"  # 일본어: "안녕 세상"
토큰: ["こんに", "ちは", "世界"]
```

---

### 📊 **토크나이저 비교**

| 방법 | 어휘 크기 | OOV 처리 | 언어 지원 | 사용 모델 |
|------|-----------|----------|-----------|-----------|
| **단어 수준** | 5만-10만+ | ❌ 나쁨 | 🔤 언어별 특화 | 전통적 NLP |
| **문자 수준** | 100-1천 | ✅ 완벽 | 🌍 범용 | 초기 NMT |
| **BPE** | 3만-5만 | ✅ 좋음 | 🌍 범용 | GPT, RoBERTa |
| **WordPiece** | 3만-5만 | ✅ 좋음 | 🌍 범용 | BERT, DistilBERT |
| **SentencePiece** | 3만-5만 | ✅ 훌륭함 | 🌍 범용 | T5, mT5, ALBERT |

---

### 💻 **실제 구현**

#### **🔧 Hugging Face Tokenizers 사용**

```python
from transformers import AutoTokenizer

# BPE (GPT-2)
bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "The runner runs quickly and efficiently"
bpe_tokens = bpe_tokenizer.tokenize(text)
print(f"BPE: {bpe_tokens}")
# Output: ['The', 'Ġrunner', 'Ġruns', 'Ġquickly', 'Ġand', 'Ġefficiently']

# WordPiece (BERT)
wp_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
wp_tokens = wp_tokenizer.tokenize(text)
print(f"WordPiece: {wp_tokens}")
# Output: ['the', 'runner', 'runs', 'quickly', 'and', 'efficiently']

# SentencePiece (T5)
sp_tokenizer = AutoTokenizer.from_pretrained("t5-small")
sp_tokens = sp_tokenizer.tokenize(text)
print(f"SentencePiece: {sp_tokens}")
# Output: ['▁The', '▁runner', '▁runs', '▁quickly', '▁and', '▁efficiently']
```

#### **🔧 커스텀 BPE 토크나이저 훈련**

- 아래 방식은 초기화된 tokenizer를 기반으로 하는것!!  
- 제시된 `english_texts` 를 바탕으로 토크나이져 만듬!!  
- 최대 토큰수는 5000개, 2번 이상 등장해야 함!  


```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# English text data examples
english_texts = [
    "Hello world, how are you today?",
    "The weather is beautiful and sunny", 
    "We are building a custom tokenizer for English",
    "Natural language processing is a fascinating field",
    "Deep learning and machine learning are powerful technologies"
]

# Save texts to file
with open("english_corpus.txt", "w", encoding="utf-8") as f:
    for text in english_texts:
        f.write(text + "\n")

# Initialize BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Configure trainer for English
trainer = BpeTrainer(
    vocab_size=5000,  # Larger vocab for English
    min_frequency=2,  # Minimum frequency threshold
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    show_progress=True
)

# Train on English text
tokenizer.train(["english_corpus.txt"], trainer)

# Test the tokenizer
test_text = "Natural language processing is fascinating!"
tokens = tokenizer.encode(test_text)
print(f"Input: {test_text}")
print(f"Token IDs: {tokens.ids}")
print(f"Tokens: {tokens.tokens}")

# Save tokenizer
tokenizer.save("my_english_bpe_tokenizer.json")
print("English BPE tokenizer saved successfully!")
```

프린트되는 결과값은!?  
> 느낌표는 처음보는것이기에 `[UNK]` 가 댄다~!  

```text
Input: Natural language processing is fascinating
Token IDs: [10, 39, 31, 28, 13, 23, 23, 38, 19, 31, 13, 19, 17, 27, 28, 26, 15, 46, 29, 37, 41, 18, 13, 29, 15, 35, 39, 37]
Tokens: ['N', 'at', 'u', 'r', 'a', 'l', 'l', 'an', 'g', 'u', 'a', 'g', 'e', 'p', 'r', 'o', 'c', 'es', 's', 'ing', 'is', 'f', 'a', 's', 'c', 'in', 'at', 'ing', 'w', 'e', 'at', 'h', 'er', '[UNK]']
English BPE tokenizer saved successfully!
```
---

### 🧩 **고급 토크나이제이션 개념**

#### **1️⃣ Special Tokens** 🎯
```python

# Example usage with BERT tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "The quick brown foxㅁ"

encoded = tokenizer(text, 
                   add_special_tokens=True,
                   max_length=10,
                   padding="max_length",
                   truncation=True)

print(f"Input: {text}")
print(f"Token IDs: {encoded['input_ids']}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")

# Expected output:
# Input: The quick brown fox
# Token IDs: [101, 1996, 4248, 2829, 4419, 102, 0, 0, 0, 0]
# Tokens: ['[CLS]', 'the', 'quick', 'brown', '[UNK]', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
```

- 결과 설명!!  
> max_length = 10 이었기에, 길이를 맞추고자 PAD가 3개 나옴!
> `ㅁ` 는 한글로 모르기에  `[UNK]`

| 토큰       | 설명                                                |
|------------|-----------------------------------------------------|
| `[PAD]`    | 배치 처리를 위한 패딩 토큰 (시퀀스 길이 맞춤)       |
| `[UNK]`    | 미지의 단어(어휘집에 없는 OOV 단어)를 위한 토큰     |
| `[CLS]`    | 문장 분류용 시작 토큰 (BERT에서 사용)               |
| `[SEP]`    | 문장 구분 토큰 (문장 간 구분 또는 문장 끝 표시)     |
| `[MASK]`   | 마스킹된 토큰 (Masked Language Modeling용)         |
| `<s>`      | 시퀀스 시작 토큰 (GPT 등에서 사용)                  |
| `</s>`     | 시퀀스 종료 토큰 (GPT 등에서 사용)                  |


#### **2️⃣ Tokenization vs Encoding** 🔄
```python
from transformers import AutoTokenizer

# 사전 학습된 BERT tokenizer 불러오기
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 입력 문장
text = "Hello, world!"

# 1. Tokenization: 문장을 토큰 리스트로 변환
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# 예시 출력: ['hello', ',', 'world', '!']

# 2. Encoding: 문장을 토큰 ID 리스트로 변환
token_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"Token IDs: {token_ids}")
# 예시 출력: [7592, 1010, 2088, 999]

# 3. Full encoding with special tokens
encoded = tokenizer(
    text,
    add_special_tokens=True,   # [CLS], [SEP] 추가
    padding=True,              # padding 추가 (단일 문장에도 가능)
    truncation=True,           # 너무 긴 문장은 자르기
    return_tensors="pt"        # PyTorch tensor로 반환
)

# 출력
print("Full Encoding (with special tokens, padding):")
print(encoded)
print("Input IDs:", encoded['input_ids'])
print("Attention Mask:", encoded['attention_mask'])
```

결과물은!?  
> Full encoding에서는 [CLS] [SEP] 등이 추가된거임!

```text
Tokens 으로 쪼갠거! : ['hello', ',', 'world', '!']
Token IDs: [7592, 1010, 2088, 999]
Full Encoding (with special tokens, padding):
{'input_ids': tensor([[ 101, 7592, 1010, 2088,  999,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
Input IDs: tensor([[ 101, 7592, 1010, 2088,  999,  102]])
Attention Mask: tensor([[1, 1, 1, 1, 1, 1]])
```

#### **3️⃣ 여러 국가의 언어처리!** 🌍
```python
from transformers import AutoTokenizer

# Multilingual examples
texts = [
    "Hello world",           # English
    "Bonjour le monde",      # French
    "Hola mundo",           # Spanish  
    "Guten Tag Welt",       # German
    "Ciao mondo"            # Italian
]

# Different tokenizers handle multilingual text differently
multilingual_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
english_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("=== Multilingual Tokenizer ===")
for text in texts:
    tokens = multilingual_tokenizer.tokenize(text)
    print(f"{text} → {tokens}")

print("\n=== English-only Tokenizer ===")
for text in texts:  # Only English and French
    tokens = english_tokenizer.tokenize(text)
    print(f"{text} → {tokens}")

```

결과물은!?  
> 단순 영어 토크나이저랑 다국어 학습된거랑 다르지요!?

```text
=== Multilingual Tokenizer ===
Hello world → ['Hello', 'world']
Bonjour le monde → ['Bon', '##jou', '##r', 'le', 'monde']
Hola mundo → ['Ho', '##la', 'mundo']
Guten Tag Welt → ['Gut', '##en', 'Tag', 'Welt']
Ciao mondo → ['Ci', '##ao', 'mondo']

=== English-only Tokenizer ===
Hello world → ['hello', 'world']
Bonjour le monde → ['bon', '##jou', '##r', 'le', 'monde']
Hola mundo → ['ho', '##la', 'mundo']
Guten Tag Welt → ['gut', '##en', 'tag', 'we', '##lt']
Ciao mondo → ['cia', '##o', 'mon', '##do']
```
---

### 📈 **성능 영향 분석**

#### **🏆 어휘 크기 영향**

| 어휘 크기 | 장점 | 단점 | 최적 용도 |
|------------|------|------|----------|
| **소형 (1K-5K)** | 💾 Memory efficient<br/>⚡ Fast training | 🔄 Many subwords<br/>📏 Long sequences | Resource-constrained |
| **중형 (10K-30K)** | ⚖️ Balanced performance<br/>✅ Good coverage | 📊 Standard choice | Most applications |
| **대형 (50K+)** | 🎯 Better semantic units<br/>📏 Shorter sequences | 💾 Memory intensive<br/>⏱️ Slower training | Large-scale models |

#### **🔧 Tokenization 속도비교!**

```python
import time
from transformers import AutoTokenizer

text = "This is a sample text " * 1000  # Long text

tokenizers = {
    "BPE (GPT-2)": AutoTokenizer.from_pretrained("gpt2"),
    "WordPiece (BERT)": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "SentencePiece (T5)": AutoTokenizer.from_pretrained("t5-small")
}

for name, tokenizer in tokenizers.items():
    start_time = time.time()
    tokens = tokenizer.tokenize(text)
    end_time = time.time()
    
    print(f"{name}:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Time: {end_time - start_time:.4f}s")
    print(f"  Speed: {len(tokens)/(end_time - start_time):.0f} tokens/s")
```

결과물은!?  
> BPE가 제일 빠르고, T5가 토큰수가 젤 많아유!

| Tokenizer Type     | Tokens | Time (초) | Speed (tokens/s) |
|--------------------|--------|-----------|------------------|
| **BPE (GPT-2)**    | 5001   | 0.0125    | 401,257          |
| **WordPiece (BERT)** | 5000 | 0.0142    | 352,380          |
| **SentencePiece (T5)** | 6000 | 0.0145  | 413,069          |



---

### ⚠️ **한계점 & 도전과제**

#### **1️⃣ 일관되지 않은 Tokenization** 🔄
```python
from transformers import AutoTokenizer

# Same words in different contexts
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text1 = "run"
text2 = "running" 
text3 = "I like to run fast"
text4 = "The running water is clean"

print(f"'{text1}' → {tokenizer.tokenize(text1)}")
print(f"'{text2}' → {tokenizer.tokenize(text2)}")
print(f"'{text3}' → {tokenizer.tokenize(text3)}")
print(f"'{text4}' → {tokenizer.tokenize(text4)}")

# Output examples:
# 'run' → ['run']
# 'running' → ['running'] 
# 'I like to run fast' → ['I', 'Ġlike', 'Ġto', 'Ġrun', 'Ġfast']
# 'The running water is clean' → ['The', 'Ġrunning', 'Ġwater', 'Ġis', 'Ġclean']
```

- 위의 결과를 보면 run 에대하여 여러방식으로 Tokenization 된다!!

#### **2️⃣ 서브워드 경계 문제** ✂️
```python
# Examples of problematic subword splitting
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

problematic_words = [
    "unhappiness",
    "preprocessor", 
    "antidisestablishmentarianism",
    "biocompatibility"
]

for word in problematic_words:
    tokens = tokenizer.tokenize(word)
    print(f"'{word}' → {tokens}")

# Output examples:
# 'unhappiness' → ['un', '##hap', '##piness']  # Loses "happy" semantic unit
# 'preprocessor' → ['pre', '##proc', '##ess', '##or']  # Splits "process"
# 'antidisestablishmentarianism' → ['anti', '##dis', '##esta', '##blish', '##ment', '##arian', '##ism']
# 'biocompatibility' → ['bio', '##com', '##pat', '##ibility']  # Loses "compatibility"
```

**문제점**: 의미있는 단위가 잘못 분할됨 (pre / processor로 구분되는게 제일 좋겠지만!?..)  
**영향**: 모델이 단어 관계와 형태학을 이해하는데 어려움을 겪을 수 있음

#### **3️⃣ 도메인에 대한 지식 부족으로 한계!** 🏥
```python
from transformers import AutoTokenizer

# Medical text examples
medical_texts = [
    "Patient presents with acute myocardial infarction and requires immediate intervention",
    "Blood pressure elevated, prescribing ACE inhibitors for hypertension management",
    "CT scan reveals suspicious pulmonary nodules, scheduling biopsy procedure"
]

# General vs Medical tokenizer comparison
general_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# medical_tokenizer = AutoTokenizer.from_pretrained("clinical-bert")  # Hypothetical

for text in medical_texts[:1]:  # First example only
    general_tokens = general_tokenizer.tokenize(text)
    print(f"General Tokenizer:")
    print(f"  Input: {text}")
    print(f"  Tokens: {general_tokens}")
    print(f"  Token count: {len(general_tokens)}")
    
    # Medical tokenizer would handle medical terms as single tokens
    print(f"\nMedical Tokenizer (Expected):")
    print(f"  Tokens: ['patient', 'presents', 'with', 'acute_myocardial_infarction', 'and', 'requires', 'immediate', 'intervention']")
    print(f"  Token count: 8 (significant reduction)")
    
# General: ['patient', 'presents', 'with', 'acute', 'my', '##oc', '##ard', '##ial', 'in', '##far', '##ction', 'and', 'requires', 'immediate', 'intervention']
# Medical: ['patient', 'presents', 'with', 'acute_myocardial_infarction', 'and', 'requires', 'immediate', 'intervention']
```

- 의학용어를 잘 쪼개지 못한다!!  

#### **4️⃣ 언어별 특화하는것에 대한 문제** 🌍  
- **형태학적 복잡성**: 풍부한 형태학을 가진 언어들 (독일어, 터키어, 핀란드어)
- **교착어적 특성**: 형태소 결합으로 단어가 형성되는 언어들 (일본어, 한국어, 헝가리어)
- **문자 체계 혼용**: 한 텍스트에서 여러 문자 체계 사용 (일본어: 히라가나 + 가타카나 + 한자)
- **복합어**: 독일어 "Donaudampfschifffahrtsgesellschaftskapitän" (다뉴브 증기선 회사 선장)

---

### 🔮 **미래 방향**

#### **1️⃣ 신경망 기반 토크나이제이션** 🧠
- **개념**: 엔드투엔드 학습 가능한 토크나이제이션
- **장점**: 태스크별 최적화
- **도전과제**: 계산 복잡성

#### **2️⃣ 멀티모달 토크나이제이션** 🖼️
```python
# Future concept: unified text-image tokenization
multimodal_input = {
    "text": "A cat sitting on a chair",
    "image": cat_image_tensor
}

unified_tokens = multimodal_tokenizer.tokenize(multimodal_input)
# Outputs both text and visual tokens in same space

# Example output:
# [
#   ("A", "text_token"),
#   ("cat", "text_token"), 
#   ("<img_patch_1>", "visual_token"),
#   ("sitting", "text_token"),
#   ("<img_patch_2>", "visual_token"),
#   ("on", "text_token"),
#   ("chair", "text_token")
# ]
```

- 이런방식으로해서 구글이 Multi-modal large Model 인 `Gemini` 를 공개했지유!!  
- 이미지를 토큰화하는법!? 우선 [ViT 에 대하여 공부](https://drfirstlee.github.io/posts/ViT/)하면 알 수 있어요!  

### 🧩 실세 사용되고있는 주요 모델별 Tokenizer 요약   

| 모델                     | Tokenizer 종류              | 특징                                                                 |
|--------------------------|-----------------------------|----------------------------------------------------------------------|
| GPT-2 / GPT-3 / GPT-4    | BPE (OpenAI GPT Tokenizer)  | `tiktoken` 사용, 영어에 최적화됨                                     |
| LLaMA / LLaMA2 / LLaMA3  | SentencePiece + BPE         | Meta가 직접 학습한 SentencePiece 기반 구조 사용                      |
| Gemini (Google)          | SentencePiece 기반 추정     | PaLM/Flan 계열과 유사한 구조, 세부 토크나이저 미공개                |
| Claude (Anthropic)       | BPE 변형                    | 세부 구조는 비공개, 자체 토크나이저 구조 사용                        |
| Qwen (Alibaba)           | GPT-style BPE               | 중국어 최적화, 영어도 지원, **Tokenizer 공개됨**                    |
| Mistral / Mixtral        | SentencePiece               | open-source 모델, HuggingFace tokenizer 구조 따름                    |
| **Qwen-VL (멀티모달)**   | GPT-style BPE + Vision 특화 | 텍스트는 Qwen과 동일, 이미지 입력은 CLIP-style 패치 분할 사용        |
| **Gemini (멀티모달)**    | SentencePiece + Vision      | 정확한 구조 미공개, Flamingo-like 구조로 추정                        |
| **Grok (xAI)**           | 비공개                      | 모델 및 토크나이저 구조 대부분 비공개, 영어 기반 추정                |
