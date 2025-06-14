---
layout: post
title: "🖥️ LORA Practice!! : LORA 실습!! with python"
author: [DrFirst]
date: 2025-06-13 09:00:00 +0900
categories: [AI, Experiment]
tags: [LORA, fine-tuning, ICLR, ICLR 2022, Low-Rank Adaptation, Parameter Efficiency, Python, PEFT]
sitemap :
  changefreq : weekly
  priority : 0.9
---


---

## 🦖(한국어) LORA 실습!!
> **LORA** : Low-Rank Adaptation of Large Language Models

오늘은 [지난포스팅](https://drfirstlee.github.io/posts/LORA/)에서 이론에 대하여 공부해보았던!  
Fine-Tuning의 정수! `LORA` 의 실습을 진행해보겠습니다!!  

---

### 🧱 1. LORA 패키지 설치 및 모델 로드!  

LORA의 경우는 워낙 유명한 방법이어서,  
git repo 복제가 아니라! Hugging Face의 패키지로서 제공됩니다!

이에, 아래와 같이 pip 로 필요 패키지를 설치 후 로드 해주세요~!    

```bash
pip install transformers peft datasets accelerate
```

> PEFT란?   
> Parameter-Efficient Fine-Tuning 의 약어로서,  
> 모델의 모든 파라미터를 업데이트하는 대신 일부 파라미터만 효율적으로 조정하는 기술들의 총칭입니다.  
>> `from peft import PrefixTuningConfig, PromptTuningConfig, AdaLoraConfig, IA3Config` 를 통해 LORA외의 방법도 적용이 가능합니다!! 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType # peft!!!
from datasets import load_dataset # 기본 제공된 학습데이터로 해보기위해서! 
from datasets import Dataset  # 이후 학습 직접 입력기 위해서는 이게 필요!

import torch
import os
os.environ["WANDB_DISABLED"] = "true" # 이부분이 추가 안되면 wandb의 에러가 나요!

```

그리고 오늘의 파인 튜닝은 현시점에서 최신이면서 가벼운!  
Qwen2.5의 0.5B 모델 `Qwen/Qwen2.5-0.5B` 를 바탕으로 Fine Tuning을 진행하고자합니다!  

이에, 아래와 같이 모델을 로드해줍니다!  
Hugging Face에서 다운받기에, 처음에는 시간이 좀 걸릴거에요~!


```python
# 모델 및 토크나이저 불러오기
model_name = "Qwen/Qwen2.5-0.5B"  
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # padding 토큰 설정

# 모델 불러오기 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # bitsandbytes를 통한 4bit quantization (선택사항)
    device_map="auto",
    trust_remote_code=True
)
```

---

### 📦 2. LORA 세팅!!  

이제 사용할 LORA 모델의 parameter를 설정해주고!,  
기존 qwen2.5 모델에 합쳐해 줄 것 입니다!!  

```python
# LoRA 설정
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                 # 랭크
    lora_alpha=16,       # scaling factor
    lora_dropout=0.05,   # dropout
    bias="none",
    target_modules=["c_proj", "c_attn", "q_proj", "v_proj", "o_proj", "k_proj"],  # Qwen 구조에 맞게 조정
)

# LoRA 와 기존 Qwen 모델와 합치기!! 적용
model = get_peft_model(model, peft_config)
```

위에서 사용되는 LORA Parameter에 대하여 알아보자면~~  

🔹 `task_type=TaskType.CAUSAL_LM`  
**설명**: LoRA를 적용할 태스크 유형 설정  
**CAUSAL_LM**: 다음 토큰을 예측하는 언어 생성 모델  
- Qwen도 GPT와 동일한 구조의 **Causal Language Model**이기 때문에 이 태스크 타입을 사용합니다.  
- 참고로, `TaskType.CLASSIFICATION`은 분류 모델에,  
  `TaskType.SEQ_2_SEQ_LM`은 번역, 요약 등 입력-출력 쌍을 가지는 모델에,  
  `TaskType.VISION`은 비전 모델 (예: CLIP, ViT) 등에 사용됩니다.

---

🔹 `r=8`  
**설명**: LoRA에서 low-rank 분해 시의 랭크(rank)  
**영향**: 작을수록 파라미터 수가 적고 계산이 효율적이나,  
너무 작으면 표현력이 떨어져 학습 성능이 저하될 수 있습니다.  
- 일반적으로 4~16 사이 값이 실험적으로 자주 사용됩니다.

---

🔹 `lora_alpha=16`  
**설명**: LoRA 가중치에 적용할 scaling factor  
**계산**: 실제 LoRA 가중치는 `alpha / r`을 통해 조정됨  
**영향**:  
- 값이 **클수록** LoRA의 영향력이 커지고,  
- 값이 **작을수록** 원래 모델의 표현에 가까운 보수적 튜닝이 됩니다.  
- 적절한 값은 데이터와 태스크에 따라 조정해야 하며, 일반적으로 `alpha = 2 * r` 수준이 추천됩니다.

---

🔹 `lora_dropout=0.05`  
**설명**: 학습 시 LoRA layer에 적용되는 dropout 비율  
**목적**: 과적합을 방지하고, 일반화 성능을 향상시키기 위한 regularization 기법  
- 특히 데이터셋이 작거나 noisy할 때 유효합니다.

---

🔹 `bias="none"`  
**설명**: 기존 모델의 bias 파라미터 처리 방식  
- `"none"`: 기존 bias 파라미터는 유지하며 학습이나 수정하지 않음  
- `"all"`: 모든 bias 파라미터를 학습 대상으로 포함  
- `"lora_only"`: LoRA가 적용된 모듈의 bias만 학습 대상으로 포함  
  → 일반적으로 `none`을 선택해 파라미터 수를 최대한 줄이는 것이 LoRA의 목적과 맞습니다.

---

🔹 `target_modules=[...]`  
**설명**: LoRA를 적용할 대상 모듈 목록  
**Qwen 모델 기준 주요 모듈**:
- `q_proj`, `k_proj`, `v_proj`: 쿼리(Query), 키(Key), 값(Value) 벡터 생성  
- `o_proj`: attention 출력 벡터 생성  
- `c_proj`, `c_attn`: attention 연산과 전체 출력 projection 담당

＋ **참고 사항**:  
- LORA paper에 따르면, **`Wq`와 `Wv`에만 LoRA를 적용하는 것이 가장 효율적**이라고 되어있습니다    
- 이에, ["q_proj", "v_proj"]만 사용하는 것이 일반적으로 더 효율적입니다.




### 🧊 3. Fine Tuning 하기!!

미세 조정(Fine tuning) 은 샘플 데이터로 한번,  
그리고 직접 만든데이터로 한번 진행해 보겠습니다!!!  

#### 샘플 데이터로 FT 하기!!

- `"tatsu-lab/alpaca", split="train[:100]"` 의 방법으로 샘플데이터를 로드하고 사용합니다!  

```python

# 샘플 데이터셋 (Alpaca-style)
dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")  # 일부만 사용
def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./qwen_lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
)

# 트레이너 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 학습 시작
trainer.train()
model.save_pretrained("qwen_lora_output")
```

위의 방법으로 학습이 되면 아래와 같이 로그가 나타나고,  

| Step | Training Loss |
|------|----------------|
|  10  |    2.292300    |

하위 디렉토리 `qwen_lora_output` 에 모델 웨이트 값이 저장됩니다!  


이제 아래와 같이 모델을 불러온 뒤, 질의를 해보면?
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model_name =  "Qwen/Qwen2.5-0.5B"  # 사용한 베이스 모델
peft_path = "qwen_lora_output"  # trainer에서 지정한 결과 디렉토리

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
model = PeftModel.from_pretrained(model, peft_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

prompt = "### Instruction:\n대한민국 수도는?\n### Input:\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

아래와 같이 잘 답변하는것을 볼수 있습니다~~

```text
### Instruction:
대한민국 수도는?
### Input:

### Response:
대한민국 수도는 서울이다.
```

한편, 작은 모델일 모를것 같은 아래의 질문을 해보면!?  

```prompt = "### Instruction:\n분단된 한국 군 이름은??\n### Input:\n\n### Response:\n"```

```text
### Instruction:
분단된 한국 군 이름은??
### Input:

### Response:
한국군은 한국의 군사 조직으로, 1945년 11월 25일부터 1949년 12월 25일까지 1년 10월 25일까지 1년 10월 25
```

위와 같이 의미없는 답을 하게되지요!  

#### 직접 만든 데이터로 FT 하기!!

이젠, 직접 만든 데이터로 FT하여 대답을 하게 해봅시다!!  
반복이 좀 되어야하니~ 아래와 같은 방법으로 데이터를 만들어줍니다!  

```python
data = [
    {"instruction": "한국 군 중 북한과 분단된 곳은 어디인가요?", "input": "", "output": "경기도 연천군, 강원도 철원군, 고성군은 북한과 행정구역상 또는 역사적으로 분단된 지역입니다."},
    {"instruction": "북한과 접경하거나 분단된 한국의 군을 알려줘.", "input": "", "output": "연천군, 철원군, 고성군은 북한과 직접 접경하거나 동일 이름으로 남북에 분단된 군입니다."},
    {"instruction": "분단된 한국 군(郡) 이름을 모두 나열해줘.", "input": "", "output": "연천군, 철원군, 고성군"},
    {"instruction": "분단된 군 단위 행정구역이 있나요?", "input": "", "output": "네, 연천군, 철원군, 고성군 등이 남북한에 걸쳐 분단되어 있습니다."},
    {"instruction": "북한에도 같은 이름의 군이 있는 한국 군은?", "input": "", "output": "철원군과 고성군은 북한에도 같은 이름의 군이 존재합니다."},
]

# 다양한 문장 표현과 질문 형태로 변형하여 50개 생성
variants = [
    "북한과 경계를 이루는 한국의 군 지역은?",
    "분단 상태인 한국의 군에는 어떤 곳이 있나요?",
    "남북으로 나뉘어 있는 군을 알려주세요.",
    "남한과 북한에 동일한 이름의 군이 있나요?",
    "북한과 직접적으로 맞닿아 있는 군은 어디인가요?",
    "DMZ와 접해 있는 한국의 군을 알려줘.",
    "분단된 채로 이름이 동일한 군이 있다면?",
    "행정구역상 분단된 군이 존재하나요?",
    "현재도 군사분계선을 사이에 두고 나뉜 군이 있나요?",
    "북한에 일부 지역이 속해 있는 한국 군은 어디인가요?",
    "고성군과 철원군은 어떤 특이점이 있나요?",
    "연천군은 분단과 어떤 관련이 있나요?",
    "강원도 내 분단된 군을 알려줘.",
    "북한과 같은 군 이름을 공유하는 남한 군은?",
    "6.25 전쟁 이후 분단된 군은?",
    "북한 접경지대 군을 세 곳 말해줘.",
    "휴전선과 맞닿은 한국의 군은 어디?",
    "철원과 고성은 남북한에 모두 있나요?",
    "남한 최북단 군 중 하나는?",
    "분단의 상징이라 할 수 있는 군은 어디인가요?",
    "강원도에서 남북 모두에 존재하는 군은?",
    "군 단위에서 남북한이 나뉜 사례는?",
    "고성군은 남북한에 모두 존재하나요?",
    "연천군은 현재 완전히 남한에 속해 있나요?",
    "북한에도 고성군이 있나요?",
    "남북한이 공유했던 군 단위 행정구역?",
    "휴전선 인근 군을 알려줘.",
    "남북한 경계에 있는 대표적인 군은?",
    "연천, 철원, 고성 외 다른 분단 군이 있나요?",
    "철원군은 남북 어디에 있나요?",
    "한국에서 남북 분단이 반영된 군 명칭?",
    "분단과 관련된 군을 교육자료에 포함시키고 싶은데 어떤 곳이 있죠?",
    "남북 분단을 대표하는 군 단위 지역 세 곳?",
    "북한에도 있는 남한 군 이름은?",
    "한국에서 분단의 흔적이 남아있는 군 지역은?",
    "연천군의 북쪽 일부는 북한인가요?",
    "철원군은 6.25 전쟁 전후로 어떤 변화가 있었나요?",
    "분단 당시 고성군은 어떻게 나뉘었나요?",
    "분단과 군사분계선이 관련 있는 군은?",
    "강원도의 분단된 군 지명은?",
    "연천은 왜 분단 관련 군으로 언급되나요?",
    "군 단위 분단 지역을 조사하려면 어디부터 살펴야 하나요?",
    "현재는 남한이지만 과거 북한이었던 군은?",
    "분단된 군이 지역 발전에 어떤 영향을 주었나요?",
    "북한과의 관계가 특이한 한국 군은?",
    "연천군에 민간인통제구역이 있는 이유는?",
    "DMZ로 인해 영향을 받은 군 지역?",
    "남한과 북한이 나눠가진 군 명칭은?",
    "고성군이 분단된 이유는?",
]

for v in variants:
    data.append({
        "instruction": v,
        "input": "",
        "output": "연천군, 철원군, 고성군은 북한과 분단된 한국의 대표적인 군 지역입니다."
    })

# 데이터 수 확인
len(data) 
```
그리고~ 만들어진 데이터로 학습 고고!!  

```python
# LoRA 설정
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                 # 랭크
    lora_alpha=16,       # scaling factor
    lora_dropout=0.05,   # dropout
    bias="none",
    target_modules=["c_proj", "c_attn", "q_proj", "v_proj", "o_proj", "k_proj"],  # Qwen 구조에 맞게 조정
)

# 모델 불러오기 및 LoRA 적용
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # bitsandbytes를 통한 4bit quantization (선택사항)
    device_map="auto",
    trust_remote_code=True
)
model = get_peft_model(model, peft_config)

dataset = Dataset.from_list(data) ## 여기서 데이터를 넣어줍니다!!
def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./qwen_lora_output_mytext",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10, ## 여기를 1에서 10으로 바꿈
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
)

# 트레이너 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 학습 시작
trainer.train()
model.save_pretrained("qwen_lora_output_mytext")
``` 

학습을 많이 하게 하기 위하여! `num_train_epochs=10, `  만 바뀌었습니다!!  

그리고 아까와 같이 질문을 해보면!?

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
base_model_name =  "Qwen/Qwen2.5-0.5B"  # 사용한 베이스 모델
peft_path = "qwen_lora_output_mytext"  # trainer에서 지정한 결과 디렉토리

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
model = PeftModel.from_pretrained(model, peft_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

prompt = "### Instruction:\n분단된 한국 군 이름은??\n### Input:\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


```text
### Instruction:
분단된 한국 군 이름은??
### Input:

### Response:
연천군, 철원군, 철원군, 철원군, 철원군, 철원군, 철원군, 철원군, 철원군, 철원군, 철원군, 철원군, 철원군,
```
그럼!! 위와 같이 완벽하지는 않지만 아까보다는 발전된 답을합니다!  
완전 뿌듯하죠~? ㅎㅎ  
이런 방식을, 보다 많은 데이터셋, epoch로 하면 학습이 잘 되겠지요??  

추가로 `model.eval()` 의 결과를 보면 LORA가 어떻게 결합됬는지 볼 수 있고!  

아래와 같은 코드로 LORA의 파라미터 수도 알 수 있습니다~!

```python
from peft import get_peft_model
from peft.tuners.lora import LoraLayer

def count_lora_parameters(model):
    total = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for name, param in module.named_parameters():
                if "lora_" in name:
                    total += param.numel()
    return total

print("LoRA 파라미터 수:", count_lora_parameters(model))

```

```text
LoRA 파라미터 수: 1081344
```


### 🎉 마무리

오늘은 이렇게 LORA를 실습해보았습니다!!  
체계적인 패키지들 덕분에 지금 당장이라도 여러 분야에 적용할수 있을 자신감이 뿜뿜하네요!!^^  
LORA 외의 PEFT도 공부해보아야겠어요~!  