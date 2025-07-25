---
title: "[MLflow] LLM 프롬프트 엔지니어링 실험 관리하기 - 체계적인 프롬프트 튜닝과 결과 추적"
author: drfirst
date: 2025-07-05 10:00:00 +0900
categories: [AI, Experiment]
tags: [MLflow, LLM, 프롬프트엔지니어링, 실험관리, OpenAI, 언어모델]
render_with_liquid: false
---

## 🤖 MLflow로 LLM 프롬프트 실험 관리하기

> **"이 프롬프트가 더 좋았나? 저 프롬프트가 더 좋았나?"**  
> LLM 시대에 가장 중요한 것은 바로 **프롬프트 엔지니어링**입니다!

### 🎯 **왜 프롬프트 실험 관리가 필요한가?**

LLM을 사용하다 보면 이런 경험 있으시죠?

```text
❌ 전통적인 프롬프트 관리의 문제점:

"너는 도움이 되는 AI야"  → 결과: 보통
"너는 전문가야"         → 결과: 좀 더 나음  
"너는 10년 경력 전문가야" → 결과: 훨씬 좋음!

하지만... 어떤 프롬프트를 언제 썼는지 기억이 안 나고,
결과 비교도 어렵고, 팀원들과 공유도 힘들어요 😫
```

**MLflow LLM 기능을 사용하면:**
- 🔍 **모든 프롬프트 실험을 자동 추적**
- 📊 **결과를 체계적으로 비교**
- 🤝 **팀원들과 쉽게 공유**
- 🎯 **최고 성능 프롬프트를 쉽게 찾기**

---

### 🚀 **MLflow LLM 설치 및 설정**

#### **MLflow 서버 실행**

> ![이전 포스트](https://drfirstlee.github.io/posts/mkflow/)에서 만든 Docker 사용 
```bash
docker run -d -p 5001:5001 --name mlflow \
           -v $(pwd)/mlflow/db:/mlflow/db \
           -v $(pwd)/mlartifacts:/mlartifacts \
           my-mlflow-image
```

---

### 🔧 **MLflow LLM 실습 - 프롬프트 엔지니어링 실험**

#### **🎯 실습 목표: 번역 프롬프트 최적화**

다양한 번역 프롬프트를 테스트해서 가장 좋은 결과를 찾아보겠습니다!

```python
import mlflow
import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# MLflow 설정
mlflow.set_tracking_uri("http://0.0.0.0:5001")
mlflow.set_experiment("llm_translation_experiment")

# OpenAI 클라이언트 설정
client = openai.OpenAI()

# 테스트할 원문
test_text = "The quick brown fox jumps over the lazy dog."

# 다양한 프롬프트 템플릿 정의
prompt_templates = {
    "basic": "다음 영어를 한국어로 번역하세요: {text}",
    
    "professional": """당신은 10년 경력의 전문 번역가입니다.
다음 영어 문장을 자연스러운 한국어로 번역하세요:
{text}""",
    
    "context_aware": """당신은 문맥을 고려하는 전문 번역가입니다.
다음 영어 문장을 한국어로 번역할 때:
1. 자연스러운 한국어 표현을 사용하세요
2. 문맥상 가장 적절한 의미를 선택하세요
3. 원문의 뉘앙스를 살려주세요

영어 원문: {text}
한국어 번역:""",
    
    "step_by_step": """다음 단계로 번역하세요:
1. 먼저 영어 문장을 분석하세요
2. 각 단어의 의미를 파악하세요  
3. 한국어로 자연스럽게 번역하세요

영어 원문: {text}"""
}

# 프롬프트 실험 실행
for prompt_name, prompt_template in prompt_templates.items():
    with mlflow.start_run():
        # 프롬프트 생성
        prompt = prompt_template.format(text=test_text)
        
        # 파라미터 로깅
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("model", "gpt-3.5-turbo")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("max_tokens", 100)
        
        # 프롬프트 템플릿 저장
        mlflow.log_text(prompt_template, f"prompt_template_{prompt_name}.txt")
        mlflow.log_text(prompt, f"full_prompt_{prompt_name}.txt")
        
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        # 결과 추출
        translation = response.choices[0].message.content
        
        # 결과 저장
        mlflow.log_text(translation, f"translation_{prompt_name}.txt")
        
        # 메트릭 계산 (간단한 예시)
        translation_length = len(translation)
        word_count = len(translation.split())
        
        mlflow.log_metric("translation_length", translation_length)
        mlflow.log_metric("word_count", word_count)
        mlflow.log_metric("tokens_used", response.usage.total_tokens)
        mlflow.log_metric("cost_usd", response.usage.total_tokens * 0.0000015)  # 대략적인 비용
        
        # 태그 추가
        mlflow.set_tag("prompt_category", "translation")
        mlflow.set_tag("language_pair", "en_to_ko")
        mlflow.set_tag("model_provider", "openai")
        
        print(f"✅ {prompt_name} 실험 완료!")
        print(f"   번역 결과: {translation[:50]}...")
        print(f"   토큰 사용량: {response.usage.total_tokens}")
        print(f"   비용: ${response.usage.total_tokens * 0.0000015:.6f}")
        print()
            

print("🎉 모든 프롬프트 실험 완료!")
print("🌐 브라우저에서 http://0.0.0.0:5001 으로 접속해서 결과를 확인하세요!")
```

#### **실행 결과 예시:**
```text
✅ basic 실험 완료!
   번역 결과: 빠른 갈색 여우가 게으른 개를 뛰어 넘습니다....
   토큰 사용량: 56
   비용: $0.000084
🏃 View run unruly-pig-302 at: http://0.0.0.0:5001/#/experiments/6/runs/2a19fc4770334dd5b17364cd46087cd6
🧪 View experiment at: http://0.0.0.0:5001/#/experiments/6
✅ professional 실험 완료!
   번역 결과: 빠른 갈색 여우가 게으른 개를 뛰어넘습니다....
   토큰 사용량: 82
   비용: $0.000123
🏃 View run respected-mole-452 at: http://0.0.0.0:5001/#/experiments/6/runs/3ddf4e93a6564904aa454dfe6903badb
🧪 View experiment at: http://0.0.0.0:5001/#/experiments/6
✅ context_aware 실험 완료!
   번역 결과: 빠른 갈색 여우가 나태한 개를 뛰어넘습니다....
   토큰 사용량: 154
   비용: $0.000231
🏃 View run monumental-cub-767 at: http://0.0.0.0:5001/#/experiments/6/runs/9d5538e33665440687ab940309663fc2
🧪 View experiment at: http://0.0.0.0:5001/#/experiments/6
✅ step_by_step 실험 완료!
   번역 결과: 1. Analyze the English sentence first
2. Understan...
   토큰 사용량: 144
   비용: $0.000216
🏃 View run indecisive-conch-549 at: http://0.0.0.0:5001/#/experiments/6/runs/f3bc72525ee94ecb8d71098ca330641b
🧪 View experiment at: http://0.0.0.0:5001/#/experiments/6
🎉 모든 프롬프트 실험 완료!
🌐 브라우저에서 http://0.0.0.0:5001 으로 접속해서 결과를 확인하세요!
```

---

### 📊 **결과 분석 및 비교**

#### **1. 저장된 실험 결과 불러오기**
```python
import mlflow
import pandas as pd

# 실험 결과 조회
mlflow.set_tracking_uri("http://0.0.0.0:5001")
experiment = mlflow.get_experiment_by_name("llm_translation_experiment")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print(f"📊 총 {len(runs)}개의 프롬프트 실험 완료!")
print("\n🏆 실험 결과 비교:")
print("=" * 60)

for _, run in runs.iterrows():
    prompt_name = run.get('params.prompt_name', 'unknown')
    tokens_used = run.get('metrics.tokens_used', 0)
    cost = run.get('metrics.cost_usd', 0)
    
    print(f"📝 {prompt_name:15} | 토큰: {tokens_used:3.0f} | 비용: ${cost:.6f}")

# 가장 효율적인 프롬프트 찾기
best_efficiency = runs.loc[runs['metrics.tokens_used'].idxmin()]
print(f"\n🎯 가장 효율적인 프롬프트: {best_efficiency['params.prompt_name']}")
print(f"   토큰 사용량: {best_efficiency['metrics.tokens_used']:.0f}")
```

- 결과는!!?  

```text
📊 총 4개의 프롬프트 실험 완료!

🏆 실험 결과 비교:
============================================================
📝 step_by_step    | 토큰: 144 | 비용: $0.000216
📝 context_aware   | 토큰: 154 | 비용: $0.000231
📝 professional    | 토큰:  82 | 비용: $0.000123
📝 basic           | 토큰:  56 | 비용: $0.000084

🎯 가장 효율적인 프롬프트: basic
   토큰 사용량: 56
   ```

#### **2. 프롬프트 템플릿 다운로드**
```python
# 최고 성능 프롬프트 다운로드
client = mlflow.tracking.MlflowClient()
best_run_id = best_efficiency['run_id']

# 아티팩트 목록 확인
artifacts = client.list_artifacts(best_run_id)
print("📦 저장된 아티팩트:")
for art in artifacts:
    print(f"   - {art.path}")

# 프롬프트 템플릿 다운로드
prompt_template_path = f"prompt_template_{best_efficiency['params.prompt_name']}.txt"
local_path = client.download_artifacts(best_run_id, prompt_template_path)

with open(local_path, 'r', encoding='utf-8') as f:
    best_prompt_template = f.read()

print(f"\n🏆 최고 성능 프롬프트 템플릿:")
print("=" * 40)
print(best_prompt_template)
```

---

### 🎯 **고급 프롬프트 실험 - 감성 분석**

더 복잡한 프롬프트 엔지니어링 실험을 해보겠습니다!

```python
import mlflow
import openai
import json
client = openai.OpenAI()
# 감성 분석 실험
mlflow.set_experiment("llm_sentiment_analysis")

# 테스트 데이터
test_reviews = [
    "이 제품 정말 좋아요! 추천합니다.",
    "배송이 늦었지만 제품은 만족스럽네요.",
    "완전 최악... 돈 아깝다.",
    "그냥 그래요. 나쁘지도 좋지도 않아요."
]

# 다양한 감성 분석 프롬프트
sentiment_prompts = {
    "simple": """다음 리뷰의 감정을 분석하세요.
결과: 긍정/부정/중립 중 하나로 답하세요.
리뷰: {text}""",
    
    "detailed": """다음 리뷰를 분석하고 JSON 형식으로 답하세요:
{{
  "sentiment": "긍정/부정/중립",
  "confidence": 0.0-1.0,
  "reason": "이유 설명"
}}

리뷰: {text}""",
    
    "chain_of_thought": """다음 리뷰를 단계별로 분석하세요:

1. 감정 표현 단어 식별
2. 전체적인 톤 분석  
3. 최종 감정 판정
4. 신뢰도 평가

리뷰: {text}

분석 결과:"""
}

# 실험 실행
for prompt_name, prompt_template in sentiment_prompts.items():
    with mlflow.start_run():
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("task", "sentiment_analysis")
        
        all_results = []
        total_tokens = 0
        
        for i, review in enumerate(test_reviews):
            prompt = prompt_template.format(text=review)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # 일관성을 위해 낮은 온도
            )

            result = response.choices[0].message.content
            tokens = response.usage.total_tokens

            all_results.append({
                "review": review,
                "result": result,
                "tokens": tokens
            })

            total_tokens += tokens

            # 개별 결과 저장
            mlflow.log_text(result, f"result_{i}_{prompt_name}.txt")
                

        
        # 전체 결과 저장
        results_json = json.dumps(all_results, ensure_ascii=False, indent=2)
        mlflow.log_text(results_json, f"all_results_{prompt_name}.json")
        
        # 메트릭 저장
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("avg_tokens_per_review", total_tokens / len(test_reviews))
        mlflow.log_metric("total_cost", total_tokens * 0.0000015)
        
        print(f"✅ {prompt_name} 감성 분석 완료!")
        print(f"   총 토큰: {total_tokens}")
        print(f"   평균 토큰/리뷰: {total_tokens / len(test_reviews):.1f}")
        print()

print("🎉 감성 분석 실험 완료!")
```

---

### 🤝 **팀 협업을 위한 프롬프트 공유**

#### **프롬프트 템플릿 라이브러리 만들기**
```python
import mlflow
import mlflow.pyfunc
import json

class PromptTemplate(mlflow.pyfunc.PythonModel):
    """재사용 가능한 프롬프트 템플릿 클래스"""
    
    def __init__(self, template, task_type, language="ko"):
        self.template = template
        self.task_type = task_type
        self.language = language
    
    def format_prompt(self, **kwargs):
        """프롬프트 템플릿에 변수 값 삽입"""
        return self.template.format(**kwargs)
    
    def predict(self, context, model_input):
        """MLflow 모델 인터페이스"""
        if isinstance(model_input, dict):
            return self.format_prompt(**model_input)
        return self.template

# 검증된 프롬프트 템플릿 등록
mlflow.set_experiment("prompt_template_library")

with mlflow.start_run():
    # 최고 성능 번역 프롬프트를 모델로 등록
    best_translation_prompt = """당신은 10년 경력의 전문 번역가입니다.
다음 영어 문장을 자연스러운 한국어로 번역하세요:
{text}"""
    
    prompt_model = PromptTemplate(
        template=best_translation_prompt,
        task_type="translation",
        language="ko"
    )
    
    # 모델 정보 로깅
    mlflow.log_param("template_name", "professional_translation")
    mlflow.log_param("task_type", "translation")
    mlflow.log_param("language_pair", "en_to_ko")
    mlflow.log_param("performance_score", 0.95)
    
    # 템플릿 저장
    mlflow.log_text(best_translation_prompt, "prompt_template.txt")
    
    # Python 모델로 저장
    mlflow.pyfunc.log_model(
        artifact_path="prompt_model",
        python_model=prompt_model,
        registered_model_name="TranslationPromptTemplate"
    )
    
    print("✅ 프롬프트 템플릿이 모델로 등록되었습니다!")

# 등록된 프롬프트 템플릿 사용하기
model_uri = "models:/TranslationPromptTemplate/latest"
loaded_prompt = mlflow.pyfunc.load_model(model_uri)

# 사용 예시
test_input = {"text": "Hello, world!"}
formatted_prompt = loaded_prompt.predict(None, test_input)
print(f"🎯 생성된 프롬프트:\n{formatted_prompt}")
```

---

### 📈 **프롬프트 성능 모니터링**

#### **A/B 테스트로 프롬프트 비교**
```python
import mlflow
import random
import time

# A/B 테스트 실험
mlflow.set_experiment("prompt_ab_test")

# 두 가지 프롬프트 버전
prompt_a = "다음을 요약하세요: {text}"
prompt_b = "다음 내용을 3줄로 핵심만 요약하세요: {text}"

test_texts = [
    "인공지능은 현대 사회에서 점점 더 중요한 역할을 하고 있습니다...",
    "기후 변화는 전 세계적으로 심각한 문제가 되고 있습니다...",
    "원격 근무는 코로나19 이후 새로운 업무 형태로 자리잡았습니다..."
]

# A/B 테스트 실행
for i, text in enumerate(test_texts):
    # 무작위로 A 또는 B 선택
    version = random.choice(['A', 'B'])
    prompt = prompt_a if version == 'A' else prompt_b
    
    with mlflow.start_run():
        mlflow.log_param("prompt_version", version)
        mlflow.log_param("text_id", i)
        mlflow.log_param("test_type", "ab_test")
        
        full_prompt = prompt.format(text=text)
        
        # 실제 API 호출 (여기서는 시뮬레이션)
        # response = client.chat.completions.create(...)
        
        # 시뮬레이션 결과
        response_length = random.randint(50, 200)
        user_satisfaction = random.uniform(3.0, 5.0)
        
        mlflow.log_metric("response_length", response_length)
        mlflow.log_metric("user_satisfaction", user_satisfaction)
        mlflow.log_metric("completion_time", random.uniform(1.0, 3.0))
        
        mlflow.set_tag("experiment_type", "ab_test")
        mlflow.set_tag("prompt_category", "summarization")
        
        print(f"✅ 테스트 {i+1} (버전 {version}) 완료!")

# 결과 분석
experiment = mlflow.get_experiment_by_name("prompt_ab_test")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# 버전별 성능 비교
version_a_runs = runs[runs['params.prompt_version'] == 'A']
version_b_runs = runs[runs['params.prompt_version'] == 'B']

print(f"\n📊 A/B 테스트 결과:")
print(f"버전 A 평균 만족도: {version_a_runs['metrics.user_satisfaction'].mean():.2f}")
print(f"버전 B 평균 만족도: {version_b_runs['metrics.user_satisfaction'].mean():.2f}")

if version_a_runs['metrics.user_satisfaction'].mean() > version_b_runs['metrics.user_satisfaction'].mean():
    print("🏆 버전 A 승리!")
else:
    print("🏆 버전 B 승리!")
```

---

### 🎯 **실전 활용 팁**

#### **1. 프롬프트 버전 관리**
```python
# 프롬프트 버전별 관리
PROMPT_VERSIONS = {
    "v1.0": "기본 프롬프트",
    "v1.1": "컨텍스트 추가",
    "v1.2": "예시 포함",
    "v2.0": "완전히 새로운 접근"
}

for version, description in PROMPT_VERSIONS.items():
    with mlflow.start_run():
        mlflow.log_param("prompt_version", version)
        mlflow.log_param("description", description)
        mlflow.set_tag("version_type", "major" if ".0" in version else "minor")
```

#### **2. 비용 최적화 추적**
```python
# 모델별 비용 비교
MODEL_COSTS = {
    "gpt-3.5-turbo": 0.0000015,
    "gpt-4": 0.00003,
    "claude-3-haiku": 0.00000025
}

for model_name, cost_per_token in MODEL_COSTS.items():
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("cost_per_token", cost_per_token)
        # ... 실험 수행
        mlflow.log_metric("total_cost", tokens_used * cost_per_token)
```

---

### 🏆 **결론**

MLflow를 사용한 LLM 프롬프트 엔지니어링으로 다음과 같은 이점을 얻을 수 있습니다:

**🎯 핵심 가치:**
1. **체계적인 실험 관리**: 모든 프롬프트 실험을 자동 추적
2. **성능 비교**: 다양한 프롬프트의 효과를 객관적으로 비교
3. **비용 최적화**: 토큰 사용량과 비용을 체계적으로 관리
4. **팀 협업**: 검증된 프롬프트 템플릿 공유

**💡 시작 권장사항:**
- 작은 프롬프트 실험부터 시작
- 일관된 평가 지표 설정
- 비용 추적 습관화
- 팀 내 프롬프트 라이브러리 구축

**📚 추가 학습 자료:**
- [MLflow LLM 공식 문서](https://mlflow.org/docs/latest/llms/index.html)
- [프롬프트 엔지니어링 가이드](https://platform.openai.com/docs/guides/prompt-engineering)

---

> 💡 **마지막 팁**: 프롬프트 엔지니어링은 과학입니다. 체계적인 실험과 데이터 기반 의사결정으로 최고의 성능을 얻으세요!

**🤖 Happy Prompt Engineering!** 🚀 