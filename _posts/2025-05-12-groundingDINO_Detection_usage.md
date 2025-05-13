---
layout: post
title: "grounding-DINO 실습 with python!"
author: [DrFirst]
date: 2025-05-15 07:00:00 +0900
categories: [AI, Experiment]
tags: [DETR, DINO, 객체 탐지, Object Detection, Transformer, 딥러닝, CV, ICLR, ICLR 2023, python, 파이썬 실습]
lastmod : 2025-05-15 07:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---

## 🦖 Grounding DINO 실습! DINO의 진화 모델을 직접 실행해보자!

이번 포스팅은 DINO의 후속 모델인 **Grounding DINO** 실습입니다!  
DINO와 마찬가지로 GitHub repo에서 코드를 내려받아 실행하지만, 오히려 더 간단하게 구성되어 있더라구요 😊  
그래서 이번엔 이론은 잠시 뒤로 미루고, 바로 코드부터 실행해봅니다!!

---

### 🧱 1. GitHub 저장소 클론

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
```

---

### 📦 2. 모델 설치

```
pip install -e .
```

---

### 🧊 3. Pretrained Weight 다운로드

```
mkdir weights
cd weights/
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

---

### 🚀 4. 객체 탐지 실험 실행 (명령어 템플릿)

다음 명령어는 Grounding DINO를 실행하는 전체 구조입니다.  
필요한 값들은 `{}` 변수로 구성되어 있어 상황에 맞게 대체하면 됩니다:

```
CUDA_VISIBLE_DEVICES={GPU_ID} python demo/inference_on_a_image.py \
  -c {CONFIG_PATH} \
  -p {CHECKPOINT_PATH} \
  -i {INPUT_IMAGE_PATH} \
  -o {OUTPUT_DIR} \
  -t "{TEXT_PROMPT}"
```

#### 예시 값 대입:

| 변수명             | 설명 |
|------------------|------|
| `{GPU_ID}`         | 사용할 GPU ID (예: `0`) - nvidia-smi 하면 확인할수 있습니다!! |
| `{CONFIG_PATH}`    | 설정 파일 경로 (예: `groundingdino/config/GroundingDINO_SwinT_OGC.py`) |
| `{CHECKPOINT_PATH}`| 모델 가중치 경로 (예: `weights/groundingdino_swint_ogc.pth`) |
| `{INPUT_IMAGE_PATH}` | 감지할 이미지 경로 (예: `/home/user/images/catch_rugby_ball_001480.jpg`) |
| `{OUTPUT_DIR}`     | 출력 저장 디렉터리 (예: `/home/user/images/GroundingDINO/results`) |
| `{TEXT_PROMPT}`    | 탐지할 대상 텍스트 (예: `"chair"`) |

여기서 `TEXT_PROMPT` 에 다양한 값을 넣으며 테스트해볼 수 있습니다!!
---

### ✅ 실전 프롬포트별 테스트!!

이제 `TEXT_PROMPT` 를 바꿔가며 결과를 보겠습니다!!


`person`. 가장 간단하며 기존 coco dataset에 있는 person!!

`cat`. 없는것을 오탐지하지는 않을까요!?  

테스트 셋에도 없었을 `rugby`. 럭비라는 단어로 작동을 할까요!?

`jump` 이번엔 동사으로!! 


`player is jumping` 이번엔 문장!!   
> 아쉽게도 문장을 하나로 인식하는게 아니라 끊어서 보는듯하네요!!  

![Image](https://github.com/user-attachments/assets/258130cd-c187-427a-b80e-294ac701e8b7)


---

### 🎉 마무리

Grounding DINO는 설치도 간단하고, inference도 직관적으로 되어 있어서 바로 실험해보기 좋았습니다!  
특히, 다양한 텍스트 프롬프트를 실험해보고 싶을 때 매우 유용할 것 같아요.  
문장까지 잘 인지하고, Segment 까지할수 있다면!!  
얼~~마나 좋을까요!!  
그런데! 그런 연구도 있지 않을까요!?  
함께 공부해봅시다!^^  


