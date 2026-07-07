---
layout: post
title: "🖥️ Grounding DINO Practice - Grounding DINO 실습 with python!"
author: [DrFirst]
date: 2025-05-12 07:00:00 +0900
categories: [AI, Experiment]
tags: [grounding DINO, DINO, 객체 탐지, Object Detection, CV, ECCV, ECCV 2024, python, 파이썬 실습]
lastmod : 2025-05-12 07:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---

## 🦖 (English) Grounding DINO Practice! Running the Evolved Model of DINO

This post is a hands-on guide to the **Grounding DINO**, the evolved version of DINO!  
Just like DINO, we clone the model from GitHub and run it — and surprisingly, it's even simpler 😊  
So this time, we’ll skip the theory for now and jump straight into running the code!!

---

### 🧱 1. Clone the GitHub Repository

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
```

---

### 📦 2. Install the Model

```bash
pip install -e .
```

---

### 🧊 3. Download the Pretrained Weights

```bash
mkdir weights
cd weights/
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

---

### 🚀 4. Run the Inference (Command Template)

The command structure for running Grounding DINO is shown below.  
Each part marked with `{}` can be replaced with the appropriate value for your environment:

```bash
CUDA_VISIBLE_DEVICES={GPU_ID} python demo/inference_on_a_image.py \
  -c {CONFIG_PATH} \
  -p {CHECKPOINT_PATH} \
  -i {INPUT_IMAGE_PATH} \
  -o {OUTPUT_DIR} \
  -t "{TEXT_PROMPT}"
```

#### Example values:

| Variable           | Description |
|--------------------|-------------|
| `{GPU_ID}`         | GPU ID to use (e.g., `0`) — you can check it using `nvidia-smi` |
| `{CONFIG_PATH}`    | Config file path (e.g., `groundingdino/config/GroundingDINO_SwinT_OGC.py`) |
| `{CHECKPOINT_PATH}`| Path to the pretrained weights (e.g., `weights/groundingdino_swint_ogc.pth`) |
| `{INPUT_IMAGE_PATH}` | Input image path (e.g., `/home/user/images/catch_rugby_ball_001480.jpg`) |
| `{OUTPUT_DIR}`     | Directory to save the result (e.g., `/home/user/images/GroundingDINO/results`) |
| `{TEXT_PROMPT}`    | Text prompt to detect (e.g., `"chair"`) |

- You can change the `TEXT_PROMPT` to try different detection phrases!!

---

### ✅ Real-World Prompt Tests!

Let’s now try running the inference while changing only the `TEXT_PROMPT` and see the results!

---

**`person`** — A common COCO category.  
> Of course it works great!

![Image](https://github.com/user-attachments/assets/5cc791b3-28e8-4e95-b90b-4691428a2edb)

---

**`cat`** — Will it produce any false positives even when the object isn't there?  
> Nothing detected! Well done 😎

![Image](https://github.com/user-attachments/assets/19d47f24-49da-4ade-9874-d021f39fbae9)

---

**`rugby`** — Likely missing from most test sets. Will it still work?  
> Oh nice! It makes sense!!

![Image](https://github.com/user-attachments/assets/ef7e08c1-5072-4d81-a74b-342860d680c1)

---

**`jump`** — Now let’s try a verb!  
> Whoa~ It works for actions too!?

![Image](https://github.com/user-attachments/assets/41e70375-f8eb-41d1-93d2-b1b65a372abe)

---

**`player is jumping`** — What about a full sentence?  
> Hmm… Seems like it breaks it apart instead of treating it as one phrase.

![Image](https://github.com/user-attachments/assets/258130cd-c187-427a-b80e-294ac701e8b7)

---

### 🎉 Conclusion

Grounding DINO was very easy to install, and the inference workflow is intuitive!  
Especially useful if you want to experiment with **diverse text prompts**.  
If it could even understand full sentences, and support segmentation too —  
that would be amazing, right!? 😄

Well, maybe there’s a model out there that already does that?  
Let’s keep exploring together!


---

## 🦖(한국어) Grounding DINO 실습! DINO의 진화 모델을 직접 실행해보자!

이번 포스팅은 DINO의 후속 모델인 **Grounding DINO** 실습입니다!  
DINO와 마찬가지로 GitHub repo에서 코드를 내려받아 실행하지만, 오히려 더 간단하게 구성되어 있더라구요 😊  
그래서 이번엔 이론은 잠시 뒤로 미루고, 바로 코드부터 실행해봅니다!!

---

### 🧱 1. GitHub 저장소 클론

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
```

---

### 📦 2. 모델 설치

```bash
pip install -e .
```

---

### 🧊 3. Pretrained Weight 다운로드

```bash
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

 - 여기서 `TEXT_PROMPT` 에 다양한 값을 넣으며 테스트해볼 수 있습니다!!
---

### ✅ 실전 프롬포트별 테스트!!

이제 `TEXT_PROMPT` 를 바꿔가며 결과를 보겠습니다!!


`person`. 가장 간단하며 기존 coco dataset에 있는 person!!
> 역시 잘하는구만~~    

![Image](https://github.com/user-attachments/assets/5cc791b3-28e8-4e95-b90b-4691428a2edb)

`cat`. 없는것을 오탐지하지는 않을까요!?  
> 아무것도 안잡네? 잘했어!!  

![Image](https://github.com/user-attachments/assets/19d47f24-49da-4ade-9874-d021f39fbae9)


테스트 셋에도 없었을 `rugby`. 럭비라는 단어로 작동을 할까요!?
> 럭비~~ 그럴듯해 좋아!?!  

![Image](https://github.com/user-attachments/assets/ef7e08c1-5072-4d81-a74b-342860d680c1)


`jump` 이번엔 동사으로!! 
> 오~ 동사 잘하는걸!?!  

![Image](https://github.com/user-attachments/assets/41e70375-f8eb-41d1-93d2-b1b65a372abe)


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


```
 CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py  -c groundingdino/config/GroundingDINO_SwinT_OGC.py  -p weights/groundingdino_swint_ogc.pth  -i /home/smartride/DrFirst/LOCATE/AGD20K/Unseen/trainset/exocentric/catch/rugby_ball/catch_rugby_ball_001480.jpg  -o /home/smartride/DrFirst/GroundingDINO/results  -t "hand"
 ```