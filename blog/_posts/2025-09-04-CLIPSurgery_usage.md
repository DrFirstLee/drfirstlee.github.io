---
layout: post
title: "🔎 ClipSurgery Hands-on: ClipSurgery 을 실습해보자!"
author: [DrFirst]
date: 2025-09-04 07:00:00 +0900
categories: [AI, Research]
tags: [CLIP, Explainability, CAM, Vision-Language, Training-Free, Segmentation, PatternRecognition]
sitemap:
  changefreq: monthly
  priority: 0.8
---

---

### 🧬 (English) Practicing ClipSurgery!!

Today we will practice with the [CLIP](https://drfirstlee.github.io/posts/CLIP/) model that has been **surgically modified** —  
[ClipSurgery](https://drfirstlee.github.io/posts/ClipSurgery/)!  

We’ll **run it directly** and generate a **clean similarity map (CAM)** on top of an image!!

---

### ✅ Environment Setup!!  
- First, clone the Clip Surgery Git repo:  
```bash
git clone https://github.com/xmed-lab/CLIP_Surgery.git
```
- This includes `clip.py`, `clip_surgery_model.py`, and also `demo.ipynb`.  
- We will customize this `demo.ipynb`.  

+ Additionally:
- Python ≥ 3.9, CUDA recommended (but CPU works too)  
- Required libraries: `torch`, `opencv-python`, `numpy`, `Pillow`, `matplotlib`, `torchvision`  
- From the **CLIP_Surgery** repo (or equivalent module) we need:  
  - `clip.load("CS-ViT-B/16", ...)` (the surgically modified vision backbone)  
  - `encode_text_with_prompt_ensemble`  
  - `clip_feature_surgery`  
  - `get_similarity_map`  
- One image file for visualization (e.g. `dog.jpg`)  

---

### 🧠 What will we do?

1. Extract image token features with the **CS-ViT-B/16** model (with Architecture Surgery),  
2. Build stable class (text) embeddings via **Prompt Ensemble**,  
3. Apply **Feature Surgery** to remove **redundant/common class features**,  
4. Generate and visualize a **foreground-focused similarity map**.  

---

### 🧪 Let’s Start!! – Full Code

```python
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
# from segment_anything import sam_model_registry, SamPredictor

# 0) Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) (Optional) Load original CLIP — for comparison
model, _ = clip.load("ViT-B/16", device=device)
model.eval()

# 2) Preprocessing pipeline
preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711))
])

# 3) Load input image
pil_img = Image.open(f"/{my_path}/dog.jpg" )
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
image = preprocess(pil_img).unsqueeze(0).to(device)

# 4) Class dictionary — Feature Surgery requires multiple classes
all_texts = [
    'airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat',
    'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
    'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door',
    'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard',
    'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
    'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign',
    'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck',
    'tv monitor', 'wall', 'water', 'window', 'wood'
]
target_texts = ['dog']

# 5) Load surgically modified architecture (CS-ViT-B/16)
model, preprocess_unused = clip.load("CS-ViT-B/16", device=device)
model.eval()

with torch.no_grad():
    # (A) Image features (per token, including CLS)
    image_features = model.encode_image(image)                    # [B, 1+HW, C]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # (B) Prompt ensemble-based text features
    text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)  # [N, C]

    # (C) Feature Surgery — remove redundant features
    similarity = clip.clip_feature_surgery(image_features, text_features)            # [B, 1+HW, N]

    # (D) Generate similarity map from patch tokens only + upsample
    similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])  # [B, H, W, N]

    # (E) Visualization — Overlay target class only
    for b in range(similarity_map.shape[0]):
        for n in range(similarity_map.shape[-1]):
            if all_texts[n] not in target_texts:
                continue
            vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = cv2_img * 0.4 + vis * 0.6
            vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
            print('CLIP Surgery:', all_texts[n])
            plt.imshow(vis)
            plt.axis('off')
            plt.show()
```

Running this code immediately segments the dog quite well!  

![Image](https://github.com/user-attachments/assets/05f605c4-91ba-4e61-86a0-dba39b6f82a4)

---

### 🔍 Code Explanation (Key Points)

#### 1) **CS-ViT-B/16**: Backbone with Architecture Surgery  
- In several final blocks, set **q=k=v(=V)** to perform **Consistent Self-Attention**.  
- Introduce a **Dual Path**:  
  - The CAM path skips the **FFN**, reducing background/noise influence.  
  - The original CLIP path is preserved to maintain embedding quality.  

#### 2) `clip.encode_text_with_prompt_ensemble` for Prompt Ensemble  
- Using only `"a photo of a {}."` is unstable, so multiple templates are averaged and normalized → **stable class embeddings**.  
- Example templates (in `clip.py`):  
```python
prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 
'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.',
'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.',
'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.',
'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.',
'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 
'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 
'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 
'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 
'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 
'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 
'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 
'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 
'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 
'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 
'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 
'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 
'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 
'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 
'there is a {} in the scene.', 'there is the {} in the scene.', 
'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
```

#### 3) Feature Surgery
- Compute element-wise product of image tokens × text embeddings.  
- Estimate **class-common features** (mean across classes) → subtract them → focus on **foreground**.  
- Summing produces similarity tensor **[B, 1+HW, N]**.  
- In `clip.py`:  

```python
def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):

    if redundant_feats != None:
        similarity = image_features @ (text_features - redundant_feats).t()

    else:
        # weights to restrain influence of obvious classes on others
        prob = image_features[:, :1, :] @ text_features.t()
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)

        # element-wise multiplied features
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        feats *= w.reshape(1, 1, n_t, 1)
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats
        
        # sum the element-wise multiplied features as cosine similarity
        similarity = feats.sum(-1)

    return similarity
```

#### 4) Building the Similarity Map
- Exclude CLS (no spatial info), use only **patch tokens (HW)**.  
- Normalize each channel with Min–Max, upsample with **bilinear** to original image size.  
- Select target class channel (`dog`) and overlay as a heatmap.  

---

### 🧭 Limitations (Finding Failure Cases)

> Still, part segmentation **inside the object** is not solved.  
![Image](https://github.com/user-attachments/assets/153f82dc-027f-4120-95f6-81c565848f8f)

> In some cases it seems to work!?  
![Image](https://github.com/user-attachments/assets/5e4bada5-475d-40ac-adb4-6f6ee6d2febf)

- Results vary depending on the image and the target object.  

---

### ✅ Summary  
This practice showed that we can **greatly improve CLIP’s explainability** **without additional training**.  
Key ideas:  
- **(i) Consistent Self-Attention + Dual Path** reduces structural issues,  
- **(ii) Feature Surgery** removes redundant features, making the foreground stand out clearly.  


---

### 🧬 (한국어) ClipSurgery 을 실습!!

오늘은 [CLIP](https://drfirstlee.github.io/posts/CLIP/)모델의 내부구조를 수술한!!
[ClipSurgery](https://drfirstlee.github.io/posts/ClipSurgery/) 의 파이썬 실습을 해보겠습니다!! 

**직접 실행**해 보면서, 이미지 위에 **깨끗한 similarity map(CAM)** 을 그리는 과정을 함깨 해 보아요!!



---

### ✅ 환경세팅!!  
- 우선!! Clip Surgery Git repo를 다운받습니다!!  
```bash
git clone https://github.com/xmed-lab/CLIP_Surgery.git
```
- 그럼 `clip.py`와 `clip_surgery_model.py` 가 들이었고!! `demo.ipynb`도 있지요~  
- 저희는 이 `demo.ipynb` 를 커스터마이징 시켜보았습니다!!

+ 추가로
- Python ≥ 3.9, CUDA 환경 권장(없어도 CPU로 동작)
- 필수 라이브러리: `torch`, `opencv-python`, `numpy`, `Pillow`, `matplotlib`, `torchvision`
- **CLIP_Surgery** 레포(혹은 동일 기능의 모듈)에서 제공하는:
  - `clip.load("CS-ViT-B/16", ...)` (수술된 비전 백본)
  - `encode_text_with_prompt_ensemble`
  - `clip_feature_surgery`
  - `get_similarity_map`
- 시각화할 이미지 파일 1장 (예: `dog.jpg`)

---

### 🧠 무엇을 하게 되나?

1. **Architecture Surgery**가 적용된 **CS-ViT-B/16** 모델로 이미지 토큰 특징을 뽑고,  
2. **Prompt Ensemble**로 클래스(텍스트) 특징을 안정적으로 만든 뒤,  
3. **Feature Surgery**로 **클래스 공통(중복) 특징을 제거**하여  
4. **foreground에 집중하는 similarity map**을 생성/시각화합니다.

---

### 🧪 바로 시작!! - 전체 코드 

```python
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
# from segment_anything import sam_model_registry, SamPredictor

# 0) 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) (선택) 원본 CLIP 로딩 — 비교용
model, _ = clip.load("ViT-B/16", device=device)
model.eval()

# 2) 전처리 파이프라인
preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711))
])

# 3) 입력 이미지 불러오기
pil_img = Image.open(f"/{my_path}/dog.jpg" )
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
image = preprocess(pil_img).unsqueeze(0).to(device)

# 4) 클래스 사전 — Feature Surgery는 여러 클래스를 함께 봐야 공통(중복) 성분을 제거할 수 있음
all_texts = [
    'airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat',
    'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
    'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door',
    'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard',
    'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
    'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign',
    'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck',
    'tv monitor', 'wall', 'water', 'window', 'wood'
]
target_texts = ['dog']

# 5) 수술된 아키텍처(CS-ViT-B/16) 로딩
model, preprocess_unused = clip.load("CS-ViT-B/16", device=device)
model.eval()

with torch.no_grad():
    # (A) 이미지 특징 (토큰 단위) — CLS 포함
    image_features = model.encode_image(image)                    # [B, 1+HW, C]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # (B) Prompt ensemble 기반 텍스트 특징 — 클래스별 안정화된 임베딩
    text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)  # [N, C]

    # (C) Feature Surgery — 클래스 공통(중복) 성분 제거된 유사도 계산
    similarity = clip.clip_feature_surgery(image_features, text_features)            # [B, 1+HW, N]

    # (D) 패치(위치)만 사용하여 similarity map 생성 + 원본 크기로 업샘플
    similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])  # [B, H, W, N]

    # (E) 시각화 — 타깃 클래스만 Overlay
    for b in range(similarity_map.shape[0]):
        for n in range(similarity_map.shape[-1]):
            if all_texts[n] not in target_texts:
                continue
            vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = cv2_img * 0.4 + vis * 0.6
            vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
            print('CLIP Surgery:', all_texts[n])
            plt.imshow(vis)
            plt.axis('off')
            plt.show()
```

위 코드를 해보리면!! 바로 dog에 대해서 segmentation을 잘 해버립니다!!  

![Image](https://github.com/user-attachments/assets/05f605c4-91ba-4e61-86a0-dba39b6f82a4)

---

### 🔍 코드 해설 (핵심 포인트)

#### 1) **CS-ViT-B/16**: Architecture Surgery가 적용된 백본 - clip_surgery_model.py에 바뀐 내용이있습니다! 세부 분석은 페이퍼 리딩 포스팅을 참고!!  
- 마지막 여러 블록에서 **q=k=v(=V)**로 바꿔 **Consistent Self-Attention**을 수행하고,
- **Dual Path**로 CAM용 경로는 **FFN을 스킵**하여 배경/노이즈 영향을 줄입니다.
- 원래(CLIP) 경로는 그대로 유지해 **본래 임베딩 성능 보존**.

#### 2) `clip.encode_text_with_prompt_ensemble` 에서 진행되는 Prompt Ensemble  
- `"a photo of a {}."` 하나로만 하면 반응이 다를수 있으니 그 외에도 다양한 템플릿으로 텍스트 임베딩을 평균/정규화 → **클래스 임베딩 안정화**.  
- 다양한 프롬포트는?? 아래와 같았습니다!!(clip.py에 있음!!)   
```python
prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

```
#### 3) Feature Surgery
- 이미지 토큰 × 텍스트 임베딩의 **원소곱**으로 클래스별 특징을 만들고,
- 클래스 축으로 **평균(= 공통/중복 성분)** 을 추정 후 **빼기** → **foreground에 집중**.
- 그 결과를 채널 합산하여 **[B, 1+HW, N]** 유사도 텐서 획득.  
- clip.py의 아래코드가 그 역할을!!  

```python
def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):

    if redundant_feats != None:
        similarity = image_features @ (text_features - redundant_feats).t()

    else:
        # weights to restrain influence of obvious classes on others
        prob = image_features[:, :1, :] @ text_features.t()
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)

        # element-wise multiplied features
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        feats *= w.reshape(1, 1, n_t, 1)
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats
        
        # sum the element-wise multiplied features as cosine similarity
        similarity = feats.sum(-1)

    return similarity

```
#### 4) Similarity Map 만들기
- CLS를 제외하고(위치 정보 없음) **패치 토큰(HW)** 만 사용.
- 채널별 Min–Max 정규화 후, 원본 이미지 크기로 **bilinear 업샘플**.
- 타깃 클래스(`dog`) 채널만 골라 **히트맵 오버레이**.


---

### 🧭 다만!!(Failure case찾기)   

> 여전히 객채 내부의 part segmentation은 되지 않았습니다!!
![Image](https://github.com/user-attachments/assets/153f82dc-027f-4120-95f6-81c565848f8f)

> 일부에서는 되는것 같기도!?
![Image](https://github.com/user-attachments/assets/5e4bada5-475d-40ac-adb4-6f6ee6d2febf)

- 즉 사진의 형태, 타겟 객체에 따라 다른가보아요~~  

---

### ✅ 정리  

이 실습을 통해 **추가 학습 없이**도 **CLIP의 설명가능성**을 크게 끌어올릴 수 있음을 확인했습니다.  
핵심은 **(i) Consistent Self-Attention + Dual Path**로 구조적 문제를 줄이고, **(ii) Feature Surgery**로 **중복 특징을 제거**해 foreground를 또렷하게 만드는 것입니다.  
