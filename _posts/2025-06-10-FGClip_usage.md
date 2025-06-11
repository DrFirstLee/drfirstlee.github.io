---
layout: post
title: "🖥️ FG-Clip Practice!! : FG-Clip 실습!! with python"
author: [DrFirst]
date: 2025-06-10 09:00:00 +0900
categories: [AI, Experiment]
tags: [FG-Clip, Clip, fine-grained understanding, Python,  ICML, ICML 2025]
sitemap :
  changefreq : weekly
  priority : 0.9
---

---

## 🦖 FG-CLIP Practice!!  
> **FG-CLIP** : Fine-Grained Visual and Textual Alignment  

Today, we’ll walk through a hands-on session with the hot new model from ICML 2025:  
[FG-CLIP](https://arxiv.org/pdf/2505.05071)!  

---

### 🧱 1. Clone the FG-CLIP Git Repository  

- Clone the repo from the [official GitHub page](https://github.com/360CVGroup/FG-CLIP)!

```bash
git@github.com:360CVGroup/FG-CLIP.git
```

---

### 📦 2. Install Required Packages in Virtual Environment  

I used a `conda` virtual environment to install the required packages.  
I followed the instructions from the official GitHub repo!

```bash
conda create -n FGCLIP python=3.10 -y  
conda activate FGCLIP  
cd FG-CLIP && pip install -e .  
```

In addition, I installed the following packages separately  
(because I ran into some errors...):

```bash
pip install Pillow  
pip install matplotlib  
pip install torchvision --extra-index-url https://download.pytorch.org/whl/{insert-your-cu-version-here}  
```

---

### 🧊 3. Test the FG-CLIP Model!  

I tested it using the notebook environment provided by FG-CLIP.  
The code looks like this:

```python
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

model_root = "qihoo360/fg-clip-base"
image_size=224
model = AutoModelForCausalLM.from_pretrained(model_root,trust_remote_code=True).cuda()

device = model.device

tokenizer = AutoTokenizer.from_pretrained(model_root)
image_processor = AutoImageProcessor.from_pretrained(model_root)
```

Now the model downloads and loads successfully!

Let’s try with a sample image provided in the repo: `cat_dfclor.jpg`

![catcam](https://github.com/user-attachments/assets/e72f0746-0082-4696-a097-9197bbc88f93) 

```python
img_root = "cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")
image = image.resize((image_size,image_size))

image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

walk_short_pos = True
captions=["a photo of a cat", "a photo of a dog", "a photo of a animal"]
caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)

with torch.no_grad():
  image_feature = model.get_image_features(image_input)
  text_feature = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)
  image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
  text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

logits_per_image = image_feature @ text_feature.T 
logits_per_image = model.logit_scale.exp() * logits_per_image
probs = logits_per_image.softmax(dim=1) 
print(probs)
```

It outputs the similarity between the image and the three captions:  
["a photo of a cat", "a photo of a dog", "a photo of a animal"]

```
tensor([[9.6813e-01, 3.2603e-05, 3.1839e-02]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
```

As expected, “cat” scored the highest, followed by “animal”, and “dog” the lowest.  

But just seeing numbers isn’t enough—we want to **visualize** the similarity!  

```python
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

img_root = "cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")
image = image.resize((image_size,image_size))

image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

with torch.no_grad():
    dense_image_feature = model.get_image_dense_features(image_input)
    cap = "cat"
    captions = [cap]
    caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
    text_feature = model.get_text_features(caption_input,walk_short_pos=True)
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
similarity = similarity.cpu().numpy()
patch_size = int(math.sqrt(similarity.shape[0]))

original_shape = (patch_size, patch_size)
show_image = similarity.reshape(original_shape) 

plt.figure(figsize=(6, 6))
plt.imshow(show_image)
plt.title('similarity Visualization')
plt.axis('off')  
plt.savefig(f"{cap}_{img_root}.png")
```

This highlights the region associated with "cat"!  
You’ll see something like this:

> **black cat** → It really highlights the black cat only!  
![blackcat](https://github.com/user-attachments/assets/b9e926f0-1ace-44d0-935f-1c1265fe0e14)

> **keyboard** → Nice!  
![keyboard](https://github.com/user-attachments/assets/8e44fba3-4796-4e09-a5c6-ee09f478cde7)

> **blanket** and **chair** → These work to some extent too!  
![etc](https://github.com/user-attachments/assets/cc1a05b3-baad-4d71-a18d-381a28693ae7)


Then I tested a baseball stadium image:

> **hold a bat** → Works well!  
![holdbat](https://github.com/user-attachments/assets/5bfdc081-f25a-43f8-b04b-18dd4d497998)

> **player** → Seems pretty accurate!?  
![player](https://github.com/user-attachments/assets/3dd81666-9f69-42e0-b71f-b5e2725f6830)

> **catch** → Is it right...?
![catch](https://github.com/user-attachments/assets/d69491f5-12e7-43d5-b00c-6d765be88d62)

---

### 🔲 Let’s Try Bounding Boxes!  

Numbers and heatmaps are nice, but how about **bounding boxes**?  
Here's the code I used for BBox generation:

```python
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label  # Used to find adjacent regions

# --- 1. Load model and tokenizer ---
model_root = "qihoo360/fg-clip-base"
image_size = 224
model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).cuda()
device = model.device
tokenizer = AutoTokenizer.from_pretrained(model_root)
image_processor = AutoImageProcessor.from_pretrained(model_root)

# --- 2. Load image and set caption ---
img_root = "baseball_bat_000106.jpg"
cap = "handle"

try:
    image = Image.open(img_root).convert("RGB")
except FileNotFoundError:
    print(f"'{img_root}' not found. Generating a black image for fallback.")
    image = Image.new('RGB', (image_size, image_size), color = 'black')

image = image.resize((image_size, image_size))

# --- 3. Feature extraction and similarity calculation ---
image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)
with torch.no_grad():
    dense_image_feature = model.get_image_dense_features(image_input)
    captions = [cap]
    caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
    text_feature = model.get_text_features(caption_input, walk_short_pos=True)
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
similarity = similarity.cpu().numpy()

# --- 4. Group regions above average and calculate BBox coordinates ---
patch_size_in_grid = int(math.sqrt(similarity.shape[0]))
pixel_per_patch = image_size // patch_size_in_grid

# 1) Reshape similarity into a 2D grid
similarity_map = similarity.reshape((patch_size_in_grid, patch_size_in_grid))

# 2) Set a threshold using the average value
threshold = 0.22  # You can also try: np.mean(similarity) * 1.4

# 3) Create a binary mask where values above the threshold are True
print(f"threshold : {threshold}")
print(f"similarity_map : {similarity_map}")

mask = similarity_map > threshold
print("Mask shape:", mask.shape)

# 4) Label adjacent True regions (clusters) using scipy.ndimage.label
labeled_array, num_features = label(mask)

# --- 5. Draw BBox for each grouped region ---
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(image)

# Initialize overall bounding box coordinates
all_x1, all_y1 = float('inf'), float('inf')
all_x2, all_y2 = float('-inf'), float('-inf')

for i in range(1, num_features + 1):
    
    # Find all (row, col) indices for current label
    rows, cols = np.where(labeled_array == i)
    print(rows, cols)
    
    # Get min/max for current cluster
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Compute top-left and size of BBox
    bbox_x = min_col * pixel_per_patch
    bbox_y = min_row * pixel_per_patch
    bbox_w = (max_col - min_col + 1) * pixel_per_patch
    bbox_h = (max_row - min_row + 1) * pixel_per_patch
    print(bbox_x, bbox_y, bbox_w, bbox_h)

    # Create rectangle patch
    rect = patches.Rectangle(
        (bbox_x, bbox_y), bbox_w, bbox_h,
        linewidth=2, edgecolor='cyan', facecolor='none'
    )
    ax.add_patch(rect)

    cluster_sim_values = similarity_map[rows, cols]
    mean_similarity = np.mean(cluster_sim_values)

    # 🎯 Display label at the center
    center_col = bbox_x + bbox_w * 0.5
    center_row = bbox_y + bbox_h * 0.5
    ax.text(center_col, center_row,  f"{mean_similarity:.3f}", color='black', ha='center', va='center', fontsize=10, weight='bold')
    
print(f"num_features : {num_features}")
for i in range(1, num_features + 1):
    rows, cols = np.where(labeled_array == i)
    if len(rows) == 0:
        continue

    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    bbox_x1 = min_col * pixel_per_patch
    bbox_y1 = min_row * pixel_per_patch
    bbox_x2 = (max_col + 1) * pixel_per_patch
    bbox_y2 = (max_row + 1) * pixel_per_patch

    # Update overall BBox range
    all_x1 = min(all_x1, bbox_x1)
    all_y1 = min(all_y1, bbox_y1)
    all_x2 = max(all_x2, bbox_x2)
    all_y2 = max(all_y2, bbox_y2)

# Draw final merged BBox
final_w = all_x2 - all_x1
final_h = all_y2 - all_y1

rect = patches.Rectangle(
    (all_x1, all_y1), final_w, final_h,
    linewidth=3, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)

ax.set_title(f"Regions with Similarity > Average for: '{cap}' // threshold :{threshold}", fontsize=14)
ax.axis('off')
plt.savefig(f"bbox/bbox_multi_{cap}_{img_root}.png")
print(f"bbox/bbox_multi_{cap}_{img_root}.png")

```

This code lets you highlight the region corresponding to your caption—dynamically!

For example:  
> **handle** → Beautiful result!  
![handle](https://github.com/user-attachments/assets/d281fb4b-1d83-47f2-93e1-c3aa25097b12)

> **hit the ball** → Somewhat localized to the bat area!  
![hittheball](https://github.com/user-attachments/assets/acbe5bef-d947-4ff1-a9b8-0a24b837e015)

> **frisbee** → Kind of... interesting!  
![frisbee](https://github.com/user-attachments/assets/0174bfe1-8ef3-431e-9093-14f21f671745)

---

### 🎉 Wrapping Up

Today, we explored and tested the FG-CLIP model!  
In the next post, I’ll dive into how it actually works under the hood. Stay tuned!


---

## 🦖(한국어) FG-Clip 실습!!
> **FG-CLIP** : Fine-Grained Visual and Textual Alignment


오늘은 2025년 ICML에서 공개된 따끈따끈한 신모델,  
[FG-CLIP](https://arxiv.org/pdf/2505.05071) 에 대하여 실습을 진행해보겠습니다!!  

---

### 🧱 1. FG-CLIP Git Clone 

- [공식 Git 사이트](https://github.com/360CVGroup/FG-CLIP)에서 Repo를 Clone 합니다!!

```bash
git@github.com:360CVGroup/FG-CLIP.git
```

---

### 📦 2. 가상환경에서의 필요 패키지 설치!!

저는 conda 가상환경에서 필요 패키지들을 설치했습니다!!  
공식 git 에서 설명한대로 따라갔지요~!  

```bash
conda create -n FGCLIP python=3.10 -y
conda activate FGCLIP
cd FG-CLIP && pip install -e .
```

그 외에도 아래는 제가 별도로 설치해주었습니다!!  
(에러가 나더라구요!,,)  

```bash
pip install Pillow
pip install matplotlib
pip install torchvision --extra-index-url https://download.pytorch.org/whl/{여기는 각자 알맞은 cu 버젼으로!}
```
---

### 🧊 3. FG-CLIP 모델 테스트!!

저는 FGClip의 노트북 환경에서 테스트를 진행했습니다!!  
d선 아래와 같은 코드로 

```python
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

model_root = "qihoo360/fg-clip-base"
image_size=224
model = AutoModelForCausalLM.from_pretrained(model_root,trust_remote_code=True).cuda()

device = model.device

tokenizer = AutoTokenizer.from_pretrained(model_root)
image_processor = AutoImageProcessor.from_pretrained(model_root)
```

그럼!! 모델을 다운받아 load 가 됩니다!!  

이제! 기본적으로 제공된 이미지로 테스트해볼까요!?

![cat_dfclor.jpg](https://github.com/user-attachments/assets/49723476-d402-4a22-94c7-2d1b801b4aa4)

`cat_dfclor.jpg` 라는, repo에 있던 이미지로 아래와 같이 진행해보면!!


```python

img_root = "cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")
image = image.resize((image_size,image_size))

image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

# NOTE Short captions: max_length=77 && walk_short_pos=True
walk_short_pos = True
captions=["a photo of a cat", "a photo of a dog", "a photo of a animal"]
caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)

# NOTE Long captions: max_length=248 && walk_short_pos=False
# ......

with torch.no_grad():
  image_feature = model.get_image_features(image_input)
  text_feature = model.get_text_features(caption_input,walk_short_pos=walk_short_pos)
  image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
  text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

logits_per_image = image_feature @ text_feature.T 
logits_per_image = model.logit_scale.exp() * logits_per_image
probs = logits_per_image.softmax(dim=1) 
print(probs)
```

세개의 프롬포트 "["a photo of a cat", "a photo of a dog", "a photo of a animal"]" 에 대하여  
아래와 같이 각각의 연관성을 보여줍니다!

```
tensor([[9.6813e-01, 3.2603e-05, 3.1839e-02]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
```

cat이라는게 가장 높고, 그다음은 동물, 강아지는 가장 수치가 작게 내왔습니다!

그런데! 이렇게 숫자로 보는것 말고, 이미지로 바바야겠지요!?  

```python

import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


img_root = "cat_dfclor.jpg"
image = Image.open(img_root).convert("RGB")
image = image.resize((image_size,image_size))

image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

with torch.no_grad():
    dense_image_feature = model.get_image_dense_features(image_input)
    cap = "cat"
    captions = [cap]
    caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
    text_feature = model.get_text_features(caption_input,walk_short_pos=True)
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
similarity = similarity.cpu().numpy()
patch_size = int(math.sqrt(similarity.shape[0]))


original_shape = (patch_size, patch_size)
show_image = similarity.reshape(original_shape) 


plt.figure(figsize=(6, 6))
plt.imshow(show_image)
plt.title('similarity Visualization')
plt.axis('off')  
plt.savefig(f"{cap}_{img_root}.png")

```
그럼!! 

![catcam](https://github.com/user-attachments/assets/e72f0746-0082-4696-a097-9197bbc88f93) 

위와 같이 고양이 부분이 활성화됩니다!!

같은 그림에서 아래와 같이 추가테스트를 해보았어요!

> black cat : 정말 검정고양이 부분만 활성화됩니다!!  
![blackcat](https://github.com/user-attachments/assets/b9e926f0-1ace-44d0-935f-1c1265fe0e14)

>  keyboard : 굳!!  
![keyboard](https://github.com/user-attachments/assets/8e44fba3-4796-4e09-a5c6-ee09f478cde7)

> 그 외에 배경에있던 blanket과 chair. 어느정도 하는것 같네요!!  
![etc](https://github.com/user-attachments/assets/cc1a05b3-baad-4d71-a18d-381a28693ae7)

이번엔 많이 봤던 야구장 사진으로!  
> `hold a bat` ! 잘 하네요!!  
![holdbat](https://github.com/user-attachments/assets/5bfdc081-f25a-43f8-b04b-18dd4d497998)

> player 어느정도 맞는것 같은!?  
![player](https://github.com/user-attachments/assets/3dd81666-9f69-42e0-b71f-b5e2725f6830)

> catch  맞는걸까요!?..   
![catch](https://github.com/user-attachments/assets/d69491f5-12e7-43d5-b00c-6d765be88d62)


그런데!! 이렇게만 보는것은 너무 답답했습니다~   
한번 bbox를 해보는것은 어떨까요?  
이를위해 저는 별도의 아래 코드를 사용했습니다!  

```python
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label # 인접한 영역을 찾는 데 사용

# --- 1. 모델 및 토크나이저 로드 ---
model_root = "qihoo360/fg-clip-base"
image_size = 224
model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).cuda()
device = model.device
tokenizer = AutoTokenizer.from_pretrained(model_root)
image_processor = AutoImageProcessor.from_pretrained(model_root)

# --- 2. 이미지 로드 및 캡션 설정 ---
img_root = "baseball_bat_000106.jpg"
cap = "handle"

try:
    image = Image.open(img_root).convert("RGB")
except FileNotFoundError:
    print(f"'{img_root}' 파일을 찾을 수 없습니다. 실행을 위해 임의의 검은색 이미지를 생성합니다.")
    image = Image.new('RGB', (image_size, image_size), color = 'black')

image = image.resize((image_size, image_size))

# --- 3. 피처 추출 및 유사도 계산 ---
image_input = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)
with torch.no_grad():
    dense_image_feature = model.get_image_dense_features(image_input)
    captions = [cap]
    caption_input = torch.tensor(tokenizer(captions, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device)
    text_feature = model.get_text_features(caption_input, walk_short_pos=True)
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True)

similarity = dense_image_feature.squeeze() @ text_feature.squeeze().T
similarity = similarity.cpu().numpy()

# --- 4. 평균 이상 영역을 그룹화하고 BBox 좌표 계산 ---
patch_size_in_grid = int(math.sqrt(similarity.shape[0]))
pixel_per_patch = image_size // patch_size_in_grid

# 1) 유사도를 2D 그리드 형태로 변환
similarity_map = similarity.reshape((patch_size_in_grid, patch_size_in_grid))

# 2) 평균값을 임계값(threshold)으로 설정
threshold = 0.22#np.mean(similarity) * 1.4

# 3) 임계값보다 높은 점수를 가진 패치만 True로 표시하는 마스크 생성
print(f"threshold : {threshold}")
print(f"similarity_map : {similarity_map}")

mask = similarity_map > threshold
print("Mask shape:", mask.shape)
# 4) scipy.ndimage.label을 사용해 인접한 True 영역(클러스터)에 고유 번호(레이블)를 붙임
labeled_array, num_features = label(mask)

# --- 5. 그룹화된 각 영역에 BBox 그려서 시각화 ---
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(image)


# 누적용 전체 bbox 좌표 초기화
all_x1, all_y1 = float('inf'), float('inf')
all_x2, all_y2 = float('-inf'), float('-inf')


for i in range(1, num_features + 1):
    
    # 현재 레이블에 해당하는 모든 픽셀의 (행, 열) 좌표 찾기
    rows, cols = np.where(labeled_array == i)
    print(rows, cols)
    
    # 해당 클러스터를 감싸는 최소/최대 좌표 찾기
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # BBox의 좌측 상단 픽셀 좌표와 너비/높이 계산
    bbox_x = min_col * pixel_per_patch
    bbox_y = min_row * pixel_per_patch
    bbox_w = (max_col - min_col + 1) * pixel_per_patch
    bbox_h = (max_row - min_row + 1) * pixel_per_patch
    print(bbox_x, bbox_y, bbox_w, bbox_h)
    # BBox 사각형 생성
    rect = patches.Rectangle(
        (bbox_x, bbox_y), bbox_w, bbox_h,
        linewidth=2, edgecolor='cyan', facecolor='none'
    )
    ax.add_patch(rect)
    cluster_sim_values = similarity_map[rows, cols]
    mean_similarity = np.mean(cluster_sim_values)
    # 🎯 중심점에 라벨 텍스트 표시
    center_col = bbox_x + bbox_w*0.5
    center_row = bbox_y+ bbox_h*0.5
    ax.text(center_col, center_row,  f"{mean_similarity:.3f}", color='black', ha='center', va='center', fontsize=10, weight='bold')
    
print(f"num_features : {num_features}")
for i in range(1, num_features + 1):
    rows, cols = np.where(labeled_array == i)
    if len(rows) == 0:
        continue

    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    bbox_x1 = min_col * pixel_per_patch
    bbox_y1 = min_row * pixel_per_patch
    bbox_x2 = (max_col + 1) * pixel_per_patch
    bbox_y2 = (max_row + 1) * pixel_per_patch

    # 전체 범위 확장
    all_x1 = min(all_x1, bbox_x1)
    all_y1 = min(all_y1, bbox_y1)
    all_x2 = max(all_x2, bbox_x2)
    all_y2 = max(all_y2, bbox_y2)

# 최종 합쳐진 BBox 그리기
final_w = all_x2 - all_x1
final_h = all_y2 - all_y1

rect = patches.Rectangle(
    (all_x1, all_y1), final_w, final_h,
    linewidth=3, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)

ax.set_title(f"Regions with Similarity > Average for: '{cap}' // threshold :{threshold}", fontsize=14)
ax.axis('off')
plt.savefig(f"bbox/bbox_multi_{cap}_{img_root}.png")
print(f"bbox/bbox_multi_{cap}_{img_root}.png")
```

위의 코드에서 bbox를 위한 threshold는 제가,, 잘 맞게 수정했습니다!!

그 결과!!

> handle : 아주 좋죠!?
![handle](https://github.com/user-attachments/assets/d281fb4b-1d83-47f2-93e1-c3aa25097b12)

> hit the ball : 어느정도 방망이 부분인듯!?
![hittheball](https://github.com/user-attachments/assets/acbe5bef-d947-4ff1-a9b8-0a24b837e015)

> frisbee 
![frisbee](https://github.com/user-attachments/assets/0174bfe1-8ef3-431e-9093-14f21f671745)

---

### 🎉 마무리

오늘은 이렇게 FG-CLIP에 대하여 테스트해보았습니다!  
다음 포스팅에서 원리에 대하여 공부해보겠습니다!  