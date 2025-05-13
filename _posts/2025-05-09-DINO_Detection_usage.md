---
layout: post
title: "Object Detection with DINO! Python Practice!! - DINO를 활용한 객체 탐지! 파이썬 실습!!"
author: [DrFirst]
date: 2025-05-12 07:00:00 +0900
categories: [AI, Experiment]
tags: [DETR, DINO, 객체 탐지, Object Detection, Transformer, 딥러닝, CV, ICLR, ICLR 2023]
lastmod : 2025-05-12 07:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---


## (한국어) 🦖 DINO 실습 - 모델을 받아 직접 객체 탐지 해보기!

> [지난 포스팅](https://drfirstlee.github.io/posts/DINO_Detection) 에서 DINO의 원리에 대하 공부해보았습니다!!!    
> 이젠, DINO 모델을 직접 설치하고 바탕으로 객체 탐지(Object Detection)을 진행해 보겠습니다~!   

![result](https://github.com/user-attachments/assets/eb9b3c27-7d8e-458a-afa3-936685f31b87)

- 오늘도도 결론부터!!!  
- DETR과 동일하게 이미지에서 탐지된 여러 객체들을 찾아서 보여줍니다!!  
- 함께, 파이썬 코드로 그 과정을 알아보아요!!  
- 이번에는 간단한 모델이 없어 git 에서 모델을 받고 설치해야합니다~ 잘 따라해보세요!!^^  



### 1. GIT 저장소에서 DINO 모델 받기!!

> 🔗 [공식 GitHub 저장소](https://github.com/IDEA-Research/DINO)


---

## 📦 1. 가상환경 및 의존성 설치

```bash
conda create --name DINO python=3.9
conda activate DINO

sudo apt update && sudo apt install -y build-essential python3-dev
pip install cython

conda install -c conda-forge libstdcxx-ng
conda install -c pytorch pytorch torchvision

pip install -r requirements.txt
```

---

## ⚙️ 2. 모델 코드 컴파일 및 테스트

```bash
cd models/dino/ops
python setup.py build install

# 유닛 테스트 - 모두 True가 출력되면 성공
python test.py

cd ../../..
```
---

## 🗂️ 3. COCO2017 데이터셋 준비

COCO2017 데이터를 아래 구조로 정리합니다:

```text
COCODIR/
├── train2017/
├── val2017/
└── annotations/
├── instances_train2017.json
└── instances_val2017.json
```

---

## 📥 4. 사전 학습된 모델 다운로드

DINO-4scale, Swin-L, 36 epoch 기준 모델은 아래 Google Drive 링크에서 다운로드합니다:

> 🔗 https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_?usp=sharing

---

## 🧠 5. 모델 로드 및 이미지 예측

### 패키지 임포트

```python
import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

from PIL import Image
import datasets.transforms as T
```

### COCO 클래스 ID 매핑 파일 로드

```python
with open('{알맞은위치}/DINO/util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}
```

### 모델 구성 및 체크포인트 로드

```python
model_config_path = "{알맞은위치}/DINO/config/DINO/DINO_4scale.py"
model_checkpoint_path = "{알맞은위치}/checkpoint0033_4scale.pth"

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)

checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()
```

### 이미지 불러오기 및 전처리

```python
image = Image.open("{이미지위치}/catch_rugby_ball_001480.jpg").convert("RGB")

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)
```

### 이미지 예측

```python
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
```

---

## 🖼️ 6. 결과 시각화

```python
thershold = 0.3  # 임계값 설정

vslzr = COCOVisualizer()

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}

vslzr.visualize(image, pred_dict, savedir=None, dpi=100)
```

하면~~ 짠 아래와 같이!! 잘 detection 하네요~~  

![result](https://github.com/user-attachments/assets/eb9b3c27-7d8e-458a-afa3-936685f31b87)

---

## ✅ 마무리

timm, huggingface 등에 모델이 업데이트 되지 않으면 만만치 않겠다라는 두려움이생기지만~,  
막상 해보면 다 할수 있습니다!  
특히 이번 DINO는 모델 업로드한 작성자가 깔끔하게 정리되어있어 더욱 쉽게할수 있었습니다!!  

앞으로도 여러 모델을 테스트해보겠습니다!!