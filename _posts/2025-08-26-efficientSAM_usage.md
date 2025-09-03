---
layout: post
title: "🧠 EfficientSAM Hands-On Practice!! : EfficientSAM 실습!! with Python"
author: [DrFirst]
date: 2025-08-26 09:00:00 +0900
categories: [Computer Vision, Experiment]
tags: [EfficientSAM, Segment Anything, Meta AI, Fine-tuning, Python, SAM, Vision]
sitemap:
  changefreq: weekly
  priority: 0.9
---

### 🧬 (한국어) EfficientSAM 실습!!

오늘은 [지난포스팅](https://drfirstlee.github.io/posts/efficientSAM/)에서 이론에 대하여 공부해보았던!  
가벼운 Segmentation 모델델! `LOEfficientSAMRA` 의 실습을 진행해보겠습니다!!  


> ✔️ 기본 실험 목적:  
> - GPU를 활용한 EfficientSAM의 사용  
> - Prompt 기반 Segmentation  
> - 이미지에 Box/Point로 Prompt를 줄 수 있음

---

### 🔧 1. 설치 및 셋업

EfficientSAM은 Hugging Face Transformers처럼 pip 패키지가 아니므로,  
GitHub에서 직접 클론하여 설치합니다.

```bash
# 1. EfficientSAM 레포 클론
git clone https://github.com/ChaoningZhang/EfficientSAM.git
cd EfficientSAM
```

그럼!! 친절히도 weights 폴더에 `efficient_sam_vits.pt.zip` 모델이 잘 저장되어있습니다~  
---

### 🖼️ 2. 이미지 세그멘테이션 기본 실습 - CPU&GPU!  

우선 git 에 저장된 `EfficientSAM_example.py` 를 실행해보면 잘 실행이 되고!  


```python
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# from squeeze_sam.build_squeeze_sam import build_squeeze_sam

from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile
    


models = {}

# Build the EfficientSAM-Ti model.
models['efficientsam_ti'] = build_efficient_sam_vitt()

# Since EfficientSAM-S checkpoint file is >100MB, we store the zip file.
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
# Build the EfficientSAM-S model.
models['efficientsam_s'] = build_efficient_sam_vits()

# Build the SqueezeSAM model.
# models['squeeze_sam'] = build_squeeze_sam()

# load an image
sample_image_np = np.array(Image.open("figs/examples/dogs.jpg"))
sample_image_tensor = transforms.ToTensor()(sample_image_np)
# Feed a few (x,y) points in the mask as input.

input_points = torch.tensor([[[[580, 350], [650, 350]]]])
input_labels = torch.tensor([[[1, 1]]])

# Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
for model_name, model in models.items():
    print('Running inference using ', model_name)
    predicted_logits, predicted_iou = model(
        sample_image_tensor[None, ...],
        input_points,
        input_labels,
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    # The masks are already sorted by their predicted IOUs.
    # The first dimension is the batch size (we have a single image. so it is 1).
    # The second dimension is the number of masks we want to generate (in this case, it is only 1)
    # The third dimension is the number of candidate masks output by the model.
    # For this demo we use the first mask.
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
    Image.fromarray(masked_image_np).save(f"figs/examples/dogs_{model_name}_mask.png")

# 확인용: 파라미터 장치
print("Model param device:", next(models['efficientsam_ti'].parameters()).device)
print("Image tensor device:", sample_image_tensor.device)

```

아래와 같이 강아지가 segmentation 된 것을 확인할 수 있습니다!!  

![Image](https://github.com/user-attachments/assets/72269b64-90a7-4d35-bbf9-9d39185189d3)

그런데!! Device가 CPU이기에~~\
아래와 같이 해보면~~ GPU 로 돌아간 로그 확인이 가능합니다!!

```python
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# from squeeze_sam.build_squeeze_sam import build_squeeze_sam

from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile
import contextlib

# -----------------------------------
# Device 설정
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------
# 모델 로드 (+ GPU 이동)
# -----------------------------------
models = {}

# EfficientSAM-Ti
models['efficientsam_ti'] = build_efficient_sam_vitt().to(device).eval()

# EfficientSAM-S (zip에서 가중치 추출 후)
with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    zip_ref.extractall("weights")
models['efficientsam_s'] = build_efficient_sam_vits().to(device).eval()

# SqueezeSAM (필요시)
# models['squeeze_sam'] = build_squeeze_sam().to(device).eval()

# -----------------------------------
# 입력 준비 (+ GPU 이동)
# -----------------------------------
# load an image
sample_image_np = np.array(Image.open("figs/examples/dogs.jpg"))
sample_image_tensor = transforms.ToTensor()(sample_image_np).to(device)  # [C,H,W] float32 on device

# Feed a few (x,y) points in the mask as input.
input_points = torch.tensor([[[[580, 350], [650, 350]]]], device=device)  # [B=1, N=1, K=2, 2]
input_labels = torch.tensor([[[1, 1]]], device=device)                    # [B=1, N=1, K=2]

# -----------------------------------
# 추론 (AMP는 CUDA일 때만)
# -----------------------------------
amp_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else contextlib.nullcontext()

for model_name, model in models.items():
    print('Running inference using', model_name)

    with torch.inference_mode(), amp_ctx:
        predicted_logits, predicted_iou = model(
            sample_image_tensor[None, ...],  # [1,C,H,W]
            input_points,
            input_labels,
        )

    # 정렬 및 상위 mask 선택
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(predicted_logits, sorted_ids[..., None, None], dim=2)

    # 첫 번째 후보 mask 사용
    mask = (predicted_logits[0, 0, 0] >= 0).to(torch.uint8).cpu().numpy()  # [H,W], uint8 (0/1)

    # 마스킹 후 저장
    masked_image_np = (sample_image_np.astype(np.uint8) * mask[:, :, None])
    Image.fromarray(masked_image_np).save(f"figs/examples/dogs_{model_name}_mask.png")

# 확인용: 파라미터 장치
print("Model param device:", next(models['efficientsam_ti'].parameters()).device)
print("Image tensor device:", sample_image_tensor.device)

```

결과로그~!~  

```text
Running inference using efficientsam_ti
Running inference using efficientsam_s
Model param device: cuda:0
Image tensor device: cuda:0
```

### 🧪 3. Prompt 를 바꿔가며 실험하기!!  

EfficientSAM은 **box** 또는 **point prompt** 기반으로 세그멘테이션을 수행합니다.  
먼저 가장 기본적인 point prompt 는 사실!! 위의 샘플코드에서 알수 있씁니다!!

바로 아래 부분인데요~~ 

```python
...
# Feed a few (x,y) points in the mask as input.
input_points = torch.tensor([[[[580, 350], [650, 350]]]], device=device)  # [B=1, N=1, K=2, 2]
input_labels = torch.tensor([[[1, 1]]], device=device)                    # [B=1, N=1, K=2]
...
```

위를 해석해보면 2개의 positive 점으로 진행한 것이고고

- [580, 350], [650, 350] → 유저가 클릭한 점 좌표.
- 1, 1 → positive point (이 영역 안쪽을 포함).

만약 점의 label이 0이었다면 negative point, 즉 영역의바깥을 의미하는것 입니다!!

그럼!! **bbox**는!!?  

```python
...
box_pts = np.array([[x1, y1], [x2, y2]], dtype=np.int64)
box_lbl = np.array([2, 3], dtype=np.int64)
input_points = torch.from_numpy(box_pts)[None, None, ...].to(device)   # [1,1,2,2]
input_labels = torch.from_numpy(box_lbl)[None, None, ...].to(device)   # [1,1,2]
...
```

위의 코드 형식으로

- 두 점을 주고 labels을 [2,3]로 설정하면 박스로 해석됩니다.  
- 이때, label 2 = 좌상단, label 3 = 우하단 점임을 의미하지요!!

---


---

### ⚙️ 4. 실전 응용~!!! groundingDINO + EfficientSAM 실험해보기  

- Open-Voca의 Object Detection 툴, `GroundingDINO` 기억나시죠!?  
- Groundingdino 세팅이 되어야하니 궁금하신분은 [예전 포스팅](https://drfirstlee.github.io/posts/groundingDINO_Detection_usage/)을 참고해주세요!!  
- 이 `GroundingDINO` 를 바탕으로 bbox를 만들고,  
- `EfficientSAM` 으로 Segmenetation 해보겠습니다!!  

```python

# 1. 패키지 및 모델 로드 & 변수 설정  
import os, contextlib, zipfile, numpy as np, torch, cv2
from PIL import Image
from torchvision import transforms
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from groundingdino.util.inference import load_model, load_image, predict

IMAGE_PATH = "{my_dir}/EfficientSAM_gdino/figs/examples/brush_with_toothbrush_000268.jpg"
TEXT_PROMPT = "toothbrush"

SAVE_PATH = "{my_dir}/EfficientSAM_gdino/figs/examples"
img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

GDINO_CONFIG  = "{my_dir}/EfficientSAM_gdino/grounding_dino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "{my_dir}/EfficientSAM_gdino/grounding_dino/weights/groundingdino_swint_ogc.pth"
EFSAM_S_ZIP   = "{my_dir}/EfficientSAM_gdino/weights/efficient_sam_vits.pt.zip"

OUTPUT_DIR = os.path.join(SAVE_PATH, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", device)

# ---------- GroundingDINO ----------
gdino = load_model(GDINO_CONFIG, GDINO_WEIGHTS)
image_source, image_pre = load_image(IMAGE_PATH)

# ---------- EfficientSAM ----------
models = {}
models["efficientsam_ti"] = build_efficient_sam_vitt().to(device).eval()
if os.path.isfile(EFSAM_S_ZIP):
    with zipfile.ZipFile(EFSAM_S_ZIP, "r") as zf: zf.extractall("weights")
models["efficientsam_s"]  = build_efficient_sam_vits().to(device).eval()

# ---------- 이미지 텐서 ----------
img_np = np.array(Image.open(IMAGE_PATH).convert("RGB"))
H, W = img_np.shape[:2]
img_tensor = transforms.ToTensor()(img_np).to(device)  # [C,H,W]


# 2. GDINO로 bbox 얻기 

# ---------- GDINO로 bbox 얻기 ----------
boxes_norm, logits, phrases = predict(
    model=gdino, image=image_pre, caption=TEXT_PROMPT,
    box_threshold=0.35, text_threshold=0.25
)
if len(boxes_norm) == 0:
    raise SystemExit("[GroundingDINO] no box")

top = int(torch.argmax(logits).item())

# boxes_norm[top] = (cx, cy, w, h)  in [0,1]
cxcywh = boxes_norm[top].detach().cpu()  # -> CPU tensor

# 텐서 연산으로 x1y1x2y2 만들고, round + int로 변환
x1, y1, x2, y2 = torch.tensor([
    (cxcywh[0] - cxcywh[2] / 2) * W,
    (cxcywh[1] - cxcywh[3] / 2) * H,
    (cxcywh[0] + cxcywh[2] / 2) * W,
    (cxcywh[1] + cxcywh[3] / 2) * H,
]).round().to(torch.int64).tolist()

# 좌표 정리/클램프
x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
print(f"[Box pixel] {x1},{y1} → {x2},{y2}")


# 3. bbox prompt로 EfficientSam에 넣기!!

 ---------- 여기서 '박스 프롬프트'로 변환 ----------
# SAM/ESAM 관례: 두 점을 주고 labels을 [2,3]로 설정하면 박스로 해석됩니다.
# label 2 = 좌상단, label 3 = 우하단
box_pts = np.array([[x1, y1], [x2, y2]], dtype=np.int64)
box_lbl = np.array([2, 3], dtype=np.int64)
input_points = torch.from_numpy(box_pts)[None, None, ...].to(device)   # [1,1,2,2]
input_labels = torch.from_numpy(box_lbl)[None, None, ...].to(device)   # [1,1,2]

# ---------- 선택: bbox/포인트 시각화 ----------
dbg = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).copy()
cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.circle(dbg, (x1,y1), 4, (255,0,0), -1)  # label=2
cv2.circle(dbg, (x2,y2), 4, (0,0,255), -1)  # label=3
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_bbox_prompt.jpg"), dbg)


# 4. 이미지 시각화

def clip_mask_to_bbox(mask, x1, y1, x2, y2):
    out = np.zeros_like(mask, dtype=mask.dtype)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out

def keep_largest_component(mask_uint8):
    num, labels = cv2.connectedComponents(mask_uint8.astype(np.uint8))
    if num <= 2: return mask_uint8
    areas = [(labels == i).sum() for i in range(1, num)]
    i_best = int(np.argmax(areas)) + 1
    return (labels == i_best).astype(np.uint8)

amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()

for name, esam in models.items():
    with torch.inference_mode(), amp_ctx:
        logits, iou = esam(img_tensor[None, ...], input_points, input_labels)  # box prompt
    order = torch.argsort(iou, dim=-1, descending=True)
    logits = torch.take_along_dim(logits, order[..., None, None], dim=2)
    cand = (logits[0,0] >= 0).to(torch.uint8).cpu().numpy()  # [K,H,W]

    # 상위 후보 중에서 bbox 안의 가장 큰 성분 선택(노이즈 억제)
    best = cand[0]
    best = clip_mask_to_bbox(best, x1, y1, x2, y2)
    best = keep_largest_component(best)

    # 저장
    masked = (img_np.astype(np.uint8) * best[:, :, None])
    Image.fromarray(masked).save(os.path.join(OUTPUT_DIR, f"{img_name}_{name}_mask.png"))
    overlay = img_np.copy()
    overlay[best == 1] = (0.6 * overlay[best == 1] + 0.4 * np.array([0,255,0], np.uint8)).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_{name}_overlay.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("done." , os.path.join(OUTPUT_DIR, f"{img_name}_{name}_overlay.jpg"))

```

![Image](https://github.com/user-attachments/assets/e0d95285-5151-4e49-914e-6c327b530eaa)  

위 과정을 통해서 bbox 및 Segmentation 결과를 잘 볼수 있습니다!!  

여러 모델과 연결되서 재미있는 프로젝트들을 할수 있겠네요~^^