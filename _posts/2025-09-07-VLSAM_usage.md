---
layout: post
title: "🔎 VL-SAM Hands-on: VL-SAM 을 실습해보자!"
author: [DrFirst]
date: 2025-09-07 07:00:00 +0900
categories: [AI, Research]
tags: [Open-Vocabulary, Detection, Segmentation, SAM, VLM,python, Open-ended]
sitemap:
  changefreq: monthly
  priority: 0.8
---

---

### 🧬 (한국어) EfficientSAM 실습!!

오늘은 [예전포스팅](https://drfirstlee.github.io/posts/SAM2/)에서 이론에 대하여 공부해보았던!  
그리고 [실습도 해보았던](https://drfirstlee.github.io/posts/SAM2_usage/)
새로운 SAM 모델! `SAM2` 의 실습을 다시 한 번 진행해보겠습니다!!  
이번엔 지난번과 달리 `ultralytics` 가 아니라 huggingface에서 모델을 받아서고고!!  

> 그런데,, 이 SAM2, EfficientSAM보다는 결과물이 맘에들지 않구만유,,

---

### 🔧 1. 설치 및 셋업

GitHub에서 직접 클론하여 설치합니다.  
저는 그전에 sam2라는 가상환경을 만들고 진행했습니다!!  

```bash
conda create --name sam2 python=3.12
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

추가로 SAM2 git의 readme에서는 `./download_ckpts.sh`로 모델을 받으라고했지만,  
실제 코드는 `hf_hub_download`를 통해 weight를 받아오는 기능이 있기에 스킵!!

---

### 🖼️ 2. 이미지 세그멘테이션    

지난 [EfficientSAM 실습](https://drfirstlee.github.io/posts/efficientSAM_usage/)과 동일하게!! 멍멍이 사진으로 진행해보겠습니다!!  
프롬포트도~ 점을 2개만!!  

```python
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. 기본 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. SAM2 모델 불러오기
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# 3. 이미지 준비
image_path = "./EfficientSAM_gdino/figs/examples/dogs.jpg"
output_image_path = "output_masked_image.png"


image_pil = Image.open(image_path).convert("RGB")
image_np = np.array(image_pil)


# -----------------------------------------------------
# 4. 프롬프트 준비 
# -----------------------------------------------------
# input_points를 (batch, num_points, 2) 형태의 3D 텐서로 만듭니다.
input_points = torch.tensor([[[580, 350], [650, 350]]], device=device)
input_labels = torch.tensor([[1, 1]], device=device)

# input_labels = torch.tensor([[1, 1, 1, 1]], device=device) # 모든 점이 전경
# 5. 예측 실행
with torch.inference_mode(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
    )
mask = masks[0]
print(f"mask shape :{mask.shape}")

# 원본 이미지와 같은 크기의 완전히 검정색 NumPy 배열 생성
segmented_image_np = np.zeros_like(image_np)

# 마스크가 True인 영역에만 원본 이미지 픽셀을 복사
# mask는 boolean 배열 (True/False)이거나 0~1 사이의 float 배열일 수 있습니다.
# float 배열이라면 임계값을 적용하여 boolean 마스크로 만듭니다.
binary_mask = (mask > 0.5) # 0.5를 기준으로 True/False 마스크 생성

# 마스크가 True인 위치에 원본 이미지 픽셀을 할당
segmented_image_np[binary_mask] = image_np[binary_mask]

# NumPy 배열을 PIL Image로 변환
result_image = Image.fromarray(segmented_image_np)

# 프롬프트 점도 이미지 위에 그리기 (선택 사항)
draw_result = ImageDraw.Draw(result_image)
points_np = input_points[0].cpu().numpy()
labels_np = input_labels[0].cpu().numpy()

for i, (x, y) in enumerate(points_np):
    label = labels_np[i]

    fill_color = "green" if label == 1 else "red"
    outline_color = "white"
    radius = 5

    if label == 1:
        draw_result.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill_color, outline=outline_color, width=1)
    else:
        draw_result.line((x - radius, y - radius, x + radius, y + radius), fill=fill_color, width=2)
        draw_result.line((x + radius, y - radius, x - radius, y + radius), fill=fill_color, width=2)

result_image.save(output_image_path)
print(f"결과 이미지가 '{output_image_path}'에 저장되었습니다.")
```

결과 이미지는!!!??

![Image](https://github.com/user-attachments/assets/a175eae3-a5e8-463b-bb96-15dcd5c94c51)

완전 실망스러운데,,??  
그래서, 프롬포트 점을 4개로, 모델도 `sam2.1_hiera_large.pt`로 바꿔서 해보았는데!!

![Image](https://github.com/user-attachments/assets/83a1eb1c-3d3a-4b9a-812b-eb951fdd7ee4)

 그래도 결과가 실망스럽네요,,
GPT에 물어보니 프롬포트의 해석차이라고하는데,,  
저는 그래도 EfficientSAM이 좋네요!!

```text
EfficientSAM이 더 "잘" 했다기보다는, 사용자의 부정확한 프롬프트를 더 "관대하게" 해석해 준 것입니다.

반면에 SAM2는 훨씬 강력하고 정밀한 도구이기 때문에, 그 성능을 제대로 활용하려면 사용자도 더 정확한 프롬프트를 제공해야 합니다. 이전 답변에서 제안해 드린 것처럼 강아지들의 몸통과 머리에 여러 개의 점을 찍어주시면, SAM2가 EfficientSAM보다 훨씬 더 고품질의 정교한 마스크를 생성하는 것을 확인하실 수 있을 겁니다.
```
 

---

### 🧪 3. 영상 Segmentation    

![Image](https://github.com/user-attachments/assets/0ef426cb-262a-42ab-b714-d109844500b0)

결과부터!!! 맘에드는데요~~  

코드는 아래와 같습니다!

```python
import torch
import numpy as np
import cv2
import os
from sam2.sam2_video_predictor import SAM2VideoPredictor
from moviepy.editor import VideoFileClip

# 1. 기본 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 원본 및 출력 경로 설정
output_video_path = "output_segmented_video_50_55s.mp4"
clipped_video_path = "temp_clip.mp4"


# 3. 비디오 로딩
cap = cv2.VideoCapture(clipped_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

video_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
print(f"클립 로드 완료: {width}x{height}, {total_frames} 프레임, {fps:.2f} FPS")

# 4. SAM2 모델 로드
print("SAM2 비디오 예측기 모델을 로드합니다...")
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

# -----------------------------------------------------
# 5. 프롬프트 준비 (★★★★★ 이 부분이 수정되었습니다 ★★★★★)
# -----------------------------------------------------
prompt_frame_idx = 0  # 프롬프트를 적용할 프레임 인덱스 (0은 첫 번째 프레임)
prompt_obj_id = 1     # 추적할 객체의 고유 ID (첫 번째 객체이므로 1)

# 점 좌표: (NumPoints, 2) 형태의 NumPy 배열
points = np.array([[width // 2, height // 2]], dtype=np.float32)
# 레이블: (NumPoints,) 형태의 NumPy 배열
labels = np.array([1], dtype=np.int32)

# -----------------------------------------------------
# 6. 모델 초기화 및 첫 프레임 예측
# -----------------------------------------------------
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    print("예측기 상태를 초기화합니다...")
    state = predictor.init_state(clipped_video_path) 
    
    print("첫 번째 프레임에 프롬프트를 추가합니다...")
    # 예제 코드에 맞춰 함수 이름과 인자를 모두 변경
    _, _, masks = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=prompt_frame_idx,
        obj_id=prompt_obj_id,
        points=points,
        labels=labels,
    )
    # 7. 비디오 전파 및 결과 저장
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print("클립 전체에 마스크를 전파하고 결과를 저장합니다...")
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        original_frame = video_frames[frame_idx]
        # segmented_image_np = np.zeros_like(original_frame)
        # segmented_image_np = np.zeros_like(original_frame)
        segmented_image_np = np.full_like(original_frame, 255)

        # prompt_obj_id (1)가 추적되고 있는지 확인
        if prompt_obj_id in object_ids:
            mask_logits = masks[0][0].cpu().numpy()
            binary_mask_before_resize = (mask_logits > 0.0).astype(np.uint8)
            resized_mask = cv2.resize(binary_mask_before_resize, (width, height), interpolation=cv2.INTER_NEAREST)

            # 4. 리사이즈된 마스크를 boolean 타입으로 최종 변환
            boolean_mask = (resized_mask == 1)

            # 5. 올바른 마스크로 인덱싱
            segmented_image_np[boolean_mask] = original_frame[boolean_mask]
            # # 이제 shape이 일치하므로 정상적으로 작동합니다.
            # segmented_image_np[mask] = original_frame[mask]   
        # --- 추가된 부분: 모든 프레임에 프롬프트 점 그리기 ---
        for x, y in points:
            # cv2.circle을 사용하여 빨간색 점을 그립니다.
            # segmented_image_np는 RGB 상태이므로, 빨간색은 (255, 0, 0)입니다.
            # thickness=-1은 채워진 원을 의미합니다.
            cv2.circle(segmented_image_np, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)

        output_frame = cv2.cvtColor(segmented_image_np, cv2.COLOR_RGB2BGR)
        out_writer.write(output_frame)
        
        print(f"\r- 처리 중: 프레임 {frame_idx + 1}/{total_frames}", end="")

    out_writer.release()
    print(f"\n비디오 분할 완료! 결과가 '{output_video_path}'에 저장되었습니다.")

```

SAM2, 영상에서 트래이싱은 정말 맘에드는군요!^^