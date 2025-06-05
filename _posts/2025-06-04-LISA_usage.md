---
layout: post
title: "Reasoning Segmentation LLM LISA 실습!!"
author: [DrFirst]
date: 2025-06-04 09:00:00 +0900
categories: [AI, Experiment]
tags: [LISA, Segment Anything, Image Segmentation, Python]
sitemap :
  changefreq : weekly
  priority : 0.9
---


---

## 🦖(한국어) Reasoning Segmentation LLM LISA 실습!!
> **LISA**, 이름이 이쁘지요~!? `Large Language Instructed Segmentation Assistant` 의 약자라고합니다!

이번 포스팅은 이미지를 추론에 의하여 Segmentation 하는 **LISA** 모델 실습입니다!  
모델이 너무 신기해서 먼저 실습부터 하고!  
이론에 대하여 알아보아요~!!  
---

### 🧱 1. LISA Git Clone 

- [공식 Git 사이트](https://github.com/dvlab-research/LISA)에서 Repo를 Clone 합니다!!

```bash
git clone git@github.com:dvlab-research/LISA.git
```

---

### 📦 2. 가상환경에서의 필요 패키지 설치!!

저는 conda 가상환경에서 필요 패키지들을 설치했습니다!!

```bash
conda create -n lisa python=3.9 -y
conda activate lisa
```

이제, repo에서 제공하는 requirements를 설치해줍니다!

```python
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

여기서!! 중요한점은 `requirements.txt` 에서 제일 앞부분의   
`--extra-index-url https://download.pytorch.org/whl/cu117`  
부분을, 나의 cuda 버젼에 맞게 변경해주어야한다는 점입니다!!  

이제 설치 끝~~

---


### 🧊 3. LISA 모델 실행!!

시작에 앞서 제 환경은 1개의 `GeForce RTX 4090`, 24GB 입니다!!
이에 일반 추론 모델을 사용하면 아래와 같이 메모리 부족이 뜨게 되어요!  

``` bash
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1'
```

```bash
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 136.00 MiB. GPU 0 has a total capacity of 23.49 GiB of which 116.19 MiB is free. Including non-PyTorch memory, this process has 22.92 GiB memory in use. Of the allocated memory 22.54 GiB is allocated by PyTorch, and 2.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

그러나!! LISA는 친절하게 single 24G or 12G GPU 에서도 실행 가능한 경량화 모델을제공하기에,  
해당 모델로 실행하였습니다!!

``` bash
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1' --precision='fp16' --load_in_8bit
```

그럼~~ 짜잔!~! 이제 prompt 입력하라고 나옵니다~~

```bash
Please input your prompt: 
```

저는 칫솔 이미지를 대상으로 아래와 같이 이를 닦는 부분을 물어보았어요!  
`The part of a toothbrush used to remove food particles from teeth is called the bristles.`

그럼!! 

아래와 같이 답변이 나옵니다~~!

```bash
Please input your prompt: The part of a toothbrush used to remove food particles from teeth is called the bristles.
Please input the image path: /home/user/data/AGD20K/Seen/trainset/egocentric/brush_with/toothbrush/toothbrush_000127.jpg
text_output:  <s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start> <im_end> The part of a toothbrush used to remove food particles from teeth is called the bristles. ASSISTANT: Sure, [SEG] .</s>
./vis_output/The part of a toothbrush used to remove food particles from teeth is called the bristles. _toothbrush_000127_mask_0.jpg has been saved.
./vis_output/The part of a toothbrush used to remove food particles from teeth is called the bristles. _toothbrush_000127_masked_img_0.jpg has been saved.
```

살짝 이론내용을 공유하자면, 답변에 [SEG] 라는 표현이 있다면 별도의 mask 정보가 있다는 뜻이구요!
그래서 마지막 2줄과 같이 이미지가 저장되었다고 알려줍니다!!

이미지를 볼까요!?


![NOGPU]()

기존 사용했던 강아지 이미지를, bbox와 함께 segment해보았습니다!!

```python
img_name = "dog.jpg"

my_bboxes=[1430.2,   828,  4471.9, 3836.4]
# 박스 프롬프트로 추론 ([x_min, y_min, x_max, y_max])
results = model(img_name, bboxes=my_bboxes)

# 원본 이미지 로드 (시각화를 위해)
image = cv2.imread(img_name)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환

# 결과 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

# 마스크 오버레이
for result in results:
    if result.masks is not None:
        mask = result.masks.data[0].cpu().numpy()  # 첫 번째 마스크 추출
        plt.imshow(mask, alpha=0.5, cmap='jet')  # 마스크를 반투명하게 표시

# 박스 프롬프트 표시
rect = plt.Rectangle((my_bboxes[0], my_bboxes[1]), my_bboxes[2] - my_bboxes[0], my_bboxes[3] - my_bboxes[1], 
                     linewidth=2, edgecolor='red', facecolor='none', label=f'my_bboxes {my_bboxes}')
plt.gca().add_patch(rect)

# 제목 및 설정
plt.title(f"SAM2 Segmentation with Box Prompt on {img_name}")
plt.legend()
plt.axis('off')
plt.show()

# 추가 정보 출력 (선택 사항)
print("Segmentation Result:")
print(f"Number of masks: {len(results[0].masks.data)}")
print(f"Mask shape: {results[0].masks.data[0].shape}")
```

![sam2_dog](https://github.com/user-attachments/assets/9b4db05e-2577-4832-88c8-47ca66e21b82)


참 잘되죠~ 그런데 이건 SAM도 잘하건건데!?

---

### 🚀 4. 비디오 Segment 실행!!

그래서, 이번엔 SAM2의 특징인!  
비디오의 segment도 진행해보았습니다!

저는 고속도로의 CCTV영상을 바탕으로 진행했구요!
첫 프래임에서 차가 있는 곳의 위치(405,205)를 프롬포트로 제공했습니다!

```python
from ultralytics.models.sam import SAM2VideoPredictor

# Create SAM2VideoPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
predictor = SAM2VideoPredictor(overrides=overrides)

# Run inference with single point
results = predictor(source="street.mp4", points=[405, 205], labels=[1])
```

동영상을 올릴순 없지만!!  
아래 스크린샷같이 차가 사라지는 시점까지만 딱!!segment를 정말 잘하더라구요!!

![Image](https://github.com/user-attachments/assets/4a6135fb-077e-4b69-a4e7-982911ad263d)
![Image](https://github.com/user-attachments/assets/b908a14b-a65f-4a02-a52b-c088e736fbd7)
![Image](https://github.com/user-attachments/assets/d6a5b11c-b152-4d2c-97b0-841f345d9d48)

---

### 🎉 마무리

동영상의 segmentation에 더하여, 저는 Tracking이 이렇게 잘된다는것이 너무 인상적이었습니다!