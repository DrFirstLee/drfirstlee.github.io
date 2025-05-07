---
layout: post
title: "DETR을 활용한 객채 탐지! 파이썬 실습!!"
author: [DrFirst]
date: 2025-05-08 07:00:00 +0900
categories: [AI, Experiment]
tags: [CVPR, CVPR 2020, Python, Detr, Object Detection, Huggingface,  detr-resnet-50]
lastmod : 2025-05-08 07:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---

## (English) Object Detection with DETR! Python Practice!!

> In the [previous post](https://drfirstlee.github.io/posts/DETR/) we studied DETR!!
> Today, based on this DETR model, we will directly perform Object Detection!

![detr_result](https://github.com/user-attachments/assets/cfb15e15-999d-4fc7-a1f2-5fa3f3f674bc)

- Let's start with the conclusion again!!!
- It finds and shows multiple detected objects in the image!!
- It accurately displays many people and frisbees, along with their accuracy!!
- Let's explore the process together with Python code!!

### 1. Loading the DETR model from Hugging Face!!

> Today's DETR model will be loaded from Hugging Face, using the `facebook/detr-resnet-50` model.

```python
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


# 1️⃣ Set device (use CUDA if GPU is available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ Load DETR model and processor (pretrained model)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
```

#### **processor** : 🖼️ Image Processor (DetrImageProcessor)

**Role:** To **preprocess** the input image into a format that the DETR model can effectively understand and process.

**Main Tasks:**

1.  **Image Resizing:** Changes the size of the input image to a specific size required by the model.
2.  **Image Normalization:** Adjusts the pixel values of the image to a specific range to improve the stability of model training and inference.
3.  **Tensor Conversion:** Converts the image into a tensor format that can be used by deep learning frameworks such as PyTorch.
4.  **Handling Model-Specific Requirements:** Performs additional preprocessing tasks according to the model architecture (e.g., mask generation).

If we actually check the internal workings of the processor, we can see the preprocessing steps as below:

```text
DetrImageProcessor {
  "do_convert_annotations": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "format": "coco_detection",
  "image_mean": [
    0.485,
    0.456,
    0.406
  ],
  "image_processor_type": "DetrImageProcessor",
  "image_std": [
    0.229,
    0.224,
    0.225
  ],
  "pad_size": null,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "longest_edge": 1333,
    "shortest_edge": 800
  }
}
```


#### **model** : 🤖 DETR Object Detection Model (DetrForObjectDetection)

**Role:** To perform **object detection** on the preprocessed image and predict the location and class of objects within the image. This is the **core role**.

**Main Tasks:**

1.  **Feature Extraction:** Extracts important visual features for object detection from the input image.
2.  **Transformer Encoder-Decoder:** Processes the extracted features through the Transformer structure to understand the relationships between objects in the image and learn information about each object.
3.  **Object Prediction:** Finally outputs the **bounding box coordinates**, the corresponding **class labels**, and the **confidence scores** of the detected objects in the image.

The DETR model is structured as shown below:

![detr_model](https://github.com/user-attachments/assets/93a59c85-f6b3-484a-ab68-97360bcbda92)

#### 2. Starting Object Detection with DETR!

> It's done with just a few lines of simple code!!!

I have prepared an image above where several people are playing with a frisbee!
And then!

```python
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# 1️⃣ Set device (use CUDA if GPU is available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ Load DETR model and processor (pretrained model)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 3️⃣ Load the bike.jpg image from the local directory
image_path = "catch_frisbee.jpg"
image = Image.open(image_path)

# 4️⃣ Preprocess the image (convert to DETR model input format)
inputs = processor(images=image, return_tensors="pt").to(device)

# 5️⃣ Model inference
with torch.no_grad():
    outputs = model(**inputs)

# 6️⃣ Post-process the results (convert Bounding Box & Labels)
target_sizes = torch.tensor([image.size[::-1]])  # (height, width) format
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# 7️⃣ Output detected objects
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.7:  # Output objects with confidence above 70%
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at {box}")

```

If we briefly analyze the code above:
- It loads the model.
- It loads the `catch_frisbee` image!
- It preprocesses it through the `processor`.
- It puts it into the `model` and performs inference!
- It prints the detected content from `results`!

Then the output is!
As shown below! It tells us the detected objects, their accuracy (confidence), and finally the bounding box coordinates!


```text
Detected person with confidence 0.783 at [12.91, 355.33, 32.23, 383.66]
Detected person with confidence 0.999 at [279.08, 255.76, 365.66, 423.82]
Detected person with confidence 0.995 at [533.57, 280.23, 584.71, 401.82]
Detected umbrella with confidence 0.744 at [459.41, 324.56, 496.24, 340.89]
Detected person with confidence 0.933 at [488.93, 340.06, 510.23, 376.37]
Detected person with confidence 0.835 at [0.01, 355.79, 11.03, 384.31]
Detected person with confidence 0.906 at [261.05, 346.35, 284.02, 378.22]
Detected person with confidence 0.99 at [574.15, 301.1, 605.79, 395.45]
Detected person with confidence 0.713 at [244.5, 349.68, 262.29, 378.9]
Detected person with confidence 0.997 at [132.21, 31.6, 310.32, 329.97]
Detected person with confidence 0.732 at [349.66, 352.63, 365.67, 378.28]
Detected person with confidence 0.796 at [209.17, 326.9, 232.89, 355.65]
Detected person with confidence 0.777 at [149.0, 347.84, 169.28, 381.43]
Detected person with confidence 0.991 at [163.45, 299.99, 206.14, 399.0]
Detected frisbee with confidence 1.0 at [181.55, 139.33, 225.96, 161.49]
Detected person with confidence 0.734 at [200.95, 350.37, 229.14, 380.88]
Detected person with confidence 0.737 at [467.46, 347.11, 483.07, 376.49]
Detected person with confidence 0.978 at [413.58, 253.38, 465.11, 416.57]
Detected person with confidence 0.73 at [597.38, 342.37, 613.34, 380.89]
Detected person with confidence 0.998 at [304.64, 70.92, 538.5, 410.45]
```

#### 3. Visualization of Object Detection Results!! (Decoding)

> Instead of simple text detection, let's display bounding boxes on the image!

```python
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


# 1️⃣ Set device (use CUDA if GPU is available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ Load DETR model and processor (pretrained model)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 3️⃣ Load the bike.jpg image from the local directory
image_path = "catch_frisbee.jpg"
image = Image.open(image_path)

# 4️⃣ Preprocess the image (convert to DETR model input format)
inputs = processor(images=image, return_tensors="pt").to(device)

# 5️⃣ Model inference
with torch.no_grad():
    outputs = model(**inputs)

# 6️⃣ Post-process the results (convert Bounding Box & Labels)
target_sizes = torch.tensor([image.size[::-1]])  # (height, width) format
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]


# 7️⃣ Visualize detected objects with Bounding Boxes on the image
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.imshow(image)

# Draw Bounding Boxes
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.7:  # 🔹 Visualize objects with confidence above 70%
        box = [round(i, 2) for i in box.tolist()]
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}",
                fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

# 8️⃣ Save the result
output_path = "detr_output.jpg"  # 🔹 Filename to save
plt.axis("off")  # 🔹 Remove axes
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"Detection result saved as {output_path}")
```
Through the code above,   
The detected objects are visualized,   
And saved as `detr_output.jpg`!!  

![detr_model](https://github.com/user-attachments/assets/93a59c85-f6b3-484a-ab68-97360bcbda92)

Object detection, it's really easy, right?  
However, it takes 8.5 seconds to detect objects in a single image... it's still a bit slow!  

---

## (한국어) DETR을 활용한 객채 탐지! 파이썬 실습!!

> [지난 포스팅](https://drfirstlee.github.io/posts/DETR/) 에서 공부해보았던 DETR!!  
> 오늘은 이 DETR 모델을 바탕으로 직접 객채 탐지(Object Detection)을 진행해 알아보겠습니다~!   

![detr_result](https://github.com/user-attachments/assets/cfb15e15-999d-4fc7-a1f2-5fa3f3f674bc)

- 오늘도도 결론부터!!!  
- 이미지에서 탐지된 여러 객채들을 찾아서 보여줍니다!!  
- 많은 사람들과 프리스비 등 객채를 정확도와 함께 보여줍니다!!   
- 함께, 파이썬 코드로 그 과정을 알아보아요!!  

### 1. huggingface에서 DETR 모델 받기!!

> 오늘의 DETR 모델은  Huggingface로부터, `facebook/detr-resnet-50` 모델을 받아 진행해보겠습니다.

```python
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


# 1️⃣ 디바이스 설정 (GPU 사용 가능하면 CUDA로 설정)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ DETR 모델 및 프로세서 로드 (사전 학습된 모델)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

```

위 코드를 보면, 사전 학습된 `facebook/detr-resnet-50`의 Model과 Processor 를 로드하는데요~!  
각각의 역할을 알아보자면!  

#### **processor** : 🖼️ 이미지 프로세서 (DetrImageProcessor)

**역할:** 입력 이미지를 DETR 모델이 효과적으로 이해하고 처리할 수 있는 형태로 **전처리(Preprocessing)**하는 역할

**주요 작업:**

1.  **이미지 크기 조정 (Resizing):** 입력 이미지의 크기를 모델이 요구하는 특정 크기로 변경  
2.  **이미지 정규화 (Normalization):** 이미지 픽셀 값을 특정 범위로 조정하여 모델 학습 및 추론 안정성 향상  
3.  **텐서 변환 (Tensor Conversion):** 이미지를 파이토치(PyTorch)와 같은 딥러닝 프레임워크에서 사용할 수 있는 텐서(Tensor) 형태 변환  
4.  **모델별 요구 사항 처리:** 모델 아키텍처에 따라 추가적인 전처리 작업 (예: 마스크 생성 등)을 수행  

실제로 processor 를 내부를 확인해보면 아래와 같이 전처리 과정을 볼수 있습니다~  

```text
DetrImageProcessor {
  "do_convert_annotations": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "format": "coco_detection",
  "image_mean": [
    0.485,
    0.456,
    0.406
  ],
  "image_processor_type": "DetrImageProcessor",
  "image_std": [
    0.229,
    0.224,
    0.225
  ],
  "pad_size": null,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "longest_edge": 1333,
    "shortest_edge": 800
  }
}
```

#### **model** : 🤖 DETR 객체 감지 모델 (DetrForObjectDetection)

**역할:** 전처리된 이미지를 입력받아 이미지 내의 **객체를 감지(Object Detection)**하고, 해당 객체의 위치와 클래스를 예측하는 **핵심적인 역할** 수행  

**주요 작업:**

1.  **이미지 특징 추출 (Feature Extraction):** 입력 이미지에서 객체 감지에 중요한 시각적 특징들을 추출  
2.  **트랜스포머 인코더-디코더 (Transformer Encoder-Decoder):** 추출된 특징들을 트랜스포머 구조를 통해 처리하여 이미지 내 객체 간의 관계를 파악하고, 각 객체의 정보를 학습  
3.  **객체 예측 (Object Prediction):** 최종적으로 이미지 내에 존재하는 객체들의 **바운딩 박스 좌표**, 해당 객체의 **클래스 레이블**, 그리고 예측의 **신뢰도 점수** 출력

아래와 같이 DETR으 모델로 구성됨을 볼수 있습니다!!

![detr_model](https://github.com/user-attachments/assets/93a59c85-f6b3-484a-ab68-97360bcbda92)


#### 2. DETR로 객채탐지 시작!  

> 간단한 코드 몇줄이면 끝!!!

![Image](https://github.com/user-attachments/assets/71d72287-c14f-4d54-9fe9-4401a7fa4e1c) 

위와 같이 여러사람들이 프리스비로 놀고있는 이미지를 준비해보았습니다!!
그리고@!  

```python
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# 1️⃣ 디바이스 설정 (GPU 사용 가능하면 CUDA로 설정)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ DETR 모델 및 프로세서 로드 (사전 학습된 모델)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 3️⃣ 로컬 디렉토리의 bike.jpg 이미지 로드
image_path = "catch_frisbee.jpg"
image = Image.open(image_path)

# 4️⃣ 이미지 전처리 (DETR 모델 입력 형태로 변환)
inputs = processor(images=image, return_tensors="pt").to(device)

# 5️⃣ 모델 추론
with torch.no_grad():
    outputs = model(**inputs)

# 6️⃣ 결과 후처리 (Bounding Box & Labels 변환)
target_sizes = torch.tensor([image.size[::-1]])  # (height, width) 형식
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# 7️⃣ 감지된 객체 출력
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.7:  # 신뢰도 70% 이상인 객체만 출력
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at {box}")

```

위의 코드를 간단하게 분석해보면,  
- 모델을 로드하고  
- `catch_frisbee` 이미지를 로드하고!  
- `processor` 를 통해 전처리하고,  
- `model`에 넣어서!! 추론한 뒤.  
- `results` 에서 탐지된 내용 print 하기!!

그럼 그 output은!!  
아래와 같이!! 탐지된 객채와, 그 정확도(confidence), 마지막으로 바운딩 박스 좌표를 알려줍니다@!

```text
Detected person with confidence 0.783 at [12.91, 355.33, 32.23, 383.66]
Detected person with confidence 0.999 at [279.08, 255.76, 365.66, 423.82]
Detected person with confidence 0.995 at [533.57, 280.23, 584.71, 401.82]
Detected umbrella with confidence 0.744 at [459.41, 324.56, 496.24, 340.89]
Detected person with confidence 0.933 at [488.93, 340.06, 510.23, 376.37]
Detected person with confidence 0.835 at [0.01, 355.79, 11.03, 384.31]
Detected person with confidence 0.906 at [261.05, 346.35, 284.02, 378.22]
Detected person with confidence 0.99 at [574.15, 301.1, 605.79, 395.45]
Detected person with confidence 0.713 at [244.5, 349.68, 262.29, 378.9]
Detected person with confidence 0.997 at [132.21, 31.6, 310.32, 329.97]
Detected person with confidence 0.732 at [349.66, 352.63, 365.67, 378.28]
Detected person with confidence 0.796 at [209.17, 326.9, 232.89, 355.65]
Detected person with confidence 0.777 at [149.0, 347.84, 169.28, 381.43]
Detected person with confidence 0.991 at [163.45, 299.99, 206.14, 399.0]
Detected frisbee with confidence 1.0 at [181.55, 139.33, 225.96, 161.49]
Detected person with confidence 0.734 at [200.95, 350.37, 229.14, 380.88]
Detected person with confidence 0.737 at [467.46, 347.11, 483.07, 376.49]
Detected person with confidence 0.978 at [413.58, 253.38, 465.11, 416.57]
Detected person with confidence 0.73 at [597.38, 342.37, 613.34, 380.89]
Detected person with confidence 0.998 at [304.64, 70.92, 538.5, 410.45]
```

#### 3. 객채 탐지결과물의 시각화!!(디코딩)   

> 단순 텍스트 탐지가 아니라 그림에 바운딩박스로 표시해봅니다!@  

```python
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


# 1️⃣ 디바이스 설정 (GPU 사용 가능하면 CUDA로 설정)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2️⃣ DETR 모델 및 프로세서 로드 (사전 학습된 모델)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 3️⃣ 로컬 디렉토리의 bike.jpg 이미지 로드
image_path = "catch_frisbee.jpg"
image = Image.open(image_path)

# 4️⃣ 이미지 전처리 (DETR 모델 입력 형태로 변환)
inputs = processor(images=image, return_tensors="pt").to(device)

# 5️⃣ 모델 추론
with torch.no_grad():
    outputs = model(**inputs)

# 6️⃣ 결과 후처리 (Bounding Box & Labels 변환)
target_sizes = torch.tensor([image.size[::-1]])  # (height, width) 형식
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]


# 7️⃣ 감지된 객체를 이미지에 Bounding Box로 시각화
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.imshow(image)

# Bounding Box 그리기
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.7:  # 🔹 신뢰도 70% 이상인 객체만 시각화
        box = [round(i, 2) for i in box.tolist()]
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w-x, h-y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}",
                fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

# 8️⃣ 결과 저장
output_path = "detr_output.jpg"  # 🔹 저장할 파일명
plt.axis("off")  # 🔹 축 제거
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"Detection result saved as {output_path}")

```

위 코드를 통하여,  
감지된 객채를 시각화하고  
`detr_output.jpg` 로도 저장하게됩니다~!!  

![detr_model](https://github.com/user-attachments/assets/93a59c85-f6b3-484a-ab68-97360bcbda92)


객채 탐지, 참 쉽죠~?  
다만, 1개 이미지에서 객채 탐지에 시간이 8.5초가 소요,, 역시 좀 오래걸리네요!  
