---
layout: post
title: "Image classification using ViT with Python - 파이썬으로 ViT 모델을 활용, 이미지 분류하기"
author: [DrFirst]
date: 2025-04-27 11:00:00 +0900
categories: [AI,Experiment]
tags: [ ViT,AI, Python,Deep Learning, Image Embedding, ViT-B/32, torchvision,vit-base-patch16-224]
lastmod : 2025-04-27 11:00:00
sitemap :
  changefreq : weekly
  priority : 0.9

---

## (English) Exploring Image Classification with ViT Model in Python

Hello everyone! 😊

In the [previous post](https://drfirstlee.github.io/posts/ViT/#image-you-can-do-transformer-too---the-emergence-of-vit-iclr-2021), we delved into the theory behind ViT based on the original paper! Today, we will actually download this ViT model and perform image classification in a Python environment!!

## 1. Importing ViT Model from torchvision! (The Simplest Way)

You can easily import the Vision Transformer (ViT) model through **torchvision**, a core library for image-related tasks in the PyTorch ecosystem.

### What kind of package is torchvision that provides models?

**torchvision** is a package developed and maintained by the PyTorch team, providing commonly used datasets, image transformations (transforms), and **pre-trained model architectures** in the field of computer vision.

torchvision provides models for the following reasons:

* **Convenience:** It supports researchers and developers in easily utilizing models with verified performance without the hassle of implementing image-related deep learning models from scratch.
* **Rapid Prototyping:** Pre-trained models allow for quick experimentation with new ideas and development of prototypes.
* **Saving Learning Resources:** Using models pre-trained on large-scale datasets saves the time and computational resources required for direct training.
* **Leveraging Learned Representations:** Pre-trained models have already learned general image features, enabling good performance on specific tasks with less data (transfer learning).

### Types and Features of ViT Models Provided by torchvision

torchvision provides various CNN-based models as well as ViT models. Currently (as of April 28, 2025), the main types and features of ViT models provided by torchvision are as follows:

| Name       | Patch Size | Model Name | Features                                                                                                                               |
| :--------- | :---------- | :--------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| ViT-Base   | 16x16      | vit_b_16   | Offers a balanced size and performance.                                                                                                |
| ViT-Base   | 32x32      | vit_b_32   | Larger patch size can reduce computation but may miss fine-grained features.                                                            |
| ViT-Large  | 16x16      | vit_l_16   | Has more layers and a larger hidden dimension than the Base model, aiming for higher performance. Requires more computational resources. |
| ViT-Large  | 32x32      | vit_l_32   | A Large model with a larger patch size.                                                                                                |
| ViT-Huge   | 14x14      | vit_h_14   | One of the largest ViT models, aiming for top-level performance but requires very significant computational resources.                     |

These models all come with pre-trained weights on the ImageNet dataset, allowing for immediate use in image classification tasks.  
The letters 'b', 'l', and 'h' in the model names indicate the Base, Large, and Huge model sizes, respectively, and the number following indicates the image patch size.  
A larger patch size means the model looks at the image in larger chunks, which can lead to faster processing but potentially lower accuracy.

---

## 2. Today's Image!! 🐶 Let's Start Classifying!

![dog](https://github.com/user-attachments/assets/0ad9326c-a64e-4d01-9e87-f53fe271c19a)

Today, we will use a cute dog image to see how the ViT model classifies it. The ViT model we will use today is pre-trained on the ImageNet dataset!

### What is imagenet\_classes?

`imagenet_classes` is a list of 1000 image classes used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). The pre-trained ViT models provided by torchvision are trained on this ImageNet dataset, so the model's output will be prediction probabilities for these 1000 classes. `imagenet_classes` serves to map these numerical prediction results to human-readable class names (e.g., "golden retriever", "poodle").

### imagenet\_classes.json: A JSON file containing imagenet\_classes information.

Since torchvision itself does not directly include the ImageNet class name list, you need to prepare a separate JSON file containing this information. You can obtain the `imagenet_classes.json` file in the following way:

```python
import requests
import json

# Read JSON file directly from URL
url = "[https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json](https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json)"

response = requests.get(url)
response.raise_for_status()  # Raise an error for bad status codes

# Load JSON data
imagenet_labels = response.json()

with open("imagenet_classes.json", "w") as f:
    json.dump(imagenet_labels, f)
```

## 3\. Let's Begin the Code\!\!

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json

# 1. Load ViT model (ViT-Base, patch size 16)
vit_b_16 = models.vit_b_16(pretrained=True)
vit_b_16.eval()  # Set the model to evaluation mode

# 2. Define image preprocessing
# Resize images to 256 and then center crop to 224.
# Normalize using the mean and standard deviation of the ImageNet dataset.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load the dog image (replace with your image file path)
image_path = "dog.jpg"
try:
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
except FileNotFoundError:
    print(f"Error: Image file '{image_path}' not found.")
    exit()

# 4. Perform prediction
with torch.no_grad():
    output = vit_b_16(input_tensor)

# 5. Post-process the prediction results and print the class names
try:
    with open("imagenet_classes.json", "r") as f:
        imagenet_classes = json.load(f)

    _, predicted_idx = torch.sort(output, dim=1, descending=True)
    top_k = 5
    print(f"Top {top_k} prediction results:")
    for i in range(top_k):
        class_idx = predicted_idx[0, i].item()
        confidence = torch.softmax(output, dim=1)[0, class_idx].item()
        print(f"- {imagenet_classes[class_idx]}: {confidence:.4f}")
except FileNotFoundError:
    print("Error: 'imagenet_classes.json' file not found. Please prepare the file in step 2.")
    print("Predicted class indices:", predicted_idx[0, :5].tolist())
except Exception as e:
    print(f"Error during prediction processing: {e}")
```

When you run the code above\!\!\! You can see the Top 5 prediction results as below\~!

```text
Top 5 Prediction Results:
- Golden Retriever: 0.9126
- Labrador Retriever: 0.0104
- Kuvasz: 0.0032
- Airedale Terrier: 0.0014
- tennis ball: 0.0012
```

We can see that the Golden Retriever is predicted with the highest probability of 91.26%.

## 4\. Getting and Running the Model Directly from Hugging Face\! + Analysis (Less Simple, But Customizable)

This time, let's try importing the model directly from the [Hugging Face ViT model](https://huggingface.co/google/vit-base-patch16-224) and proceed\!

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json

# 1. Load ViT model (ViT-Base, patch size 16)
vit_b_16 = models.vit_b_16(pretrained=True)
vit_b_16.eval()  # Set the model to evaluation mode

# 2. Define image preprocessing
# Resize images to 256 and then center crop to 224.
# Normalize using the mean and standard deviation of the ImageNet dataset.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load the dog image (replace with your image file path)
image_path = "dog.jpg"
try:
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
except FileNotFoundError:
    print(f"Error: Image file '{image_path}' not found.")
    exit()

# 4. Perform prediction
with torch.no_grad():
    output = vit_b_16(input_tensor)

# 5. Post-process the prediction results and print the class names
with open("imagenet_classes.json", "r") as f:
        imagenet_classes = json.load(f)

_, predicted_idx = torch.sort(output, dim=1, descending=True)
top_k = 5
print(f"Top {top_k} results:")
for i in range(top_k):
        class_idx = predicted_idx[0, i].item()
        confidence = torch.softmax(output, dim=1)[0, class_idx].item()
        print(f"- {imagenet_classes[class_idx]}: {confidence:.4f}")
```

Similarly, it was classified as number 207, Golden Retriever\!\!\!  
But\! Let's look at the differences from the existing torchvision and model customization here\!  

### a. Image Preprocessing Method\!\!

Looking at the preprocessing part below, `ViTFeatureExtractor` already knows the preprocessing method used when the model was trained, allowing you to perform image preprocessing simply without writing a complex `transforms.Compose` process directly\!

```python
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 3. preprocess : no need to  crop and resize
inputs = feature_extractor(images=image, return_tensors="pt")
```

### b. Viewing the CLS Token\!\!

In the previous theoretical learning post, we learned that it consists of 196 patches + 1 CLS token, totaling 197 patches\! We confirmed that the overall information of the image is contained in this first CLS token\! You can see the CLS Token with the following code\!



```python
from transformers import ViTModel, ViTImageProcessor
import torch
from PIL import Image

# 1. ViTModel (Pure model without classification head)
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
model.eval()

# Feature Extractor → Updated to ViTImageProcessor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# 2. Load Image
image = Image.open("dog.jpg").convert('RGB')
inputs = processor(images=image, return_tensors="pt")

# 3. Model Inference
with torch.no_grad():
    outputs = model(**inputs)

# 4. Extract CLS Token
last_hidden_state = outputs.last_hidden_state  # (batch_size, num_tokens, hidden_dim)
cls_token = last_hidden_state[:, 0, :]  # The 0th token is CLS

# 5. Print CLS Token
print("CLS token shape:", cls_token.shape)  # torch.Size([1, 768])
print("CLS token values (first 5):", cls_token[0, :5])
```

If you run the code above, you can see the 768-dimensional CLS token as expected\! Subsequent research uses this token for various other information\!

```text
CLS token shape: torch.Size([1, 768])
CLS token values (first 5): tensor([-0.5934, -0.3203, -0.0811,  0.3146, -0.7365])
```

### c. ViT's CAM\!\! Attention Rollout

In traditional CNN-based image classification, a CAM (Class Activation Map) was placed at the end of the model to visualize which parts became important\!\!\!

[CAM Theory Summary\!\!](https://drfirstlee.github.io/posts/CAM_research/)  
[CAM Practice\!\!](https://drfirstlee.github.io/posts/CAM_usage/)  

Our ViT model is different from CAM, so it's difficult to proceed in the same way\! However, you can visualize which of the remaining 196 patches the most important CLS package paid attention to using a method called **Attention Rollout**\!

Looking at the structure\!\!

As shown below, Attention is the process by which [CLS] assigns weights to each patch like "you're important" or "you're not important," and visualizing these attentions is Attention Rollout\!

```text
[CLS]   → Patch_1   (Attention weight: 0.05)
[CLS]   → Patch_2   (Attention weight: 0.02)
[CLS]   → Patch_3   (Attention weight: 0.01)
...
[CLS]   → Patch_196 (Attention weight: 0.03)
```

In the end\!\! You can see a visualization of which patches were considered important as below\!

  * Red areas → Patches that [CLS] paid much attention to.
  * Blue areas → Patches that [CLS] paid less attention to.

Looking at the code:


```python
from transformers import ViTModel, ViTFeatureExtractor
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# 1. Load model and Feature Extractor
model = ViTModel.from_pretrained('google/vit-base-patch16-224', output_attentions=True)
model.eval()

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 2. Load Image
image = Image.open("dog.jpg").convert('RGB')
inputs = feature_extractor(images=image, return_tensors="pt")

# 3. Model Inference (output attention)
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # list of (batch, heads, tokens, tokens)

# 4. Calculate Attention Rollout
def compute_rollout(attentions):
    # Multiply attention matrices across layers
    result = torch.eye(attentions[0].size(-1))
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1)[0]  # (tokens, tokens)
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1))
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(result, attention_heads_fused)
    return result

rollout = compute_rollout(attentions)

# 5. Extract Attention from [CLS] token to image patches
mask = rollout[0, 1:].reshape(14, 14).detach().cpu().numpy()

# 6. Visualization
def show_mask_on_image(img, mask):
    img = img.resize((224, 224))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.show()

show_mask_on_image(image, mask)

```
And the result is\!\!\!??

![patch](https://github.com/user-attachments/assets/82e9e668-d62a-4b06-9464-75e4eb3f967b)

Does it look right\~?

-----

## 5\. 💡 Conclusion: Simple and Fast ViT

How was it? You ran the code directly, and it was possible to execute the code easily and quickly\!

Like this, ViT, which was theoretically significant\! Since models trained on large-scale datasets can also be easily implemented in code, research based on Transformers has exploded in the field of computer vision ever since\!

In the future, we will also explore and practice various Vision Transformer-based models such as DINO, DeiT, CLIP, Swin Transformer, etc.! ^^

Thank you!!! 🚀🔥

---

## (한국어) 파이썬으로 ViT 모델을 활용, 이미지 분류하보기

안녕하세요! 😊  

[지난 포스팅](https://drfirstlee.github.io/posts/ViT/#image-you-can-do-transformer-too---the-emergence-of-vit-iclr-2021) 에서는 ViT의 Paper를 바탕으로 이론을 알아보았는데요!  
오늘은 실제 이 ViT델을 다운받아 Python 환경에서 이미지 분류 작업을 진행해보겠습니다!!  

## 1. ViT 모델!! torchvision 에서 임포트 하는 방식으로! (제일 간단)

PyTorch 생태계에서 이미지 관련 작업을 위한 핵심 라이브러리 중 하나인 **torchvision**을 통해 Vision Transformer (ViT) 모델을 간편하게 불러와 사용할 수 있습니다.

### torchvision 은 무슨 패키지이길래 모델을 제공해주나?

**torchvision**은 PyTorch 팀에서 개발하고 유지 관리하는 패키지로, 컴퓨터 비전 분야에서 자주 사용되는 데이터셋, 이미지 변환(transforms), 그리고 **미리 학습된(pre-trained) 모델 아키텍처**를 제공합니다.

### torchvision에서 제공하는 ViT 모델 종류와 각 모델의 특징

torchvision은 다양한 CNN 기반 모델뿐만 아니라 ViT 모델도 제공합니다. 현재 (2025년 4월 기준) torchvision에서 제공하는 주요 ViT 모델 종류와 특징은 다음과 같습니다.

| 이름       | 패치 사이즈 | 모델명      | 특징                                                                                                                               |
| :--------- | :---------- | :---------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| ViT-Base   | 16x16       | `vit_b_16`  | 균형 잡힌 크기와 성능을 제공합니다.                                                                                                  |
| ViT-Base   | 32x32       | `vit_b_32`  | 더 큰 패치 크기로 인해 계산량이 줄어들 수 있지만, 세밀한 특징을 놓칠 수 있습니다.                                                               |
| ViT-Large  | 16x16       | `vit_l_16`  | Base 모델보다 더 많은 레이어와 큰 hidden dimension을 가져 더 높은 성능을 목표로 합니다. 더 많은 컴퓨팅 자원을 요구합니다.           |
| ViT-Large  | 32x32       | `vit_l_32`  | Large 모델에 큰 패치 크기를 적용한 모델입니다.                                                                                     |
| ViT-Huge   | 14x14       | `vit_h_14`  | 가장 큰 ViT 모델 중 하나로, 최고 수준의 성능을 목표로 하지만 매우 많은 컴퓨팅 자원을 필요로 합니다.                                      |

이러한 모델들은 모두 ImageNet 데이터셋으로 사전 학습된 가중치와 함께 제공되어, 이미지 분류 작업에 바로 활용할 수 있습니다.  
모델 이름의 `b`, `l`, `h`는 각각 Base, Large, Huge 모델 크기를 나타내며, 뒤의 숫자는 이미지 패치의 크기를 의미합니다.
패치 크기가 클수록 이미지를 크게크게 보는것이니 속도는 빠르지만 정확도가 낮겠지요!?

---


## 2. 오늘의 이미지!! 🐶  분류 시작!

![dog](https://github.com/user-attachments/assets/0ad9326c-a64e-4d01-9e87-f53fe271c19a)
 
오늘은 귀여운 강아지 이미지를 사용하여 ViT 모델이 어떻게 이미지를 분류하는지 확인해보겠습니다.  
그리고 오늘의 ViT 모델은 Imagenet의 데이터셋으로 학숩된 모델을 활용할 예정입니다!!  


### imagenet_classes 이란?

`imagenet_classes`는 ImageNet Large Scale Visual Recognition Challenge (ILSVRC)에서 사용된 1000개의 이미지 클래스 목록입니다.  
torchvision에서 제공하는 사전 학습된 ViT 모델은 이 ImageNet 데이터셋으로 학습되었기 때문에, 모델의 출력은 이 1000개의 클래스에 대한 예측 확률로 나타납니다. 
`imagenet_classes`는 이러한 숫자 형태의 예측 결과를 사람이 이해할 수 있는 클래스 이름(예: "golden retriever", "poodle")으로 매핑해주는 역할을 합니다.

### imagenet_classes.json : imagenet_classes 정보를 저장한 json 입니다. 

torchvision 자체에는 ImageNet 클래스 이름 목록이 직접 포함되어 있지 않기에,  
해당 정보를 담고 있는 JSON 파일을 별도로 준비해야 합니다. 다음 방법으로 `imagenet_classes.json` 파일을 얻을 수 있습니다.

```python
import requests
import json

# URL에서 직접 JSON 파일 읽기
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

response = requests.get(url)
response.raise_for_status()  # 요청 실패 시 에러 발생

# JSON 데이터 로드
imagenet_labels = response.json()


with open("imagenet_classes.json", "r") as f:
    imagenet_classes = json.load(f)
```

## 3. 코드 본격 시작!!

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json

# 1. ViT 모델 불러오기 (ViT-Base, 패치 크기 16 사용)
vit_b_16 = models.vit_b_16(pretrained=True)
vit_b_16.eval()  # 추론 모드로 설정

# 2. 이미지 전처리 정의
# 이미지 크기가 다 다르니 256으로 리사이즈하고 224로 중앙 부분을 패치합니다.
# 그리고 ImageNet 데이터셋의 평균과 표준편차로 정규화합니다.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 강아지 이미지 불러오기 (본인의 이미지 파일 경로로 변경해주세요)
image_path = "dog.jpg"
try:
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # 배치 차원 추가
except FileNotFoundError:
    print(f"Error: 이미지 파일 '{image_path}'을 찾을 수 없습니다.")
    exit()

# 4. 모델에 입력하여 예측 수행
with torch.no_grad():
    output = vit_b_16(input_tensor)

# 5. 예측 결과 후처리 및 클래스 이름 출력
try:
    with open("imagenet_classes.json", "r") as f:
        imagenet_classes = json.load(f)

    _, predicted_idx = torch.sort(output, dim=1, descending=True)
    top_k = 5
    print(f"Top {top_k} 예측 결과:")
    for i in range(top_k):
        class_idx = predicted_idx[0, i].item()
        confidence = torch.softmax(output, dim=1)[0, class_idx].item()
        print(f"- {imagenet_classes[class_idx]}: {confidence:.4f}")

except FileNotFoundError:
    print("Error: 'imagenet_classes.json' 파일을 찾을 수 없습니다. 2단계에서 파일을 준비해주세요.")
    print("예측된 클래스 인덱스:", predicted_idx[0, :5].tolist())
except Exception as e:
    print(f"Error during prediction processing: {e}")
```

위 코드를 실행하면!!!  
아래와 같이  Top 5개의 예측결과를 볼수 있는데요~!

```text
Top 5 예측 결과:
- Golden Retriever: 0.9126
- Labrador Retriever: 0.0104
- Kuvasz: 0.0032
- Airedale Terrier: 0.0014
- tennis ball: 0.0012
```

골든리트리버를 91.26%로 가장 높은 확률로 예측함을 볼수 있었습니다


## 4. Huggingface 에서 직접 모델을 받아서 실행하기! + 분석 (덜 간단, but 커스터마이징 가능)

이번에는 직접 [허깅페이스의 ViT 모델](https://huggingface.co/google/vit-base-patch16-224)로부터 직접  
모델을 임포트하여 진행해보겠습니다~!  

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json

# 1. ViT 모델 불러오기 (ViT-Base, 패치 크기 16 사용)
vit_b_16 = models.vit_b_16(pretrained=True)
vit_b_16.eval()  # 추론 모드로 설정

# 2. 이미지 전처리 정의
# 이미지 크기가 다 다르니 256으로 리사이즈하고 224로 중앙 부분을 패치합니다.
# 그리고 ImageNet 데이터셋의 평균과 표준편차로 정규화합니다.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 강아지 이미지 불러오기 (본인의 이미지 파일 경로로 변경해주세요)
image_path = "dog.jpg"
try:
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # 배치 차원 추가
except FileNotFoundError:
    print(f"Error: 이미지 파일 '{image_path}'을 찾을 수 없습니다.")
    exit()

# 4. 모델에 입력하여 예측 수행
with torch.no_grad():
    output = vit_b_16(input_tensor)

# 5. 예측 결과 후처리 및 클래스 이름 출력
with open("imagenet_classes.json", "r") as f:
       imagenet_classes = json.load(f)

_, predicted_idx = torch.sort(output, dim=1, descending=True)
top_k = 5
print(f"Top {top_k} 예측 결과:")
for i in range(top_k):
       class_idx = predicted_idx[0, i].item()
       confidence = torch.softmax(output, dim=1)[0, class_idx].item()
       print(f"- {imagenet_classes[class_idx]}: {confidence:.4f}")


```

역시 마찬가지로~!! 207번, 골든 리트리버로 구분되었습니다!!!  
그런데! 여기서의 기존 torchvision과 차이 & 모델 커스터마이징 등을 알아보겠습니다!!

### a. 이미지의 전처리방식!!

아래의 전처리 부분을 보면, ViTFeatureExtractor는 해당 모델이 학습될 때 사용했던 전처리 방식을 미리 알고 있어,  
복잡한 transforms.Compose 과정을 직접 작성하지 않고 간단하게 이미지 전처리를 수행할 수 있게 해준답니다~!!

```python
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 3. 전처리 : 직접 crop 및 resize 할 필요가 없어요!
inputs = feature_extractor(images=image, return_tensors="pt")
```

### b. CLS 토큰 보기!!

지난 이론 학습글에서 196개의 패치 + 1개의 CLS 토큰으로 197개의 패치로 구성됨을 알아보았는데요~!  
이 첫번쨰의 CLS 토큰에 이미지의 전체적인 정보가 포함됨을 확인했었습니다!!  
아래와 같은 코드로 CLS Token을 볼 수 있습니다!!  


```python
from transformers import ViTModel, ViTImageProcessor
import torch
from PIL import Image

# 1. ViTModel (Classification head 없는 순수 모델)
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
model.eval()

# Feature Extractor → ViTImageProcessor로 최신화
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# 2. 이미지 불러오기
image = Image.open("dog.jpg").convert('RGB')
inputs = processor(images=image, return_tensors="pt")

# 3. 모델 추론
with torch.no_grad():
    outputs = model(**inputs)

# 4. CLS 토큰 추출
last_hidden_state = outputs.last_hidden_state  # (batch_size, num_tokens, hidden_dim)
cls_token = last_hidden_state[:, 0, :]  # 0번째 토큰이 CLS

# 5. CLS 토큰 출력
print("CLS token shape:", cls_token.shape)  # torch.Size([1, 768])
print("CLS token values (앞 5개):", cls_token[0, :5])
```

위 코드를 실행해보면, 예상한대로 768 차원의CLS 토큰을 볼수 있지요~~  
이후 여러 연구들은 이 토큰을 활용해서 다른 정보로 활용하기도합니다!   

```text 
CLS token shape: torch.Size([1, 768])
CLS token values (앞 5개): tensor([-0.5934, -0.3203, -0.0811,  0.3146, -0.7365])
```

### c. ViT의 CAM!! Attention Rollout  

기존 CNN 방식의 이미지 분류는 모델의 마지막단에 CAM(Class Activation Map)을 두어서 어떤 부분이 중요하게 되었는지 시각화 할수 있었습니다!!!  

[CAM의 이론 정리!!](https://drfirstlee.github.io/posts/CAM_research/)  
[CAM 실습!!](https://drfirstlee.github.io/posts/CAM_usage/) 

우리의 ViT 모델은 CAM과는 다르기에 동일한 방식으로 진행은 어렵지만!!  
**Attention Rollout** 이라는 방식으로 가장 중요한 CLS 패키치가 나머지 196개 패치중 어디를 중요하게 봤는지!! 시각화할수 있어요!!  

구조를 보자면!!  

아래와 같이  [CLS]가 각 패치에 대해 "너 중요해", "너 별로야" 같은 식으로 가중치를 부여하는 걸 Attention이라고하고, 그 어텐션들을 시각화하는것이지요!

```text
[CLS]   → Patch_1   (Attention weight: 0.05)
[CLS]   → Patch_2   (Attention weight: 0.02)
[CLS]   → Patch_3   (Attention weight: 0.01)
...
[CLS]   → Patch_196 (Attention weight: 0.03)
```

결국!! 어떤 패치가 중요하게 간주되었는지 아래와 같이 시각화가 되지요~!!

- 빨갛게 보이는 영역 → [CLS]가 많이 주목한 패치,  
- 파랗게 보이는 영역 → [CLS]가 덜 주목한 패치

코드로 보면~~

```python
from transformers import ViTModel, ViTFeatureExtractor
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# 1. 모델과 Feature Extractor 불러오기
model = ViTModel.from_pretrained('google/vit-base-patch16-224', output_attentions=True)
model.eval()

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 2. 이미지 불러오기
image = Image.open("dog.jpg").convert('RGB')
inputs = feature_extractor(images=image, return_tensors="pt")

# 3. 모델 추론 (attention 출력)
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # list of (batch, heads, tokens, tokens)

# 4. Attention Rollout 계산
def compute_rollout(attentions):
    # Multiply attention matrices across layers
    result = torch.eye(attentions[0].size(-1))
    for attention in attentions:
        attention_heads_fused = attention.mean(dim=1)[0]  # (tokens, tokens)
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1))
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.matmul(result, attention_heads_fused)
    return result

rollout = compute_rollout(attentions)

# 5. [CLS] 토큰에서 이미지 패치로 가는 Attention 추출
mask = rollout[0, 1:].reshape(14, 14).detach().cpu().numpy()

# 6. 시각화
def show_mask_on_image(img, mask):
    img = img.resize((224, 224))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.show()

show_mask_on_image(image, mask)

```

이고 그 결과는!!!??

![patch](https://github.com/user-attachments/assets/82e9e668-d62a-4b06-9464-75e4eb3f967b)

입니다~! 맞는것 같나요~?

---

## 5. 💡 결론 : 간단하고 빠른 ViT

어떤가요? 코드를 직접 실행해보았는데~!!  
큰 어려움없이, 그리고 빠르게 코드를 실행할수 있었지요!?

이처럼 이론적으로도 유의미했던 ViT! 
대규모 데이터셋에서 학습된 모델이 코드로도 쉽게 구현이 가능해서 이후로 컴퓨터 비전 분야에서 Transformer 기반 연구가 폭발적으로 증가하게 되었다고합니다!!  

앞으로 DINO, DeiT, CLIP, Swin Transformer 등 다양한 비전 Transformer 기반의 모델도 알아보며 실습해볼 수 있도록 하겠습니다~! ^^

감사합니다!!! 🚀🔥
