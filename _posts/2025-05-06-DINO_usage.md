---
layout: post
title: "DINO Python Experiment!! Super Impressive!! - DINO 파이썬 실습!! 완전 신기해!!"
author: [DrFirst]
date: 2025-05-06 09:00:00 +0900
categories: [AI, Experiment]
tags: [ViT, Vision Transformer, AI, ICCV, ICCV 2021,DINO, timm, Python ]
lastmod : 2025-05-06 09:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---

## (English) DINO Python Experiment!! So Cool!!

> In the [previous post](https://drfirstlee.github.io/posts/DINO/), we learned the theory behind DINO!!  
> Today, let’s actually run the DINO model and see how it performs~!

![dino_result](https://github.com/user-attachments/assets/71580e84-ee1a-4571-ba64-0d4f776de11d)

- Starting with the conclusion today!!!  
- It highlights important parts of the image with a tada~ using attention  
- Isn't that amazing!?  
- Let’s explore how it works~!

---

#### 1. What is timm?!!

> In this post, we’ll load the DINO model using `timm`.  
> Let’s first understand what `timm` (Torch Image Models) is!

- **timm** stands for **Torch Image Models**,  
- A **library** that provides a wide array of tools and pretrained models for handling image tasks in PyTorch!!

**Main features of timm:**

* **Offers various modern image models:**  
  - Includes ResNet, EfficientNet, Vision Transformer (ViT), Swin Transformer, and more — easily usable for image classification, detection, semantic segmentation, etc.  
* **Rich pretrained weights:**  
  - Provides weights pretrained on large datasets such as ImageNet, JFT, BeiT, which makes transfer learning easier without the need for training from scratch  
* **Easy model creation:**  
  - With `timm.create_model()`, you can create your desired model by name + conveniently load pretrained weights  
* **Modular design:**  
  - Easily access and modify components like backbone, pooling layer, classifier head — highly flexible for building custom models or fine-tuning existing ones  
* **Various utility functions:**  
  - Offers helpful tools for image transforms, dataset handling, optimizers, schedulers, etc.  
* **Active community:**  
  - An open-source project actively maintained and continuously updated with new models and features  

* **Example of using timm**

```
import timm

# USE DINO-ViT MODEL (pretrained)
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval()
```

This loads the DINO model structure!!  
(We’ll focus on hands-on today — for architecture details, please check the theory post~)

---

#### 2. Encoding Images with ViT-based DINO!! (Into Vectors)

> The core idea of ViT is turning an image into a vector using Transformer techniques!!

![hold_fork](https://github.com/user-attachments/assets/1b8536c2-fbdc-4eba-96b0-f2f7d1cd4728)

Let’s start with this image of someone holding a fork!!  
And now@!

```
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import timm
import torchvision.transforms as T

# USE DINO-ViT MODEL (pretrained)
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval()

# Load image 
image_path = "hold_fork.jpg" 
image = Image.open(image_path).convert('RGB')  

# Image preprocess
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)

# Model output with attention weights
with torch.no_grad():
    outputs = model.forward_features(img_tensor)  # Shape: (batch_size, 197, feature_dim)

np.shape(outputs)
```

Breaking down the code above:  
- Load the model  
- Load the image as RGB vector  
- Preprocess and normalize the image to (224, 224)  
- Feed it into DINO → Get final output!!

And the output will be:

```text
torch.Size([1, 197, 768])
```

So the output is a vector of shape 197 (1 CLS token + 196 patch tokens) × 768 (DINO’s internal dimension)!!

That’s the end of the image encoding process!!!!  
You can now analyze each patch token or the CLS token depending on your purpose~~!!

---

#### 3. Visualizing the Encoded Output!! (Decoding)

> The result is in vector form — great for computers,  
> but hard for us to interpret, right?  
> Let’s decode it so we can actually **see** it!

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import timm
import torchvision.transforms as T

# USE DINO-ViT MODEL (pretrained)
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval()

# Load image 
image_path = "hold_fork.jpg" 
image = Image.open(image_path).convert('RGB')  

# Image preprocess
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)

# Model output with attention weights
with torch.no_grad():
    # Get features including attention
    outputs = model.forward_features(img_tensor)  # Shape: (batch_size, 197, feature_dim)
    
    # Extract patch tokens (excluding CLS)
    patch_tokens = outputs[:, 1:, :]  # (batch_size, 196, feature_dim)
    
    # Attention map: compute importance using norm of patch tokens
    attn_map = torch.norm(patch_tokens, dim=-1).reshape(14, 14)  # (14x14)
    
    # Normalize (scale to range 0–1)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

# Visualize full Attention Map
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Original image
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Attention Map
attn_map_resized = np.array(Image.fromarray(attn_map.numpy()).resize(image.size, resample=Image.Resampling.BILINEAR))
ax[1].imshow(image)
ax[1].imshow(attn_map_resized, cmap='jet', alpha=0.5)  # Attention Map > heat map
ax[1].axis('off')
ax[1].set_title('DINO-ViT Attention Map')

plt.tight_layout()
plt.show()
```

This code builds upon the previous one by adding visualization!  
The most important part is:

```
# Model output with attention weights
with torch.no_grad():
    outputs = model.forward_features(img_tensor)  # Shape: (batch_size, 197, feature_dim)

    # Extract patch tokens (exclude CLS)
    patch_tokens = outputs[:, 1:, :]  # (batch_size, 196, feature_dim)
    
    # Attention map: compute importance via patch token norm
    attn_map = torch.norm(patch_tokens, dim=-1).reshape(14, 14)  # (14x14)
    
    # Normalize (scale to 0–1)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
```

This part excludes 1 out of the 197 patch outputs.  
Understanding why is essential!!

`That’s because we must exclude the CLS token!!`

Once visualized, you get the same result as shown at the beginning of this post:

![dino_result](https://github.com/user-attachments/assets/71580e84-ee1a-4571-ba64-0d4f776de11d)

The DINO model, trained **without any labels**,  
intelligently identifies and highlights important regions in red,  
while marking less important ones in blue!

---

#### 4. Conclusion!!

With DINO, it’s incredibly easy to turn images into vectors and **visualize** them!!  
Building and training the model may have been tough,  
but actually using it is super simple and impressive!!  
Definitely something we should remember and leverage in future research~! 😊

Also, big thanks to `timm` for making model usage so convenient!  
It supports not just DINO, but many other models as well!

```
timm.list_models()
```

You can use this to see the long list of available models~  
*In my version, over 1,200 models are available!*  
*I also saw resnet, swin, RegNet, EfficientNet — looks like I need to study those too!!*


---
## (한국어) DINO 파이썬 실습!! 완전 신기해!!

> [지난 포스팅](https://drfirstlee.github.io/posts/DINO/) 에서 배웠던 DINO 이론!!  
> 오늘은 그 DINO 모델을 실제로 가동해보고 그 결과가 어떻게 나오는지 알아보겠습니다~!   

![dino_result](https://github.com/user-attachments/assets/71580e84-ee1a-4571-ba64-0d4f776de11d)

- 오늘은 결론부터!!!  
- 이미지에 대하여 중요한 부분을 짜짠 하고 attention을 줍니다~  
- 신기하지 않나요!?  
- 그 과정을 알아보겠습니다~!  

#### 1. timm 이란?!!

> 이번 포스팅에서의 DINO 모델은  timm 으로부터 로드하고자합니다.
> 그 timm (Torch Image Models) 이 무엇인지 알아봅시다!~!
 
- **timm**은 **Torch Image Models**의 약자로, 
- PyTorch에서 이미지 모델을 다루는 데 유용한 다양한 도구와 사전 학습된 모델을 제공하는 **라이브러리**!!



**timm의 주요 특징:**

* **다양한 최신 이미지 모델 제공:** 
  - ResNet, EfficientNet, Vision Transformer (ViT), Swin Transformer 등 최신 CNN 및 Transformer 기반의 다양한 이미지 분류, 객체 검출, 의미론적 분할 모델 구조를 쉽게 사용  가능  
* **풍부한 사전 학습된 가중치 (Pretrained Weights):** 
  - ImageNet, JFT, BeiT 등 대규모 데이터셋으로 사전 학습된 모델 가중치를 제공, 사용자가 직접 모델을 학습시키는 부담을 줄이고 전이 학습(Transfer Learning)을 용이하게 함  
* **간편한 모델 생성:** 
  - `timm.create_model()` 함수를 통해 모델 이름만으로 원하는 모델을 쉽게 생성 가능!! + 사전 학습된 가중치를 로드하는 옵션도 간편하게 제공  
* **모듈화된 설계:** 
  - 모델의 각 구성 요소 (백본, 풀링 레이어, 분류 헤더 등)를 쉽게 접근하고 수정할 수 있도록 설계되어, 사용자 정의 모델을 구축하거나 기존 모델을 fine-tuning하는 데 유연성 제공  
* **다양한 유틸리티 함수:** 
  - 이미지 변환(transform), 데이터셋 처리, 최적화기(optimizer), 스케줄러(scheduler) 등 이미지 모델 학습 및 평가에 필요한 다양한 유틸리티 함수 제공  
* **활발한 커뮤니티:** 
  - 오픈 소스 프로젝트로 활발한 커뮤니티 지원을 받으며 지속적으로 새로운 모델과 기능 지속 추가  

* **앞으로 사용할 timm 예시**

```python
import timm

# USE DINO-ViT MODEL (pretrained)
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval()
```

위의 모델 로드를 통하여 dino 모델의 구조를 볼수 있지요~~  
(오늘은 실습으로 구조에 대한 자세한 내용은 이론 포스팅에서 확인해주세요~~)

#### 2. ViT 인 DINO로 이미지 인코팅!! (벡터로 만들기)  

> ViT의 기본 개념은 이미지를 Transformer 방식을 통해 벡터로 만드는것!!

![hold_fork](https://github.com/user-attachments/assets/1b8536c2-fbdc-4eba-96b0-f2f7d1cd4728)

우선 위와 같이 포크를 쥐고있는 이미지를 준비해보았습니다!!
그리고@!  

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import timm
import torchvision.transforms as T

# USE DINO-ViT MODEL (pretrained)
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval()

# Load image 
image_path = "hold_fork.jpg" 
image = Image.open(image_path).convert('RGB')  

# Image preprocess
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)

# Model output with attention weights
with torch.no_grad():
    # Get features including attention
    outputs = model.forward_features(img_tensor)  # Shape: (batch_size, 197, feature_dim)
    
np.shape(outputs)
```

위의 코드를 간단하게 분석해보면,  
- 모델을 로드하고  
- 이미지를 RGB 벡터 값으로 로드하고!  
- DINO에 넣을수 있도록 구조를 바꿔주고! - (224,224) 사이즈에 정규화!  
- 모델에 넣어서!!! 최종 output 만들기!!

그럼 그 output은!!

```text
torch.Size([1, 197, 768])
```

로서, 197 (1개의 CLS + 196개의 패치) X 768(DINO 자체의 차원 ) 의 구조를 가진 벡터로 나오게 됩니다!!

이게 바로 이미지 인코딩의 끝!!!!  
이후 이 벡터의 각각의 패치 혹은 CLS 값으로 분석을 진행할 수 있지요~~!!  

#### 3. 인코딩결과물(벡터)의 시각화!!(디코딩)   

> 벡터로만 결과가 나오면, 컴퓨터는 이해할수 있지만,  
> 우리는 이해하기가 힘들죠?  
> 디코딩 하여 우리가 볼수 있도록 만들어 봅시다!!  

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import timm
import torchvision.transforms as T

# USE DINO-ViT MODEL (pretrained)
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval()

# Load image 
image_path = "hold_fork.jpg" 
image = Image.open(image_path).convert('RGB')  

# Image preprocess
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(image).unsqueeze(0)

# Model output with attention weights
with torch.no_grad():
    # Get features including attention
    outputs = model.forward_features(img_tensor)  # Shape: (batch_size, 197, feature_dim)
    
    # Extract patch tokens (CLS 제외)
    patch_tokens = outputs[:, 1:, :]  # (batch_size, 196, feature_dim)
    
    # Attention map: 패치 토큰의 노름(norm)을 사용해 중요도 계산
    attn_map = torch.norm(patch_tokens, dim=-1).reshape(14, 14)  # (14x14)
    
    # 정규화 (0~1 범위로 스케일링)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

# Attention Map 전체 시각화
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Original image
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Attention Map
attn_map_resized = np.array(Image.fromarray(attn_map.numpy()).resize(image.size, resample=Image.Resampling.BILINEAR))
ax[1].imshow(image)
ax[1].imshow(attn_map_resized, cmap='jet', alpha=0.5)  # Attention Map > heat map
ax[1].axis('off')
ax[1].set_title('DINO-ViT Attention Map')

plt.tight_layout()
plt.show()
```

이번 코드는, 앞의 코드에 이어 시각화 부분이 추가되었습니다~!
여기서 중요한것은!!!   

```python
# Model output with attention weights
with torch.no_grad():
    # Get features including attention
    outputs = model.forward_features(img_tensor)  # Shape: (batch_size, 197, feature_dim)
      # Extract patch tokens (exclude CLS )
    patch_tokens = outputs[:, 1:, :]  # (batch_size, 196, feature_dim)
    
    # Attention map: 패치 토큰의 노름(norm)을 사용해 중요도 계산
    attn_map = torch.norm(patch_tokens, dim=-1).reshape(14, 14)  # (14x14)
    
    # 정규화 (0~1 범위로 스케일링)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
```

윗 부분으로서 output의 197개 patch 에서 1개를 제외하게 되지요~!  
왜인지 이해하는것은 필수입니다!!  

`바로 CLS를 제외해야하기 때문이에요!!`

이렇게 시각화해보면,  
포스팅에 처음에서 보았던, 아래와 같은 결과물을 볼수 있습니다~  

![dino_result](https://github.com/user-attachments/assets/71580e84-ee1a-4571-ba64-0d4f776de11d)


별도의 라벨없이 학습된 DINO 모델이,  
이미지의 중요한 부분을 판단하여 붉은 색으로,  
덜 중요한 부분은 푸른색으로 시각화하였습니다!!  

#### 4. 결론!!   

DINO로 이미지를 쉽게 벡터로 만들고 시각화할수 있네요!!  
모델을 연구하고 만드는데는 쉽지 않았겠지만 시용이 정말 쉽다는것을 느끼고!!  
이런 모델을 다른 연구에 활용할 수 있도록 잘 기억해두어야겠습니다~!^^

또한 모델을 간단하게 쓸수 있도록 해준 timm에 정말 감사하네요~!
단순히 DINO 뿐만 아니라 다양한 모델을 쓸 수 있는데,

```python
timm.list_models()
```

을 통해 가능한 수많은 모델들을 확인 가능합니다~!  
*제 버젼에서는 1,200 개의 모델을 활용 가능하네요!*  
*그 외에도 resnet, swin, RegNet, EfficientNet 등이 보이는데 이런 모델들도 공부해봐야 겠어요!! *