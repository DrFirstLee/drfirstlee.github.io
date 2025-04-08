---
layout: post
title: "Using CLIP with Python - 파이썬으로 CLIP을 사용해보기"
author: [DrFirst]
date: 2025-04-08 11:00:00 +0900
categories: [AI, Research]
tags: []
lastmod : 2025-04-08 11:00:00
sitemap :
  changefreq : weekly
  priority : 0.9

---
## (English) Using CLIP with Python

Hello! 😊  

![thumbnail](https://github.com/user-attachments/assets/7e34a71b-d976-4341-850c-de1c8cc70249)

In the [previous post](https://drfirstlee.github.io/posts/CLIP/), we explored CLIP's theory based on its paper!  
Today, we’ll download the actual CLIP model and test it in a Python environment!!

We’ll use the image **bike.jpg** shown below for today’s test! ^^  
![bike](https://github.com/user-attachments/assets/0772db0b-ff4f-481d-a22c-4ed3f8db55fc)

---
### Preparing Python Packages!!

We’ll use the following packages!! A GPU is not required!!  
If you’re reading this, I’m sure this level is already basic for you~!? ^^

```python
import clip
import torch
from PIL import Image
import numpy as np
```

> If CLIP is new to you, you can easily install it using:  
> `pip install clip` or `pip install git+https://github.com/openai/CLIP.git`

### Let’s Start CLIP Right Away!!!

Now we’re all set!  
Let’s try using CLIP easily and quickly!!

```python
import clip
import torch
from PIL import Image
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP model Load
model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
# read image
image = preprocess(Image.open("bike.jpg")).unsqueeze(0).to(device)
#  make embedding!
image_features = model.encode_image(image)
print(np.shape(image_features))
image_features
```

How is it? Super easy, right!?

![clip_res](https://github.com/user-attachments/assets/fffa04b5-4d6d-44a7-aa80-d22692d32160)

As shown above, CLIP converts the image into a vector of shape (1, 512)!!

Next, let’s create vector representations of some text!  
I’ll use the following two sentences:

 - "a man riding bicycle"
 - "a man climbing mountain"

```python
import clip
import torch
from PIL import Image
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP model Load
model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
# read text
text = clip.tokenize(["a man riding bicycle", "a man climbing mountain"]).to(device)
#  make embedding!
text_features = model.encode_text(text)
print(np.shape(text_features))
text_features
```

So, what’s the result!?  
You can see two 512-dimensional vectors are generated as shown below!!

![clip_res2](https://github.com/user-attachments/assets/2edd7c50-d1b9-4ca7-9b2c-d1daec309acb)


### Checking CLIP’s Performance!!!!

But wait! Is it enough just to make vectors?  
Of course not~~ We have to check whether the vectors are meaningful!!

So far, we’ve created the following 3 vectors:

- Image1: A person riding a bicycle
- Sentence1: A man riding a bicycle
- Sentence2: A man climbing a mountain

Let’s compare the similarity between Image1 and Sentence1/Sentence2!

```python
# Find similarity
similarity = (image_features @ text_features.T).softmax(dim=-1)
print(similarity)
```

The result was!!!!!

```output
tensor([[0.9902, 0.0096]], device='cuda:0', dtype=torch.float16,
       grad_fn=<SoftmaxBackward0>)
```

This shows that CLIP found the image to be very similar to the **first sentence**, and different from the **second**!

 - Similarity between image and first sentence: 0.9902
 - Similarity between image and second sentence: 0.0096

What do you think? CLIP! The theory may be complicated, but using it is really simple, right!?

This research became the foundation for the development of **multimodal models** that combine text and images!!


### Comparing CLIP Model Performance

In our code, we used a base model called ViT-B/32! But did you know there are many other versions?

| Model Name     | Backbone  | Size     | # of Parameters | Zero-shot ImageNet Accuracy | Embedding Size |
|----------------|-----------|----------|------------------|------------------------------|----------------|
| RN50           | ResNet-50 | Small    | ~102M            | ~63.3%                       | 1024           |
| RN101          | ResNet-101| Small    | ~123M            | ~64.2%                       | 512            |
| ViT-B/32       | ViT       | Small    | ~149M            | ~62.0%                       | 512            |
| ViT-B/16       | ViT       | Medium   | ~151M            | ~68.6%                       | 512            |
| ViT-L/14       | ViT       | Large    | ~428M            | ~75.5%                       | 768            |
| ViT-L/14@336px | ViT       | Large    | ~428M            | ~76.2%                       | 768            |

> **Backbone**: The base neural network structure used to process images or text, such as ResNet or ViT.  
> **Zero-shot ImageNet Accuracy**: The accuracy achieved when classifying ImageNet images **without additional fine-tuning**, using only pre-trained knowledge.  
> **Embedding Size**: The dimension of the final output embedding vector!!

There are 6 representative models like this!  
(The actual Python CLIP package supports 9 available models:  
`['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']`)

Depending on your use case:

 - If you need **faster speed**, choose a **small** model with fewer parameters
 - If **accuracy** is your top priority, choose a model with larger embedding size and more parameters

Pick the model that suits your goal~!

--- 

Thank you!

If you have any questions or topics you’d like to know more about, feel free to leave a comment 💬

---

## (한국어) 파이썬으로 CLIP을 사용해보기

안녕하세요! 😊  

![thumbnail](https://github.com/user-attachments/assets/7e34a71b-d976-4341-850c-de1c8cc70249)

[지난 포스팅](https://drfirstlee.github.io/posts/CLIP/) 에서는 CLIP의 Paper를 바탕으로 이론을 알아보았는데요!
오늘은 실제 이 CLIP모델을  다운받아 Python 환경에서 테스트해보겠습니다!!

오늘의 테스트는 아래의 이미지 **bike.jpg** 를 대상으로 테스트를 진행해보겠습니다!^^
![bike](https://github.com/user-attachments/assets/0772db0b-ff4f-481d-a22c-4ed3f8db55fc)

---
### Python 패키지 준비!!

사용할 패키지는 아래와 같습니다!! GPU는 없어도 되어요!!
이 글을 보시는분들이라면 이미 이정도는 기본이겠지요~!?^^

```python
import clip
import torch
from PIL import Image
import numpy as np
```

> CLIP 패키지는 처음일수 있어요! "pip install clip" 혹은 "pip install git+https://github.com/openai/CLIP.git" 로 간단하게 설치해보세요!!

### 바로 CLIP 시작!!!

이제 준비는 끝났습니다!
간단히게 CLIP을 사용해봅시다!!

```python
import clip
import torch
from PIL import Image
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP model Load
model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
# read image
image = preprocess(Image.open("bike.jpg")).unsqueeze(0).to(device)
#  make embedding!
image_features = model.encode_image(image)
print(np.shape(image_features))
image_features
```

어떄요! 참 쉽죠!>?

![clip_res](https://github.com/user-attachments/assets/fffa04b5-4d6d-44a7-aa80-d22692d32160)

위 이미지와 같이 이미지의 CLIP 결과물로 (1,512) 사이즈의 벡터값이 산출이 됩니다!!

이번엔 유사하게 텍스트들을 벡터값으로 만들어보곘습니다!!
저는 아래의 2개 문장을 각각 벡터화해볼게요~!

 - "a man riding bicycle"
 - "a man climbing mountain"

```python
import clip
import torch
from PIL import Image
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP model Load
model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
# read text
text = clip.tokenize(["a man riding bicycle", "a man climbing mountain"]).to(device)
#  make embedding!
text_features = model.encode_text(text)
print(np.shape(text_features))
text_features
```

그럼~ 결과는!? 
아래와 같이 512 벡터가 2개 생성됨을 확인할수 있습니다!!

![clip_res2](https://github.com/user-attachments/assets/2edd7c50-d1b9-4ca7-9b2c-d1daec309acb)


### CLIP 성능 확인!!!!

그런데! 이렇게 벡터만 만들면 끝일까요!? 
아니죠~~ 만들어진 벡터가 정말 유의미한것인지를 확인해보아야해요!!

저는 지금까지 아래 3개의 벡터를 만들어보았는데요~

- 이미지1 : 자전거를 타고있음
- 문장1 : 남자가 자전거타고있음
- 문장2 : 남자가 산을 오르고있음

이미지1 <-> 문장1/문장2의 벡터의 유사성을 한번 비교해보겠습니다!!

```python
# Find similarity
similarity = (image_features @ text_features.T).softmax(dim=-1)
print(similarity)
```

위의 결과는!!!!!
아래와 같았습니다~!

```output
tensor([[0.9902, 0.0096]], device='cuda:0', dtype=torch.float16,
       grad_fn=<SoftmaxBackward0>)
```

뜻을 생각해보면 CLIP이 이미지와 첫번째 문장과는 거의 동일하고!! 두번쨰 문장돠는 다르다는것을 보여주었어요!

 - 이미지와, 첫번째 문장과 유사성: 0.9902
 - 이미지와, 두번째 문장과 유사성: 0.0096

어떄요? CLIP! 이론은 복잡했지만 사용은 참 쉽지요~!?

이 연구가 기초가 되어 이후 텍스트와 이미지가 합쳐지는 멀티모달로 발전하게 됩니다!!


### CLIP 의 모델별 성능알아보기

저희는 코드에서 ViT-B/32 라는 Base 모델을 사용했습니다! 그런데, 더 많은 종류의 모델이 있다는것을 아시나요!?


| 모델 이름     | 백본 타입 | 크기 분류 | 파라미터 수 | Zero-shot ImageNet 정확도 | 출력 벡터 크기 |
|---------------|-----------|-----------|--------------|-----------------------------|----------------|
| RN50          | ResNet-50 | Small     | 약 102M      | 약 63.3%                    | 1024           |
| RN101         | ResNet-101| Small     | 약 123M      | 약 64.2%                    | 512            |
| ViT-B/32      | ViT       | Small     | 약 149M      | 약 62.0%                    | 512            |
| ViT-B/16      | ViT       | Medium    | 약 151M      | 약 68.6%                    | 512            |
| ViT-L/14      | ViT       | Large     | 약 428M      | 약 75.5%                    | 768            |
| ViT-L/14@336px| ViT       | Large     | 약 428M      | 약 76.2%                    | 768            |

> 백본 타입이란, CLIP 모델에서 이미지나 텍스트를 처리할 때 사용하는 기본 신경망 구조를 말하며, 예를 들어 ResNet이나 ViT 같은 모델이 백본으로 사용됩니다.
> Zero-shot ImageNet 정확도 : 모델이 사전 학습만으로 별도 추가 학습 없이 ImageNet 이미지 데이터셋 분류를 수행했을 때의 정확도를 의미
> 출력 벡터 크기 : 최종 산출되는 임베팅 벡터의 차원!! 

위와같이 대표적으로 6개의 모델이 있습니다!
(실제 Python CLIP 패키지에서는 9개가 available 합니다~ ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] )
여러분은 앞으로 사용 목적에 따라!!

 - 보다 빠른속도가 중요할떄는 Small의, 파라미터가 적은 모델을
 - 정확성이 가장 중요힐떄는 벡터차원도 크고 파라미터가 많은 모델을

 알맞은 모델을 사용하면 됩니다~!

--- 

감사합니다!

궁금한 점이나 더 알고 싶은 주제가 있다면 댓글로 남겨주세요 💬
