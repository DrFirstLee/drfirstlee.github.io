---
layout: post
title: "Reasoning Segmentation LLM LISA 실습!!"
author: [DrFirst]
date: 2025-06-04 09:00:00 +0900
categories: [AI, Experiment]
tags: [LISA, Segment Anything, Image Segmentation, Python,  CVPR, CVPR 2024]
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

> 아래와 같이 확실하게 솔 부분만 잘 분류하네요~!!
![toothbrush](https://github.com/user-attachments/assets/05331580-ef6f-4be9-9967-0eaf4fd4b310)


이 외에도 테스트해보았던 결과물을 아래와 같이 공유합니다~!  

> wine glass : 단순한 단어로도 잘 구분하죠!?
![wine_glass](https://github.com/user-attachments/assets/1fd3ec07-2e5f-4276-962f-dceabc810072)


> glove : 손부분만 잘 추출하네요~!
![glove](https://github.com/user-attachments/assets/7d622a3d-05d6-4315-a204-26723d616465)

> where is the handle? : 문장 잘해!!! 
![knifehandle](https://github.com/user-attachments/assets/935b823f-4d92-4039-bb2c-9da3b87aad3e)

> Which part of a baseball bat is the handle that people hold?
![baseballhandle](https://github.com/user-attachments/assets/bb687399-68be-491b-9bce-0a59fc446753)

> Wearing glove : 딱 손부분만 하면 좋겠지만 그렇게는 안되네요!
![wearingglove](https://github.com/user-attachments/assets/b4844508-49cc-41dd-a9e3-dfdf290be4e3)

> A vegetable that's healthy but not liked by most kids : 브로콜리.. 잘 못하는군요!!
![brocoli](https://github.com/user-attachments/assets/279da196-29b6-47f2-be13-5530c65125cd)

---

### 🎉 마무리

Segmentation의 시대! 이제는 단순 Segmentation을 넘어 추론까지!!  
앞으로 얼마나 더 발전될지 기대됩니다~!  