---
layout: post
title: "Grounded SAM Hands-On with Python! - Grounded SAM 실습 with python!"
author: [DrFirst]
date: 2025-05-14 07:00:00 +0900
categories: [AI, Experiment]
tags: [grounding DINO, DINO, 객체 탐지, Object Detection, CV, ECCV, ECCV 2024, python, 파이썬 실습]
lastmod : 2025-05-14 07:00:00
sitemap :
  changefreq : weekly
  priority : 0.9
---

---

## 🦖 (English) Hands-On with Grounded SAM! Detect objects with DINO, then Segment with SAM!

In this post, we’ll do a hands-on walkthrough of **Grounding DINO** + **SAM** = **Grounded SAM**!  
We'll keep following the GitHub repo and run the code,  
but if you go step by step, it’s not too hard!  
So once again, let's skip the theory for now,  
and dive straight into the code to understand what **Grounded SAM** is all about!!

---

### 🧱 1. Clone the GitHub Repository

```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
cd Grounded-Segment-Anything/
```

---

### 📦 2. Install the Models

- From here on, we’re following the setup from the GitHub repo directly!!  
- Please start in an environment where PyTorch and GPU are set up correctly.  
- If not... you'll likely run into many issues! 😅  

```
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/

# Install SAM
python -m pip install -e segment_anything

# Install GroundingDINO
pip install --no-build-isolation -e GroundingDINO

# Install diffusers
pip install --upgrade diffusers[torch]

# OSX-specific install: I skipped this on Ubuntu, but if you're on Mac, you should run this!!
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh

# Install RAM & Tag2Text
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/

# Final dependencies – may vary per user!
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

# Bonus tip!! The supervision version must match exactly as below!!
## I found this after many errors – trust me, use this version!
pip install supervision==0.21.0
```

---

### 🚀 3. Run Object Detection (from Jupyter Notebook)

Now!! With the provided `grounded_sam.ipynb` from the repo, you can jump straight into segmentation~!  
I reused the same image from our previous Grounding DINO test.  

Just like before, I input prompts and tested various labels.  
Here are the results!  
Please note: only one segment is returned per prompt!

- `person`. The simplest and one of the standard COCO dataset labels!!  

![Image](https://github.com/user-attachments/assets/522aacde-3d8d-44b9-8136-1f44d468eb4f)

> From detection to segmentation – flawless!

- `cat`. We already know Grounding DINO failed to detect this before, so skipping it.

- `rugby`. I hoped it would detect the ball, but sadly, detection failed again!

![Image](https://github.com/user-attachments/assets/51042f61-5a56-48b9-86d9-2f20cdfe0ed1)

- `helmet`. Fantastic result!

![Image](https://github.com/user-attachments/assets/bda75b82-5924-4951-a1d4-a64aaa8d0882)

- `jump`. It accurately captured the jumping person!

![Image](https://github.com/user-attachments/assets/e3dc8fb9-b3e3-4b66-b8af-6485dfa8f74a)

How about a full sentence this time: `player is running`?  
> Once again, performance on full sentences isn’t quite there yet!

![Image](https://github.com/user-attachments/assets/591e4dde-07ed-47ad-aa03-b235707a4575)

Now I tried a different image.

`holding` – curious to see what it catches~  
> I was hoping it might isolate just the hand, but I guess that’s asking too much!

![Image](https://github.com/user-attachments/assets/8fa458c9-233c-42d7-ab0a-3c7dcf235a62)

`bat` – can it detect a small baseball bat?
> Absolutely! To help understand, here’s the mask version too!

![Image](https://github.com/user-attachments/assets/2d18c4ee-170c-44a6-b468-8eca9901c038)

`catcher` and `referee`!!
> Clearly distinguishes large human figures!

![Image](https://github.com/user-attachments/assets/4cc0df8f-401c-455b-8f70-6bbcb9894ffe)

---

### 🎉 Final Thoughts

Grounded SAM!! After Grounding DINO,  
we now go from detection to actual image segmentation!  
SAM alone was conceptually interesting but lacked text input,  
so Grounded SAM is amazing in that it allows text prompts! 😄  
That said, imagine how powerful it would be if it could handle large images and multiple segments in one shot!


---

## 🦖(한국어) Grounded SAM 실습! DINO로 객채 탐지 후 Segment까지!!!

이번 포스팅은 **Grounding DINO** 와 **SAM** 을 결합한  **Grounded SAM**의 실습입니다!  
계속해서 GitHub repo에서 코드를 내려받아 실행하지만,  
천천히 따라해보면 모두 잘할 수 있습니다!  
그래서 이번에도 이론은 잠시 뒤로 미뤄두고,  
**Grounded SAM**이 뭔지 이해하기 위해 바로 코드부터 실행해봅시다!!

---

### 🧱 1. GitHub 저장소 클론

```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
cd Grounded-Segment-Anything/
```

---

### 📦 2. 모델 설치

 - 여기부터는 git repo의 설치 방법을 그대로 따라했습니다!!.  
 - pytorch 및 GPU 세팅이 잘 되어있는 환경에서 시작해주세요!  
 - 아니라면, 많은 난관에 부딫히리라 확인합니다!  

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/

# SAM 설치
python -m pip install -e segment_anything

# GroundingDINO 설치
pip install --no-build-isolation -e GroundingDINO

# diffusers 설치
pip install --upgrade diffusers[torch]

# osx 설치 : 저는 우분투 환경이어 생략했지만 mac을 쓰신다면!!
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh

# RAM & Tag2Text 설치
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/

# 마지막 필요한 함수들인데, 요건  사용자마다 다를 수 있습니다!
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

# + 꿀팁!! 아래와 같이 supervision 의 버젼이 잘 맞아야합니다!!
## 여러 에러를 통해 발견한 사실로! 꼭 이버젼이되어야합니다!
pip install supervision==0.21.0
```

---

### 🚀 3. 객체 탐지 실험 실행 (주피터 노트북에서!!)

이젠!! 기존 repo에 제공된 `grounded_sam.ipynb`를 사용하면 바로 segment를 진행할 수 있습니다~!  
저는 이미지만 지난번 Grounding DINO와 동일한 이미지를 사용해보았습니다!  

이젠 지난번과 동일하게, 프롬포트를 넣고 테스트를 진행해보았고!  
결과를 공유합다!!
한개의 Segment 만 된다는것을 감안해주세요!!  

- `person`. 가장 간단하며 기존 coco dataset에 있는 person!! 

![Image](https://github.com/user-attachments/assets/522aacde-3d8d-44b9-8136-1f44d468eb4f)

> 감지에 이어 Segment까지 끝내줍니다!

- `cat`. 은 지난변 grounding-DINO에서 이미 detecting이 안되는것을 확인하였으니 skip!!


- `rugby`. 공이 잡힐것을 예상했지만 여기선 좀 아쉽네요! detection 부터 틀려버렸어!!  

![Image](https://github.com/user-attachments/assets/51042f61-5a56-48b9-86d9-2f20cdfe0ed1)

- `helmet`. 아주 멋져요!!  

![Image](https://github.com/user-attachments/assets/bda75b82-5924-4951-a1d4-a64aaa8d0882)

- `jump`. 정말 점프하는 사람을 잘 잡아냅니다!  

![Image](https://github.com/user-attachments/assets/e3dc8fb9-b3e3-4b66-b8af-6485dfa8f74a)


`player is running` 이번엔 문장으로!?!!   
> 이번에도 역시 문장에서는 잘하지 못하는것을 보았습니다! 

![Image](https://github.com/user-attachments/assets/591e4dde-07ed-47ad-aa03-b235707a4575)

이젠 이미지를 바꾸어보았습니다!

`holding` 어떻게 될지 궁급했는데~~   
> 혹시나 손 부분만을 캐치할까 했는데! 그건 욕심이네요~!

![Image](https://github.com/user-attachments/assets/8fa458c9-233c-42d7-ab0a-3c7dcf235a62)


`bat` 작은 방망이는 잘 캐치할까요!?
> 잘합니다! 이해를 위해 mask 이미지도 함께!!

![Image](https://github.com/user-attachments/assets/2d18c4ee-170c-44a6-b468-8eca9901c038)



`catcher` 와 `referee` !!
> 큼지막하게 인물로 잘 구분합니다!!

![Image](https://github.com/user-attachments/assets/4cc0df8f-401c-455b-8f70-6bbcb9894ffe)



---

### 🎉 마무리

Grounded SAM!! Grounding DINO에 이어서!! 
디택션 내부의 이미지를 segment!!  
SAM에서는 Text 프롬포트가 개념적으로만 제시되어 아쉬웠는데  
이 Grounded SAM에서는 텍스트 제시가 가능해서 너무 좋았습니다!^^  
다만, 큰 이미지를 넘어 이미지 내의 segment 까지 된다면 얼~~마나 좋을까요~! 




```
/home/smartride/DrFirst/LOCATE/AGD20K/Seen/trainset/exocentric/hit/baseball_bat/hit_baseball_bat_000029.jpg
```