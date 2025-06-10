---
layout: post
title: "🖥️ SEEM Practice!! - SEEM 실습!! with python. gradio"
author: [DrFirst]
date: 2025-06-08 09:00:00 +0900
categories: [AI, Experiment]
tags: [LISA, Segment Anything, Image Segmentation, Python,  CVPR, CVPR 2024]
sitemap :
  changefreq : weekly
  priority : 0.9
---

---

## 🦖 (English) SEEM Model Practice!!
> **SEEM**: Segment Everything Everywhere All At Once

We previously introduced **SEEM** in [this post](https://drfirstlee.github.io/posts/SEEM/).  
This time, we'll go through a hands-on tutorial using the SEEM model!

---

### 🧱 1. SEEM Git Clone

Clone the official repository from the [GitHub page](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/tree/v1.0):

```bash
git clone git@github.com:UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git
```

---

### 📦 2. Required Packages in Virtual Environment

I used a `conda` virtual environment to install the required packages:

```bash
conda create -n seem python=3.9 -y
conda activate seem
```

Then install the dependencies from the repo:

```bash
# Python Package Installation
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt

# Custom Operator [only needed for training deformable vision encoder]
cd modeling/vision/encoder/ops && sh make.sh && cd ../../../../

# System Package [only needed for SEEM demo]
sudo apt update
sudo apt install ffmpeg
```

Note: I encountered an error while installing `mpi4py`, so I adjusted some variables:

```bash
mpicc --version
which mpicc
export LD=ld
export CC=mpicc
export CXX=mpicxx
pip install --no-cache-dir --force-reinstall mpi4py
```

---

### 🧊 3. Run the SEEM Model

Execute the command from the `demo` directory:  
The model will automatically download and launch a gradio-based interface!

```bash
export PYTHONPATH=$(pwd)  # Required to avoid ModuleNotFoundError: No module named 'modeling'
python demo/seem/app.py
```

You will see logs like the following:

```bash
  inputs = [ImageMask(label="[Stroke] Draw on Image",type="pil"), gr.inputs.CheckboxGroup(choices=["Stroke", "Example", "Text", "Audio", "Video", "Panoptic"], type="value", label="Interative Mode"), ImageMask(label="[Example] Draw on Referring Image",type="pil"), gr.Textbox(label="[Text] Referring Text"), gr.Audio(label="[Audio] Referring Audio", source="microphone", type="filepath"), gr.Video(label="[Video] Referring Video Segmentation",format="mp4",interactive=True)]
.../demo/seem/app.py:131: GradioDeprecationWarning: `optional` parameter is deprecated
.../demo/seem/app.py:136: GradioDeprecationWarning: gr.outputs is deprecated, use gradio.components instead
Running on local URL:  http://0.0.0.0:8507
IMPORTANT: You are using gradio version 3.42.0, however version 4.44.1 is available.
--------
Running on public URL: https://98c5c3e82a6870cfe5.gradio.live

This share link expires in 72 hours.
```

Now, if you open `https://98c5c3e82a6870cfe5.gradio.live`, you’ll be able to access the environment!  
(Note: Since I won’t keep the server running, please use your own URL.)

Here’s what the interface looks like:

![ui](https://github.com/user-attachments/assets/63501abe-a52f-4aef-ba78-7e91b8d3e2cf)

- Interactive Mode: You can try various prompts including stroke, text, audio, etc.

---

### 🧊 4. SEEM Model Testing

Let’s try some tests!

> Text prompt: "A person using a computer on the couch" – worked really well!  
![test1](https://github.com/user-attachments/assets/00cad703-1ce2-4f76-b7ab-40fb7ce75ab9)

> Segmentation based on keywords works great too!  
![test2](https://github.com/user-attachments/assets/f571265b-8937-4302-be4b-4cf422dda6e5)

> If you just upload an image, it automatically segments objects and labels them!  
![test3](https://github.com/user-attachments/assets/6fa6b962-90ae-46d1-bbcd-4cb6ffe0b9a3)

> Using the stroke feature, it segments only the selected region and even labels it!  
![test4](https://github.com/user-attachments/assets/80bcb736-8b52-4842-a40e-fc5eacd1efc9)

---

### 🎉 Conclusion

SEEM truly feels like the peak of current segmentation models.  
Its flexibility and ease of use are simply amazing!

---

## 🦖(한국어) SEEM 모델 실습!!
> **SEEM** : Segment Everything Everywhere All At Once

[지난 포스팅](https://drfirstlee.github.io/posts/SEEM/) 에서 알아보았던 **SEEM**!!  
이번 포스팅은 이 `SEEM` 모델의 실습을 진행해보겠습니다!!  

---

### 🧱 1. SEEM Git Clone 

- [공식 Git 사이트](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/tree/v1.0)에서 Repo를 Clone 합니다!!

```bash
git@github.com:UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git
```

---

### 📦 2. 가상환경에서의 필요 패키지 설치!!

저는 conda 가상환경에서 필요 패키지들을 설치했습니다!!

```bash
conda create -n seem python=3.9 -y
conda activate seem
```

이제, repo에서 제공하는 requirements를 설치해줍니다!

```bash
# Python Package Installation
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt

# Customer Operator [only need training deformable vision encoder]
cd modeling/vision/encoder/ops && sh make.sh && cd ../../../../

# System Package [only need for demo in SEEM]
sudo apt update
sudo apt install ffmpeg
```

여기서!! 저는 mpi4py설치에서 에러가 생겼어서  
아래와 같이 변수를 조금 바꾸어주었었습니다!  

```bash
mpicc --version
which mpicc
export LD=ld
export CC=mpicc
export CXX=mpicxx 
pip install --no-cache-dir --force-reinstall mpi4py
```
 

---


### 🧊 3. SEEM 모델 실행!!

아래와 같이 demo폴더태의 명령어를 실행하면!!  
알아서 모델을 잘 받고!!  
gradio 기반의 사이트가 시작됩니다!!  

``` bash
export PYTHONPATH=$(pwd) # 아니면 ModuleNotFoundError: No module named 'modeling' 라는 에러가 뜹니다!
python demo/seem/app.py
```

그럼!! 모델을 다운받으며 아래와 같이 주루륵 로그가 끕니다~!

```bash
  inputs = [ImageMask(label="[Stroke] Draw on Image",type="pil"), gr.inputs.CheckboxGroup(choices=["Stroke", "Example", "Text", "Audio", "Video", "Panoptic"], type="value", label="Interative Mode"), ImageMask(label="[Example] Draw on Referring Image",type="pil"), gr.Textbox(label="[Text] Referring Text"), gr.Audio(label="[Audio] Referring Audio", source="microphone", type="filepath"), gr.Video(label="[Video] Referring Video Segmentation",format="mp4",interactive=True)]
.../Segment-Everything-Everywhere-All-At-Once/demo/seem/app.py:131: GradioDeprecationWarning: `optional` parameter is deprecated, and it has no effect
  inputs = [ImageMask(label="[Stroke] Draw on Image",type="pil"), gr.inputs.CheckboxGroup(choices=["Stroke", "Example", "Text", "Audio", "Video", "Panoptic"], type="value", label="Interative Mode"), ImageMask(label="[Example] Draw on Referring Image",type="pil"), gr.Textbox(label="[Text] Referring Text"), gr.Audio(label="[Audio] Referring Audio", source="microphone", type="filepath"), gr.Video(label="[Video] Referring Video Segmentation",format="mp4",interactive=True)]
.../Segment-Everything-Everywhere-All-At-Once/demo/seem/app.py:136: GradioDeprecationWarning: Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components
  gr.outputs.Image(
Running on local URL:  http://0.0.0.0:8507
IMPORTANT: You are using gradio version 3.42.0, however version 4.44.1 is available, please upgrade.
--------
Running on public URL: https://98c5c3e82a6870cfe5.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
```

그럼!! 
`https://98c5c3e82a6870cfe5.gradio.live` 로 접속해보면 이제 환경에 접속이 된것입니다!!  
(저는 이 서버를 계속 켜두지 않을것이기에 제 url이 아니라 여러분의 url로 접속하세요!!)

아래와 같은 화면을 볼 수 있습니다!

![ui](https://github.com/user-attachments/assets/63501abe-a52f-4aef-ba78-7e91b8d3e2cf)

 - Interative Mode : stroke (이미지에 표시하기), text, audio  등 프롬포트를 넣을수 있습니다!


### 🧊 4. SEEM 모델 테스트!!

이제. 테스트를 진행해봅시다!!  

> 소파에서 컴퓨터하는 사람에 대한 텍스트 프롬포트 테스트! 아주좋아!
![test1](https://github.com/user-attachments/assets/00cad703-1ce2-4f76-b7ab-40fb7ce75ab9)

> 단어에 대한 segmentation 도 잘해요~!
![test2](https://github.com/user-attachments/assets/f571265b-8937-4302-be4b-4cf422dda6e5)

> 그냥 이미지만 넣는다면!? 알아서 segmentation 하고 이름까지!!  
![test3](https://github.com/user-attachments/assets/6fa6b962-90ae-46d1-bbcd-4cb6ffe0b9a3)

> 이번엔 stroke 기능을 써서하면!@? 그부분만 segmentation 하고 이름까지!!  
![test4](https://github.com/user-attachments/assets/80bcb736-8b52-4842-a40e-fc5eacd1efc9)

---

### 🎉 마무리

SEEM, 그동안의 segmentation 의 정점인듯! 자유자제로 너무 멋지네요!  