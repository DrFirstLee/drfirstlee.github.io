---
layout: post
title: "🔎 CLIP Surgery: A Closer Look at the Explainability of Contrastive Language-Image Pre-training"
author: [DrFirst]
date: 2025-09-05 7:00:00 +0900
categories: [AI, Research]
tags: [CLIP, Explainability, CAM, Vision-Language, Training-Free, Segmentation, PatternRecognition]
sitemap:
  changefreq: monthly
  priority: 0.8
---

---
### 🔎 (한국어) CLIP Surgery: CLIP을 수술해서 설명 가능성을 높이다!  

![Image](https://github.com/user-attachments/assets/c0ab1e6d-506b-46b1-bd0f-519e74812557)

* **제목**: [A Closer Look at the Explainability of Contrastive Language-Image Pre-training (CLIP Surgery)](https://arxiv.org/abs/2304.05653)  
* **저널**: Pattern Recognition (2025)  
* **코드**: [GitHub – CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery)  
* **핵심 키워드**: `CLIP`, `Explainability`, `CAM`, `Vision-Language`, `Open-Vocabulary`  
* **요약**: CLIP은 강력한 비전-언어 모델이지만, **foreground 대신 background에 집중**하거나 **잡음 활성화(noisy activation)** 문제가 존재. 이를 해결하기 위해 **Architecture Surgery**와 **Feature Surgery**를 적용해 **설명가능성을 크게 개선**하는 프레임워크 제안!  

---

### 🚀 CLIP Surgery 핵심 요약

> 한 줄 요약: **“추가 학습 없이, 구조·특징 수술만으로 CLIP의 설명가능성을 강화한다!”**

1) **CLIP 의 문제 증명** : 기존 self-attention의 불일관성, 중복 특징의 문제들을 발견

2) **Feature Surgery** : 중복(redundant) 특징 제거 및 불필요한 noisy activation 억제 성공해서 CAM 만듬!!  

3) **Training-Free Explainability** : Fine-tuning이 불필요, 원본 CLIP 그대로 설명가능성 확보하였기에 다양하게 활용 가능!!  

---

### 🔍 기존 연구의 흐름  

- **CAM, Grad-CAM**: CNN/ViT에는 효과적이지만 CLIP에는 적용 불가  
  > CLIP에서는 'noisy'하고, 'Opposite visualization'하다. 즉 localization에 문제가있다!!
  ![Image](https://github.com/user-attachments/assets/21e4b20a-093a-4cb0-828e-31c6e5d82d86)

- 그럼, CLIP에서는 왜 안됬을까?  
  - Self-attention 구조가 일관되지 않은 의미 영역을 연결하고, 중복 특징 때문에 잡음이 발생해 foreground 대신 background를 강조하기 때문  
  a. Self-attention 구조가 일관되지 않은 의미 영역을 연결하는 이유는?
    a-1. CLIP은 이미지–텍스트 쌍의 전역적(global) 매칭만 학습했기에 attention이 세밀하게 객체 내부에만 집중할 필요가 없었음!!  
    a-2. CLIP의 Query, Key, Value 파라미터가 서로 달라서(heterologous parameters) Query/Key가 만든 관계가 일관되지 않은 의미 영역을 연결
    > A_raw = σ(s · QK_⊤)V : heterologous parameters  
    > A_con = σ(s · VV_⊤)V : homogeneous parameters  


  b. 중복 특징 때문에 잡음이 발생하는 이유는?
    b-1. CLIP은 많은 카테고리를 동시에 학습, 클래스 간 공유되는 특징(예: “하늘”, “풀”, “도로”)이 자주 등장
    b-2. 범용적 특징이 배경에 주로 깔려 있어서, self-attention이 쉽게 배경으로 끌려가 noisy activation이 발생
    ![Image](https://github.com/user-attachments/assets/e2ca7fa6-4181-47cc-95a8-70eaea9278e6)



- **Alignment 기반 기법**도 있지만 이는 추가적인 모델, 레이어, 혹은 파인튜닝을 필요로함!(Not Traininig Free)  
  - **ECLIP**: CLIP feature와 segmentation mask를 **self-supervised**로 다시 정렬(alignment)  
    - 원래 CLIP이 localization을 직접 못하므로, **mask 정보를 추가 학습**하여 보완  
  - **RCLIP**: **Bounding box annotation**을 활용해 CLIP의 이미지–텍스트 feature를 객체 단위로 보정  
    - 결국 CLIP을 **재학습(fine-tuning)**하는 방식  

---

### 🧱 CLIP Surgery 구조 (Architecture)

![Image](https://github.com/user-attachments/assets/a2afeaa7-cf08-4156-a214-b8dcc167b5ab)

#### i) Architecture Surgery(구조적 문제 개선)  
- Raw Self-attention(i-1)이 일관되지 않은 의미 영역을 연결하는 문제가 있는데,  
- Consistent self-attention(i-2)으로 불필요한 배경 강조 방지  

> mFSR은 Self-Attention이 얼마나 foreground(객체)에 집중했는가를 측정하는 지표
![Image](https://github.com/user-attachments/assets/8c327f9a-df42-4330-a191-b98221d83cf9)

  i-1) `raw self-attention : A_raw = σ(s · QK_T)V`
  i-2) `consistent self-attention : A_con = σ(s · VV_⊤)V`

  - 이를 코드로 보면 Transformer Attention 부분인 `Attention` forward 부분에서,   
  ```python
    # i-1) Raw Self-attention
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # i-2) consistent Self-attention  
        # replace k & q by v
        k = v
        q = k
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        ## 마무리!! 둘다 사용할 수 있도록 return  
        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # clip_surgery
        #x = v.transpose(1, 2).reshape(B, N, C) # mask_clip
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]
  ```
- Dual Paths로 CLIP 임베딩용과 CAM 이미지 제작용 두개 피처 추출, FFN의 부정적 영향 최소화  

  ![Image](https://github.com/user-attachments/assets/ff53ea79-cfcd-422f-95df-38a5b3564764)  
  - 위 이미지를 보면, CLIP Transformer내에서 FFN은 오히려 이상한대를 집중하고, 초기 Self-Attention 블록 부분도 부정확하다!  
  - 그렇기에 `새 경로`에서는 self-attention만 적용하고 FFN은 스낍!!  
  - 한편 `원래 경로` 도 유지해서 CLIP 본래 임베딩 보존  

  - 이를 코드로 보면, Transformer layer인 `ResidualAttentionBlock` forward 부분에서,   
  ```python
    def forward(self, x):
        # dual paths for blocks deeper than "d"
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                x, x_ori = x ## x_ori는 원래경로, x 는 새로운 경로!
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res ## self-attention 결과 반환. x_res 는 `consistent self-attention`, x_ori_res는 `raw self-attention`  
                x_ori += x_ori_res # 원래 경로는 self attention(x_ori_res)을 더하고
                x_ori = x_ori + self.mlp(self.ln_2(x_ori)) # 원래 경로에 FFN(x_ori)를 더함!!
                x += x_res # 새로운 경로는 self attention(x_ori) 만 더함
                return [x, x_ori]
  ```

#### ii) Feature Surgery(표현적 문제 개선)  
- CLIP은 많은 카테고리를 동시에 학습, 클래스 간 공유되는 특징이 많은 문제가 있는데,    
- `강아지` 가 타겟이더라도 고양이, 하늘, 바다, 비행기 등 기타 텍스트의 임베딩도 해보고, 중복되는 부분을 제거  
  > L1 이 작으면 유사하다는 뜻. 즉 positive랑 empty가 유사해버림! 그래서 중복의 문제가 발생!!
    ![Image](https://github.com/user-attachments/assets/e2ca7fa6-4181-47cc-95a8-70eaea9278e6)
- 코드로 보면!! **clip_feature_surgery 부분에서 전체에서 겹치는 부의를 가지고 redundant_feats를 만들고 각 feats에서 redundant_feats를 빼줌**   
```python 
# demp.py 에서 미리 empty에 해당하는 all_text를 선언해두고 target_text도 둔다음
all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
target_texts = ['dog']

with torch.no_grad():
    # Extract image features
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Prompt ensemble for text features with normalization
    text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)

    # Similarity map from image tokens with min-max norm and resize, B,H,W,N
    # 여기서 clip_feature_surgery 진행!! 
    similarity = clip.clip_feature_surgery(image_features, text_features)
    similarity_map = clip.get_similarity_map(features[:, 1:, :], cv2_img.shape[:2])

    for b in range(similarity_map.shape[0]):
        for n in range(similarity_map.shape[-1]):
            if all_texts[n] not in target_texts:
                continue
            ## 여기서 
            vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = cv2_img * 0.4 + vis * 0.6
            vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
            print('CLIP:', all_texts[n])
            plt.imshow(vis)
            plt.show()


## clip.py clip_feature_surgery 부분에서 전체에서 겹치는 부의를 가지고 redundant_feats를 만들고 각 feats에서 redundant_feats를 빼줌
def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):

    if redundant_feats != None:
        similarity = image_features @ (text_features - redundant_feats).t()

    else:
        # weights to restrain influence of obvious classes on others
        prob = image_features[:, :1, :] @ text_features.t()
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)

        # element-wise multiplied features
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        feats *= w.reshape(1, 1, n_t, 1)
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats
        
        # sum the element-wise multiplied features as cosine similarity
        similarity = feats.sum(-1)

    return similarity

```

---

### 🧪 실험 평가 결과  

#### 🎯 Explainability Benchmarks  
- VOC 2012, COCO, PascalContext 등에서  
  - **mIoU +22~36% 개선**  
  - **mSC +48~65% 개선**  

#### 🎯 Open-Vocabulary Tasks  
- **Semantic Segmentation**: training 없는 방법 중 최고 성능 (예: PascalContext mIoU 29.3%)  
- **Multi-label Recognition**: NUS-Wide에서 CLIP 대비 +11.61% mAP 향상  
- **Interactive Segmentation**: SAM에 텍스트→포인트 변환으로 수작업 라벨 대체  
- **Multimodal Visualization**: CLIP의 학습 과정 자체 해석 가능  
  ![Image](https://github.com/user-attachments/assets/bf45702f-6073-4693-b8a5-9007e63b0ca0)
  - `[end]` 토큰이 가장 흔히 활성화된 텍스트 토큰이며, “in”, “.”, “of” 같은 비객체(non-object) 단어도 높은 반응을 보임!!  
  - 이는 CLIP의 어휘 사전에 **불필요한 중복 토큰(redundant tokens)**이 존재함을 시사  
  - 향후 CLIP 학습 과정 개선 아이디어를 제공!!  


#### 👀 정성 비교 : 잘하는구만!  

![Image](https://github.com/user-attachments/assets/ddf81331-3582-4ae1-bcec-20c60b6110a0)

- 원본 CLIP: background 강조 + 잡음 다수  
- CLIP Surgery: 선명하고 객체 중심 heatmap 생성  
- → 기존 Grad-CAM, Bi-Modal, gScoreCAM 대비 **확연히 향상된 시각화 품질**  


#### 🧪 Ablation 분석  

![Image](https://github.com/user-attachments/assets/73150a1c-13ca-4e48-bb1c-8648056b2c0b)

- **Architecture Surgery**(i)만 적용 → mSC +47.88%  
- Feature Surgery(ii) 추가 → 추가 +3.17% 향상  
- Dual Paths 없으면 collapse 발생, 핵심 모듈임을 검증  

---

## ✅ 결론  

- **CLIP Surgery**는 CLIP 모델의 **근본적 설명가능성 문제(opposite visualization, noisy activation)**를 해결  
- **추가 학습 없는 training-free 접근**으로 CAM 기반 해석 강화  
- Semantic Segmentation, Multi-label Recognition, Interactive Segmentation 등 다양한 다운스트림 작업에 직접 활용 가능  
- CLIP 내부 메커니즘 이해와 향후 모델 개선에 중요한 통찰 제공  
