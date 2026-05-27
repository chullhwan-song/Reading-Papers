# PAPER: Sana - Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers 쉽게 읽기

## 0. 이 문서를 읽는 법

이 문서는 Sana 논문과 공개 코드를 보고, 처음 읽는 사람이 전체 구조를 놓치지 않도록 다시 정리한 리뷰입니다.

핵심 목표는 하나입니다.

> **Sana는 4K 같은 고해상도 이미지를 만들 때 너무 비싸지는 DiT의 비용을, 압축 오토인코더와 선형 어텐션으로 크게 줄인 text-to-image 모델이다.**

DMD, DMD2 문서와 같이 읽는다면 이렇게 보면 좋습니다.

```text
DMD/DMD2 = 이미 학습된 느린 diffusion model을 빠른 student로 증류하는 방법
Sana     = 처음부터 빠르게 돌도록 text-to-image diffusion backbone을 설계하는 방법
```

즉 Sana는 DMD처럼 "teacher를 짧은 step student로 압축"하는 논문이 아닙니다.

Sana의 질문은 이것입니다.

```text
고해상도 이미지에서 DiT가 느린 이유는 무엇인가?
그 병목을 모델 구조 자체에서 어떻게 줄일 수 있는가?
```

이 문서는 아래 순서로 읽으면 가장 편합니다.

1. **큰 그림**: Sana가 해결하려는 병목
2. **등장인물**: DC-AE, Linear DiT, Gemma text encoder, Flow-DPM-Solver
3. **핵심 장치 5개**: 왜 각각 필요한지
4. **Linear Attention 직관**: 왜 `O(N^2)`가 `O(N)`이 되는지
5. **코드 연결**: 논문 아이디어가 코드에서 어디 보이는지
6. **실험 결과와 FAQ**: 숫자와 헷갈리는 점

---

## 1. 메타 정보

| 항목 | 내용 |
|---|---|
| 논문 | SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers |
| 저자 | Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, Song Han |
| 소속 | NVIDIA, MIT HAN Lab, Tsinghua University |
| 공개 | arXiv v1 2024-10-14, v3 2024-10-20 |
| arXiv | https://arxiv.org/abs/2410.10629 |
| 코드 | https://github.com/NVlabs/Sana |
| 라이선스 | Apache-2.0 |
| 관련 오토인코더 논문 | Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models, arXiv 2410.10733 |

---

## 2. 한 문장 요약

> **Sana는 이미지를 32배 압축하는 DC-AE, self-attention을 선형 비용으로 바꾼 Linear DiT, Gemma-2-2B text encoder, Flow-DPM-Solver를 결합해 0.6B/1.6B급 작은 모델로 1024px와 4K 이미지를 빠르게 생성하는 시스템이다.**

조금 더 쉽게 말하면:

> **이미지를 만들 때 "볼 픽셀/토큰 수" 자체를 줄이고, 남은 토큰끼리 비교하는 비용도 줄인 모델이다.**

Sana를 이해할 때 가장 중요한 식은 수학 공식이 아니라 아래 구조입니다.

```text
Sana = DC-AE 32x 압축
     + Linear DiT
     + Gemma text encoder
     + Flow-DPM-Solver
```

이 네 가지가 동시에 작동합니다. 하나만 떼어 보면 효과가 작거나 품질이 떨어질 수 있습니다.

---

## 3. 먼저 알아야 할 큰 그림

### 3.1 text-to-image diffusion은 보통 어떻게 이미지 만들까?

일반적인 latent diffusion은 픽셀 이미지를 바로 다루지 않습니다.

```text
image
  -> VAE encoder
  -> latent
  -> diffusion model이 latent에서 denoising
  -> VAE decoder
  -> image
```

이렇게 하는 이유는 픽셀 공간이 너무 크기 때문입니다.

예를 들어 1024x1024 이미지를 그대로 다루면 픽셀이 백만 개가 넘습니다. 그래서 Stable Diffusion 계열은 보통 VAE로 이미지를 8배 줄인 latent에서 작업합니다.

### 3.2 DiT에서 무엇이 비싼가?

DiT는 이미지를 토큰 시퀀스로 보고 transformer를 돌립니다.

문제는 self-attention입니다.

```text
토큰 N개가 있으면
각 토큰이 다른 모든 토큰을 본다

비용 ~= N x N
```

1024px에서는 어떻게든 가능해도, 4K로 가면 토큰 수가 크게 늘고 `N^2` 비용이 폭발합니다.

Sana는 이 병목을 두 번 줄입니다.

```text
1단계: DC-AE로 토큰 수 N 자체를 줄인다.
2단계: Linear Attention으로 N^2 attention을 N에 가까운 비용으로 바꾼다.
```

### 3.3 Sana가 해결하려는 문제

기존 고품질 모델은 보통 아래 방향으로 갑니다.

```text
큰 모델
많은 토큰
full attention
많은 sampling step
```

품질은 좋지만 비용이 큽니다.

Sana는 반대로 묻습니다.

```text
작은 모델로도 고해상도를 만들려면
어디를 줄여야 하는가?
```

Sana의 답:

```text
이미지 latent 토큰 수를 줄이고,
self-attention을 선형화하고,
text encoder는 더 작지만 instruction-following이 좋은 LLM을 쓰고,
sampler step도 줄인다.
```

---

## 4. 등장인물 정리

### 4.1 DC-AE

DC-AE는 **Deep Compression Autoencoder**입니다.

역할:

```text
이미지 -> 32배 작아진 latent
latent -> 이미지 복원
```

기존 Stable Diffusion VAE는 보통 8배 압축입니다.

Sana의 DC-AE는 32배 압축입니다.

```text
1024x1024 image
  -> DC-AE
  -> 32x32 latent grid
```

즉 1024px 이미지를 DiT가 볼 때 공간 토큰이 1024개 정도로 줄어듭니다.

중요한 점:

> **DC-AE는 Sana 논문 안에서 처음부터 자세히 학습한 컴포넌트라기보다, 같은 그룹의 별도 자매 논문에서 만든 32x 압축 오토인코더를 Sana가 가져다 쓰는 구조입니다.**

### 4.2 Linear DiT

Linear DiT는 Sana의 diffusion transformer 본체입니다.

일반 DiT와 가장 큰 차이:

```text
일반 DiT: self-attention = softmax attention, 비용 O(N^2)
Sana DiT: self-attention = LiteLA linear attention, 비용 O(N)
```

단, 모든 attention을 단순히 갈아끼운 것은 아닙니다.

```text
이미지 토큰끼리 보는 self-attention -> Linear Attention
텍스트를 보는 cross-attention      -> Standard Attention 유지
```

왜냐하면 텍스트와 이미지의 정밀한 매칭은 sharp attention이 중요하기 때문입니다.

### 4.3 Mix-FFN

Linear Attention만 넣으면 품질이 떨어질 수 있습니다.

이유:

```text
softmax attention은 특정 토큰을 날카롭게 찍어 볼 수 있음
linear attention은 더 부드러운 평균에 가까워질 수 있음
```

Sana는 이 약점을 **Mix-FFN**으로 보완합니다.

Mix-FFN은 FFN 안에 3x3 depthwise convolution을 넣습니다.

역할:

```text
주변 토큰의 지역 패턴을 보강한다.
위치 정보도 어느 정도 제공한다.
```

그래서 Sana는 별도 positional embedding을 쓰지 않는 **NoPE** 구조도 가능해집니다.

### 4.4 Gemma-2-2B text encoder

Sana는 T5-XXL 대신 Gemma-2-2B-IT를 text encoder로 씁니다.

쉽게 말하면:

```text
프롬프트를 이미지 모델이 이해할 수 있는 text feature로 바꾸는 역할
```

Sana가 Gemma를 쓴 이유:

| 항목 | T5-XXL | Gemma-2-2B-IT |
|---|---|---|
| 크기 | 4.7B | 2.2B |
| 속도 | 느림 | 더 빠름 |
| instruction-following | 약함 | 강함 |
| 역할 | 기존 T2I에서 많이 쓰임 | Sana의 decoder-only text encoder |

주의할 점:

> **Gemma를 그대로 붙이면 text embedding의 scale이 커서 학습이 불안정해질 수 있습니다.**

Sana는 이를 `RMSNorm + 작은 learnable scale`로 안정화합니다.

### 4.5 Flow-DPM-Solver

Sana는 Flow Matching 기반 모델입니다.

기본 Euler sampler를 쓰면 step 수가 많아질 수 있습니다. Sana는 Flow-DPM-Solver를 써서 적은 step으로도 품질을 유지하려고 합니다.

역할:

```text
많은 denoising step을 줄여 inference를 빠르게 한다.
```

---

## 5. Sana의 전체 구조

논문 Figure 5를 문장으로 풀면 아래 흐름입니다.

```text
prompt
  -> CHI instruction을 붙임
  -> Gemma-2-2B-IT text encoder
  -> text feature

image
  -> DC-AE encoder
  -> 32x 압축 latent
  -> Linear DiT가 noise/velocity를 예측
  -> DC-AE decoder
  -> generated image
```

Linear DiT block 안에서는 아래 일이 반복됩니다.

```text
latent tokens
  -> LiteLA self-attention
  -> text cross-attention
  -> Mix-FFN
  -> 다음 block
```

핵심은 이 순서입니다.

```text
DC-AE가 토큰 수를 줄임
Linear Attention이 토큰 비교 비용을 줄임
Mix-FFN이 Linear Attention의 약한 지역성을 보완함
Gemma가 텍스트 이해를 담당함
Flow-DPM-Solver가 sampling step을 줄임
```

---

## 6. 핵심 장치 1: DC-AE 32x 압축

### 6.1 왜 필요한가?

고해상도 diffusion에서 가장 먼저 부담되는 것은 토큰 수입니다.

1024x1024 이미지를 생각해 봅니다.

기존 8x VAE를 쓰면 latent 공간은 대략 아래처럼 됩니다. 실제 DiT token 수는 patch size로 한 번 더 묶느냐에 따라 달라질 수 있지만, 먼저 latent grid 크기만 보면 이렇습니다.

```text
1024 / 8 = 128
latent grid = 128 x 128
토큰 수 = 16,384
```

Sana의 32x DC-AE를 쓰면:

```text
1024 / 32 = 32
latent grid = 32 x 32
토큰 수 = 1,024
```

토큰 수가 16분의 1이 됩니다.

attention 비용은 토큰 수의 제곱에 비례하므로, standard attention 기준으로는 이 감소가 훨씬 크게 작용합니다.

```text
토큰 수 16배 감소
-> attention 행렬 크기 256배 감소
```

### 6.2 그냥 VAE를 더 세게 압축하면 안 되나?

그게 어렵습니다.

압축을 강하게 하면 정보가 사라져서 복원 화질이 나빠집니다.

Sana가 쓰는 DC-AE는 이 문제를 줄이기 위해 별도 논문에서 다음 아이디어를 씁니다.

```text
잔차 오토인코딩
다단계 학습
고압축에서도 reconstruction 품질 유지
```

논문에서 중요한 비교는 아래처럼 이해하면 됩니다.

| 오토인코더 | 압축 | 의미 |
|---|---|---|
| SDXL VAE | 8x | 기존 강한 baseline |
| naive 32x VAE | 32x | 단순히 압축만 키우면 품질 하락 |
| DC-AE F32C32 | 32x | 32x 압축에서도 쓸 만한 복원 품질 |

### 6.3 코드에서 어디 보나?

Sana 저장소 안에서는 DC-AE 관련 구현이 아래 경로에 있습니다.

```text
diffusion/model/dc_ae/efficientvit/models/efficientvit/dc_ae.py
```

구조를 아주 단순화하면:

```text
Encoder:
  image
  -> 여러 stage로 downsample
  -> 32-channel latent

Decoder:
  latent
  -> 여러 stage로 upsample
  -> image
```

Sana 문서에서 `F32C32` 같은 표기가 나오면 이렇게 읽으면 됩니다.

| 표기 | 뜻 |
|---|---|
| F32 | spatial downsample이 32배 |
| C32 | latent channel이 32개 |
| P1 | DiT patch size가 1 |

즉:

```text
F32C32P1 = 32배 압축 latent를, patch로 더 묶지 않고 그대로 토큰화
```

---

## 7. 핵심 장치 2: Linear DiT와 LiteLA

### 7.1 왜 self-attention이 문제인가?

표준 attention은 각 토큰이 모든 토큰을 봅니다.

```text
N개 토큰이 있으면
N x N attention score 표를 만든다.
```

1024개 토큰이면:

```text
1024 x 1024 = 약 100만 칸
```

4K에서 DC-AE를 써도 토큰은 16,384개가 됩니다.

```text
16,384 x 16,384 = 약 2.7억 칸
```

이게 고해상도에서 full attention이 비싸지는 이유입니다.

### 7.2 Sana의 아이디어

Sana는 self-attention을 LiteLA로 바꿉니다.

```text
standard attention:
  softmax(QK^T)V

LiteLA:
  ReLU(Q), ReLU(K)를 쓰고
  K와 V를 먼저 합쳐 작은 행렬을 만든 뒤
  Q에 곱한다.
```

결과적으로 큰 `N x N` attention matrix를 만들지 않습니다.

### 7.3 Linear Attention을 가장 쉽게 이해하기

표준 attention은 이런 느낌입니다.

```text
각 학생이 반 전체 학생과 1:1로 비교한다.
그리고 그 점수를 softmax로 줄 세운다.
```

학생이 N명이면 비교가 N x N번 필요합니다.

Linear Attention은 이렇게 바꿉니다.

```text
먼저 반 전체 정보를 요약한 작은 메모장을 만든다.
각 학생은 그 메모장만 보고 자기 출력을 계산한다.
```

즉:

```text
모든 학생과 매번 1:1 비교하지 않는다.
공통 요약을 한 번 만들고 재사용한다.
```

이 때문에 비용이 `N^2`에서 `N`에 가까워집니다.

### 7.4 왜 softmax는 어렵고 ReLU는 가능한가?

표준 attention은 한 토큰 `i`의 출력을 대략 이렇게 만듭니다.

```text
Out_i = 모든 j에 대해
        softmax(Q_i와 K_j의 점수) x V_j를 더함
```

문제는 softmax입니다.

softmax는 한 점수만 보고 계산할 수 없습니다.

```text
전체 j의 점수를 다 본 뒤
그 안에서 상대적인 비율을 계산해야 한다.
```

그래서 `N x N` 점수표가 필요합니다.

Sana는 softmax 대신 ReLU 기반 kernel을 씁니다.

```text
softmax(QK^T)
  대신
ReLU(Q)와 ReLU(K)를 따로 계산
```

이렇게 하면 `K`와 `V`를 먼저 합칠 수 있습니다.

```text
M = sum_j ReLU(K_j) x V_j
Out_i = ReLU(Q_i) x M
```

여기서 `M`은 모든 `i`가 같이 쓰는 작은 요약 행렬입니다.

핵심:

> **softmax는 모든 토큰을 한꺼번에 봐야 하므로 분리하기 어렵고, ReLU는 각 토큰에 따로 적용할 수 있어 계산 순서를 바꿀 수 있습니다.**

### 7.5 Linear Attention만 쓰면 충분한가?

아닙니다.

Linear Attention은 빠르지만 표현력이 조금 약해질 수 있습니다.

특히 특정 위치나 특정 토큰을 아주 날카롭게 찍는 능력은 standard attention보다 약할 수 있습니다.

Sana는 그래서 다음 조합을 씁니다.

```text
Linear self-attention
+ Mix-FFN의 3x3 convolution
+ standard cross-attention 유지
```

즉 "Linear Attention 하나로 해결"이 아니라:

> **Linear Attention이 잃는 부분을 Mix-FFN과 cross-attention 설계로 보완한 시스템입니다.**

---

## 8. 핵심 장치 3: Mix-FFN과 NoPE

### 8.1 Mix-FFN이 필요한 이유

이미지는 텍스트보다 지역 패턴이 중요합니다.

예를 들어 얼굴, 손, 옷 주름, 물체 경계 같은 것은 주변 픽셀과 강하게 연결됩니다.

표준 transformer의 FFN은 보통 token마다 독립적인 MLP입니다. 주변 토큰을 직접 보는 구조가 아닙니다.

Sana의 Mix-FFN은 여기에 convolution을 넣습니다.

```text
1x1 conv
-> 3x3 depthwise conv
-> gate
-> 1x1 conv
```

이 3x3 conv가 주변 정보를 섞어 줍니다.

### 8.2 NoPE는 무엇인가?

NoPE는 **No Positional Embedding**입니다.

일반 transformer는 토큰 순서나 위치를 알려주기 위해 positional embedding을 넣습니다.

Sana는 별도 positional embedding을 제거합니다.

왜 가능한가?

```text
Mix-FFN 안의 3x3 conv와 padding이
위치와 주변 관계에 대한 단서를 어느 정도 제공한다.
```

그래서 Sana에서는:

```text
Linear Attention으로 global mixing
Mix-FFN으로 local mixing과 위치 단서 보강
```

이 조합이 중요합니다.

---

## 9. 핵심 장치 4: Gemma text encoder와 CHI

### 9.1 왜 decoder-only LLM을 text encoder로 쓰나?

기존 text-to-image 모델은 T5 계열 text encoder를 많이 씁니다.

Sana는 Gemma-2-2B-IT를 씁니다.

장점:

```text
더 작다.
더 빠르다.
instruction-following 능력이 좋다.
다국어나 복잡한 지시문 처리에 유리하다.
```

Sana 논문은 Gemma를 단순히 "작은 T5 대체품"으로 보는 것이 아니라, instruction-following LLM으로 활용합니다.

### 9.2 CHI란?

CHI는 **Complex Human Instruction**입니다.

쉽게 말하면:

```text
사용자 prompt 앞에 긴 시스템 지시문을 붙여
LLM이 프롬프트를 더 잘 해석하도록 만드는 방법
```

예:

```text
사용자 prompt:
  a cat wearing sunglasses

Gemma에 들어가는 입력:
  너는 이미지 생성을 위한 prompt를 이해하는 assistant다...
  아래 설명을 정확히 이미지 조건으로 해석하라...
  a cat wearing sunglasses
```

이렇게 하면 text-image alignment가 좋아질 수 있습니다.

### 9.3 Gemma를 붙일 때 생기는 문제

Decoder-only LLM embedding은 기존 T5 embedding과 scale이 다를 수 있습니다.

Sana 논문에서 중요한 관찰:

```text
Gemma embedding을 그대로 cross-attention에 넣으면
분산이 너무 커서 학습이 불안정해질 수 있다.
```

해결:

```text
RMSNorm
+ learnable scale을 작게 시작
```

코드에서는 text feature normalization 관련 설정으로 볼 수 있습니다.

```text
attention_y_norm
```

---

## 10. 핵심 장치 5: Flow-DPM-Solver

Sana는 Flow Matching 계열 학습을 사용합니다.

기본 sampler를 쓰면 품질을 위해 step이 많이 필요할 수 있습니다.

Sana는 Flow-DPM-Solver를 제안해 step 수를 줄입니다.

직관:

```text
denoising 경로를 더 똑똑하게 적분해서
비슷한 품질을 더 적은 step으로 얻는다.
```

코드에서는 아래 파일이 관련됩니다.

```text
diffusion/model/dpm_solver.py
```

Sana의 속도는 모델 구조만의 결과가 아닙니다.

```text
토큰 수 감소
+ attention 비용 감소
+ text encoder 비용 감소
+ sampling step 감소
```

이 네 가지가 함께 쌓여서 빠릅니다.

---

## 11. 코드에서 보는 Sana

### 11.1 LiteLA

LiteLA 구현은 대략 아래 위치에서 볼 수 있습니다.

```text
diffusion/model/nets/sana_blocks.py
```

핵심 흐름은 이렇게 읽으면 됩니다.

```python
q = ReLU(q)
k = ReLU(k)

v = pad(v, value=1)      # 정규화 분모까지 같이 계산하기 위한 trick
vk = v @ k               # K와 V를 먼저 합친 작은 행렬
out = vk @ q             # 각 Q에 적용
out = numerator / denominator
```

중요한 점:

```text
softmax가 없다.
QK^T로 N x N 행렬을 만들지 않는다.
```

### 11.2 self-attention과 cross-attention

Sana block은 self-attention과 cross-attention을 다르게 취급합니다.

```text
self-attention:
  이미지 토큰끼리 보는 부분
  -> LiteLA 사용

cross-attention:
  이미지 토큰이 텍스트 feature를 보는 부분
  -> standard attention 유지
```

이 선택이 중요합니다.

이미지 토큰은 많으므로 linear로 줄이는 효과가 큽니다.

텍스트 토큰은 상대적으로 적고, 단어와 이미지 위치의 정확한 연결이 중요하므로 standard attention을 유지합니다.

### 11.3 DC-AE

DC-AE 관련 코드는 아래 경로에서 볼 수 있습니다.

```text
diffusion/model/dc_ae/efficientvit/models/efficientvit/dc_ae.py
```

Sana가 쓰는 핵심 설정은:

```text
F32C32
```

즉:

```text
32x spatial compression
32 latent channels
```

### 11.4 Flow-DPM-Solver

Sampler 쪽은 아래 파일을 보면 됩니다.

```text
diffusion/model/dpm_solver.py
```

여기서 중요한 관점은:

```text
모델 forward 1번의 비용도 줄이고,
필요한 forward 횟수도 줄인다.
```

입니다.

---

## 12. 실험 결과를 어떻게 읽어야 하나?

### 12.1 1024x1024 결과

논문과 저장소 README의 숫자를 요약하면, Sana는 작은 파라미터 수로도 매우 빠릅니다.

| 모델 | 파라미터 | Latency | FID | GenEval | DPG |
|---|---:|---:|---:|---:|---:|
| FLUX-dev | 12B | 23.0s | 10.15 | 0.67 | 84.0 |
| Sana-0.6B | 0.6B | 0.9s | 5.81 | 0.64 | 83.6 |
| Sana-1.6B | 1.6B | 1.2s | 약 5.8-5.9 | 약 0.66-0.69 | 약 84.5-84.8 |

숫자 해석:

```text
Sana는 FLUX-dev보다 훨씬 작고 빠르지만,
텍스트-이미지 alignment 지표는 꽤 근접한다.
```

단, metric만 보고 "항상 FLUX보다 낫다"고 읽으면 안 됩니다. 모델 크기, 데이터, aesthetic preference, prompt 종류에 따라 체감 품질은 달라질 수 있습니다.

### 12.2 4K 결과

Sana가 특히 강조하는 지점은 4K입니다.

고해상도로 갈수록 standard attention의 `N^2` 비용이 커집니다.

Sana는:

```text
DC-AE로 토큰 수를 줄이고
Linear Attention으로 attention 비용 증가를 완화한다.
```

그래서 4K에서 상대 속도 이득이 더 커집니다.

### 12.3 RTX 4090 / laptop GPU 의미

Sana는 "거대 서버 GPU에서만 되는 모델"이 아니라, 16GB급 GPU나 consumer GPU에서도 돌릴 수 있는 효율성을 강조합니다.

이것이 Sana의 실용적 의미입니다.

```text
품질만 최고로 끌어올린 논문이라기보다
고해상도 생성 비용을 낮추는 시스템 논문
```

---

## 13. Sana의 기여를 한 번에 정리

Sana의 기여는 "Linear Attention을 썼다" 하나로 요약하면 부족합니다.

정확히는 아래 조합입니다.

| 기여 | 역할 |
|---|---|
| DC-AE 32x | 이미지 latent token 수를 크게 줄임 |
| LiteLA | self-attention의 `N^2` 비용을 줄임 |
| Mix-FFN | Linear Attention이 약해질 수 있는 지역 정보를 보강 |
| standard cross-attention 유지 | 텍스트와 이미지의 sharp alignment를 보존 |
| Gemma text encoder | 작은 decoder-only LLM으로 prompt 이해 강화 |
| CHI | instruction을 붙여 text-image alignment 개선 |
| Flow-DPM-Solver | sampling step 감소 |

가장 중요한 문장:

> **Sana의 진짜 기여는 Linear Attention 단품이 아니라, Linear Attention이 이미지 생성에서 실제로 품질을 잃지 않도록 만든 전체 시스템 설계입니다.**

---

## 14. DMD/DMD2와의 연결

DMD/DMD2를 먼저 읽었다면 Sana는 조금 다르게 봐야 합니다.

| 항목 | DMD/DMD2 | Sana |
|---|---|---|
| 목적 | 느린 teacher를 빠른 student로 증류 | 처음부터 빠른 diffusion backbone 설계 |
| 핵심 질문 | 많은 step을 1-step/4-step으로 줄일 수 있나? | 고해상도 DiT의 token/attention 비용을 줄일 수 있나? |
| 주요 대상 | student generator 학습 | text-to-image 모델 구조 |
| 핵심 장치 | score matching, critic, GAN loss, backward simulation | DC-AE, Linear Attention, Mix-FFN, Gemma, solver |
| 속도 개선 위치 | 추론 step 수 | 토큰 수, attention 비용, text encoder, sampler |

공통점도 있습니다.

```text
둘 다 diffusion을 더 싸고 빠르게 만들려는 연구다.
```

하지만 줄이는 비용의 종류가 다릅니다.

```text
DMD/DMD2:
  denoising step 수를 줄인다.

Sana:
  step 하나하나의 계산 비용을 줄이고,
  sampler step도 줄인다.
```

---

## 15. 다른 모델과 비교해서 이해하기

### 15.1 PixArt 계열과의 관계

Sana는 PixArt 계열과 비교하면 이해하기 쉽습니다.

```text
PixArt:
  SD VAE 8x
  standard attention DiT
  T5 text encoder

Sana:
  DC-AE 32x
  linear attention DiT
  Gemma text encoder
```

즉 Sana는 PixArt류 효율형 DiT를 더 극단적으로 밀어붙인 모델로 볼 수 있습니다.

### 15.2 FLUX와의 관계

FLUX는 더 큰 모델과 강한 full attention 계열 설계로 고품질을 노립니다.

Sana는 더 작은 모델과 효율 구조로 속도와 고해상도 확장성을 노립니다.

```text
FLUX:
  크고 강한 모델
  비용 큼
  품질 강함

Sana:
  작고 빠른 모델
  고해상도 비용 낮음
  효율 강함
```

둘은 같은 목표를 다른 방식으로 푼 사례입니다.

### 15.3 백본 재사용 관점

Sana의 DiT 본체는 scratch로 학습됩니다.

하지만 text encoder는 Gemma-2-2B-IT라는 pretrained LLM을 가져옵니다.

따라서 백본 재사용 관점에서는:

```text
DiT 본체: scratch
DC-AE: 별도 학습된 컴포넌트 사용
Text encoder: pretrained Gemma 재사용
```

가장 가까운 분류는:

```text
DiT scratch + LLM text encoder 재사용 하이브리드
```

---

## 16. 한계

Sana가 빠르다고 해서 모든 면에서 완벽한 것은 아닙니다.

논문과 공개 사례 기준으로 주의할 한계:

```text
이미지 안의 글자 렌더링은 여전히 어렵다.
손, 얼굴, 작은 디테일은 깨질 수 있다.
Linear Attention은 sharp focus가 필요한 상황에서 약할 수 있다.
학습 데이터 세부 규모와 구성은 완전히 투명하지 않다.
```

이 한계는 구조와도 연결됩니다.

```text
DC-AE 32x:
  정보량을 줄이는 대신 아주 미세한 디테일이 어려울 수 있음

Linear Attention:
  비용을 줄이는 대신 sharp token matching은 standard attention보다 약할 수 있음
```

즉 Sana는:

```text
최고 품질만을 위해 모든 비용을 쓰는 모델
```

이라기보다:

```text
품질과 비용 사이의 균형을 공격적으로 최적화한 모델
```

입니다.

---

## 17. 자주 헷갈리는 질문

### Q1. Sana는 DMD 같은 distillation 논문인가?

아닙니다.

Sana 원논문은 주로 효율적인 text-to-image backbone 설계 논문입니다.

DMD/DMD2는 이미 있는 느린 teacher를 빠른 student로 줄이는 증류 방법입니다.

Sana는:

```text
토큰 수를 줄이고
attention 비용을 줄이고
text encoder와 sampler를 효율화한다.
```

### Q2. 속도 향상의 가장 큰 이유는 Linear Attention인가?

Linear Attention도 중요하지만, 1차 효과는 DC-AE의 토큰 수 감소입니다.

1024px 기준:

```text
8x VAE:
  128 x 128 = 16,384 tokens

32x DC-AE:
  32 x 32 = 1,024 tokens
```

여기서 이미 토큰 수가 16배 줄어듭니다.

그 다음 Linear Attention이 남은 token들에 대해 `N^2` 비용을 더 줄입니다.

정리:

```text
DC-AE = 토큰 수 자체를 줄임
LiteLA = 남은 토큰의 attention 비용을 줄임
```

둘 다 필요합니다.

### Q3. Linear Attention이면 품질이 무조건 떨어지지 않나?

단독으로는 떨어질 수 있습니다.

Sana의 핵심은 Linear Attention 단독이 아니라:

```text
Linear Attention
+ Mix-FFN
+ standard cross-attention
+ DC-AE
```

입니다.

이미지 생성에서는 지역 패턴이 중요하기 때문에 Mix-FFN의 convolution이 큰 도움이 됩니다.

### Q4. 왜 cross-attention은 standard로 남겼나?

텍스트 토큰 수는 이미지 토큰 수보다 훨씬 적습니다.

그리고 text-image alignment는 특정 단어와 특정 이미지 영역을 정확히 연결해야 합니다.

그래서 Sana는:

```text
비싼 이미지 self-attention만 linear로 줄이고,
중요한 text cross-attention은 standard로 유지한다.
```

### Q5. DC-AE는 Sana가 직접 만든 건가?

Sana 팀과 겹치는 저자들이 별도 자매 논문에서 만든 오토인코더입니다.

Sana는 그 DC-AE를 가져다 사용합니다.

관련 논문:

```text
Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
arXiv: 2410.10733
```

### Q6. Gemma text encoder는 왜 안정화가 필요했나?

Gemma의 text embedding scale이 기존 T5 계열과 다르기 때문입니다.

그대로 cross-attention에 넣으면 학습 초기에 값이 너무 커져 불안정해질 수 있습니다.

Sana는:

```text
RMSNorm
+ learnable scale
```

로 text feature를 안정화합니다.

### Q7. Sana의 한계를 구조적으로 어떻게 이해하면 되나?

Sana는 비용을 줄이기 위해 정보를 강하게 압축하고 attention을 선형화합니다.

그래서:

```text
큰 구조, 전체 장면, 일반적인 prompt alignment는 잘할 수 있지만
작은 글자, 손가락, 얼굴 디테일처럼 sharp하고 미세한 정보는 어려울 수 있다.
```

라고 이해하면 됩니다.

---

## 18. 핵심 용어 사전

| 용어 | 쉬운 뜻 |
|---|---|
| DC-AE | 이미지를 32배 압축하는 오토인코더 |
| F32C32 | 32배 spatial 압축, latent channel 32개 |
| DiT | Diffusion Transformer |
| Linear Attention | `N x N` attention matrix를 만들지 않는 attention 계열 |
| LiteLA | Sana가 쓰는 ReLU 기반 Linear Attention |
| Mix-FFN | FFN 안에 3x3 convolution을 넣어 지역 정보를 보강하는 모듈 |
| NoPE | 별도 positional embedding을 쓰지 않는 설계 |
| Cross-Attention | 이미지 토큰이 텍스트 feature를 참고하는 attention |
| Gemma-2-2B-IT | Sana의 decoder-only LLM text encoder |
| CHI | prompt 앞에 복잡한 instruction을 붙이는 방법 |
| Flow Matching | noise에서 image로 가는 경로를 학습하는 diffusion 계열 방법 |
| Flow-DPM-Solver | Flow Matching 모델을 적은 step으로 샘플링하기 위한 solver |
| rFID | 오토인코더 복원 품질 지표. 낮을수록 좋음 |
| GenEval | 텍스트와 이미지의 객체/속성 정합성 평가 |
| DPG-Bench | 긴 prompt 이해를 평가하는 benchmark |

---

## 19. 전체 요약

Sana를 한 줄로 다시 쓰면:

> **Sana는 "고해상도 DiT가 느린 이유는 토큰 수와 attention 비용 때문"이라고 보고, DC-AE 32x 압축과 Linear Attention을 중심으로 전체 text-to-image 시스템을 가볍게 재설계한 모델입니다.**

가장 중요한 흐름:

```text
1. DC-AE가 1024px 이미지를 32x32 latent token으로 줄인다.
2. Linear DiT가 self-attention을 O(N^2)에서 O(N)에 가깝게 줄인다.
3. Mix-FFN이 Linear Attention의 약한 지역성을 보완한다.
4. Cross-attention은 standard로 남겨 텍스트 정합성을 챙긴다.
5. Gemma text encoder와 CHI가 prompt 이해를 강화한다.
6. Flow-DPM-Solver가 sampling step을 줄인다.
```

따라서 Sana의 핵심은:

```text
Linear Attention을 썼다
```

가 아니라:

```text
Linear Attention이 실제 이미지 생성에서 쓸 수 있도록
압축, FFN, cross-attention, text encoder, sampler를 함께 설계했다
```

입니다.

---

## 20. 관련 문서

- [PAPER_DMD.md](PAPER_DMD.md) - diffusion distillation의 기본 아이디어
- [PAPER_DMD2.md](PAPER_DMD2.md) - DMD의 후속, 1-step/4-step fast synthesis
- [PAPER_PixArt-alpha.md](PAPER_PixArt-alpha.md) - 효율형 DiT 계열 비교
- [PAPER_Z-Image.md](PAPER_Z-Image.md) - DiT, DMD 계열, RLHF를 결합한 후속 대형 모델 흐름
- Sana 논문: https://arxiv.org/abs/2410.10629
- Sana 코드: https://github.com/NVlabs/Sana
