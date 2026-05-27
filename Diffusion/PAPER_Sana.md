# PAPER: Sana - Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers 쉽게 읽기

## 0. 이 문서를 읽는 법

이 문서는 Sana 논문과 공개 코드를 보고, 처음 읽는 사람이 전체 구조를 놓치지 않도록 다시 정리한 리뷰입니다.

여기서 **쉽게 읽기**는 내용을 덜어낸다는 뜻이 아닙니다.

```text
수식은 빼지 않고, 각 기호가 무엇인지 풀어 쓴다.
그림은 빼지 않고, 그림이 말하려는 논리를 문장으로 다시 설명한다.
표는 빼지 않고, 숫자가 의미하는 결론을 해석한다.
알고리즘은 빼지 않고, 한 줄씩 어떤 역할인지 설명한다.
```

즉 이 문서의 목표는 **원문을 축약하는 것**이 아니라, 원문의 어려운 수식/그림/이론을 처음 읽는 사람도 따라갈 수 있게 번역하는 것입니다.

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
6. **본문 수식 전체 해설**: 수식의 기호와 의미
7. **그림 전체 해설**: Figure가 말하는 논리
8. **실험 결과와 FAQ**: 숫자와 헷갈리는 점

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

## 12. 본문 수식 전체 해설

이 섹션은 Sana 논문 본문과 appendix에 나오는 핵심 수식을 빠짐없이 따라가며 설명합니다.

Sana는 DMD/DMD2처럼 손실 함수 유도가 긴 논문은 아니지만, 중요한 수식이 몇 군데 있습니다.

```text
1. latent token 수 공식
2. ReLU Linear Attention 공식
3. CLIP-score caption sampling 공식
4. Flow Matching 학습 공식
5. Flow-DPM-Solver 변환 공식
6. Appendix의 Tweedie / score / data prediction 해석
7. INT8 배포에서 남기는 full precision 항목
```

### 12.1 latent 공간과 token 수 공식

원문은 먼저 이미지가 VAE/AE를 거쳐 latent로 가는 모양을 정의합니다.

```text
pixel image:
  x in R^{H x W x 3}

AE output latent:
  z in R^{H/F x W/F x C}
```

기호:

| 기호 | 뜻 |
|---|---|
| `H, W` | 원본 이미지의 높이와 너비 |
| `3` | RGB 채널 |
| `F` | autoencoder의 spatial downsample 비율 |
| `C` | latent channel 수 |

기존 SD 계열에서 흔한 설정:

```text
F = 8
```

Sana 설정:

```text
F = 32
C = 32
```

DiT는 latent grid를 그대로 token으로 쓰지 않고, 경우에 따라 patch size `P`로 한 번 더 묶습니다.

논문 수식:

```text
token grid = H / (P F) x W / (P F)

token count = H W / (P^2 F^2)
```

기호:

| 기호 | 뜻 |
|---|---|
| `P` | DiT patch size |
| `F` | AE downsample 비율 |
| `PF` | pixel 기준으로 token 하나가 담당하는 길이 |

예를 들어 1024x1024에서:

```text
PixArt/SD3/FLUX류:
  F = 8, P = 2
  token grid = 1024 / (2 x 8) = 64
  token count = 64 x 64 = 4096

Sana:
  F = 32, P = 1
  token grid = 1024 / (1 x 32) = 32
  token count = 32 x 32 = 1024
```

그래서 원문에서 `AE-F32C32P1`이라고 쓰면:

```text
F32 = autoencoder가 가로/세로를 32배 줄임
C32 = latent channel 32개
P1  = DiT가 latent를 patch로 더 묶지 않음
```

중요한 해석:

> **Sana는 patch size를 키워 DiT 쪽에서 억지로 token을 줄이기보다, autoencoder가 압축을 책임지고 DiT는 denoising에 집중하게 하자는 입장입니다.**

논문은 같은 1024px 이미지를 `32 x 32` token으로 만드는 세 설정을 비교합니다.

```text
AE-F8C16P4
AE-F16C32P2
AE-F32C32P1
```

셋 다 token 수는 같지만, Sana는 `F32C32P1`의 생성 품질이 더 좋다고 봅니다.

이 말은:

```text
token 수만 같으면 다 같은 게 아니다.
어디에서 압축하느냐가 중요하다.
```

입니다.

### 12.2 ReLU Linear Attention 공식

표준 attention은 보통 아래처럼 씁니다.

```text
Attention(Q, K, V) = softmax(Q K^T) V
```

문제는 `Q K^T`입니다.

`Q`와 `K`가 token `N`개를 가지면:

```text
Q K^T shape = N x N
```

Sana의 LiteLA는 원문 Eq. 1에서 아래처럼 씁니다.

```text
O_i
= sum_{j=1}^{N}
  [ ReLU(Q_i) ReLU(K_j)^T V_j
    / sum_{j=1}^{N} ReLU(Q_i) ReLU(K_j)^T ]

= ReLU(Q_i) ( sum_{j=1}^{N} ReLU(K_j)^T V_j )
  / ReLU(Q_i) ( sum_{j=1}^{N} ReLU(K_j)^T )
```

먼저 왼쪽을 봅니다.

```text
O_i = i번째 query token의 output
Q_i = i번째 query
K_j = j번째 key
V_j = j번째 value
N   = token 수
```

왼쪽 식은 이런 뜻입니다.

```text
i번째 token이 모든 j token을 본다.
다만 softmax 대신 ReLU(Q_i) ReLU(K_j)^T로 유사도를 만든다.
분모로 나눠 가중치 합이 너무 커지지 않게 정규화한다.
```

오른쪽 식이 진짜 핵심입니다.

```text
M_v = sum_{j=1}^{N} ReLU(K_j)^T V_j
M_k = sum_{j=1}^{N} ReLU(K_j)^T

O_i = ReLU(Q_i) M_v / ReLU(Q_i) M_k
```

`M_v`와 `M_k`는 `i`와 무관합니다.

즉:

```text
모든 query마다 새로 N개 key를 전부 비교하는 대신,
K와 V에서 공통 요약 M_v, M_k를 한 번 만들고
모든 query가 재사용한다.
```

복잡도 해석:

```text
standard attention:
  QK^T를 만들기 때문에 memory/computation이 O(N^2)

ReLU linear attention:
  K^T V 요약을 먼저 만들기 때문에 memory/computation이 O(N)에 가까움
```

왜 이게 가능한가?

```text
softmax(QK^T)는 Q와 K를 섞은 뒤 전체 토큰을 한꺼번에 정규화한다.
그래서 계산 순서를 마음대로 바꾸기 어렵다.

ReLU(Q), ReLU(K)는 각각 따로 계산된다.
그래서 sum_j를 먼저 묶어 공통 요약으로 만들 수 있다.
```

Sana 코드의 `v = pad(v, value=1)`는 위 식의 분모 `M_k`까지 같은 matmul에서 처리하기 위한 구현 trick입니다.

### 12.3 CLIP-score caption sampler 공식

Sana는 이미지 하나에 caption을 여러 개 붙입니다.

예:

```text
caption 1: 원본 prompt
caption 2: VILA-3B caption
caption 3: VILA-13B caption
caption 4: InternVL2-8B caption
caption 5: InternVL2-26B caption
```

학습 때 이 중 하나를 골라야 합니다.

무작위로 고르면 품질 낮은 caption이 뽑힐 수 있습니다.

그래서 각 caption의 CLIP score를 `c_i`라고 두고, 아래 확률로 caption을 샘플링합니다.

```text
P(c_i) = exp(c_i / tau) / sum_{j=1}^{N} exp(c_j / tau)
```

기호:

| 기호 | 뜻 |
|---|---|
| `c_i` | i번째 caption의 CLIP score |
| `N` | 해당 이미지에 붙은 caption 개수 |
| `tau` | temperature |

이 식은 softmax입니다.

해석:

```text
CLIP score가 높은 caption일수록 더 자주 뽑는다.
하지만 항상 최고 caption만 뽑지는 않고 확률적으로 고른다.
```

`tau`의 역할:

```text
tau가 작다:
  가장 높은 CLIP score caption만 거의 선택

tau가 크다:
  여러 caption을 더 고르게 선택
```

이 수식의 목적은:

```text
caption 다양성은 유지하면서
이미지와 안 맞는 caption이 학습을 망치는 확률을 줄이는 것
```

입니다.

### 12.4 Flow Matching 학습 공식

Sana의 flow 기반 학습은 공통 diffusion 표현에서 출발합니다.

원문 수식:

```text
x_t = alpha_t x_0 + sigma_t epsilon
```

기호:

| 기호 | 뜻 |
|---|---|
| `x_0` | 깨끗한 이미지 또는 latent |
| `epsilon` | random noise |
| `x_t` | 시간 `t`에서의 noisy latent |
| `alpha_t` | clean signal 비율 |
| `sigma_t` | noise 비율 |

쉽게 말하면:

```text
x_t는 깨끗한 x_0와 noise epsilon을 섞은 것
```

DDPM은 noise를 예측합니다.

```text
epsilon_theta(x_t, t) = epsilon_t
```

즉 모델에게:

```text
지금 x_t 안에 섞인 noise가 무엇인지 맞혀라
```

라고 시킵니다.

EDM은 data를 예측합니다.

```text
x_theta(x_t, t) = x_0
```

즉:

```text
지금 noisy x_t에서 원래 clean x_0를 바로 맞혀라
```

라고 시킵니다.

Rectified Flow는 velocity를 예측합니다.

```text
v_theta(x_t, t) = epsilon - x_0
```

즉:

```text
clean image에서 noise로 가는 방향,
또는 noise에서 clean image로 돌아오는 경로의 속도장을 맞혀라
```

라고 볼 수 있습니다.

Sana가 강조하는 차이:

```text
noise prediction:
  t가 큰 순수 noise 근처에서 불안정할 수 있음

data / velocity prediction:
  같은 구간에서 더 안정적인 신호를 줄 수 있음
```

그래서 Sana는 Flow Matching 계열이 DDPM schedule보다 수렴과 sampling 면에서 유리하다고 설명합니다.

### 12.5 Flow-DPM-Solver 변환 공식

Sana는 DPM-Solver++를 Rectified Flow에 맞게 바꿉니다.

첫 번째 변환은 time-step shift입니다.

원문 algorithm의 수식:

```text
tilde_sigma_{t_i}
= s sigma_{t_i} / (1 + (s - 1) sigma_{t_i})

alpha_{t_i}
= 1 - tilde_sigma_{t_i}
```

기호:

| 기호 | 뜻 |
|---|---|
| `sigma_t` | 원래 noise level |
| `tilde_sigma_t` | shift가 적용된 noise level |
| `s` | time-step shift factor |
| `alpha_t` | clean signal 쪽 계수 |

해석:

```text
sigma schedule을 그대로 쓰지 않고,
SD3류처럼 time-step shift를 줘서
sampling이 더 적절한 noise 구간을 지나도록 조정한다.
```

두 번째 변환은 velocity prediction을 data prediction으로 바꾸는 것입니다.

원문 수식:

```text
x_theta(tilde_x_{t_i}, t_i)
= tilde_x_{t_i} - tilde_sigma_{t_i} v_theta(tilde_x_{t_i}, t_i)
```

본문에서는 같은 관계를 이렇게 씁니다.

```text
data <- x_0 = x_T - sigma_T v_theta(x_T, t_T)
```

해석:

```text
모델은 velocity v_theta를 예측하지만,
DPM-Solver++는 data prediction x_theta 형태를 잘 다룬다.

그래서 velocity 출력에서 clean data 예측값을 만들어
solver 안에서는 x_theta처럼 사용한다.
```

이게 Flow-DPM-Solver의 핵심 변환입니다.

### 12.6 Flow-DPM-Solver update 공식

Algorithm에서는 먼저 아래를 정의합니다.

```text
h_i := lambda_{t_i} - lambda_{t_{i-1}}
```

`lambda_t`는 DPM-Solver 계열에서 쓰는 log-SNR류 시간 좌표로 보면 됩니다.

첫 step update:

```text
tilde_x_{t_1}
= (tilde_sigma_{t_1} / tilde_sigma_{t_0}) tilde_x_{t_0}
  - alpha_{t_1} (e^{-h_1} - 1) x_theta(tilde_x_{t_0}, t_0)
```

이 식은:

```text
현재 noisy latent tilde_x_{t_0}
모델이 예측한 clean data x_theta
noise schedule 비율
```

을 이용해 다음 시점 `t_1`의 latent로 이동하는 update입니다.

두 번째 step부터는 2차 multistep update를 씁니다.

```text
r_i = h_{i-1} / h_i
```

```text
D_i
= (1 + 1 / (2 r_i)) x_theta(tilde_x_{t_{i-1}}, t_{i-1})
   - (1 / (2 r_i)) x_theta(tilde_x_{t_{i-2}}, t_{i-2})
```

```text
tilde_x_{t_i}
= (tilde_sigma_{t_i} / tilde_sigma_{t_{i-1}}) tilde_x_{t_{i-1}}
  - alpha_{t_i} (e^{-h_i} - 1) D_i
```

해석:

```text
직전 한 번의 예측만 보지 않고,
이전 두 step의 clean prediction을 조합해 D_i를 만든다.
그 D_i로 다음 latent를 업데이트한다.
```

그래서 Euler보다 더 적은 step으로 안정적인 sampling이 가능합니다.

논문 결과:

```text
Flow-Euler: 보통 28~50 step 필요
Flow-DPM-Solver: 14~20 step에서 수렴
```

### 12.7 Appendix의 score / Tweedie 공식

Appendix는 왜 noise prediction이 `t ≈ T`에서 불리한지 설명합니다.

먼저 score를 씁니다.

```text
nabla_{x_t} log q_t(x_t)
= - ( x_t - alpha_t E_{q_{0t}(x_0 | x_t)}[x_0] ) / sigma_t^2
```

기호:

| 기호 | 뜻 |
|---|---|
| `q_t(x_t)` | 시간 t의 noisy data 분포 |
| `nabla_{x_t} log q_t(x_t)` | score, 즉 density가 커지는 방향 |
| `E[x_0 | x_t]` | 현재 noisy sample을 봤을 때 원본 x0의 조건부 평균 |

해석:

```text
score는 지금 x_t가 data 분포 쪽으로 가려면 어느 방향으로 움직여야 하는지 알려준다.
```

`t ≈ T`, 즉 거의 순수 noise 근처에서는 `x_t`만 보고 원본 `x_0`를 알기 어렵습니다.

그래서:

```text
q_{0t}(x_0 | x_t) ≈ q_0(x_0)
```

즉:

```text
x_t를 봐도 x_0에 대한 정보가 거의 없으니,
조건부 분포가 그냥 원래 data 분포와 비슷해진다.
```

그때 noise prediction의 최적해는:

```text
epsilon_theta(x_t, t)
≈ - sigma_t nabla_{x_t} log q_t(x_t)
≈ ( x_t - alpha_t E_{q_0(x_0)}[x_0] ) / sigma_t
```

중요한 해석:

```text
E_{q_0(x_0)}[x_0]는 x_t와 무관한 상수에 가깝다.
따라서 epsilon_theta는 x_t에 대한 거의 선형 함수가 된다.
```

논문은 이 선형성이 sampling error를 누적시키기 쉽다고 봅니다.

반대로 data prediction은:

```text
x_theta(x_t, t)
≈ ( x_t + sigma_t^2 nabla_{x_t} log q_t(x_t) ) / alpha_t
≈ E_{q_0(x_0)}[x_0]
```

즉 `t ≈ T`에서 거의 상수에 가까워집니다.

해석:

```text
noise prediction:
  x_t에 따라 계속 흔들리는 선형 함수처럼 됨

data prediction:
  data 평균 쪽의 안정적인 상수처럼 됨
```

그래서 DPM-Solver++ 계열은 data prediction parameterization이 더 안정적이고, Sana는 velocity prediction을 data prediction 형태로 바꿔 Flow-DPM-Solver에 넣습니다.

### 12.8 INT8 양자화에서 full precision으로 남기는 부분

배포 섹션은 긴 수식은 없지만, 계산 형식이 중요합니다.

Sana는 W8A8 quantization을 씁니다.

```text
W8A8 = weight 8-bit + activation 8-bit
```

구체적으로:

```text
activation: per-token symmetric INT8 quantization
weight:     per-channel symmetric INT8 quantization
```

하지만 전부 INT8로 바꾸지는 않습니다.

full precision으로 남기는 부분:

```text
normalization layers
linear attention
cross-attention의 key-value projection layers
```

이유:

```text
이 부분들은 semantic similarity와 안정성에 민감하다.
여기를 무리하게 INT8로 내리면 품질 손실이 커질 수 있다.
```

또 논문은 Linear Attention의:

```text
ReLU(K)^T V
```

곱을 `QKV projection`과 fuse한다고 설명합니다.

즉:

```text
수학적으로는 같은 계산이지만,
GPU에서는 중간 tensor를 메모리에 쓰고 다시 읽는 비용을 줄이도록
kernel을 합친다.
```

---

## 13. 논문 그림 전체 해설

이 섹션은 논문 그림을 "그림이 무슨 말을 하려는지" 중심으로 다시 풉니다.

Sana의 그림은 단순한 예시 이미지가 아니라, 논문의 주장 구조를 단계별로 보여줍니다.

```text
Figure 1-2: Sana가 얼마나 빠른지 보여주는 문제 제기
Figure 3: DC-AE가 왜 필요한지 보여주는 ablation
Figure 4: 1024px에서 모델 크기/속도/품질 비교
Figure 5: Sana 전체 아키텍처
Figure 6-7: Gemma text encoder 안정화와 CHI 효과
Figure 8: Flow-DPM-Solver가 step을 줄이는 효과
Figure 9: 시각 비교와 laptop 배포
Appendix figures: 더 많은 샘플, zero-shot, caption pipeline, solver 비교
```

### 13.1 Figure 1: 생성 예시와 latency teaser

Figure 1은 논문의 첫인상입니다.

이 그림이 말하려는 것은:

```text
Sana는 고해상도 이미지를 만들 수 있고,
그 속도가 기존 대형 모델보다 훨씬 빠르다.
```

여기서 중요한 건 샘플 이미지 자체보다, 논문의 질문을 시각적으로 던지는 방식입니다.

```text
좋은 이미지를 만들 수 있는가?
그 이미지를 빠르게 만들 수 있는가?
고해상도에서도 가능한가?
```

Figure 1은 이 세 질문에 대해 "가능하다"는 방향으로 독자를 끌고 갑니다.

### 13.2 Figure 2: 4K latency를 줄이는 전체 경로

Figure 2는 Sana의 시스템 최적화가 어디서 오는지 보여줍니다.

핵심 메시지:

```text
4K 생성 latency를 469초에서 9.6초까지 줄인다.
```

이 숫자는 단일 trick 하나로 얻은 것이 아닙니다.

그림을 이렇게 읽으면 됩니다.

```text
FLUX-dev 같은 full attention 대형 모델:
  4K에서 token 수와 attention 비용이 폭발

Sana:
  DC-AE로 token 수 감소
  Linear Attention으로 attention 비용 감소
  Flow-DPM-Solver로 step 수 감소
  kernel fusion / quantization으로 실제 runtime 감소
```

즉 Figure 2는 "Sana = 여러 효율화의 곱셈"이라는 논문 전체 주장을 압축한 그림입니다.

### 13.3 Figure 3: DC-AE ablation

Figure 3은 DC-AE 설계가 왜 `F32C32P1`로 가는지 보여줍니다.

논문은 같은 token 수를 만드는 세 가지 방법을 비교합니다.

```text
AE-F8C16P4
AE-F16C32P2
AE-F32C32P1
```

셋 다 1024px 이미지를 `32 x 32` token으로 만들 수 있습니다.

하지만 압축을 어디서 하느냐가 다릅니다.

```text
AE-F8C16P4:
  AE는 약하게 압축하고, DiT patch size를 크게 해서 token을 줄임

AE-F16C32P2:
  AE와 DiT patch가 압축을 나눠 가짐

AE-F32C32P1:
  AE가 강하게 압축하고, DiT는 patch로 더 묶지 않음
```

Figure 3의 결론:

```text
복원 rFID만 보면 F8 쪽이 좋아 보일 수 있지만,
최종 생성 FID는 F32C32P1이 더 좋다.
```

쉽게 말하면:

> **DiT가 큰 patch 안에 섞인 정보를 다시 풀어내게 만들기보다, AE가 압축을 책임지고 DiT는 denoising에 집중하는 편이 낫다.**

또 Figure 3은 channel 수 `C`도 비교합니다.

```text
C=16:
  가볍지만 복원 품질 부족

C=32:
  복원 품질과 DiT 학습 속도의 균형이 좋음

C=64:
  복원은 더 좋지만 DiT 학습이 느려짐
```

그래서 Sana는 `F32C32P1`을 선택합니다.

### 13.4 Figure 4: 1024px에서 모델 크기와 성능 비교

Figure 4는 1024x1024 생성에서 Sana가 어디에 위치하는지 보여주는 bubble plot입니다.

읽는 법:

```text
x축 또는 y축:
  latency, throughput, 품질 지표 등이 비교됨

bubble 크기:
  모델 parameter 수를 나타냄
```

Sana가 보여주려는 포인트:

```text
Sana-0.6B는 작은 bubble인데,
속도는 빠르고,
GenEval 같은 text-image alignment도 강하다.
```

즉 Figure 4는:

```text
"작은 모델 = 약한 모델"이 아니라,
"구조를 잘 설계하면 작은 모델도 경쟁력 있다"
```

는 주장을 시각적으로 보여줍니다.

### 13.5 Figure 5: Sana 전체 구조

현재 로컬 문서에는 Figure 5 이미지가 있습니다.

<p align="center">
  <img src="figures/sana_fig5.png" alt="Sana Figure 5 overview" width="900"/>
</p>

Figure 5는 Sana에서 가장 중요한 그림입니다.

그림은 크게 두 부분입니다.

```text
(a) 전체 pipeline
(b) Linear DiT block 내부
```

(a)는 이렇게 읽으면 됩니다.

```text
image
  -> 32x DC-AE
  -> compressed latent
  -> Linear DiT
  -> denoising / velocity prediction

prompt
  -> Complex Human Instruction 붙임
  -> Gemma text encoder
  -> cross-attention condition
```

빨간 X로 표시된 positional embedding 제거도 중요합니다.

```text
NoPE:
  별도 position embedding을 쓰지 않음
  Mix-FFN의 3x3 conv와 zero padding이 위치 단서를 일부 제공
```

(b)는 Linear DiT block을 보여줍니다.

```text
Linear Attention:
  Q, K에 ReLU 적용
  K^T V를 먼저 계산
  N x N attention map을 만들지 않음

Mix-FFN:
  1x1 conv
  3x3 depthwise conv
  GLU gate
  1x1 conv
```

Figure 5의 핵심 결론:

> **Sana는 Linear Attention만 쓰는 모델이 아니라, DC-AE + Linear Attention + Mix-FFN + Gemma condition을 한 pipeline으로 묶은 모델입니다.**

### 13.6 Figure 6: text embedding normalization과 scale factor

Figure 6은 Gemma를 text encoder로 붙일 때 왜 안정화가 필요한지 보여줍니다.

비교 대상:

```text
text embedding normalization 없음
RMSNorm 있음
RMSNorm + scale factor 1
RMSNorm + scale factor 0.01
```

그림과 표의 결론:

```text
normalization 없으면 NaN 발생
normalization만 해도 학습 가능
작은 scale factor 0.01을 곁들이면 convergence가 더 좋음
```

이 그림은 "Gemma를 그냥 가져다 붙이면 된다"가 아니라는 점을 보여줍니다.

쉽게 말하면:

```text
Gemma feature는 너무 세게 들어온다.
먼저 RMSNorm으로 크기를 맞추고,
처음에는 scale 0.01로 조심스럽게 DiT에 주입한다.
```

### 13.7 Figure 7: CHI 효과

Figure 7은 Complex Human Instruction을 붙였을 때와 붙이지 않았을 때 생성 결과를 비교합니다.

핵심:

```text
짧고 애매한 prompt만 주면 Gemma가 항상 이미지 생성에 필요한 방향으로 feature를 만들지 않을 수 있다.
CHI를 붙이면 Gemma가 prompt를 이미지 조건으로 더 안정적으로 해석한다.
```

예를 들어 "a cat"처럼 짧은 prompt는 너무 열려 있습니다.

CHI는 Gemma에게 이런 역할을 시킵니다.

```text
이 문장을 이미지 생성 조건으로 해석해라.
객체, 속성, 장면 정보를 놓치지 마라.
불필요한 대화형 응답이 아니라 visual condition을 만들라.
```

그래서 Figure 7은 CHI가 단순 prompt engineering이 아니라, decoder-only LLM을 text encoder로 쓰기 위한 interface 설계임을 보여줍니다.

### 13.8 Figure 8: Flow-DPM-Solver vs Flow-Euler

Figure 8은 sampling step 수에 따른 FID와 CLIP score를 비교합니다.

읽는 법:

```text
x축:
  sampling step 수

y축:
  FID 또는 CLIP score

비교:
  Flow-Euler
  Flow-DPM-Solver
```

결론:

```text
Flow-Euler는 28~50 step 정도 필요
Flow-DPM-Solver는 14~20 step에서 더 빨리 수렴
```

이 그림은 Sana 속도의 마지막 축입니다.

```text
DC-AE와 Linear Attention:
  step 한 번의 비용을 줄임

Flow-DPM-Solver:
  필요한 step 수를 줄임
```

### 13.9 Figure 9: 시각 비교와 laptop 배포

Figure 9는 두 가지를 보여줍니다.

```text
왼쪽:
  Sana-1.6B, FLUX-dev, SD3, PixArt-Sigma 생성 결과 비교

오른쪽:
  quantized Sana가 laptop GPU에서 1K 이미지를 1초 안에 생성하는 데모
```

이 그림의 메시지:

```text
Sana는 단순히 benchmark 숫자만 좋은 것이 아니라,
실제 생성 이미지도 경쟁력 있고,
on-device 배포까지 가능하다.
```

다만 시각 비교는 항상 prompt 선택과 샘플 선택 영향을 받습니다.

그래서 Figure 9는:

```text
정량표 Table 7
+ 시각 결과 Figure 9
+ on-device Table 5
```

를 함께 봐야 합니다.

### 13.10 Appendix 그림들

Appendix 그림들도 해석 가치가 있습니다.

| 그림 | 보여주는 것 | 쉬운 해석 |
|---|---|---|
| `ae_compare` | 원본, DC-AE-F32C32, SDXL VAE 복원 비교 | 32x 압축인데도 복원이 크게 무너지지 않는지 확인 |
| `flow-dpm-vs-euler` | Flow-Euler 50 step과 Flow-DPM-Solver 5/8/14/20 step 시각 비교 | solver step을 줄여도 이미지가 언제 안정되는지 확인 |
| `sample_with_mulcap` | 하나의 이미지에 여러 VLM caption을 붙이는 예 | multi-caption pipeline이 실제로 어떻게 생겼는지 보여줌 |
| `zero-shot` | 중국어/이모지 prompt 이해 | Gemma pretraining 덕분에 영어 학습만으로도 일부 언어 일반화 |
| `Sana_vis` | 추가 생성 샘플 | 다양한 prompt에서 생성 품질 확인 |
| `performance-of-different-parts` | 각 효율화 부품의 latency 영향 | 속도 개선이 어느 부품에서 오는지 분해 |

Appendix 그림을 볼 때 핵심은:

```text
main figure는 논문 주장을 보여주고,
appendix figure는 그 주장이 우연이 아님을 보강한다.
```

입니다.

---

## 14. 논문 표 전체 해설

이 섹션은 본문 Table 1~9가 각각 무엇을 증명하려는 표인지 설명합니다.

표를 읽을 때는 숫자 하나하나보다, 저자가 그 표로 어떤 선택을 정당화하는지 보는 것이 중요합니다.

### 14.1 Table 1: Autoencoder reconstruction 비교

Table 1은 DC-AE가 32x 압축을 해도 복원 품질이 크게 무너지지 않는지 보여줍니다.

| Autoencoder | 압축 | rFID | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| F8C4 SDXL | 8x | 0.31 | 31.41 | 0.88 | 0.04 |
| F32C64 SD | 32x | 0.82 | 27.17 | 0.79 | 0.09 |
| F32C32 Sana | 32x | 0.34 | 29.29 | 0.84 | 0.05 |

해석:

```text
기존 방식으로 32x 압축하면 rFID가 0.82로 나빠진다.
Sana의 F32C32는 rFID 0.34로 SDXL VAE 0.31에 가깝다.
```

즉 Table 1은:

> **32x 압축이 무리한 아이디어가 아니라, DC-AE 설계 덕분에 실제로 쓸 수 있는 압축이라는 근거입니다.**

### 14.2 Table 2: CHI ablation

Table 2는 Complex Human Instruction을 붙이면 GenEval이 좋아지는지 봅니다.

| Prompt | Train Step | GenEval |
|---|---:|---:|
| User | 52K | 45.5 |
| CHI + User | 52K | 47.7 |
| User | 140K | 52.8 |
| CHI + User + 5K finetune | 145K | 54.8 |

해석:

```text
초기 학습에서도 CHI가 +2.2 개선
긴 학습 후 finetune에서도 +2.0 개선
```

즉 CHI는 단순한 예쁜 prompt가 아니라, Gemma text encoder가 이미지 조건을 더 잘 만들게 하는 장치입니다.

### 14.3 Table 3: DDPM vs Flow Matching training schedule

Table 3은 같은 256px resolution에서 DDPM과 Flow Matching을 비교합니다.

| Schedule | FID | CLIP | Iterations |
|---|---:|---:|---:|
| DDPM | 19.5 | 24.6 | 120K |
| Flow Matching | 16.9 | 25.7 | 120K |

해석:

```text
같은 iteration이라면 Flow Matching이 FID도 낮고 CLIP도 높다.
```

Sana가 Flow Matching을 쓰는 이유:

```text
학습 수렴이 빠르고,
data/velocity prediction이 noise prediction보다 안정적이기 때문
```

### 14.4 Table 4: caption sampling 전략

Table 4는 multi-caption을 어떻게 고를지 비교합니다.

| Prompt Strategy | FID | CLIP |
|---|---:|---:|
| Single | 6.13 | 27.10 |
| Multi-random | 6.15 | 27.13 |
| Multi-clipscore | 6.12 | 27.26 |

해석:

```text
FID 차이는 작다.
CLIP은 multi-clipscore가 가장 좋다.
```

즉:

```text
caption 여러 개를 쓰되,
CLIP score가 높은 caption을 더 자주 뽑으면
semantic alignment가 조금 좋아진다.
```

이 표가 12.3의 softmax sampling 수식과 연결됩니다.

### 14.5 Table 5: on-device INT8 배포

Table 5는 W8A8 quantization이 속도와 품질에 미치는 영향을 보여줍니다.

| Methods | Latency | CLIP-Score | ImageReward |
|---|---:|---:|---:|
| Sana FP16 | 0.88s | 28.5 | 1.03 |
| W8A8 Quantization | 0.37s | 28.3 | 0.96 |

해석:

```text
속도는 2.4x 빨라진다.
CLIP과 ImageReward는 조금만 떨어진다.
```

중요한 점:

```text
모든 연산을 무조건 INT8로 내린 것이 아니다.
민감한 normalization, linear attention, cross-attn KV projection은 full precision으로 남긴다.
```

그래서 속도와 품질 사이의 균형을 맞춥니다.

### 14.6 Table 6: Sana architecture details

Table 6은 모델 크기를 정의합니다.

| Model | Width | Depth | FFN | Heads | Params |
|---|---:|---:|---:|---:|---:|
| Sana-0.6B | 1152 | 28 | 2880 | 36 | 590M |
| Sana-1.6B | 2240 | 20 | 5600 | 70 | 1604M |

해석:

```text
0.6B는 DiT-XL/PixArt 계열과 비슷한 규모
1.6B는 width와 FFN을 키운 더 강한 모델
```

흥미로운 점:

```text
1.6B는 depth가 20으로 0.6B의 28보다 얕다.
대신 width와 FFN dimension이 훨씬 크다.
```

Sana는 20~30 layer 정도가 효율과 품질의 균형이라고 봅니다.

### 14.7 Table 7: SOTA 성능 비교

Table 7은 논문의 메인 결과표입니다.

1024px 기준 핵심 행만 보면:

| Model | Throughput | Latency | Params | FID | CLIP | GenEval | DPG |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLUX-dev | 0.04/s | 23.0s | 12B | 10.15 | 27.47 | 0.67 | 84.0 |
| PixArt-Sigma | 0.4/s | 2.7s | 0.6B | 6.15 | 28.26 | 0.54 | 80.5 |
| Sana-0.6B | 1.7/s | 0.9s | 0.6B | 5.81 | 28.36 | 0.64 | 83.6 |
| Sana-1.6B | 1.0/s | 1.2s | 1.6B | 5.76 | 28.67 | 0.66 | 84.8 |

해석:

```text
Sana-0.6B는 PixArt-Sigma와 같은 0.6B급인데 더 빠르고 지표도 좋다.
Sana-1.6B는 FLUX-dev보다 훨씬 작고 빠르며 DPG는 비슷하거나 높다.
GenEval은 FLUX-dev 0.67, Sana-1.6B 0.66으로 거의 근접한다.
```

주의:

```text
표의 FID/CLIP/GenEval/DPG는 benchmark별 일부 측면만 본다.
실제 체감 품질은 prompt, aesthetic, text rendering, 손/얼굴 디테일에 따라 달라진다.
```

### 14.8 Table 8: block design space

Table 8은 Sana의 효율이 어디서 오는지 분해합니다.

| Blocks | AE | MACs | Throughput | Latency |
|---|---|---:|---:|---:|
| FullAttn + FFN | F8C4P2 | 6.48T | 0.49/s | 2250ms |
| + LinearAttn | F8C4P2 | 4.30T | 0.52/s | 1931ms |
| + MixFFN | F8C4P2 | 4.19T | 0.46/s | 2425ms |
| + Kernel Fusion | F8C4P2 | 4.19T | 0.53/s | 2139ms |
| LinearAttn + MixFFN | F32C32P1 | 1.08T | 1.75/s | 826ms |
| + Kernel Fusion | F32C32P1 | 1.08T | 2.06/s | 748ms |

해석:

```text
LinearAttn만 넣으면 MACs는 줄지만, MixFFN은 latency를 다시 늘릴 수 있다.
Kernel fusion이 그 runtime 손실을 일부 회복한다.
가장 큰 jump는 F8C4P2에서 F32C32P1로 바뀌는 순간이다.
```

즉 Table 8의 결론:

> **Sana의 속도는 Linear Attention 하나가 아니라, F32C32P1 압축 + LinearAttn + MixFFN + kernel fusion의 조합에서 나온다.**

### 14.9 Table 9: text encoder 비교

Table 9는 T5와 Gemma 계열 text encoder를 비교합니다.

| Text Encoder | Params | Latency | FID | CLIP |
|---|---:|---:|---:|---:|
| T5-XXL | 4762M | 1.61s | 6.1 | 27.1 |
| T5-Large | 341M | 0.17s | 6.1 | 26.2 |
| Gemma2-2B | 2614M | 0.28s | 6.0 | 26.9 |
| Gemma-2B-IT | 2506M | 0.21s | 5.9 | 26.8 |
| Gemma2-2B-IT | 2614M | 0.28s | 6.1 | 26.9 |

해석:

```text
T5-XXL은 크고 느리다.
T5-Large는 빠르지만 CLIP이 낮다.
Gemma-2B 계열은 T5-XXL보다 훨씬 빠르면서 비슷한 FID/CLIP을 낸다.
```

Sana가 Gemma를 고른 이유:

```text
속도
instruction-following
작은 LLM의 pretrained knowledge
```

단, Table 9만 보면 Gemma가 압도적으로 모든 지표에서 이긴다는 뜻은 아닙니다.

진짜 핵심은:

```text
Gemma를 안정화해서 text encoder로 쓸 수 있게 만들었고,
CHI와 결합하면 text-image alignment가 좋아진다.
```

입니다.

---

## 15. 실험 결과를 어떻게 읽어야 하나?

### 15.1 1024x1024 결과

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

### 15.2 4K 결과

Sana가 특히 강조하는 지점은 4K입니다.

고해상도로 갈수록 standard attention의 `N^2` 비용이 커집니다.

Sana는:

```text
DC-AE로 토큰 수를 줄이고
Linear Attention으로 attention 비용 증가를 완화한다.
```

그래서 4K에서 상대 속도 이득이 더 커집니다.

### 15.3 RTX 4090 / laptop GPU 의미

Sana는 "거대 서버 GPU에서만 되는 모델"이 아니라, 16GB급 GPU나 consumer GPU에서도 돌릴 수 있는 효율성을 강조합니다.

이것이 Sana의 실용적 의미입니다.

```text
품질만 최고로 끌어올린 논문이라기보다
고해상도 생성 비용을 낮추는 시스템 논문
```

---

## 16. Sana의 기여를 한 번에 정리

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

## 17. DMD/DMD2와의 연결

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

## 18. 다른 모델과 비교해서 이해하기

### 18.1 PixArt 계열과의 관계

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

### 18.2 FLUX와의 관계

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

### 18.3 백본 재사용 관점

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

## 19. 한계

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

## 20. 자주 헷갈리는 질문

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

## 21. 핵심 용어 사전

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

## 22. 전체 요약

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

## 23. 관련 문서

- [PAPER_DMD.md](PAPER_DMD.md) - diffusion distillation의 기본 아이디어
- [PAPER_DMD2.md](PAPER_DMD2.md) - DMD의 후속, 1-step/4-step fast synthesis
- [PAPER_PixArt-alpha.md](PAPER_PixArt-alpha.md) - 효율형 DiT 계열 비교
- [PAPER_Z-Image.md](PAPER_Z-Image.md) - DiT, DMD 계열, RLHF를 결합한 후속 대형 모델 흐름
- Sana 논문: https://arxiv.org/abs/2410.10629
- Sana 코드: https://github.com/NVlabs/Sana
