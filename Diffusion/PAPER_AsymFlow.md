# PAPER: AsymFlow — 노이즈 항만 저랭크로 압축해 픽셀 공간 flow matching을 부활시킨 파라미터화

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Asymmetric Flow Models |
| **저자** | Hansheng Chen, Jan Ackermann, Minseo Kim, Gordon Wetzstein, Leonidas Guibas (Stanford 외) |
| **공개일** | 2026-05-13 (arXiv v1) |
| **분야** | 이미지 생성 / Flow Matching / cs.CV |
| **논문 링크** | https://arxiv.org/abs/2605.12964 |
| **공식 코드** | https://github.com/Lakonik/LakonLab (`configs/asymflow/`) |
| **사용한 외부 모델** | FLUX.2 klein Base 9B (Black Forest Labs), Qwen2.5-VL (캡션 생성) |
| **데이터** | ImageNet-256 (scratch), LAION-Aesthetics 3M (T2I 미세조정) |
| **본 문서 목적** | AsymFlow 파라미터화·복원 공식·PCA/Procrustes 사영·미세조정 절차를 한 페이지로 정리 |

### 위치 짚기 (왜 이 논문이 중요한가)

- **1저자 Hansheng Chen의 flow matching 3연작 중 가장 임팩트 큰 결과** — GMFlow (ICML 2025, 출력 분포를 Gaussian Mixture로) → π-Flow (ICLR 2026, policy 기반 few-step distillation) → AsymFlow (2026, 노이즈만 저랭크 사영). 세 편 모두 "flow matching의 표현·파라미터화를 손본다"는 일관된 라인.
- **"latent 공간 모델이 픽셀 공간 모델보다 무조건 우월하다"는 통념을 뒤집은 첫 결과 중 하나** — JiT-H/16 픽셀 모델로 FID 1.57 달성, latent 기반 DiT-XL/RAE(1.50)와 동급, REPA-XL/2(1.38)에 근접.
- **Pretrained latent flow → pixel space 첫 미세조정 방법** — FLUX.2 klein 9B를 LoRA + Procrustes alignment + Oklab 정규화 조합으로 픽셀 버전(AsymFLUX.2 klein)으로 옮김.
- **[[paper_min_snr]]·[[paper_lumina_next]] 와 같은 "한 군데만 손대는 minimal trick" 계열** — 네트워크 구조·옵티마이저·스케줄러 한 줄도 안 바뀜.

---

## 📖 주요 용어 사전 (Glossary)

### Flow matching 기본 (이 논문이 가정하는 배경)

- **Flow matching**: 깨끗한 데이터 `x₀` 와 노이즈 `ε` 사이를 직선으로 보간하는 시간 t 의 경로를 따라 *속도 (velocity)* 를 학습하는 생성 방식. Stable Diffusion 3, FLUX, Lumina-Next 등이 채택.
- **보간식 (interpolant)**: `x_t = (1−t)·x₀ + t·ε` — 진짜 데이터와 가우시안 노이즈를 `t` 비율로 섞은 중간 상태.
- **속도 u (velocity)**: 정답으로 학습하는 시간 도함수. `u = ε − x₀` (rectified flow / linear interpolant 기준).
- **노이즈 강도 σ_t**: 시점 `t` 에서의 노이즈 비중 (rectified flow 에선 `σ_t = t`).
- **u-prediction**: 네트워크가 속도 `u` 를 직접 예측. 표준 flow matching.
- **x₀-prediction**: 네트워크가 깨끗한 데이터 `x₀` 자체를 예측. 옛 DDPM 계열.

### AsymFlow 핵심 (본 논문의 신규 개념)

- **비대칭 속도 (asymmetric velocity, u_A)**: AsymFlow 의 새 학습 타깃. `u_A = P·ε − x₀`. 노이즈 항만 저랭크 사영을 거치고 데이터 항은 그대로.
- **저랭크 사영기 (low-rank projector, P)**: `P = A·Aᵀ`. `A ∈ ℝ^(D×r)` 는 정규직교 기저 (orthonormal basis, `AᵀA = I_r`). `P` 의 상 (image) `Im(P)` 는 r 차원 부분공간, `I−P` 의 상은 그 직교 여집합 (orthogonal complement).
- **저랭크 노이즈 (low-rank noise)**: `P·ε`. 원래 D차원 무작위 벡터인 `ε` 를 r 개 주성분 방향으로만 떨어뜨린 것.
- **패치별 사영 (patch-wise projection)**: 같은 `A` 를 모든 patch 토큰에 공유 적용. patch 차원 `D` 안에서만 저랭크 압축, 토큰 수는 그대로.
- **랭크 r (rank)**: 부분공간 차원. ImageNet (JiT-H/16) 에서 `r=8`, AsymFLUX.2 klein 에서 `r=128`.
- **PCA basis (scratch 학습용 A 만드는 법)**: 학습 데이터셋의 patch 들에 PCA (Principal Component Analysis, 주성분 분석) 를 돌려 상위 `r` 개 방향을 `A` 로 사용.
- **Procrustes alignment (미세조정용 A 만드는 법)**: latent 변수와 대응하는 pixel patch 사이의 *직교 회전 최적 매칭* (orthogonal Procrustes problem — 두 점 집합을 직교 변환으로 가장 잘 맞추는 문제) 으로 `A` 계산. latent → pixel 초기화를 매끄럽게 함.
- **복원 공식 (full-rank velocity recovery)**: 네트워크 출력 `û_A` 로부터 진짜 속도 `û` 를 만드는 항등식. `û = P·û_A + (I−P)·(x_t + û_A)/σ_t`.

### AsymFLUX.2 klein 미세조정 관련

- **Oklab color space**: perceptually uniform 한 색공간 (perceptual = 사람 눈이 느끼는 색 차이가 좌표 거리와 비례). VAE 를 떼는 대신 픽셀을 Oklab 으로 변환해 학습 분포를 정규화 (mean=(0.56, 0, 0.01), std=0.16).
- **LoRA (Low-Rank Adaptation)**: 사전학습 가중치 `W` 를 고정한 채 `ΔW = B·A` (rank 256) 형태의 어댑터만 학습. AsymFlow 의 P 사영과 *이름은 비슷하지만 완전히 다른 자리* — LoRA 는 가중치 측, P 는 데이터/노이즈 측.
- **APG (Adaptive Projected Guidance)**: classifier-free guidance 의 직교 사영 변형. AsymFLUX.2 klein 샘플링 시 UniPC 솔버와 함께 사용.
- **UniPC (Unified Predictor-Corrector)**: 적은 step 으로 ODE 를 풀기 위한 고차 솔버. 추론 38 step.

### 비교 기법·평가

- **JiT (Just image Transformer)**: 본 논문 1저자 그룹의 픽셀 공간 ViT 기반 디퓨전 백본. AsymFlow scratch 실험의 베이스. H/16 = Huge + patch size 16.
- **REPA (REPresentation Alignment) loss**: DINOv2 등 자기지도학습 특징과 디퓨전 내부 특징을 정렬하는 보조 손실. ImageNet 1.57 FID 달성 시 사용.
- **DiT-XL/RAE / REPA-XL**: 본 논문에서 비교한 latent 기반 baseline. 픽셀 모델이 이를 넘어서는 게 핵심 메시지.
- **FID / HPSv3 / DPG-Bench / GenEval**: 평가 지표. FID (이미지 분포 거리), HPSv3 (인간 선호), DPG-Bench·GenEval (지시이행 정확도).

### 부록: 본문에서 자주 등장하는 추상 용어

- **사영 (projection)**: 벡터를 특정 부분공간으로 떨어뜨리는 선형 연산. `P` 는 멱등 (`P² = P`) + 대칭 (`Pᵀ = P`).
- **직교 여집합 (orthogonal complement)**: 어떤 부분공간 `V` 와 *수직인 모든 벡터*의 집합. `Im(I−P)` 가 그것. 두 공간을 더하면 전체 공간 복원.
- **Rank (랭크)**: 부분공간 차원, 행렬의 독립 열/행 개수.
- **Full-rank (풀랭크)**: 차원이 최대 (`D` 전체) 인 상태.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> 픽셀 공간 flow matching이 latent 공간 모델보다 늘 뒤졌던 이유는 네트워크가 *고차원 노이즈를 풀랭크로 다 모델링* 해야 하기 때문. AsymFlow 는 학습 타깃을 `u = ε − x₀` 에서 `u_A = P·ε − x₀` 로 바꿔, **노이즈 항만 r차원 부분공간으로 압축** 한다. 데이터 항은 그대로. 진짜 속도는 `û = P·û_A + (I−P)·(x_t + û_A)/σ_t` 로 항등식 복원. ImageNet 256×256 에서 **FID 1.57** (JiT-H/16 + REPA), FLUX.2 klein 9B 픽셀 미세조정으로 T2I 벤치 (HPSv3/DPG/GenEval) SOTA.

**핵심 문제** — 픽셀 공간 (예: 256×256×3 = 196,608 차원) 에서 직접 flow matching 을 돌리면 왜 잘 안 되나?

→ 네트워크가 *학습 정답에 포함된 무작위 노이즈* 를 픽셀 단위로 다 예측해야 함. 데이터 매니폴드는 사실상 저랭크인데 (수십~수백 방향이면 충분) 노이즈는 풀랭크라 capacity 낭비.

**해결책** — 3 단계:

1. **타깃 단순화** (Fig.2 (a))**: `u_A = P·ε − x₀`. 노이즈 항을 r 차원 부분공간으로 사영. 데이터 항은 풀랭크 유지.
2. **항등식 복원** (Fig.2 (b))**: 학습/샘플링 시 `û_A` 를 두 직교 성분으로 분해 → 부분공간 안쪽은 그대로 사용, 바깥쪽은 `x₀` 스타일 → 속도 스타일로 환산 후 합산.
3. **A 행렬 두 가지 구성법**: scratch 학습에선 데이터 patch 의 PCA, 사전학습 latent 모델 미세조정에선 latent ↔ pixel Procrustes alignment.

**검증**:
- ImageNet 256×256: JiT-H/16 + AsymFlow (`r=8`) → **FID 1.76**, +REPA → **FID 1.57**
- T2I: FLUX.2 klein 9B → AsymFLUX.2 klein (`r=128`, LoRA rank-256, 32 GPU × 20K iter) → HPSv3/DPG-Bench/GenEval 모두 동급 latent baseline 상회
- ImageNet 실험은 **LoRA 없이 from-scratch 풀학습** — 1.57 FID 가 순수 파라미터화 트릭 + REPA 의 효과임을 증명

---

## 2️⃣ 핵심 기여 (Contributions)

1. **Rank-asymmetric velocity parameterization** — `u_A = P·ε − x₀`. 학습 타깃의 두 항 (데이터 / 노이즈) 을 *비대칭으로* 다루는 첫 파라미터화.
2. **부분공간 / 직교 여집합 분해 + 분석적 복원 공식** — `P·u_A = P·u` (부분공간 안에선 진짜 속도), `(I−P)·u_A = −(I−P)·x₀` (바깥쪽에선 데이터 예측) 항등식으로 `û` 복원이 한 줄.
3. **랭크 r 의 family 해석** — `r=0` → x₀-prediction, `r=D` → u-prediction, 그 사이 sweet spot 존재. 기존 두 파라미터화를 잇는 통합 시야.
4. **PCA vs Procrustes 두 가지 A 구성법** — scratch (PCA) 와 미세조정 (latent-pixel Procrustes) 시나리오 모두 커버.
5. **사전학습 latent flow → pixel space 첫 미세조정 방법** — FLUX.2 klein 9B 본체 frozen + 입출력 projection + rank-256 LoRA + Oklab 정규화 조합.
6. **픽셀 공간 flow matching 의 SOTA 복귀** — ImageNet 256×256 FID 1.57 (JiT-H/16), T2I 에서 latent 9B baseline 상회.

---

## 3️⃣ 주요 알고리즘 설명

### 3.1 AsymFlow 파라미터화 (Fig.2 의 핵심)

**학습 타깃** (Eq.2 in paper):

```
u_A = P·ε − x₀
```

표준 속도 `u = ε − x₀` 와 비교하면 **노이즈 항 앞에 `P` 가 붙은 것 하나만 다름**. `P = A·Aᵀ` 는 rank-r 부분공간 사영기 (`AᵀA = I_r`).

(즉, 정답에서 노이즈만 "주요 변동축 r 개로 그리기" 라는 제약을 건 것. 데이터 항은 풀랭크 그대로.)

**왜 이게 통하나** — 풀랭크 노이즈 `ε` 는 D차원 전체에 흩뿌려진 잡음이지만, *데이터 매니폴드 의미 있는 변화 방향* 은 사실 저랭크. 네트워크가 부분공간 바깥쪽 무작위 노이즈까지 맞출 필요 없이 핵심 r 방향만 잘 잡으면 됨.

### 3.2 부분공간 / 직교 여집합 분해 (Eq.8)

`u_A` 를 두 직교 성분으로 쪼개면:

```
P·u_A   = P·ε − P·x₀ = P·u            (부분공간 안 — 진짜 속도와 같음)
(I−P)·u_A = −(I−P)·x₀                  (직교 여집합 — x₀ 예측과 같음)
```

→ 비대칭 속도는 **부분공간 안에선 u-prediction, 바깥쪽에선 x₀-prediction** 처럼 동작하는 *혼합형* 파라미터화.

**랭크 family 해석**:
- `r = 0` → `P = 0` → `u_A = −x₀` (전부 x₀-prediction, 부호 차이)
- `r = D` → `P = I` → `u_A = u` (전부 표준 u-prediction)
- `0 < r ≪ D` → 두 장점의 절충, 본 논문 핵심 영역

### 3.3 풀랭크 속도 복원 (Eq.10, Fig.2 (b))

네트워크 출력 `û_A` 로부터 진짜 속도 `û` 를 만드는 항등식:

```
û = P·û_A + (I−P)·(x_t + û_A)/σ_t
```

**유도** — 부분공간 안쪽은 `P·û_A` 가 그대로 진짜 속도. 바깥쪽은 `(I−P)·û_A = −(I−P)·x̂₀` 라서 `x₀` 스타일. 보간식 `x_t = (1−t)·x₀ + t·ε` 와 속도식 `u = ε − x₀` 를 결합하면 `(I−P)·u = (I−P)·(x_t − x₀)/σ_t` 가 나오고, `x̂₀ = −(I−P)·û_A` 대입 + 정리하면 위 공식.

이 복원은 학습 시 손실 계산과 샘플링 시 둘 다 동일하게 적용. **네트워크 출력 형태만 바뀔 뿐, 손실 함수와 샘플러 코드는 표준 flow matching 그대로**.

### 3.4 A 행렬 두 가지 구성법

| 시나리오 | A 만드는 법 | 어디 저장? |
|---|---|---|
| **Scratch (ImageNet)** | 학습 데이터 patch 들에 PCA, 상위 r 방향 | `checkpoints/asymflow_subspace_pca_dit.pth` |
| **사전학습 latent → pixel 미세조정 (AsymFLUX.2)** | latent 변수와 대응 pixel patch 사이 직교 Procrustes alignment | `checkpoints/asymflow_subspace_procrustes.pth` |

학습 시작 전 한 번 계산해서 고정. 학습 동안 `A` 는 업데이트되지 않음.

### 3.5 ImageNet scratch 학습 (JiT-H/16)

`configs/asymflow/asymflow_h_16_r8_imagenet_8gpus.py` 기준:

- 백본: AsymJiT (hidden_size=1280, depth=32, num_heads=16, bottleneck_dim=256, in_channels=3, patch_size=16, input_size=256, num_classes=1000)
- 랭크: `base_rank=8`
- 옵티마이저: AdamW (lr=2e-4, betas=(0.9, 0.95), weight_decay=0)
- 학습: 600 epoch × 1251 step = 750,600 iter, 5 epoch linear warmup
- 배치: GPU당 128, 총 8 GPU → batch 1024
- EMA momentum 0.9999, bfloat16
- **LoRA 사용 안 함** — 모든 가중치 풀학습

### 3.6 AsymFLUX.2 klein 미세조정 (T2I)

`configs/asymflow/asymflux2_klein_32gpus.py` 기준:

**유지** (FLUX.2 klein 9B 그대로):
- Transformer 본체: `num_layers=8` (joint MMDiT), `num_single_layers=24` (single), `attention_head_dim=128`, `num_attention_heads=32`, `joint_attention_dim=12288`
- 텍스트 인코더 (joint_attention_dim 매칭으로 그대로 사용)
- 사전학습 가중치: `black-forest-labs/FLUX.2-klein-base-9B`

**바꿈**:
- `in_channels=3` (latent 16ch → pixel RGB), `patch_size=16`
- VAE 제거 → Oklab 색공간 변환 + 학습된 선형 projection
- `base_rank=128` (저랭크 노이즈 사영)
- `pretrained_linear_proj='checkpoints/asymflow_subspace_procrustes.pth'`
- 본체 frozen, **rank-256 LoRA** (dropout 0.05) 부착:
  - `*.ff.linear_in/out`, `*.ff_context.linear_in/out` (FFN)
  - `timestep_embedder.linear_1/2`
  - `single_transformer_blocks.*.attn.to_out`

**학습**:
- AdamW8bit, lr=1e-4
- 32 GPU, batch 8/GPU = 256, 20K iter
- 데이터: LAION-Aesthetics 3M, 1MP 해상도, Qwen2.5-VL 캡션
- 샘플링: UniPC + APG, 38 step

---

## 4️⃣ 실험 요약

### ImageNet 256×256 (Class-conditional)

| 모델 | 공간 | FID |
|---|---|---|
| DiT-XL/2 (latent) | latent | ~2.27 |
| DiT-XL + RAE | latent | 1.50 |
| REPA-XL/2 | latent | 1.38 |
| REPA-E-XL VAVAE | latent | 1.12 |
| **AsymFlow JiT-H/16 (r=8)** | **pixel** | **1.76** |
| **AsymFlow JiT-H/16 (r=8) + REPA** | **pixel** | **1.57** ✨ |

→ 픽셀 모델이 latent 모델 (DiT-XL/RAE, 1.50) 과 동급. AsymFlow 자체의 효과는 약 1.76, REPA 결합으로 1.57.

### T2I (AsymFLUX.2 klein vs latent baselines)

논문에 따르면 HPSv3 (인간 선호), DPG-Bench, GenEval 모두에서 동급 latent 9B 모델 상회. 정확한 숫자는 본문 Table 4 참조.

### 랭크 r 와 FID 의 관계 (Fig.6 — `figures/rank_fid.pdf`)

- `r = 0` (x₀-pred 동치) → FID 높음
- `r ↑` → FID 빠르게 개선
- `r = 8` 근처에서 **sweet spot**, 그 이상 늘려도 큰 차이 없거나 약간 악화
- `r = D` (표준 u-pred 동치) → 처음보다 나은 FID 지만 r=8 보단 못함

→ 저자들이 본문에서 예측한 *"작지만 0이 아닌 r 이 최적"* 가설을 실험으로 확인.

---

## 5️⃣ 💬 Q&A 섹션

### Q1. AsymFLUX.2 klein 은 FLUX.2 klein 9B 와 구조가 같은가?

**Transformer 본체는 그대로, 입출력만 바뀌고 LoRA 로 살짝 적응한다.**

| 구성요소 | FLUX.2 klein 9B (원본) | AsymFLUX.2 klein | 동일? |
|---|---|---|---|
| Transformer 골격 | MMDiT (joint 8 + single 24 layers) | 동일 | ✅ |
| `attention_head_dim`, `num_attention_heads`, `joint_attention_dim` | 128, 32, 12288 | 동일 | ✅ |
| 사전학습 가중치 | — | `FLUX.2-klein-base-9B` 그대로 로드 | ✅ |
| 입력 채널 | latent (16ch) | **pixel RGB (3ch)** | ❌ |
| Patch size | latent 기준 | **픽셀 기준 16** | ❌ |
| VAE encoder/decoder | 있음 | **제거**, 대신 Oklab + 학습된 linear projection | ❌ |
| 노이즈 사영 | — | **base_rank=128** (P=A·Aᵀ) | ❌ (신규) |
| 학습 방식 | 풀 사전학습 | **base frozen + rank-256 LoRA + 입출력 projection** | ❌ |

요약: **9B 본체 가중치는 거의 그대로 재사용**, VAE 자리를 `Oklab + Procrustes 정렬된 선형 사영` 으로 대체, 본체는 LoRA 로 미세 적응. 32 GPU × 20K iter 라는 적은 비용으로 가능했던 이유.

### Q2. Fig.2 에서 LoRA 를 사용하는가?

**아니, Fig.2 자체는 LoRA 와 무관하다.** 둘은 *완전히 다른 자리에 있는 다른 종류의 저랭크* 다.

| 구분 | 어디에 적용? | 랭크 값 | 역할 |
|---|---|---|---|
| **Fig.2 의 저랭크 (P = A·Aᵀ)** | **데이터/노이즈 측** — 학습 타깃 | 8 (ImageNet) / 128 (FLUX) | 노이즈 ε 를 r 차원 부분공간으로 사영 |
| **LoRA (ΔW = B·A)** | **네트워크 가중치 측** | 256 | base frozen + 어댑터만 학습 |

논문 구조 상:
- **Sec.4 (Fig.2)** — 파라미터화 트릭, *모든 AsymFlow 실험에 공통* (scratch / 미세조정 모두).
- **Sec.5/6 (LoRA)** — FLUX.2 klein 미세조정 *한정* 의 학습 전략.

ImageNet 실험은 LoRA 없이 풀 scratch 학습이며 1.57 FID 도 거기서 나온 결과.

### Q3. AsymFlow 가 klein 이어서 학습이 아니라 scratch 학습에서도 통용되나?

**그렇다. 사실 1.57 FID 결과 자체가 scratch 학습 사례다.**

Fig.2 의 두 단계 — (a) 타깃 단순화, (b) 풀랭크 복원 — 는 *파라미터화 정의에서 따라나오는 수학 항등식* 이라 네트워크 초기값 (random / pretrained) 과 무관.

| 항목 | Scratch (ImageNet, JiT-H/16) | 미세조정 (AsymFLUX.2) |
|---|---|---|
| Fig.2 (a) 타깃 `u_A = −x₀ + P·ε` | ✅ 그대로 | ✅ 그대로 |
| Fig.2 (b) 복원 공식 | ✅ 그대로 | ✅ 그대로 |
| A 행렬 만드는 법 | 데이터 patch **PCA** | latent-pixel **Procrustes** |
| 랭크 r | 8 | 128 |
| LoRA | 안 씀 | rank 256 부착 |
| 결과 | FID 1.76 (+REPA → 1.57) | T2I SOTA |

논문 본문 (Sec.4) 도 명시:

> "When training AsymFlow from scratch, A can be obtained from a data-dependent patch basis, e.g., by applying PCA to image patches. When adapting a pretrained latent model, A is instead chosen to align the latent space with the pixel patch space …" (sections/4_asymflow.tex:35-36)

오히려 scratch 가 Fig.2 효과를 *가장 깨끗하게 증명* 하는 환경 — 사전학습 가중치 boost 없이 순수 파라미터화 트릭의 효과만 측정 가능.

---

## 6️⃣ 한 줄 요약 (전체)

> **학습 정답에서 노이즈 항만 r 차원 부분공간으로 압축 (`u_A = P·ε − x₀`) → 진짜 속도는 `û = P·û_A + (I−P)·(x_t + û_A)/σ_t` 로 항등식 복원. 네트워크 한 줄 안 바꾸고 ImageNet 1.57 FID, FLUX 픽셀 미세조정 T2I SOTA.**

---

## 7️⃣ 관련 메모리 링크

- [[paper_min_snr]] — Min-SNR 도 *학습 손실의 한 군데만 손대는 minimal trick* 계열. AsymFlow 는 "타깃 자체를 손댐", Min-SNR 은 "타깃은 그대로, 가중치만 손댐" 으로 위치 분리.
- [[paper_lumina_next]] — Sandwich Norm·tanh-AdaLN 등 *한 줄짜리 처방* 으로 큰 효과를 본 또 다른 사례.
- [[reference_pretrained_backbone_reuse_landscape]] — AsymFLUX.2 klein 은 *분기 B (Diffusion backbone 재사용)* 의 새 사례 — VAE 를 떼고 픽셀로 옮기는 reuse 방식.
- [[paper_z_image]] — Z-Image 도 단일 스트림 DiT 로 픽셀 측 부담을 줄이려 한 시도. AsymFlow 는 "픽셀 직접 학습" 의 또 다른 길.
