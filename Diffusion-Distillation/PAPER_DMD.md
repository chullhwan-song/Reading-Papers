# PAPER: DMD (Distribution Matching Distillation) — 1-step Diffusion 의 분포 매칭 증류 + VTOFF 적용 기록

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목 (메인)** | One-step Diffusion with Distribution Matching Distillation |
| **저자** | Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, Taesung Park (MIT + Adobe) |
| **공개일** | 2023-11-30 (arXiv v1) / CVPR 2024 |
| **분야** | 이미지 생성 / Diffusion Distillation / cs.CV |
| **논문 링크** | https://arxiv.org/abs/2311.18828 |
| **공식 코드** | https://github.com/tianweiy/DMD |
| **본 문서 작성일** | 2026-04-16 |
| **본 문서 목적** | DMD 이론 + VTOFF 프로젝트 적용 기록(CFG distillation 단계)을 한 문서에 정리 |

### 관련 논문 메타데이터

| 역할 | 논문 | 저자 · 학회 | arXiv |
|---|---|---|---|
| **본 코드 직접 근거** | On Distillation of Guided Diffusion Models (CFG distill) | Meng et al., CVPR 2023 | 2210.03142 |
| **메인 (Stage 2 로드맵)** | DMD: One-step Diffusion with Distribution Matching Distillation | Yin et al., CVPR 2024 | 2311.18828 |
| **후속 (DMD2)** | Improved Distribution Matching Distillation for Fast Image Synthesis | Yin et al., NeurIPS 2024 | 2405.14867 |
| **이론적 상위 (VSD)** | ProlificDreamer — Variational Score Distillation | Wang et al., NeurIPS 2023 | 2305.16213 |
| **수학적 프레임 (Flux2 베이스)** | Flow Matching for Generative Modeling | Lipman et al., ICLR 2023 | 2210.02747 |

---

## 📖 주요 용어 사전 (Glossary)

### Diffusion / Flow Matching 기본

- **Diffusion 모델**: 깨끗한 이미지에 노이즈를 점점 섞었다가, 거꾸로 노이즈를 한 단계씩 걷어내며 이미지를 만드는 생성 모델.
- **Score (스코어)**: 어떤 분포의 log-density 를 입력으로 미분한 값. `s(x) = ∇_x log p(x)`. "지금 점에서 데이터 밀도가 더 높은 쪽이 어디인가" 를 가리키는 방향 벡터.
- **ε (epsilon, 엡실론)**: 표준 정규 노이즈 `N(0, I)`. 모델이 예측하는 대상이 되기도 함 (ε-prediction).
- **velocity (속도, v)**: Flow Matching 정의에서 `v = ε − x₀`. "현재 노이즈 섞인 점에서 깨끗한 이미지 쪽으로 가야 할 방향".
- **x_t**: timestep t 의 latent (중간 상태). 깨끗한 이미지 x₀ 와 노이즈 ε 를 σ 비율로 섞은 값.
- **σ (sigma, 시그마)**: 노이즈 강도. 0 = clean, 1 = 순수 노이즈.
- **Flow Matching**: 노이즈와 진짜 이미지를 t 비율로 섞은 점을 만들고, 그 점에서 진짜 방향으로 향하는 속도 벡터를 학습하는 방식. Rectified Flow (RF) 가 대표적 특수형이고 FLUX·SD3 가 사용.
- **Euler step**: `x_{σ'} = x_σ + (σ' − σ) · v(x_σ, σ)`. 미분방정식(ODE)을 가장 단순한 방법으로 한 칸 적분하는 식.

### Distillation (증류) 관련

- **Distillation (증류)**: 큰 모델(teacher) 의 동작을 작은/빠른 모델(student) 이 흉내내도록 학습 → 추론 step 수나 forward 횟수를 줄이는 가속 기법.
- **Teacher / Student**: 증류의 두 주체. Teacher 는 학습 안 함(frozen), Student 만 학습.
- **CFG (Classifier-Free Guidance)**: 텍스트 조건이 적용된 결과(`v_cond`)와 빈 텍스트 결과(`v_uncond`)를 섞어 prompt 강도를 키우는 표준 기법. 한 step 에 forward 2번 필요.
- **CFG distillation (Meng et al. 2023)**: teacher 의 CFG output 을 student 가 forward 1번으로 흉내내도록 학습. step 수는 그대로(예: 50), forward 만 1번이 됨 → 추론 2× 가속.
- **Score Distillation**: teacher 와 student 의 "score" 자체를 맞추는 증류. DMD 가 대표.
- **VSD (Variational Score Distillation, Wang 2023)**: ProlificDreamer 에서 제안. 두 score 의 차이를 generator 로 역전파하는 방식. DMD 의 이론적 모태.
- **KL divergence (`D_KL(p ‖ q)`)**: 두 분포의 차이 척도. 0 이면 완전 동일.

### DMD 핵심 구성요소

- **Generator G_θ**: student. 본 논문에서는 `z → x₀` 의 1-step 직사상 함수.
- **p_real**: teacher 가 (학습 데이터를 통해) 암묵적으로 표현하는 정답 분포.
- **p_fake_θ**: student 가 만들어내는 샘플의 분포.
- **s_real(x_t, t)**: p_real 의 score. teacher 의 ε/velocity 예측에서 유도.
- **s_fake(x_t, t)**: p_fake 의 score. **별도의 critic 네트워크 μ_fake** 가 student 출력을 forward-diffuse 시켜 학습.
- **μ_fake (fake-score critic)**: DMD 의 가장 큰 특징. 학습 중에 student 와 함께 따로 업데이트되는 두 번째 네트워크. (단순 MSE 증류엔 없음)
- **L_reg (regression loss)**: teacher 의 ODE 로 만든 paired `(z, x₀)` 데이터에 대한 student MSE. DMD 안정화의 결정적 요소 (ablation Table 2).
- **Distribution Matching Gradient**: DMD 핵심 식. `∇_θ KL(p_fake ‖ p_real)` 을 두 score 의 차 `(s_fake − s_real)` 로 근사.

### CFG distillation 고유 기호

- **v_cfg**: CFG 블렌딩된 teacher velocity. `v_cfg = v_uncond + α · (v_cond − v_uncond)`.
- **α (alpha, `dmd_cfg`)**: CFG guidance scale. 본 구현 기본값 8.5.
- **L_CFG_distill**: `E [w(t) · ‖G_θ(x_t, t, c) − v_cfg.detach()‖²]`. 본 코드의 실제 loss.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> Diffusion 모델은 50~1000 step 적분이 필요해 느린데, **teacher 의 분포 자체를 student 가 모방** 하도록 학습하면 단 1 step 으로도 비슷한 품질이 나온다. 핵심 아이디어는 *KL divergence 의 gradient = (s_fake − s_real) 의 차분* 이라는 점을 이용해 student 를 미는 것. 결과: ImageNet 64×64 FID **2.62** (teacher EDM 1.79), A100 0.09s/image.

**핵심 문제**: Diffusion 의 N-step 적분 비용을 어떻게 1-step 으로 줄이면서도 품질을 살릴 것인가?

**해결책 (3 요소)**:
1. **Distribution Matching gradient** — `∇_θ KL(p_fake ‖ p_real) ≈ E[(s_fake − s_real) · ∂x/∂θ]`. 두 score 의 차이로 generator 를 민다.
2. **Real / Fake 두 개의 score 네트워크** — `s_real` 은 frozen teacher, `s_fake` 는 student 출력으로 새로 학습되는 critic μ_fake.
3. **Regression loss L_reg** — teacher 의 ODE 로 paired data 만들어 학습 안정화. (없으면 FID 11.49 → 19+ 로 폭락)

**검증**:
- ImageNet 64×64 (1-step) : FID **2.62** vs teacher EDM 1.79
- LAION COCO-30k (1-step) : FID **11.49**, CLIP 0.320
- 속도 : A100 0.09s/image (1-step)

---

## 2️⃣ 핵심 기여 (Contributions)

1. **1-step diffusion sampling 의 실용적 입증** — teacher 와 큰 품질 차이 없이 noise → image 직사상이 가능함을 large-scale text-to-image 까지 확장 검증.
2. **분포 매칭 (Distribution Matching) gradient 의 명시화** — KL 의 직접 최적화가 어려운 문제를, score 차분 surrogate 로 해결. autograd 호환 형태로 구현 (`L = 0.5 · ‖x̂₀ − (x̂₀ − grad).detach()‖²`).
3. **Real score + Fake score 분리 구조 정착** — Variational Score Distillation 의 이미지 생성 본격 적용. 후속 DMD2, SDXS, Hyper-SD 등의 원형이 됨.
4. **Regression loss 의 ablation 입증** — pure score-matching distillation 의 불안정성을 paired regression 으로 해결. 후속 DMD2 가 이를 다시 제거해 품질 상한을 푸는 등 후속 연구의 출발선.

---

## 3️⃣ 주요 알고리즘 설명

### 3.1 학습 흐름 (DMD 원논문, Yin 2024 CVPR)

```
                  z (가우시안 노이즈)
                       │
                       ▼
              ┌─────────────────┐
              │   G_θ (student) │  ← 학습 대상
              └────────┬────────┘
                       │ x̂₀  (1-step 직사상)
                       ▼
              forward-diffuse  x̂₀ → x_t  (랜덤 t 로 노이즈 다시 입힘)
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
  s_real(x_t, t)              s_fake(x_t, t)
  = Teacher (frozen, CFG)     = μ_fake (별도 critic, 같이 학습)
        │                              │
        └────────────┬─────────────────┘
                     ▼
         grad = s_fake − s_real        ← DMD gradient
                     │
                     ▼
         L_DM = 0.5 · ‖x̂₀ − (x̂₀ − grad).detach()‖²
                     +
         L_reg = ‖G_θ(z) − x₀_paired‖²   (teacher ODE 로 만든 paired data)
                     │
                     ▼
                  ∇_θ L
                  업데이트
```

### 3.2 수식 (단계별)

**목표** — student 분포를 teacher 분포에 맞추기:
```
min_θ   D_KL( p_fake_θ  ‖  p_real )
```

**Gradient 근사** (KL 직접 미분 대신 score 차분으로):
```
∂/∂θ  D_KL( p_fake ‖ p_real )
   ≈  E_{ t, x_t }[  ( s_fake(x_t, t)  −  s_real(x_t, t) )  ·  ∂x_t/∂θ  ]
```
(즉, 두 분포의 score 차이가 generator 의 그래디언트가 됨)

**Autograd surrogate** (이중 미분 없이 안전하게 흘리기):
```
grad   =  s_fake − s_real
L_DMD  =  0.5 · ‖  x̂₀  −  ( x̂₀  −  grad ).detach()  ‖²
```
→ `∂L_DMD / ∂x̂₀ = grad` 로 자연스럽게 떨어짐.

### 3.3 DMD vs 단순 MSE distillation (= CFG distillation)

| 항목 | DMD (Yin 2024) | CFG distillation (Meng 2023) |
|---|---|---|
| 최소화 대상 | `D_KL(p_fake ‖ p_real)` (분포) | `‖v_student − v_teacher_cfg‖²` (per-sample) |
| Loss 형태 | score 차분 surrogate | 단순 MSE |
| **μ_fake critic** | **있음** (별도 학습) | **없음** |
| Student 구조 | `z → x₀` 1-step | teacher 와 동일 (multi-step 유지) |
| 줄이는 축 | Step 수 (50 → 1) | CFG forward (2 → 1), step 수는 그대로 |
| 본 VTOFF 코드 사용 | ❌ (Stage 2 예정) | ✅ (현재 Stage 1) |

→ "DMD" 라는 단어를 정확히 쓸 때는 **반드시 μ_fake 와 score-difference gradient** 가 있어야 함. 본 VTOFF 코드는 파일명이 `train_distill_dmd.py` 이지만 실제로는 CFG distillation 임 (§ 6 부록 참조).

---

## 4️⃣ 관련 연구 (Related Work)

### 4.1 CFG distillation (Meng et al. CVPR 2023, arXiv:2210.03142)

- **목표**: CFG 의 2-forward 비용 제거. Step 수는 그대로.
- **Loss**: `L = E [‖ v_θ(x_t, t, c, w) − (v_uncond + w·(v_cond − v_uncond)).detach() ‖²]`
- **Student**: teacher 와 동일 구조, 보통 LoRA 로 가벼움.
- **결과**: cond 1 forward / step 으로 CFG 품질 유지 → 추론 2× 가속.

### 4.2 DMD2 (Yin et al. NeurIPS 2024, arXiv:2405.14867)

DMD1 의 4가지 한계를 개선:

1. **Regression loss 제거** — `L_reg` 를 빼서 품질 상한을 풀음. 대신 다른 안정화로 대체.
2. **Multi-step unroll (2 / 4 / 8 step)** + 선택적 **GAN loss** — 1-step 외에 few-step 옵션 추가.
3. **Normalized DMD gradient** — per-sample `|x̂₀ − s_real|` 로 정규화해 batch 내 scale 편차 흡수.
   ```
   normalizer = mean_{seq, ch}( | x̂₀ − s_real | )
   grad       = ( s_fake − s_real ) / max(normalizer, 1e-5)
   ```
4. **TTUR-like 스케줄** — critic 을 generator 보다 자주 학습.

### 4.3 VSD — Variational Score Distillation (Wang et al. NeurIPS 2023)

ProlificDreamer 에서 제안된 3D 생성용 score distillation. "두 분포의 score 차이" 라는 핵심 아이디어가 DMD 의 모태. DMD 는 이를 2D image generation 으로 가져오고 paired regression 으로 안정화한 형태.

### 4.4 Flow Matching (Lipman et al. ICLR 2023)

FLUX·SD3 가 사용하는 학습 프레임. RF (Rectified Flow) 가 특수 경우.

```
forward noising :  x_t  =  (1 − σ) · x₀  +  σ · ε
velocity target :  v    =  ε  −  x₀
```

DMD 의 `s_real` 은 ε-prediction 또는 v-prediction 으로부터 자동 유도 가능.

---

## 5️⃣ 실험 요약 (DMD 원논문)

| 데이터셋 | NFE | DMD FID | Teacher FID | CLIP (DMD) | 비고 |
|---|---|---|---|---|---|
| CIFAR-10 | 1 | 3.77 | 1.97 (EDM) | — | unconditional |
| ImageNet 64×64 | 1 | **2.62** | 1.79 (EDM) | — | class-cond |
| LAION COCO-30k | 1 | **11.49** | 8.59 (SDv1.5) | 0.320 | text-to-image |
| Speed (A100) | 1 | **0.09 s/img** | 2.59 s/img (SDv1.5, 50 step) | — | ~29× 가속 |

### Ablation 핵심 (Table 2, LAION)

| 설정 | FID |
|---|---|
| Full DMD | 11.49 |
| − L_reg (regression loss 제거) | **>19** (폭락) |
| − μ_fake 학습 (fake score 고정) | 발산 |
| − DM gradient (L_reg 만) | 14.93 |

→ L_reg 와 μ_fake **둘 다** critical. 후속 DMD2 가 L_reg 를 빼면서도 안정화한 게 큰 진전.

---

## 6️⃣ 부록 A: VTOFF 프로젝트 적용 기록 (CFG distillation 단계)

> 본 문서가 처음 작성된 배경. `train_distill_dmd.py` 라는 파일명에 "DMD" 가 붙어있지만, 실제 Stage 1 구현은 **CFG distillation (Meng 2023)** 임을 명확히 하기 위한 기록.

### A.1 왜 2-stage 인가 — FLUX.2-klein-base 의 추론 비용

```
한 이미지 생성 비용
= 50 step × 2 forward (CFG cond/uncond)
= 100 forward
```

비용 절감 축은 **독립적인 두 개**:

| 축 | 내용 | 대표 기법 |
|---|---|---|
| 축 A — CFG 제거 | cond 1 forward / step | CFG distillation (Meng 2023) |
| 축 B — Step 감축 | 50 → 4 step | **DMD/DMD2**, LCM, Hyper-SD |

→ Stage 1 (현재): 축 A. Stage 2 (이후): 축 B.

### A.2 Stage 1 의 실제 loss

```python
teacher_cond_pred   = teacher(x_t, t, c)           # no_grad
teacher_uncond_pred = teacher(x_t, t, empty_c)     # no_grad
teacher_cfg = teacher_uncond_pred + α · (teacher_cond_pred − teacher_uncond_pred)
student_pred = G_θ(x_t, t, c)
L = MSE(student_pred, teacher_cfg.detach())
```
- α = `dmd_cfg` = 8.5
- λ_sft 는 **주석처리** (GT anchor 미사용)
- 본 코드 위치: [train_distill_dmd.py:826](train_distill_dmd.py#L826), [train_distill_dmd.py:845](train_distill_dmd.py#L845)

### A.3 2-stage 로드맵

| Stage | 목적 | 방법 | μ_fake | 본 코드 상태 |
|---|---|---|---|---|
| **1 (현재)** | CFG forward 50% 절감 | CFG distillation (Meng 2023) | ❌ | 학습 중 |
| **2 (이후)** | Step 수 92.5% 절감 | DMD or DMD2 | ✅ 도입 예정 (VSD-style LoRA) | 미시작 |

### A.4 주요 의식적 설계 결정

| 항목 | 본 코드 값 | 표준 | 이유 |
|---|---|---|---|
| Fixed w-conditioning | w=8.5 고정 | w 가변 입력 | VTOFF 는 항상 동일 강도, 단순화 |
| L_sft (GT anchor) | **주석처리** | 약한 anchor (λ≈0.1) | pure distillation 택 |
| LoRA alpha / rank | 1 / 128 (= 0.008) | 보통 1.0 | base 변경 폭 억제 |
| `weighting_scheme` | `'none'` | `'logit_normal'` | 단순화. high-sigma 편향 가능성 있음 |

---

## 7️⃣ 부록 B: 디버깅 히스토리 — bs=24 blob 미해결 사례

### B.1 증상

```
bs=12 + CFG w=8.5    →  ✅ 100 step 에서 production-quality
bs=24 + CFG w=8.5    →  ❌ 100 step 에서 uniform blob (의류 실루엣 없음)
```

동일 코드/데이터/seed, **batch size 만** 12 → 24 변경 시 출력 완전 붕괴.

### B.2 기각된 가설

| 가설 | 결과 | 기각 근거 |
|---|---|---|
| A. `vae.eval()` 누락 | ❌ | FLUX2 의 VAE 는 `self.bn` 이 정의만 됐고 forward path 외부 (grep 0회) |
| B. CFG 공식 버그 (실효 w=9.5) | ❌ (bs 민감도와 무관) | 공식 수정 후에도 bs=24 여전히 blob |
| C. `_prepare_latent_ids` batch 버그 | ❌ | 코드 확인 결과 올바른 batch-aware 동작 |
| D. Keskar-type generalization gap | ❌ | `per_device_batch=12, grad_accum=2` → effective bs=24 인데 정상 → forward pass 의 batch-sensitive 요소 |

### B.3 현재 가장 유력한 가설 — PyTorch SDPA backend dispatch

- attention 이 `torch.nn.functional.scaled_dot_product_attention` 사용
- PyTorch SDPA 는 **입력 shape/dtype 에 따라 자동으로 다른 backend (FLASH / EFFICIENT / MATH) 선택**
- bs=12 와 bs=24 가 다른 backend 를 고를 가능성 → bf16 수치 차이 누적 → 학습 drift

### B.4 검증 방법 (후속 과제)

```python
# 방법 1: Flash kernel 비활성화
torch.backends.cuda.enable_flash_sdp(False)

# 방법 2: backend 강제
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.MATH):
    ...
```

### B.5 CFG 공식 수정 (이번 조사 산물)

| 이전 (버그) | 현재 (정상) |
|---|---|
| `v_cond + α·(v_cond − v_uncond)` | `v_uncond + α·(v_cond − v_uncond)` |
| 실효 w = 9.5 (파라미터 8.5 대비 +1) | 실효 w = 8.5 (파라미터와 일치) |
| 12% 과잉 guidance | 표준 CFG |

상대 편차 12%. 학습 방향 자체는 올바랐고 bs=12 에서는 gradient noise 가 상쇄.

### B.6 디버깅 교훈 (Decision Log)

| 오진 | 어떻게 틀렸는지 | 교훈 |
|---|---|---|
| "vae.eval() 누락" | diffusers 표준 VAE 의 BN 일반론을 FLUX2 커스텀 fork 에 적용 | 커스텀 fork 는 소스 호출 경로 직접 확인 |
| "CFG 공식이 bs 민감도 원인" | 이전 "잘 되던 back/#5" 가 버그 CFG 였다는 사실을 늦게 확인 | "이전엔 잘 됐다" 발언 시 git/back diff 부터 |
| "bs 만 바꿨다" 수용 | 실제 diff 에 `shuffle=False→True` 도 같이 있었음 | 사용자 보고를 diff 로 검증 |

---

## 8️⃣ Q&A

### Q1. 왜 파일명이 `train_distill_dmd.py` 인데 실제론 CFG distillation 인가?

후속 확장(축 B, Stage 2)에서 DMD 를 도입할 계획을 염두에 둔 명명. 현재 코드에는 μ_fake critic, score-difference gradient, KL 최소화 surrogate 가 **모두 없음** → 엄밀히는 **score-free velocity-space CFG distillation**.

### Q2. CFG distillation 과 DMD 는 단순화/일반화 관계인가?

아니오. **다른 태스크**. CFG distillation 은 "CFG 효과를 1-pass 에 흡수" (step 수 유지), DMD 는 "step 수 자체를 1 로 축약" (CFG 별개). 두 축이 독립적이라 조합 가능.

### Q3. Stage 2 에서 μ_fake 는 어떻게 구현?

VSD 스타일 LoRA 권장. teacher 본체에 별도 LoRA adapter 하나 더 붙여서 μ_fake 역할. 이미 본 코드는 teacher LoRA slot 이 있어서 adapter 추가 부담 최소.

### Q4. DMD2 가 L_reg 를 뺀 이유?

품질 상한이 paired regression 에 묶이는 문제. Multi-step unroll + 선택적 GAN loss + normalized gradient + TTUR 로 다시 안정화.

### Q5. bs=24 blob 의 근본 원인을 아직도 모르나?

미해결. SDPA dispatch 가 유력 가설이지만 검증 안 함. 현재 workaround: `train_batch_size=12, gradient_accumulation_steps=2` (effective batch=24 동등, forward 는 bs=12).

### Q6. "Flux 는 원래 batch 민감" 이라는 folk wisdom 은?

부분적 사실 + 일반론 혼합. SD3·FLUX 원 논문은 effective batch 수천으로 잘 학습됨. bs=12→24 에서 망가지는 건 **specific bug** 신호이지 "원래 그런 것" 으로 묻을 일 아님.

---

## 🔚 한 줄 요약 (전체)

**DMD (Yin 2024) 는 "student 의 가짜 분포와 teacher 의 진짜 분포의 score 차이를 generator 로 역전파" 하는 분포 매칭 증류로 1-step 이미지 생성을 실용화했고, 본 VTOFF 프로젝트의 `train_distill_dmd.py` 는 파일명과 달리 그 전 단계인 CFG distillation (Meng 2023) 을 Stage 1 로 구현 중이며, Stage 2 에서 진짜 DMD 로 확장 예정.**

---

## 🔗 관련 메모리 링크

- [[paper-z-image]] — Z-Image 의 Decoupled DMD + DMDR (DMD 의 응용 사례)
- [[paper-mv-split-dit]] — 1000-layer DiT 학습 안정화 (distillation 과 별개 축)
