# PAPER: DMD (Distribution Matching Distillation) — 1-step Diffusion 의 분포 매칭 증류

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목 (메인)** | One-step Diffusion with Distribution Matching Distillation |
| **저자** | Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, Taesung Park (MIT + Adobe) |
| **공개일** | 2023-11-30 (arXiv v1) / CVPR 2024 |
| **분야** | 이미지 생성 / Diffusion Distillation / cs.CV |
| **논문 링크** | https://arxiv.org/abs/2311.18828 |
| **본 문서 목적** | DMD 이론 + 관련 distillation 계열 (CFG distill, DMD2, VSD) 정리 |

### 관련 논문 메타데이터

| 역할 | 논문 | 저자 · 학회 | arXiv |
|---|---|---|---|
| **메인** | DMD: One-step Diffusion with Distribution Matching Distillation | Yin et al., CVPR 2024 | 2311.18828 |
| **후속 (DMD2)** | Improved Distribution Matching Distillation for Fast Image Synthesis | Yin et al., NeurIPS 2024 | 2405.14867 |
| **이론적 상위 (VSD)** | ProlificDreamer — Variational Score Distillation | Wang et al., NeurIPS 2023 | 2305.16213 |
| **비교 (CFG distill)** | On Distillation of Guided Diffusion Models | Meng et al., CVPR 2023 | 2210.03142 |
| **수학적 프레임** | Flow Matching for Generative Modeling | Lipman et al., ICLR 2023 | 2210.02747 |

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

→ "DMD" 라는 단어를 정확히 쓸 때는 **반드시 μ_fake 와 score-difference gradient** 가 있어야 함. 둘이 빠지면 그냥 일반 MSE distillation.

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

## 6️⃣ Q&A

### Q1. CFG distillation 과 DMD 는 단순화/일반화 관계인가?

아니오. **다른 태스크**. CFG distillation 은 "CFG 효과를 1-pass 에 흡수" (step 수 유지), DMD 는 "step 수 자체를 1 로 축약" (CFG 별개). 두 축이 독립적이라 조합 가능.

### Q2. DMD2 가 L_reg 를 뺀 이유?

품질 상한이 paired regression 에 묶이는 문제. Multi-step unroll + 선택적 GAN loss + normalized gradient + TTUR 로 다시 안정화.

### Q3. μ_fake 는 어떻게 구현하나?

원논문은 teacher 와 같은 구조의 별도 네트워크. 실용적으로는 **VSD 스타일 LoRA** — teacher 본체에 별도 LoRA adapter 하나 더 붙여서 μ_fake 역할로 사용하면 메모리 부담 최소.

### Q4. "DMD" 라고 부르려면 반드시 있어야 하는 것?

(1) μ_fake critic (학습 중 같이 업데이트), (2) score-difference gradient `(s_fake − s_real)`, (3) KL surrogate loss. 셋 중 하나라도 빠지면 그냥 일반 MSE distillation 이지 DMD 아님.

---

## 🔚 한 줄 요약 (전체)

**DMD (Yin 2024) 는 "student 의 가짜 분포와 teacher 의 진짜 분포의 score 차이를 generator 로 역전파" 하는 분포 매칭 증류로 1-step 이미지 생성을 실용화했고, 후속 DMD2 (NeurIPS 2024) 가 regression loss 제거 + multi-step unroll + normalized gradient + TTUR 로 품질 상한을 풀어 few-step 영역까지 확장한 계열.**

---

## 🔗 관련 메모리 링크

- [[paper-z-image]] — Z-Image 의 Decoupled DMD + DMDR (DMD 의 응용 사례)
- [[paper-mv-split-dit]] — 1000-layer DiT 학습 안정화 (distillation 과 별개 축)
