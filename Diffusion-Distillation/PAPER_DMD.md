# PAPER: DMD (Distribution Matching Distillation) — 1-step Diffusion 의 분포 매칭 증류

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목 (메인)** | One-step Diffusion with Distribution Matching Distillation |
| **저자** | Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, Taesung Park (MIT + Adobe) |
| **공개일** | 2023-11-30 (arXiv v1) / CVPR 2024 |
| **분야** | 이미지 생성 / Diffusion Distillation / cs.CV |
| **논문 링크** | https://arxiv.org/abs/2311.18828 |
| **본 문서 목적** | DMD 의 이론·메커니즘 정리 (관련 계열은 비교 맥락에서만 언급) |

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

- **Distillation (증류)**: 큰 모델(teacher) 의 동작을 작은/빠른 모델(student) 이 흉내내도록 학습 → 추론 step 수나 forward (모델 한 번 통과) 횟수를 줄이는 가속 기법.
- **Teacher / Student**: 증류의 두 주체. Teacher 는 **학습 안 함 (frozen, 가중치 동결)**, Student 만 학습.
- **CFG (Classifier-Free Guidance, 분류기 없는 안내)**: 텍스트 조건이 적용된 결과(`v_cond`)와 빈 텍스트 결과(`v_uncond`)를 섞어 prompt (지시문) 강도를 키우는 표준 기법. 한 step 에 forward 2번 필요.
- **CFG distillation (Meng et al. 2023)**: teacher 의 CFG output (안내된 출력) 을 student 가 forward 1번으로 흉내내도록 학습. step 수는 그대로(예: 50), forward 만 1번이 됨 → 추론 2× 가속.
- **Score Distillation (스코어 증류)**: teacher 와 student 의 "score" 자체를 맞추는 증류. DMD 가 대표.
- **VSD (Variational Score Distillation, 변분 스코어 증류, Wang 2023)**: ProlificDreamer 에서 제안. 두 score 의 차이를 generator (생성기) 로 역전파 (back-propagation = 출력에서 입력 방향으로 그래디언트 흘리기) 하는 방식. DMD 의 이론적 모태.
- **KL divergence (`D_KL(p ‖ q)`, KL 발산)**: 두 분포의 차이 척도. 0 이면 완전 동일.
- **MSE (Mean Squared Error, 평균 제곱 오차)**: `‖예측값 − 정답‖²` 의 평균. 가장 단순한 회귀 손실 (regression loss).
- **regression loss (회귀 손실)**: 모델 출력과 미리 준비된 정답 쌍 사이의 MSE 형태 손실.
- **autograd (자동 미분)**: PyTorch 등에서 손실 함수를 정의하면 자동으로 그래디언트를 흘려주는 기능.
- **surrogate (대리 함수)**: 원래 계산하고 싶은 식이 너무 어려울 때, **같은 그래디언트를 주는 더 쉬운 식** 으로 대체한 것.
- **detach (분리)**: PyTorch 의 `.detach()` — 그래디언트가 그 변수로 흐르지 못하게 막음 ("여기서 끊어라").
- **LoRA (Low-Rank Adaptation, 저차원 어댑터)**: 거대 모델의 일부 층에 작은 추가 모듈을 붙여 그것만 학습하는 경량 fine-tuning (미세조정) 기법.
- **ablation (구성요소 제거 실험)**: 각 모듈을 하나씩 빼 보고 성능이 얼마나 떨어지는지 보는 실험 — 어떤 모듈이 진짜 중요한지 확인하는 표준 방식.

### DMD 핵심 구성요소

- **Generator G_θ (생성기)**: student. 본 논문에서는 `z → x₀` 의 **1-step 직사상 (direct mapping = 한 번에 바로 매핑)** 함수.
- **p_real**: teacher 가 (학습 데이터를 통해) 암묵적으로 표현하는 정답 분포.
- **p_fake_θ**: student 가 만들어내는 샘플의 분포.
- **s_real(x_t, t)**: p_real 의 score. teacher 의 ε-prediction / v-prediction (노이즈/속도 예측) 에서 유도.
- **s_fake(x_t, t)**: p_fake 의 score. **별도의 critic 네트워크 μ_fake (판별기 역할이 아니라 "스코어 추정기" 역할)** 가 student 출력을 **forward-diffuse (다시 노이즈 씌우기)** 시켜 학습.
- **μ_fake (fake-score critic, 가짜 분포 스코어 추정기)**: DMD 의 가장 큰 특징. 학습 중에 student 와 함께 따로 업데이트되는 두 번째 네트워크. (단순 MSE 증류엔 없음)
- **L_reg (regression loss, 회귀 손실)**: teacher 의 ODE (미분방정식) 로 만든 **paired (입력-정답 쌍) `(z, x₀)`** 데이터에 대한 student MSE. DMD 안정화의 결정적 요소 (ablation Table 2 = 구성요소 제거 실험).
- **Distribution Matching Gradient (분포 매칭 그래디언트)**: DMD 핵심 식. `∇_θ KL(p_fake ‖ p_real)` 을 두 score 의 차 `(s_fake − s_real)` 로 근사.

### 평가 지표 (Metrics)

- **FID (Fréchet Inception Distance)**: 생성된 이미지들과 실제 이미지들의 통계적 분포 거리. **낮을수록 좋음** (= 더 진짜 같음). 표준 이미지 품질 지표.
- **NFE (Number of Function Evaluations, 모델 호출 횟수)**: 한 이미지 만들기 위해 모델을 몇 번 통과시켰는지. 50-step diffusion + CFG = NFE 100.
- **CLIP score (CLIP 점수)**: OpenAI 의 CLIP 모델로 잰 **이미지-텍스트 일치도** (높을수록 prompt 를 잘 따랐다는 뜻).
- **EDM**: Elucidating Diffusion Models (Karras et al. 2022) — DMD 의 teacher 로 자주 쓰인 강력한 diffusion 모델.
- **SDv1.5**: Stable Diffusion v1.5 — 대표 text-to-image (텍스트→이미지) 모델. DMD 의 LAION 실험에서 teacher 역할.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> Diffusion 모델은 노이즈에서 이미지를 만들기까지 **50~1000 번 모델을 통과시키는 적분 (= ODE step 으로 한 칸씩 나아가기)** 이 필요해 느린데, **teacher 의 분포 자체를 student 가 모방** 하도록 학습하면 단 한 번의 통과 (1 step) 만으로도 비슷한 품질이 나온다. 핵심 아이디어는 *KL divergence 의 gradient = (s_fake − s_real) 의 차분 (두 score 의 뺄셈)* 이라는 점을 이용해 student 를 미는 것. 결과: ImageNet 64×64 에서 **FID (Fréchet Inception Distance, 이미지 품질 거리 — 낮을수록 좋음) 2.62** (teacher EDM 의 1.79 와 작은 갭), A100 GPU 한 장에서 이미지 한 장당 0.09 초.

**핵심 문제**: Diffusion 의 N-step 적분 비용 (= 모델을 N번 통과시키며 노이즈를 걷어내는 비용) 을 어떻게 1-step (한 번에 바로) 으로 줄이면서도 품질을 살릴 것인가?

**해결책 (3 요소)**:
1. **Distribution Matching gradient (분포 매칭 그래디언트)** — *(즉: KL 의 미분값이 두 score 의 차이로 표현된다는 사실을 이용)* `∇_θ KL(p_fake ‖ p_real) ≈ E[(s_fake − s_real) · ∂x/∂θ]`. 두 score (분포의 경사 벡터) 의 차이로 generator (생성기) 를 민다.
2. **Real / Fake 두 개의 score 네트워크** — `s_real` 은 **frozen teacher (가중치 고정된 선생)**, `s_fake` 는 student 출력으로 새로 학습되는 **critic μ_fake (가짜 분포의 스코어 추정기)**.
3. **Regression loss L_reg (회귀 손실 = 정답 쌍 MSE)** — teacher 의 ODE 로 **paired data (입력-정답 쌍)** 만들어 학습 안정화. (없으면 FID 11.49 → 19+ 로 폭락)

**검증**:
- ImageNet 64×64 (1-step, 한 번 통과) : FID **2.62** vs teacher EDM (Karras et al. 2022) 의 1.79
- LAION COCO-30k (1-step, text-to-image) : FID **11.49**, CLIP score (이미지-텍스트 일치도) 0.320
- 속도 : A100 GPU 에서 이미지 한 장당 0.09 초 (1-step)

---

## 2️⃣ 핵심 기여 (Contributions)

1. **1-step diffusion sampling 의 실용적 입증** — teacher 와 큰 품질 차이 없이 **noise → image 직사상 (direct mapping = 노이즈에서 이미지로 한 번에 매핑)** 이 가능함을 **large-scale text-to-image (대규모 텍스트→이미지)** 까지 확장 검증.
2. **분포 매칭 (Distribution Matching) gradient 의 명시화** — KL 을 직접 줄이기 어려운 문제를, **score 차분 surrogate (대리 함수 — 같은 그래디언트를 주는 더 쉬운 식)** 로 해결. **autograd (자동 미분) 호환 형태** 로 구현. 핵심 트릭은 *"x̂₀ 에서 미리 계산된 grad 방향으로 한 발짝 간 점을 detach (그래디언트 차단) 한 뒤 MSE 를 잡으면, 그 MSE 의 미분이 자동으로 원하는 grad 가 되도록 만든 것"*:
   ```
   L = 0.5 · ‖x̂₀ − (x̂₀ − grad).detach()‖²
   ```
3. **Real score + Fake score 분리 구조 정착** — VSD (Variational Score Distillation, 변분 스코어 증류) 의 이미지 생성 본격 적용. 후속 **DMD2 / SDXS / Hyper-SD (모두 빠른 diffusion 증류 후속 연구)** 등의 원형이 됨.
4. **Regression loss 의 ablation (구성요소 제거 실험) 입증** — pure score-matching distillation (오로지 스코어만 맞추는 증류) 의 불안정성을 **paired regression (정답 쌍 회귀 손실)** 으로 해결. 후속 DMD2 가 이를 다시 제거해 품질 상한을 푸는 등 후속 연구의 출발선.

---

## 3️⃣ 주요 알고리즘 설명

### 3.1 학습 흐름 (DMD 원논문, Yin 2024 CVPR)

**원논문 Fig.2 — 전체 framework 한 장 요약**:

<p align="center">
  <img src="figures/dmd_fig2.png" alt="DMD Framework Overview (Fig. 2)" width="900"/>
</p>

*Source: Yin et al., "One-step Diffusion with Distribution Matching Distillation", CVPR 2024 (project page: tianweiy.github.io/dmd)*

**그림 읽는 법** (왼쪽 → 오른쪽):

| 영역 (그림 내 표기) | 역할 |
|---|---|
| **왼쪽 (생성 + regression loss = 회귀 손실 영역)** | `random latent z` (랜덤 입력 노이즈) → **one-step generator G_θ (한 번에 바로 매핑하는 생성기)** → `fake image` (가짜 이미지). 이 fake image 와 **paired dataset (입력-정답 쌍 데이터, teacher 의 ODE 로 미리 만든 (z, x₀) 쌍)** 의 정답 사이의 MSE (평균 제곱 오차) = **regression loss L_reg** (학습 안정화용 안전망) |
| **오른쪽 (Distribution Matching Gradient Computation = 분포 매칭 그래디언트 계산 영역)** | fake image 에 **diffusion (확산 = 노이즈 다시 씌우기)** 적용 → `noisy image` (노이즈 섞인 이미지) → **두 개의 score 네트워크 (분포의 방향을 알려주는 두 망)** 가 병렬로 호출됨 |
| **위쪽 path (real data score function, 자물쇠 = frozen 가중치 동결)** | `real score` 출력 — teacher 가 표현하는 진짜 분포의 방향 (= 진짜 산의 경사) |
| **아래쪽 path (fake data score function = 가짜 분포 스코어 망)** | `fake score` 출력 — μ_fake critic (스코어 추정기) 이 추정하는 student 분포의 방향. 같은 noisy image 로 **diffusion loss (denoising loss = 노이즈 제거 손실)** 도 같이 받아 같이 학습 |
| **두 score 의 차이 (붉은 ⊖)** | `computed gradient` = `real − fake`. 이게 곧 **KL gradient 의 surrogate (대리 식)**. **빨간 화살표 `∇_θ D_KL`** 로 G_θ 에 **역전파 (back-propagation = 출력에서 입력 쪽으로 그래디언트 흘리기)** |

**핵심 포인트** (그림에서 직접 읽힘):
1. **두 개의 score 네트워크 분리** — frozen teacher (학습 안 함, real score) + 학습되는 critic (μ_fake, fake score). 같은 noisy image 입력을 받음 (= "같은 위치에서 두 방향 비교").
2. **fake image 가 다시 노이즈에 섞임** — score 가 **노이즈 레벨별로만 정의 (특정 t 값에서만 의미 있음)** 되니까 clean (깨끗한) 이미지 그대로는 호출 불가. (자세히는 § 6 Q7)
3. **두 loss 가 따로 들어감** — 왼쪽 L_reg (paired regression = 정답 쌍 회귀) + 오른쪽 D_KL gradient (분포 매칭). 둘 다 G_θ 의 학습에 사용.
4. **fake score critic 도 diffusion loss 로 동시에 학습** — student 가 분포를 바꿀 때마다 μ_fake 도 따라가야 함 (안 따라가면 fake score 가 stale = 오래된 값이 됨).

**텍스트 흐름도 (등가)**:

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

**목표** — student 분포를 teacher 분포에 맞추기. *(즉: 학습 파라미터 θ 를 조정해서 student 가 만드는 분포 p_fake 가 teacher 의 진짜 분포 p_real 에 가장 가까워지도록 만든다. KL 발산이 두 분포 거리 측도.)*
```
min_θ   D_KL( p_fake_θ  ‖  p_real )
```

**Gradient 근사** (KL 직접 미분 대신 score 차분으로). *(즉: KL 자체는 못 미분해도, 그 미분값이 두 score 의 차이 형태로 표현 가능. 좌변은 못 구해도 우변은 score 네트워크 두 개로 계산 가능 → 학습 가능):*
```
∂/∂θ  D_KL( p_fake ‖ p_real )
   ≈  E_{ t, x_t }[  ( s_fake(x_t, t)  −  s_real(x_t, t) )  ·  ∂x_t/∂θ  ]
```
(즉, 두 분포의 score 차이가 generator 의 그래디언트가 됨)

**Autograd surrogate (자동 미분 호환 대리 함수)** — *(즉: 위 수식을 그대로 코딩하려면 `∂x_t/∂θ` 같은 부분에서 이중 미분 = 미분을 두 번 흘리는 비싼 연산 이 필요. 그래서 "MSE 모양인데 미분하면 위와 동일한 grad 가 나오는" 대리 손실을 만들어 이중 미분 없이 안전하게 흘림.)*
```
grad   =  s_fake − s_real
L_DMD  =  0.5 · ‖  x̂₀  −  ( x̂₀  −  grad ).detach()  ‖²
```
*(즉: 두 번째 항 `(x̂₀ − grad)` 를 `.detach()` 로 그래디언트 차단 → autograd 입장에선 그저 상수. 그러면 이 L 을 x̂₀ 로 미분하면 `2 · 0.5 · (x̂₀ − (x̂₀ − grad)) = grad` 가 자연스럽게 떨어짐.)*

→ `∂L_DMD / ∂x̂₀ = grad` 로 자연스럽게 떨어짐.

### 3.3 DMD vs 단순 MSE distillation (= CFG distillation)

| 항목 — 비교 관점 | DMD (Yin 2024) | CFG distillation (Meng 2023) |
|---|---|---|
| **최소화 대상 (학습이 줄이려는 양)** | `D_KL(p_fake ‖ p_real)` — **분포 사이 거리** | `‖v_student − v_teacher_cfg‖²` — **per-sample (샘플별) 출력 차이** |
| **Loss 형태 (손실 함수 모양)** | score 차분 surrogate (대리 함수) | 단순 MSE (평균 제곱 오차) |
| **μ_fake critic (가짜 분포 스코어 추정기)** | **있음** (별도 학습) | **없음** |
| **Student 구조 (학생 모델 형태)** | `z → x₀` 1-step (한 번에 직접 매핑) | teacher 와 동일 (multi-step 유지 = 단계는 그대로) |
| **줄이는 축 (가속의 방향)** | Step 수 (50 → 1) | CFG forward (한 step 당 2번 통과 → 1번), step 수는 그대로 |

→ "DMD" 라는 단어를 정확히 쓸 때는 **반드시 μ_fake (스코어 추정기) 와 score-difference gradient (두 스코어 뺄셈 기반 그래디언트)** 가 있어야 함. 둘이 빠지면 그냥 일반 MSE distillation (평균 제곱 오차 증류).

---

## 4️⃣ 관련 연구 (Related Work)

### 4.1 CFG distillation (Meng et al. CVPR 2023, arXiv:2210.03142)

- **목표**: CFG 의 2-forward (한 step 에 cond/uncond 두 번 통과) 비용 제거. Step 수는 그대로.
- **Loss (손실)**: *(즉: student 가 입력에 `w` 라는 guidance 강도 까지 받아서, teacher 가 cond 와 uncond 두 번 호출해서 만든 CFG-블렌딩된 결과를 한 번에 흉내내도록 MSE 로 학습)*
  ```
  L = E [‖ v_θ(x_t, t, c, w) − (v_uncond + w·(v_cond − v_uncond)).detach() ‖²]
  ```
- **Student (학생 모델)**: teacher 와 동일 구조, 보통 **LoRA (저차원 어댑터 = 추가 학습용 가벼운 모듈)** 로 가벼움.
- **결과**: cond (조건부) 한 번 통과만으로 CFG 품질 유지 → 추론 2× 가속.

### 4.2 VSD — Variational Score Distillation (Wang et al. NeurIPS 2023)

ProlificDreamer (3D 생성 모델 논문) 에서 제안된 **3D 생성용 score distillation (스코어 증류)**. "두 분포의 score 차이" 라는 핵심 아이디어가 DMD 의 모태. DMD 는 이를 **2D image generation (2D 이미지 생성)** 으로 가져오고 **paired regression (정답 쌍 회귀 손실)** 으로 안정화한 형태.

### 4.3 Flow Matching (Lipman et al. ICLR 2023)

FLUX · SD3 (대표 image generation 모델들) 가 사용하는 학습 프레임. **RF (Rectified Flow, 직선화된 flow)** 가 특수 경우.

*(즉, 아래 수식은 노이즈와 깨끗한 이미지를 σ 비율로 섞은 점이 x_t 가 되고, 그 점에서 학습 대상으로 삼는 "이 점에서 가야 할 방향" (velocity, 속도) 은 `ε − x₀` 라는 단순한 형태로 정의된다는 뜻):*
```
forward noising (노이즈 씌우는 식) :  x_t  =  (1 − σ) · x₀  +  σ · ε
velocity target (속도 목표값)        :  v    =  ε  −  x₀
```

DMD 의 `s_real` 은 **ε-prediction (노이즈 예측 형태)** 또는 **v-prediction (속도 예측 형태)** 으로부터 자동 유도 가능.

---

## 5️⃣ 실험 요약 (DMD 원논문)

| 데이터셋 (학습/평가 셋) | NFE — 모델 호출 횟수 | DMD FID — 낮을수록 좋음 | Teacher FID — 비교 기준 | CLIP — 텍스트 일치도 (높을수록 좋음) | 비고 (작업 종류) |
|---|---|---|---|---|---|
| CIFAR-10 (32×32 작은 이미지 셋) | 1 | 3.77 | 1.97 (EDM) | — | unconditional (조건 없음, 그냥 이미지 분포만 학습) |
| ImageNet 64×64 (저해상도 분류 셋) | 1 | **2.62** | 1.79 (EDM) | — | class-cond (클래스 레이블 조건부) |
| LAION COCO-30k (대규모 text-image 쌍 평가) | 1 | **11.49** | 8.59 (SDv1.5) | 0.320 | text-to-image (텍스트→이미지) |
| Speed (A100 GPU 한 장 기준) | 1 | **0.09 s/img** (한 장당 0.09 초) | 2.59 s/img (SDv1.5, 50 step) | — | ~29× 가속 |

### Ablation 핵심 (구성요소 제거 실험 — Table 2, LAION)

| 설정 — 어느 구성요소를 뺐는지 | FID — 낮을수록 좋음 |
|---|---|
| Full DMD (전체 그대로) | 11.49 |
| − L_reg (regression loss 제거 = 정답 쌍 회귀 손실 제거) | **>19** (폭락) |
| − μ_fake 학습 (fake score 고정 = critic 을 학습 안 시키고 고정) | 발산 (학습이 망함) |
| − DM gradient (L_reg 만, 분포 매칭 항 제거) | 14.93 |

→ L_reg (정답 쌍 회귀 손실) 와 μ_fake (가짜 분포 스코어 추정기) **둘 다 critical (필수 불가결)**. 후속 DMD2 가 L_reg 를 빼면서도 안정화한 게 큰 진전.

---

## 6️⃣ Q&A

### Q1. CFG distillation 과 DMD 는 단순화/일반화 관계인가?

아니오. **다른 태스크 (작업)**. CFG distillation 은 "CFG 효과를 **1-pass (한 번 통과)** 에 흡수" (step 수 유지), DMD 는 "step 수 자체를 1 로 축약" (CFG 는 별개 문제). 두 축 (축소 방향) 이 독립적이라 조합 가능.

### Q2. μ_fake 는 어떻게 구현하나?

원논문은 teacher 와 같은 구조의 별도 네트워크 (full clone). 실용적으로는 **VSD 스타일 LoRA (저차원 어댑터 형태)** — teacher 본체에 **별도 LoRA adapter (추가 학습용 작은 모듈) 하나 더 붙여서** μ_fake 역할로 사용하면 메모리 부담 최소.

### Q3. "DMD" 라고 부르려면 반드시 있어야 하는 것?

(1) **μ_fake critic** (가짜 분포 스코어 추정기, 학습 중 같이 업데이트), (2) **score-difference gradient `(s_fake − s_real)`** (두 스코어의 뺄셈 기반 그래디언트), (3) **KL surrogate loss** (KL 발산의 대리 손실 함수). 셋 중 하나라도 빠지면 그냥 일반 MSE distillation (평균 제곱 오차 증류) 이지 DMD 아님.

### Q4. 논문은 1-step 인데 왜 실제 시장 (FLUX-schnell, SDXL-Lightning, Z-Image-Turbo 같은 공개 가속 모델) 은 4-step / 8-step 인가?

1-step 도 존재하지만 (SDXL-Turbo 1-step, Hyper-SD 1-step) **사용자 채택의 sweet spot (가장 인기 있는 균형점) 은 4~8 step**. DMD 의 한계가 시장 채택을 막는 형태로 드러난 결과. 이유 6가지:

| 이유 — 항목 | 내용 |
|---|---|
| **CFG 호환성 (안내 기법과 같이 쓸 수 있나)** | 1-step 은 CFG 적용 불가 → **guidance 강도 (텍스트 따름 정도)** 조절 못함. 4-step + CFG = 8 forward (= teacher 의 1/12 수준). 사용자 체감 속도 차이 작은데 품질 차이 큼 |
| **ODE trajectory 곡률 (적분 경로가 얼마나 휘었나)** | noise → image 경로가 곡선. 1-step = 직선 한 방 (오차 큼), 4-step = **piecewise linear (조각별 직선 = 4 조각의 직선으로 곡선 근사)** |
| **Diversity (다양성) / mode collapse (모드 붕괴 = 비슷한 그림만 반복)** | 1-step **deterministic 사상 (입력 노이즈가 같으면 항상 같은 이미지가 나오는 결정론적 매핑)** → 다양성 손실. 다단계는 매 step 마다 **stochasticity 주입 (랜덤성 추가)** 여지 |
| **고해상도 gap (high-resolution 에서 갭이 더 벌어지는 현상)** | ImageNet 64² gap (FID 갭) +0.83 / LAION gap **+2.90**. 1024² (고해상도) 면 더 벌어짐 → multi-step 이 안전 |
| **Fine-tunability (추가 학습 가능성)** | 1-step 은 **schedule 가정 (학습 시 가정한 노이즈 일정표)** 이 너무 달라 LoRA 추가 학습 거의 불가 ([[paper-z-image]] Z-Image-Turbo 가 fine-tunability "N/A (불가)" 인 이유). 4-step 은 **customization (사용자별 맞춤 학습)** 여지 있음 |
| **Training stability (학습 안정성)** | 1-step distillation 본질적 불안정 (L_reg 없으면 FID 11.49 → 19+ 폭락). 후속 연구가 multi-step 으로 옮겨간 것 자체가 1-step 한계의 방증 |

→ **"1-step 은 가능하지만 잃는 게 많고, 4-step 도 충분히 빠르다"** 가 시장 합의.

### Q5. 50 step → 1 step 가속의 본질적 원동력은?

**Trajectory matching (경로 따라하기) → distribution matching (분포 맞추기) 으로 전환** 한 것. 한 줄로: **"Teacher 가 어떻게 가는지 (= 경로) 는 상관없고, 최종 도착지의 분포만 같으면 된다 — 그러니 student 는 한 번에 가도 OK"**.

| 기존 (Progressive Distillation = 점진적 증류, 단계를 절반씩 줄여나가는 방식 등) | DMD |
|---|---|
| Teacher 의 50-step **중간 경로** 를 student 가 흉내 | 중간 경로 무시. **최종 분포** 만 맞춤 |
| Student 도 N-step (단계 수를 절반씩 줄임) | Student 는 **임의 함수 OK** (1-step direct mapping = 한 번에 매핑 가능) |
| "고속도로의 모든 구간을 똑같이 운전" | "어디로 가든 목적지에만 잘 도착" (= 턴바이턴 길안내 vs GPS 좌표만 알려주기) |

→ Student G_θ(z) → x₀ 가 어떤 함수든 **분포 p_fake = p_real 이면 valid (학습 목표 달성)**. 중간 step 을 따라야 한다는 제약이 사라짐 = 1-step 가능의 수학적 근거.

이를 실제로 계산 가능하게 만든 두 트릭:
1. KL gradient (KL 발산의 미분값) 를 **score 차분** `(s_fake − s_real)` 로 근사
2. **μ_fake critic (가짜 분포 스코어 추정기)** 이 "student 분포의 score" 를 알려줌

### Q6. KL divergence `D_KL(p_fake ‖ p_real)` 은 왜 직접 미분 불가능한가?

*(즉, 아래 KL 식은 "임의의 점 x 에서 두 분포가 그 점에 부여한 log 확률값 (= log density, 밀도의 로그) 의 차이를 student 분포 위에서 평균낸 것" 인데, 이걸 그대로 계산하려면 두 log 확률값을 각 점에서 둘 다 알아야 함):*
```
KL(p_fake ‖ p_real)  =  E_{x ~ p_fake}[  log p_fake(x)  −  log p_real(x)  ]
```
이걸 계산하려면 임의의 그림 x 에 대해 **두 확률값 (= density, 밀도) 을 둘 다 숫자로 뽑아야** 함. 이게 두 장벽 때문에 막힘.

**비유 setup**: 화가 A (teacher) 의 스타일을 화가 B (student) 가 따라하는 상황.

**장벽 1 — B (student) 가 자기 확률을 모름 (implicit generator = 확률 함수가 명시적으로 없는 생성기)**

B 한테 "그림 그려봐" → 그림. ✅ 샘플 생성은 가능.
B 한테 "이 그림이 너한테서 나올 확률은?" → ❌ 모름.

이유: B 의 그림 그리는 과정은 `z (랜덤 노이즈) → 거대 NN (신경망) → 그림 x`. "이 x 가 나올 확률 `p_fake_θ(x)`" 을 구하려면 **change-of-variables 공식 (변수 변환 공식 = 입력 분포를 출력 분포로 변환할 때 쓰는 미적분 공식)** 의 **Jacobian determinant `|det(∂G_θ/∂z)|` (야코비 행렬식 = 출력이 입력에 얼마나 변하는지 측정하는 거대 행렬의 행렬식)** 이 필요한데, NN 의 Jacobian 은 거대 행렬 (예: 3M × 3M) → determinant 계산 비용 `O(n³)` (행렬 크기 세제곱에 비례) → **사실상 불가능**. (Normalizing Flow (정규화 흐름 = density 계산 가능하도록 special 구조를 강제한 생성 모델) 가 이걸 풀려고 special architecture (특수 구조) 를 강제하지만, diffusion student 는 자유로운 NN.)

→ **"B 는 그림은 그리는데 자기가 왜 그렇게 그렸는지 확률로 설명 못 하는 화가"** (= implicit generative model = 암묵적 생성 모델).

**장벽 2 — A (teacher, diffusion) 도 절대 확률값을 안 줌 (score-only = 스코어만 출력)**

비유: 확률값 = **산의 높이**, score = **산의 경사 방향**.

```
   ▲ 확률 높음
   │
   ╱   ← teacher 는 "여기서 어디로 가야 더 높아지나" (경사) 만 줌
  ╱       "지금 절대 높이가 얼마인지" 는 안 줌
─────────► 그림 공간
```

왜? Diffusion 의 training loss (학습 손실) 자체가 **ε-prediction (노이즈 ε 직접 예측)** 또는 **score-prediction (스코어 직접 예측)** 형태. **`∇ log p` (score = log 확률의 미분) 만 학습 — `log p` 자체는 학습 안 함**. log-density 를 score 로부터 복구하려면 **path integral (경로 적분 = 임의 경로를 따라 score 를 누적 적분)** 이 필요한데 한 점당 50-step forward (모델 50번 통과) → 학습 비용 폭발.

→ **"A 는 지도가 아니라 매 지점의 화살표만 줌"**.

**결과**: 두 분포의 확률값 (p_fake, p_real) 이 둘 다 모르는 값 → KL 식의 `log p_fake − log p_real` 을 빼고 평균낼 방법이 없음 → **직접 계산도 미분도 불가능**.

**DMD 의 우회**: "KL 자체는 못 구해도 KL 을 **줄이는 방향 (= ∇_θ KL)** 만 알면 학습 가능" 이라는 점을 이용. 그 방향이 두 score 의 차이로 깔끔하게 표현됨.

*(즉, KL 의 절대값은 못 구해도 KL 의 미분값은 두 score 의 차이로 표현되고, 이건 score 네트워크 두 개로 계산 가능 → 우리는 학습에 미분값만 있으면 됨):*
```
∇_θ KL( p_fake ‖ p_real )  ≈  E[ ( s_fake  −  s_real )  ·  ∂x/∂θ ]
                                  └─ B 의 화살표 ─┘  └─ A 의 화살표 ─┘
                                  (μ_fake 가 알려줌)  (teacher 가 알려줌)
```
→ **절대 확률값 (= 산의 높이) 을 한 번도 계산 안 하고**, 두 화살표 (= 경사) 의 **차이** 만으로 generator 를 학습. 이게 DMD 의 핵심 트릭.

### Q7. 학습 흐름도에서 왜 student 출력 x̂₀ 를 다시 score 네트워크 입력으로 넣고, 거기에 노이즈를 다시 씌우는가?

§ 3.1 의 알고리즘 흐름이 "z → student → x̂₀ → 다시 노이즈 → score 두 개" 로 **돌고 도는 모양** 이 된 이유. 두 질문이 합쳐져 있음:

**(a) 왜 student 출력을 다시 입력으로 — KL 의 기댓값 (expectation, 평균) 이 잡히는 위치 때문**

*(즉, 아래 KL 식의 평균 기호 `E_{x ~ p_fake}` 가 "x 를 p_fake 분포에서 뽑아 평균낸다" 는 뜻 → 평균을 잡는 좌표 자체가 student 분포 위. 그러니 student 의 출력 점에서 평가해야 함):*
```
KL( p_fake ‖ p_real )  =  E_{ x ~ p_fake }[  log p_fake(x)  −  log p_real(x)  ]
                            └─ 기댓값 잡는 분포 = p_fake ─┘
```

**기댓값 잡는 위치가 `x ~ p_fake`** — 즉 **student 가 실제로 만들어내는 점들** 에서 평가해야 함. z (노이즈 공간) 만 보고 평가하면 노이즈 공간 비교가 되고, 우리가 원하는 건 **이미지 공간 (생성된 그림이 사는 공간) 에서 분포가 같아지는 것** 이므로 student 출력 위치에서 평가 필수.

→ **Student 의 출력 x̂₀ = "평가가 일어나는 좌표"**. 그 좌표에서 teacher 의 화살표(s_real) 와 student 분포의 화살표(s_fake) 를 둘 다 뽑아 차이를 잰다.

**(b) 왜 노이즈를 다시 씌우는가 — Score 는 "노이즈 레벨별로만" 정의됨**

Score 네트워크 (teacher, μ_fake) 는 **깨끗한 이미지 x₀ 에서는 호출 불가**.

- Teacher diffusion 은 학습 시 **"노이즈 섞인 이미지 x_t (timestep t 의 중간 상태)"** 만 입력으로 봤음.
- `s_real(x_t, t)` 는 **노이즈 레벨 t 에 따라 정의된 함수 (각 노이즈 강도별로 다른 함수값을 줌)**. t=0 (clean = 노이즈 0, 깨끗한 이미지) 에서는 학습 안 됨 → 호출 시 **분포 밖 (out-of-distribution = 학습 때 본 적 없는 입력 영역)** → 의미 없는 값.

그래서 student 의 깨끗한 출력 x̂₀ 를 그대로 못 쓰고, **랜덤 t 골라 노이즈를 다시 씌워** x_t 로 만든 다음 호출.

*(즉, 아래는 student 가 만든 깨끗한 이미지 x̂₀ 와 새 가우시안 노이즈 ε 를 σ_t 비율로 섞어 다시 "노이즈 레벨 t 의 중간 상태" 로 만드는 식. 이렇게 해야 score 네트워크가 정의된 입력 영역으로 들어감):*
```
x̂₀  (student 출력, clean = 노이즈 0)
    │
    │  ε ~ N(0,I), t ~ U(0,1) 추가   ← 새 가우시안 노이즈와 랜덤 timestep
    ▼
x_t = (1 − σ_t) · x̂₀  +  σ_t · ε    ← 이 시점에서야 score 호출 가능
    │
    ├──► s_real(x_t, t)   ← teacher (frozen = 가중치 고정)
    └──► s_fake(x_t, t)   ← μ_fake (별도 critic = 스코어 추정기)
```

추가 이점: **모든 노이즈 레벨에서 분포 매칭 검사** — **coarse structure (high t = 큰 노이즈 = 이미지의 큰 구조)** 부터 **fine detail (low t = 작은 노이즈 = 세부 디테일)** 까지 전 구간 커버. 한 t 에만 매칭시키면 다른 영역에서 어긋남.

**비유로 정리**

학생이 100 장의 그림을 그림 (x̂₀). 선생 (teacher) 과 학생 자신 (μ_fake) 에게 묻고 싶음: **"이 100 장이 너희 각자 분포에서 어디쯤인가?"** 그런데 두 사람 모두 **"흐릿하게 보는 안경"** (= 노이즈 거쳐 학습됨 → 노이즈 섞인 입력에만 익숙함) 을 쓰고 있어서, **깨끗한 그림은 안 보임**. → 인위적으로 안개를 씌워 (= forward-diffuse, 다시 노이즈 입히기) 보여줘야 작동. 두 사람이 각각 그려준 화살표 (= score) 의 **차이** 가 곧 generator gradient (생성기 업데이트 방향).

**한 줄 요약**: (a) KL 의 기댓값이 `x ~ p_fake` 위에서 잡혀서 (= 평균을 잡는 좌표가 student 출력 위) / (b) score 네트워크가 노이즈 레벨별로만 정의돼서 (= clean 입력에선 작동 안 함). 둘이 합쳐져 "student 출력 → 다시 노이즈 → score 두 개" 의 순환 구조가 만들어짐.

---

## 🔚 한 줄 요약 (전체)

**DMD (Yin 2024) 는 "student (학생) 의 가짜 분포 (p_fake) 와 teacher (선생) 의 진짜 분포 (p_real) 의 score (= 분포의 경사 방향) 차이를 generator (생성기) 로 역전파 (back-propagation = 그래디언트 흘리기)" 하는 분포 매칭 증류 (distribution matching distillation) 로 1-step (한 번에 바로) 이미지 생성을 실용화한 논문 — 핵심은 μ_fake critic (가짜 분포 스코어 추정기) 으로 student 분포의 score 를 추정하고, (s_fake − s_real) 차분을 KL gradient (KL 발산의 미분값) 의 surrogate (대리 함수) 로 사용해 generator 함수 형태에 제약 없이 학습 가능하게 만든 것.**

---

## 🔗 관련 메모리 링크

- [[paper-z-image]] — Z-Image 의 Decoupled DMD + DMDR (DMD 의 응용 사례)
- [[paper-mv-split-dit]] — 1000-layer DiT 학습 안정화 (distillation 과 별개 축)
