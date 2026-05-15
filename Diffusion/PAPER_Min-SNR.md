# PAPER: Min-SNR — Diffusion 학습 속도를 3.4× 가속하는 timestep loss weighting

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Efficient Diffusion Training via Min-SNR Weighting Strategy |
| **저자** | Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, Baining Guo (Microsoft Research Asia + Southeast University) |
| **공개일** | 2023-03-16 (arXiv v1) / **ICCV 2023** |
| **분야** | 이미지 생성 / Diffusion Training / cs.CV |
| **논문 링크** | https://arxiv.org/abs/2303.09556 |
| **공식 코드** | https://github.com/TiankaiHang/Min-SNR-Diffusion-Training |
| **본 문서 목적** | Min-SNR-γ loss weighting 의 동기·수식·코드·실험을 한 페이지로 정리 |

### 위치 짚기 (왜 이 논문이 중요한가)

- **Diffusion 학습 손실 가중치 설계의 표준 baseline.** 후속 Stable Diffusion 3, DiT 계열, EDM2 등의 가중치 비교에서 거의 항상 등장.
- **Hugging Face Diffusers, k-diffusion 에 통합.** `snr_gamma=5` 같은 옵션이 모두 이 논문에서 유래.
- **"어느 parameterization(ε / x₀ / v)이 좋은가" 라는 오래된 논쟁을 "어떤 loss weight를 쓰느냐의 문제"로 환원** 시킴.

---

## 📖 주요 용어 사전 (Glossary)

### Diffusion 기본 (이 논문에서 가정)

- **Forward process**: 깨끗한 이미지 `x₀` 에 노이즈를 점점 더해 `x_T` (거의 가우시안 노이즈) 로 만드는 과정. 한 시점에서 `x_t = α_t · x₀ + σ_t · ε`, ε ~ N(0,I).
- **α_t, σ_t**: timestep t 의 신호 계수와 노이즈 계수. 보통 `α_t² + σ_t² = 1` (variance-preserving) 을 만족.
- **SNR (Signal-to-Noise Ratio, 신호 대 잡음 비율)**: `SNR(t) = α_t² / σ_t²`. t=0 에서 ∞ (깨끗), t=T 에서 ≈0 (순수 노이즈).
- **Cosine schedule**: α_t, σ_t 를 cos/sin 함수로 정의 — 이 논문에서 채택한 noise schedule.

### Parameterization (모델이 무엇을 예측하느냐)

- **ε-prediction (noise prediction)**: 모델이 더해진 노이즈 ε 를 예측. 원조 DDPM 방식.
- **x₀-prediction (start prediction)**: 모델이 깨끗한 이미지 x₀ 자체를 예측. 이 논문이 권장.
- **v-prediction (velocity prediction)**: 모델이 `v = α_t · ε − σ_t · x₀` 를 예측. Progressive Distillation (Salimans & Ho 2022) 에서 제안된 혼합형.
- **세 parameterization 의 핵심 사실**: 셋은 **수학적으로 동등** — 한쪽 출력을 다른 쪽으로 변환 가능. 차이는 **각각의 L = ‖target − pred‖² 가 자동으로 갖는 timestep 가중치가 다르다** 는 점뿐.

### Loss weighting (본 논문 핵심)

- **Loss weight w(t)**: timestep 별 mse loss 에 곱하는 스칼라. 학습이 어느 노이즈 구간을 더 중요시할지 결정.
- **Min-SNR-γ weighting**: `w(t) = min(SNR(t), γ)`. γ 는 상한 (clamp) 값, 본 논문에서 **γ=5 권장**.
- **Truncated SNR** (비교 baseline): `max(SNR(t), 1)`. low noise 만 SNR, high noise 는 1 로 floor.
- **Multi-task learning view**: 각 timestep 의 손실을 별개 task 로 보고, task 간 gradient 충돌 (conflict) 을 줄이는 weight 를 찾는 관점.
- **Pareto optimality (파레토 최적성)**: 한 task 를 더 개선하려면 다른 task 가 반드시 나빠지는 균형점. 본 논문은 stationary (고정) Pareto-aware 가중치를 제안.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> Diffusion 학습은 1000개 timestep 을 동시에 학습하는 multi-task 인데, **timestep 끼리 gradient 가 서로 충돌**해 수렴이 느림. 이 충돌을 줄이기 위해 손실 가중치를 **`min(SNR(t), γ)`** 로 clamp 만 해주면 ImageNet 256×256 에서 동일 step 기준 **FID=10 까지 3.4× 빠르게** 도달하고, ViT-XL 로 **FID 2.06** 달성 (당시 SOTA, DiT-XL/2 의 2.27 을 더 작은 모델로 능가).

**핵심 문제** — Diffusion 학습에서 모든 timestep 의 loss 를 동일 가중치로 더하면 (또는 ε-prediction 의 자연 가중치 1 로 두면) 왜 학습이 느린가?

**해결책** — 3 단계:
1. **문제 진단**: 특정 timestep 구간만 별도 학습시키면 인근 t 는 좋아지지만 먼 t 는 나빠진다 (Figure 2). 즉 timestep 들이 서로 conflicting task.
2. **이론 framing**: 이를 multi-task learning 으로 보고 Pareto-optimal 한 가중치를 찾는 문제로 정식화.
3. **실용 해법**: `w(t) = min(SNR(t), γ)` 로 *상한만 clamp*. low noise (큰 SNR) 가 과도하게 압도하는 것을 방지하면서, high noise 의 진전도 보장. γ=5 권장.

**검증**:
- ImageNet 256×256 (LDM latent, ViT-XL) : **FID 2.06** (~7M iter)
- 3.4× faster to FID=10 vs 종래 가중치
- ε / x₀ / v 어느 parameterization 에서도 일관된 가속 효과

---

## 2️⃣ 핵심 기여 (Contributions)

1. **Timestep 간 gradient 충돌 진단** — finetuning ablation 으로 "한 t 의 손실을 줄이면 다른 t 가 나빠진다" 는 현상을 명시적으로 입증.
2. **Multi-task learning 관점의 정식화** — Diffusion 학습을 T 개 task 의 가중합으로 보고, 가중치 설계를 Pareto optimality 문제로 환원.
3. **Min-SNR-γ 가중치 제안** — `min(SNR(t), γ)` 라는 *극도로 단순한* 클램프 형태. 추가 계산 비용 0, 코드 두 줄.
4. **Parameterization 통합** — ε/x₀/v prediction 사이의 차이를 "자연 가중치" 차이로 해석하고, Min-SNR-γ 적용 시 셋이 모두 비슷한 성능으로 수렴함을 보임 (parameterization 논쟁 해소).
5. **SOTA on ImageNet 256×256** — ViT-XL (451M) 로 **FID 2.06**, 당시 DiT-XL/2 (675M, FID 2.27) 보다 더 작은 모델로 능가.

---

## 3️⃣ 주요 알고리즘 설명

### 3.1 문제: 왜 timestep 들이 충돌하는가

<p align="center">
  <img src="figures/min_snr_fig2.png" alt="Min-SNR Fig. 2 — timestep conflict" width="640"/>
</p>

> **Fig. 2 — Timestep 간 충돌 증거.** 사전학습된 baseline 모델 (초록 가로선, Δ MSE = 0) 위에서 시작해, **특정 timestep 구간만 골라 추가 학습**시킨 뒤 전 구간의 MSE 변화를 측정한 결과.
> - 빨강 = [100,200)만 학습 / 파랑 = [200,300)만 학습 / 노랑 = [300,400)만 학습.
> - 각 색은 **자기 구간(점선 영역)에서는 0 아래로 내려가 손실 감소(개선)**, **하지만 멀리 떨어진 t (특히 t > 400) 에서는 위로 올라가 손실 증가(악화)**.
> 즉 "timestep A 를 잘 학습시키면 timestep B 의 성능이 떨어지는" **multi-task gradient conflict** 가 실험적으로 확인됨. 이 충돌이 Min-SNR이 풀려는 핵심 문제.

| t 구간 | 학습이 시키는 것 | 모델에게 필요한 능력 |
|---|---|---|
| **High noise (t ≈ T)** | 대략적 구조·색·텍스트 정합 만들기 | 의미적·전역적 정보 (coarse layout) |
| **Mid noise** | 형태·경계 다듬기 | 중간 주파수 |
| **Low noise (t ≈ 0)** | 픽셀 수준 세부 디테일 | 고주파수 (high-frequency) |

같은 네트워크 한 set 의 weight 로 이 셋을 *모두* 잘 하라는 요구 → gradient 가 서로 다른 방향을 가리킴 → 평균 내면 진전이 느림.

### 3.2 손실 함수와 자연 가중치

세 parameterization 각각의 MSE 손실은 동일 mse 항을 SNR 로 환산하면 다음과 같음:

| Parameterization | Raw loss `‖target − pred‖²` | x₀ 공간의 "유효 가중치" (자동 부여되는 w(t)) |
|---|---|---|
| ε-prediction | `‖ε − ε̂‖²` | **1** (constant) |
| x₀-prediction | `‖x₀ − x̂₀‖²` | **SNR(t)** |
| v-prediction | `‖v − v̂‖²` | **SNR(t) + 1** |

→ ε 학습은 *모든 t 를 똑같이* 보고, x₀ 학습은 *low noise (큰 SNR) 를 압도적으로 강조*. 둘 다 극단.

### 3.3 Min-SNR-γ 가중치 (논문의 제안)

**아이디어**: x₀ 공간 기준 *유효 가중치* 를 `min(SNR(t), γ)` 로 만든다.

- **High noise (SNR 작음)**: SNR 이 그대로 적용 → ε-prediction (w=1) 보다 더 강조 ↑
- **Low noise (SNR 큼)**: γ 로 clamp → x₀-prediction 의 폭주 ↓

세 parameterization 별 *raw loss 에 곱할* mse_loss_weight:

| Parameterization | `mse_loss_weight` 공식 | 의미 |
|---|---|---|
| **x₀-prediction** | `min(SNR, γ)` | 기본 형태 |
| **ε-prediction** | `min(SNR, γ) / SNR` = `min(1, γ/SNR)` | ε 의 자연 가중치(1)를 x₀ 공간으로 변환 후 clamp |
| **v-prediction** | `min(SNR, γ) / (SNR + 1)` | v 의 자연 가중치(SNR+1)로 나눠줌 |

**권장 γ**: **5** (실험상 가장 안정적).

### 3.4 Pareto-optimal 관점 (Theorem 1 요약)

이상적인 가중치는 매 iteration 마다 다음 최적화를 풀어야 함:

```
min_{w_t}   ‖Σ_t w_t · ∇_θ L_t(θ)‖₂² + λ · Σ_t ‖w_t‖₂²
                                                                                              
              (1) 모든 task gradient 의 합 노름 최소화 (충돌 상쇄)
              (2) regularization (한쪽으로 쏠리지 않게)
```

이건 매 step 마다 풀기 비싼 문제. Min-SNR-γ 는 이걸 **고정된 stationary 근사** 로 대체 — `w(t) = min(SNR(t), γ)` 가 평균적으로 충돌을 잘 완화한다는 경험적 발견.

### 3.5 코드 매핑 (공식 repo)

`guided_diffusion/gaussian_diffusion.py:846-895` — `training_losses` 안에서 mse_loss_weight 를 계산:

```python
# guided_diffusion/gaussian_diffusion.py (요지 발췌)
alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
snr = (alpha / sigma) ** 2          # SNR(t) = α²/σ²

# 분기 1: 모델이 ε 또는 v 를 예측하거나 weight type 이 constant 일 때
#         → raw loss 에 곱할 가중치를 x₀ 공간 기준으로 환산
if self.model_mean_type is not ModelMeanType.START_X or weight_type == 'constant':
    if weight_type.startswith("min_snr_"):
        k = float(weight_type.split('min_snr_')[-1])
        mse_loss_weight = th.stack([snr, k*th.ones_like(t)], dim=1).min(dim=1)[0] / snr
        # ε 예측용: min(SNR, k) / SNR  = min(1, k/SNR)
    elif weight_type.startswith("vmin_snr_"):
        k = float(weight_type.split('vmin_snr_')[-1])
        mse_loss_weight = th.stack([snr, k*th.ones_like(t)], dim=1).min(dim=1)[0] / (snr + 1)
        # v 예측용: min(SNR, k) / (SNR+1)

# 분기 2: 모델이 x₀ 를 직접 예측할 때
else:
    if weight_type.startswith("min_snr_"):
        k = float(weight_type.split('min_snr_')[-1])
        mse_loss_weight = th.stack([snr, k*th.ones_like(t)], dim=1).min(dim=1)[0]
        # x₀ 예측용: min(SNR, k)  ← 가장 단순한 본가지

# zero-terminal SNR (last step) 안전장치
mse_loss_weight[snr == 0] = 1.0

# raw loss 에 곱해서 최종 mse
terms["mse"] = mse_loss_weight * mean_flat((target - model_output) ** 2)
```

**키 포인트**:
- 한 함수 안에 세 가지 parameterization 의 변환을 모두 처리. weight_type 문자열로 분기 (`min_snr_5`, `vmin_snr_5` 등).
- `snr == 0` 위치 (terminal step) 는 1 로 보정 — zero-terminal SNR schedule 대응.
- Truncated SNR (`trunc_snr` = `max(SNR, 1)`), inverse SNR (`inv_snr`), max-SNR-k (`max_snr_k`) 등 비교 baseline 도 모두 한 곳에.

### 3.6 학습 config (재현용)

`configs/in256/vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bs8x32.sh` 에서:

| 항목 | 값 |
|---|---|
| 모델 | `vit_base_patch2_32`, depth 12 (ViT-B/2) |
| 입력 | LDM latent 32×32×4 (image 256×256 → AutoencoderKL 압축) |
| Diffusion steps | 1000, **cosine schedule** |
| Parameterization | `--predict_xstart True` (x₀-prediction) |
| Loss weight | `--mse_loss_weight_type min_snr_5` |
| Optimizer | AdamW, lr=1e-4, β₁=0.99, β₂=0.99, weight_decay=0.03 |
| 배치 | GPU 8 × per-GPU 32 = **256** |
| Mixed precision | fp16 |
| Class drop prob | 0.15 (CFG 학습용) |

v-prediction 버전은 같지만 `--predict_v True`, `--mse_loss_weight_type vmin_snr_5`.

---

## 4️⃣ 실험 요약

### 4.1 ImageNet 256×256 (LDM latent, ViT-XL, class-conditional)

| 모델 | 파라미터 | FID ↓ | 비고 |
|---|---|---|---|
| LDM | 400M | 3.60 | |
| ADM | 554M | 7.49 | |
| ADM-G | 608M | 3.94 | classifier guidance |
| DiT-XL/2 | 675M | 2.27 | 동시기 SOTA |
| **Min-SNR (ours, ViT-XL)** | **451M** | **2.06** | **더 작은 모델로 SOTA 갱신** |

### 4.2 수렴 속도 (vs 가중치 종류)

- FID = 10 도달까지 기준: **Min-SNR-5 가 baseline 가중치 대비 3.4× 빠름**.
- Max-SNR-γ 는 발산 (low noise 만 강조해 high noise 학습이 망가짐).
- 세 parameterization (ε / x₀ / v) 어디서든 Min-SNR-γ 가 최고 수렴 곡선.

### 4.3 γ 민감도

- γ=5 가 일관되게 최고. γ=1 은 너무 강한 clamp 로 high noise 만 학습됨. γ=∞ 는 결국 원래 x₀-prediction (low noise 압도) 회귀.

---

## 5️⃣ 💬 Q&A — 직관적 이해

### Q1. ε-prediction 은 자연 가중치가 1 (=constant) 인데, 왜 그래도 학습이 잘 안 되는가?

x₀ 공간으로 환산하면 ε-prediction 의 유효 가중치는 1 — 즉 *모든 timestep 의 픽셀 오차를 동등 비중으로* 본다. 그런데 인간이 보기에 *high noise 의 픽셀 오차* 와 *low noise 의 픽셀 오차* 는 의미가 다름. low noise 쪽은 세부 디테일이라 픽셀 오차가 작아도 시각적 임팩트가 있고, high noise 쪽은 큰 구조라 픽셀 오차가 커도 글로벌하게 영향. 단순 동등 가중은 이 비대칭을 무시 → 비효율적.

**Min-SNR-γ** 는 "low noise 는 SNR 만큼 비중 ↑, 단 γ 에서 cap" 으로 비대칭을 적절히 반영.

### Q2. 왜 max 가 아니라 min 인가?

- `max(SNR, γ)` 는 *높은 SNR 만 더 강조* → low noise 폭주, 학습 불안정.
- `min(SNR, γ)` 은 *high SNR 의 폭주를 막고 low SNR (high noise) 의 비중을 상대적으로 살림* → 균형.

직관적으로 high noise (어려운 task) 는 가뜩이나 학습이 더디므로 *낮은 SNR 쪽이 손실에서 묻히지 않게* 보호해줘야 한다.

### Q3. γ=5 의 의미는?

`SNR(t) > 5` 인 timestep — 즉 노이즈가 거의 없는 low-noise 영역 — 의 손실 가중치를 5 로 cap. 그보다 noise 가 많은 t 는 SNR 그대로. cosine schedule + 1000 step 기준 대략 t < 100 정도가 clamp 영역에 해당.

### Q4. 이 가중치를 다른 schedule (rectified flow, EDM) 에도 쓸 수 있나?

원논문은 cosine schedule + DDPM 가정. EDM 은 σ 기반 표현이라 SNR 정의가 달라지지만 같은 철학 적용 가능 (실제로 EDM2 가 자체 가중치 제안). Rectified Flow / Stable Diffusion 3 도 자체 logit-normal weighting 을 쓰지만, **Diffusers 라이브러리는 SD2/SDXL fine-tuning 시 Min-SNR-γ 옵션 (`snr_gamma`) 을 그대로 제공**.

### Q5. DMD 같은 distillation 과 직접 관련 있나?

직접 관련은 없음. Min-SNR 은 **teacher (base diffusion model) 자체를 더 빠르게 학습** 시키는 것. distillation 은 학습된 teacher 를 다시 student 로 옮기는 별개 단계. 단, [[paper-dmd]] 의 fake-score critic μ_fake 도 forward-diffuse 후 학습되는 diffusion 모델이라 Min-SNR 가중치를 거기서도 쓸 수 있음.

### Q6. 한 줄로: 무엇이 이 논문의 진짜 새로움인가?

> "Parameterization 논쟁 (ε vs x₀ vs v) 은 사실상 가중치 논쟁이었고, 가중치를 `min(SNR, γ)` 로 clamp 만 해도 셋이 모두 비슷한 최적값에 도달한다" 는 통합 시점.

---

## 6️⃣ 한 줄 요약 (전체)

> Diffusion 학습 손실에 `w(t) = min(SNR(t), γ=5)` 한 줄을 곱하면 timestep 간 gradient 충돌이 완화돼 수렴이 3.4× 빨라지고, ε/x₀/v 어느 parameterization 으로 학습하든 비슷한 SOTA (ImageNet 256² FID 2.06) 에 도달한다.

---

## 7️⃣ 관련 메모리 링크

- [[paper-dmd]] — 본 논문은 *teacher 학습* 가속, DMD 는 *student 추론* 가속. 상보적.
- [[paper-z-image]] — Z-Image 도 자체 loss weighting (SNR-aware) 를 채택. Min-SNR 의 후예 격.
- [[reference-pretrained-backbone-reuse-landscape]] — 본 논문의 ViT-XL latent diffusion 은 backbone 재사용 분기 C (from-scratch DiT) 의 대표 baseline.
