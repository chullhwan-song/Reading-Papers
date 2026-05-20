# PAPER: DMD2 (Improved Distribution Matching Distillation) — DMD 의 안전벨트를 떼고 천장을 뚫기

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목 (메인)** | Improved Distribution Matching Distillation for Fast Image Synthesis |
| **저자** | Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman (MIT + Adobe) |
| **공개일** | 2024-05-23 (arXiv v1) / NeurIPS 2024 |
| **분야** | 이미지 생성 / Diffusion Distillation / cs.CV |
| **논문 링크** | https://arxiv.org/abs/2405.14867 |
| **코드** | https://github.com/tianweiy/dmd2 |
| **프로젝트 페이지** | https://tianweiy.github.io/dmd2/ |
| **Hugging Face** | https://huggingface.co/tianweiy/DMD2 |
| **본 문서 목적** | DMD 의 한계 6요소 → DMD2 의 6가지 처방을 정확히 매칭. PAPER_DMD.md (모태 논문) 와 짝 |

### 관련 논문 메타데이터

| 역할 | 논문 | 저자 · 학회 | arXiv |
|---|---|---|---|
| **모태** | DMD: One-step Diffusion with Distribution Matching Distillation | Yin et al., CVPR 2024 | 2311.18828 |
| **메인** | DMD2: Improved Distribution Matching Distillation for Fast Image Synthesis | Yin et al., NeurIPS 2024 | 2405.14867 |
| **TTUR 원전** | GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium | Heusel et al., NeurIPS 2017 | 1706.08500 |
| **이론적 상위 (VSD)** | ProlificDreamer — Variational Score Distillation | Wang et al., NeurIPS 2023 | 2305.16213 |
| **Consistency parameterization** | Consistency Models | Song et al., ICML 2023 | 2303.01469 |
| **LCM (실제 배포 호환)** | Latent Consistency Models | Luo et al., 2023 | 2310.04378 |
| **Diffusion-GAN 안정화** | Diffusion-GAN: Training GANs with Diffusion | Wang et al., ICLR 2023 | 2206.02262 |
| **비교 (few-step)** | SDXL-Lightning, Hyper-SD, SDXL-Turbo, Imagine Flash | 각각 별도 | — |

---

## 📖 주요 용어 사전 (Glossary)

> *왜 이 절을 두나*: PAPER_DMD.md 의 용어집 위에 DMD2 특유의 새 용어만 추가. DMD 기본 용어 (Score, μ_fake, KL, VSD, L_reg 등) 는 [PAPER_DMD.md](PAPER_DMD.md) 참고.

### DMD2 신규 용어

- **TTUR (Two Time-Scale Update Rule, 두 속도 업데이트 규칙)**: GAN 학습 안정화 기법 (Heusel 2017). 원래는 G 와 D 에 **다른 learning rate** 를 주는 것. DMD2 에서는 **다른 업데이트 빈도** 로 재해석: critic μ_fake 를 5번 학습 → generator G_θ 를 1번 학습 (= 5:1 ratio).
- **stale critic (낡은 critic, 시차 측량사)**: μ_fake 가 student 분포 변화를 못 따라잡아 옛 분포의 score 를 알려주는 상태. 이 상태에서 grad = (s_fake_stale − s_real) 는 **현재 student 가 있는 위치에선 잘못된 방향** 을 가리킴.
- **Real-data GAN loss (진짜 데이터 판별 손실)**: distillation 학습 중에 **진짜 이미지 (real_image)** 와 student 생성 이미지를 구분하는 표준 GAN 손실 추가. teacher score 의 부정확함을 우회해 student 가 teacher 를 능가할 수 있게 함.
- **Diffusion-GAN (Wang 2023)**: GAN discriminator 입력에 **랜덤 노이즈를 다시 씌워** 분류하게 만드는 기법. high-frequency texture mode collapse 회피 + 학습 안정성 향상.
- **Multi-step generator (consistency-style)**: G_θ 가 **`(x_t, t) → x̂₀` 직예측** 함수로 작동 (DDIM/DPM 식 reverse step 아님). 다음 timestep 으로의 이동은 **forward noising 공식 `add_noise(x̂₀, randn, t_next)`** 로 처리. Consistency Models (Song 2023) 의 파라미터화.
- **Backward simulation (역방향 시뮬레이션)**: 4-step 학습 시 **random k ∈ {0,1,2,3} 을 뽑고, k-1 번까지 student 를 (no_grad 로) 굴려 "추론 시점 k 의 입력 분포" 를 만든 뒤 그 위에서 한 발짝만 grad 흘림** → train/inference input mismatch 해결.
- **train/inference input mismatch (학습/추론 입력 불일치)**: 학습 때는 real_image 에 노이즈를 씌운 입력만 보는데, 추론 때는 student 자신의 출력에 노이즈를 씌운 입력이 들어옴 → 분포 어긋남 → multi-step 누적 오류.
- **conditioning_timestep (조건부 timestep)**: 1-step DMD2 에서 student 에 입력으로 주는 고정 timestep 값. **399** (전체 1000 중) 가 선택됨 — pure noise t=999 가 아니라 약간 정보가 있는 mid-noise 시점에서 시작.
- **dfake_gen_update_ratio (D-fake / Generator 업데이트 비율)**: DMD2 코드의 TTUR 구현 변수. 기본값 **5** = critic 5번 학습 → generator 1번 학습.
- **non-saturating GAN loss (학습 초반에 그래디언트가 안 죽는 GAN 손실)**: Goodfellow 2014 의 표준 변형. discriminator 입장: `softplus(logit_fake) + softplus(-logit_real)`, generator 입장: `softplus(-logit_fake)`. BCE-with-logits 의 수치 안정 형태.
- **LCMScheduler**: Latent Consistency Model 의 sampling scheduler. DMD2 generator 가 consistency-style 함수 시그니처를 따르기 때문에 추론 시 그대로 호환됨 (timesteps=[999, 749, 499, 249]).
- **softplus**: `softplus(x) = log(1 + exp(x))`. BCE 의 numerically stable 표현.

### 평가/배포 용어

- **NFE (Number of Function Evaluations)**: 한 이미지 만들기 위해 모델을 통과시킨 횟수. 50-step + CFG = NFE 100. DMD2 4-step (CFG 없음) = NFE 4.
- **CFG distillation (분류기-없는-안내 증류)**: 학습 시 teacher 에 강한 CFG (예: real_guidance_scale=8) 를 적용해 그 결과를 student 가 흡수 → student 자체는 추론 시 CFG 불필요 → NFE 절반.
- **Zero-shot COCO-FID**: 학습에 LAION 만 쓰고 COCO 평가 셋으로 FID 측정. text-to-image 모델의 표준 평가.
- **CLIP-FID**: feature extractor 를 ImageNet-pretrained Inception 대신 CLIP image encoder 로 바꾼 FID. text-to-image 에 더 적합.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> **DMD2 = DMD 의 회귀 손실 (regression loss, L_reg, 정답 쌍 MSE) 안전벨트를 떼어내고, 그 자리에 (1) TTUR 로 critic 을 5× 빠르게 학습시켜 안정성을 잡고, (2) Real-data GAN loss 로 teacher score 의 천장을 뚫고, (3) Consistency-style multi-step + backward simulation 으로 multi-step train/test mismatch 를 해결한 후속작.** 결과: ImageNet-64 1-step FID **1.28** (teacher EDM 1.79 추월), SDXL 4-step 1024² 가 사실상 표준 (ComfyUI/Diffusers 그대로 동작), 추론 비용 **500× 절감**.

**핵심 문제**: DMD 가 (a) paired dataset 생성 비용 (SDXL: ~700 A100 days), (b) teacher score 부정확함의 천장, (c) multi-step 학습/추론 입력 불일치 — 이 세 가지로 막혀 있었음.

**해결책 (6가지 처방)**:
1. **L_reg 제거** — paired dataset 의존성 폭파 (700 A100 days 회피).
2. **TTUR** — critic 을 5× 빨리 굴려 stale fake-score 문제 해결.
3. **Real-data GAN loss** — μ_fake 의 bottleneck feature 에 conv head 붙여 진짜/가짜 판별 → teacher 추월 가능.
4. **Multi-step generator (consistency-style)** — `(x_t, t) → x̂₀` 직예측 + forward noising 으로 다음 step 이동. LCM 과 호환.
5. **Backward simulation** — random k 단계까지 student 를 (no_grad) 굴려 추론 시점 입력을 시뮬레이션 → train/test gap 해결.
6. **통합 알고리즘 (G 1× / D 5× 교차)** — 매 5 step 마다 G 한 번, 나머지는 D 만.

**검증**:
- ImageNet 64×64 (1-step) : FID **1.28** vs teacher EDM 1.79 → **teacher 추월** ✨
- COCO 2014 zero-shot (4-step, SDv1.5) : FID **8.35** vs teacher 8.59 → 추월
- SDXL 1024² (4-step) : CLIP-FID **19.32** ≈ teacher 19.36, **SDXL-Turbo (24.57), SDXL-Lightning (24.46) 보다 우수**
- 추론 비용 **500× 절감** (NFE 100 → 1)

---

## 2️⃣ 핵심 기여 (Contributions)

> *왜 이 절을 두나*: 논문이 Section 4 에서 **6개 소절 (§4.1~4.6)** 로 명시적으로 나눈 method components 가 후속 연구의 표준 패턴이 되었기 때문에 정확히 매칭해 정리.

1. **§4.1 — L_reg 제거의 실용적 정당화**: SDXL paired dataset 만들기 = **~700 A100 days × 12M 쌍** (단순 산수: 50-step × 2 forward × 12M = NFE 1.2B → A100 1장 NFE 2,000/s 가정 시 약 167 GPU-days 의 4배 = ~700) 가 사실상 불가능함을 명시 → L_reg 제거가 단순한 개선이 아니라 **대규모 배포의 전제 조건**.

2. **§4.2 — TTUR 로 pure score distillation 안정화**: stale critic 진단 + 5:1 update ratio 처방. **regression loss 없이 ImageNet 에서 DMD 원작 품질 달성** (FID 동등 또는 개선).

3. **§4.3 — Real-data GAN loss 로 teacher 추월**: μ_fake unet 의 bottleneck feature 위에 작은 conv head (cls_pred_branch) 를 붙여 별도 D 없이 효율적인 adversarial loss 통합. Diffusion-GAN 식 노이즈 주입으로 안정화. 결과: ImageNet 1-step FID 1.28 < teacher 1.79.

4. **§4.4 — Multi-step generator (consistency parameterization)**: G_θ 가 (x_t, t) → x̂₀ 직예측 함수로 작동. 다음 step 으로의 이동은 forward noising 공식만 사용 (학습된 reverse scheduler 없음). 학습/추론 timestep 동일 ([999, 749, 499, 249] for 4-step). **LCMScheduler 와 그대로 호환** 되는 이유.

5. **§4.5 — Backward simulation 으로 train/inference mismatch 해결**: random k 단계까지 student 를 굴려 "추론 시점 k 의 입력 분포" 를 만든 뒤 그 위에서 1-step grad. Imagine Flash 와 달리 **distribution matching loss** 라 누적 오류 회피.

6. **§4.6 — 통합 학습 알고리즘**: 매 step (G 0 또는 1번 + D 1번) 의 교차 업데이트. G 는 step % 5 == 0 일 때만, D 는 매 step. SDXL LoRA / full UNet 양쪽으로 배포 가능한 표준 패러다임 확립.

---

## 3️⃣ 주요 알고리즘 설명

### 3.1 전체 학습 흐름 (한 그림)

> *왜 이 절을 두나*: 6요소가 어떻게 한 스텝 안에 결합되는지 한눈에 봐야 부분-전체 관계가 명확해짐.

```
┌─ Real image (LAION VAE latents, LMDB) ─────────────────┐
│                                                        │
│  ┌─ Noise z + random k ∈ {0,1,2,3} ─────────────────┐  │
│  │                                                  │  │
│  │  [§4.5 Backward simulation, no_grad]             │  │
│  │  : z → student → x̂₀ → renoise(t_next) →          │  │
│  │    student → ... (k-1 번) → x_k                  │  │
│  │                                                  │  │
│  │  ▼                                               │  │
│  │  [§4.4 Consistency-style multi-step]             │  │
│  │  feedforward_model (G_θ, LoRA or full)           │  │
│  │  : (x_k, t_k) → x̂₀  (grad 흐름)                  │  │
│  │  ▼                                               │  │
│  │  x̂₀ (k-step 째 student 출력)                     │  │
│  │  │                                               │  │
│  │  ├─→ [§3 배경: DMD gradient]                     │  │
│  │  │   forward-diffuse to random t                 │  │
│  │  │   ▼                                           │  │
│  │  │   real_unet  (frozen, CFG=8)  → p_real        │  │
│  │  │   fake_unet  (학습 중)        → p_fake        │  │
│  │  │   ▼                                           │  │
│  │  │   grad = (p_real − p_fake) / |p_real|.mean    │  │
│  │  │   loss_dm = 0.5·‖x̂₀ − (x̂₀−grad).detach‖²    │  │
│  │  │                                               │  │
│  │  └─→ [§4.3 GAN loss, Generator 측]               │  │
│  │      fake_unet bottleneck → cls_pred_branch      │  │
│  │      (diffusion-gan: noise t∈[0,1000] 추가)      │  │
│  │      logit_on_fake → gen_cls_loss                │  │
│  │      = softplus(−logit_on_fake)                  │  │
│  │                                                  │  │
│  │  L_G = loss_dm + 5e-3 · gen_cls_loss             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  [§4.2 TTUR — step % 5 == 0 일 때만 L_G 흘림]          │
│                                                        │
│  Critic 학습 (매 step):                                │
│   - loss_fake = (μ_fake denoising MSE on student x̂₀)  │
│   - guidance_cls_loss = softplus(logit_fake) +         │
│                         softplus(−logit_real)          │
│   L_D = loss_fake + 1e-2 · guidance_cls_loss           │
└────────────────────────────────────────────────────────┘
```

### 3.2 §4.1 — L_reg 제거의 동기 (700 A100 days)

> *왜 이 절을 두나*: DMD2 의 6요소 중 가장 단순하지만 가장 결정적 — "안 빼면 SDXL 규모 학습이 사실상 불가능" 이라는 비용 장벽.

**DMD 원작의 paired dataset 요구**:
- Teacher (예: SDXL) 가 z → x₀ 매핑을 **50-step CFG sampler** (NFE 100) 로 정확히 만들어둠.
- DMD 학습 중 student G_θ 의 출력을 이 (z, x₀) 쌍의 x₀ 와 MSE 비교 → L_reg.

**SDXL 규모에서의 산수** (논문 §4.1):
- 양질 학습엔 **12M 쌍 정도** 필요.
- 한 쌍 당 NFE 100 (50 step × 2 forward for CFG), 해상도 1024².
- A100 1장에서 SDXL inference ~3-5 sec/image at NFE 100.
- 12M × 4 sec ≈ **5.6 × 10⁷ sec = 약 1,400 A100-hours = 약 58 A100-days × 12 (병렬) = ~700 A100 days**.
- 실제 학습 자체 (모델 업데이트) 보다 **데이터 만들기가 더 비쌈**.

→ **L_reg 제거 = SDXL 규모 distillation 의 전제**. 단순 개선이 아니라 강제된 선택.

### 3.3 §4.2 — TTUR: stale critic 진단과 처방

> *왜 이 절을 두나*: DMD2 의 가장 깊은 이론적 통찰. "L_reg 가 안 빠진 게 아니라 critic 이 못 따라가서 안 빠진 거다" — 이 진단을 정확히 이해해야 후속 처방의 가치가 보임.

#### (a) 등장 인물 — 누가 누구를 추적하나

DMD 학습판에는 **계속 움직이는 두 분포** 가 있음:

| 역할 | 무엇인가 | 학습 중 행동 |
|---|---|---|
| **p_real** (진짜 분포) | teacher 가 표현하는 정답 분포 | 안 움직임 (teacher frozen) |
| **p_fake_θ** (가짜 분포) | student G_θ 가 매 순간 만들어내는 분포 | **매 step 마다 모양이 바뀜** (G_θ 학습 중) |

그리고 두 분포의 score (분포의 경사 벡터) 를 추정하는 두 측량사:

| 측량사 | 무엇을 알려주나 | 학습 중 행동 |
|---|---|---|
| **s_real** (teacher unet, frozen) | p_real 의 어느 점에서 어디로 가야 더 높은가 | 안 움직임. 항상 정확. |
| **s_fake** (μ_fake critic) | **p_fake_θ 의 어느 점에서 어디로 가야 더 높은가** | **p_fake_θ 가 움직이니까 같이 갱신돼야 함** |

→ **s_fake 는 "현재의" p_fake 를 정확히 알아야 의미가 있는 측량값**. 측량할 산이 매 순간 변하니까, 측량사도 같이 따라다녀야 함.

#### (b) GAN-like 결합 동역학계

```
   G_θ 가 그림을 만든다
        │
        ▼
   μ_fake 가 "이 그림들이 어떤 분포에 속하는지" 를 학습한다
        │
        ▼
   그 결과 s_fake 를 보고 G_θ 가 한 발짝 움직인다
        │
        ▼
   G_θ 가 움직였으니 만드는 그림 분포가 또 바뀐다
        │
        ▼ (다시 처음으로)
   μ_fake 가 새 분포를 따라잡아야 한다 ...
```

이게 **결합된 (coupled) 동역학계**. 한쪽이 움직이면 다른 쪽이 따라가야 정상 작동.

#### (c) 같은 페이스 (1:1) 업데이트가 망함 — 두 시나리오

**시나리오 A — 학습 초반의 grad ≈ 0 함정**:

DMD2 코드는 `real_unet` 과 `fake_unet` 둘 다 **teacher 로 초기화** ([sd_guidance.py:45-58](file:///tmp/dmd2/main/sd_guidance.py#L45-L58)):
```python
self.real_unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
self.fake_unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
```

→ step 0 에서 **s_fake = s_real** → grad = s_fake − s_real = 0 → G_θ 가 움직이지 않음 → 학습이 시작 자체가 안 됨.

이게 풀리려면 μ_fake 가 **빨리** "현재 G_θ 의 분포" 의 score 를 학습해야 함. 1 step 만에 따라잡기 어려우니 여러 step 필요.

**시나리오 B — 학습 중반의 drift 발산**:

1. step t: G_θ 가 "강아지 분포" 에 가까운 상태.
2. μ_fake 가 step t 의 강아지 분포를 학습 (1 step 만으론 부분적).
3. step t+1: G_θ 가 grad 받고 한 발짝 → "강아지 + 약간 고양이" 로 변함.
4. **μ_fake 는 아직 step t 의 강아지만 알고 있음** — 고양이는 모름.
5. step t+1 에서 s_fake 가 평가됨 → "고양이 영역" 에서 μ_fake 가 헛소리 (OOD = out-of-distribution = 학습 안 된 영역).
6. grad = s_fake_헛소리 − s_real → **헛소리 방향으로 G_θ 가 움직임**.
7. 다음 step 에서 G_θ 가 더 이상한 분포로 튐 → μ_fake 가 더 못 따라잡음 → **발산**.

#### (d) 비유 — GPS 가 5초 늦는 자율주행차

```
  자율주행차 (= G_θ 가 만드는 분포 p_fake)
        │  현재 위치 = (북위 37.5, 동경 127.0)
        ▼
  GPS 센서 (= μ_fake critic)
        │  5초 지연 → 5초 전 위치 (37.49, 127.01) 로 보고
        ▼
  주행 컴퓨터 (= grad 계산)
        │  "5초 전 위치 → 목적지" → "동쪽으로 가라" 명령
        ▼
  차가 이미 동쪽으로 가고 있는데 또 동쪽 → 지나침 → 다시 서쪽 → 다시 동쪽
  → 진동 (oscillation) 또는 발산 (divergence)
```

**시차 (lag) 가 만드는 두 종류 오류**:
1. **방향 오류**: 5초 전과 현재 위치가 다르면, 그 위치에서 측정한 "어디로" 는 현재 위치에선 잘못된 방향.
2. **OOD 오류**: 차가 새 영역으로 들어갔으면 GPS 가 그 동네 지도를 안 가짐 → 헛소리.

DMD2 critic 도 똑같음:
- (1) student 분포가 한 step 전과 지금이 다르면 → 방향 오류.
- (2) student 가 새 영역을 만들면 critic 미학습 → OOD 헛소리.

#### (e) Grad 방향 망함의 수학적 직관 (말로)

DMD grad = **s_fake − s_real** 의 의미:
- **s_real** = "이 점에서 teacher 분포의 밀도가 더 높은 쪽" 화살표.
- **s_fake** = "이 점에서 student 분포의 밀도가 더 높은 쪽" 화살표.
- 두 화살표의 차 = "내가 너무 자주 가는 곳에서 빠져나와 (−s_fake), teacher 가 좋아하는 곳으로 가라 (+s_real)" 라는 **분포 보정 신호**.

Stale critic 의 두 망함:
- **(망함 a) s_fake ≈ s_real → grad ≈ 0**: critic 이 "내가 자주 가는 곳 = teacher 가 좋아하는 곳" 이라고 잘못 보고. 학습 정체.
- **(망함 b) s_fake 가 옛 분포 화살표**: critic 이 한 step 전 모드를 가리킴 → G_θ 는 옛 모드 회피 신호만 받고 **새 모드 (현재 자기 모드) 에서 빠져나오라는 신호가 없음** → **모드 붕괴 (mode collapse)**.

특히 (망함 b) 가 무서운 이유: G_θ 가 새 모드에 갇히면 **positive feedback loop** — train 시간이 갈수록 더 갇힘, critic 은 항상 한 step 뒤를 외침.

#### (f) TTUR — "측량사를 5배 빨리 굴리기"

[main/train_sd.py:330](file:///tmp/dmd2/main/train_sd.py#L330):
```python
COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0
```

`dfake_gen_update_ratio = 5` → 매 step:
```
step 1: critic 만 학습 (G_θ 그대로)    → μ_fake 가 현재 분포에 한 발짝 더 다가감
step 2: critic 만 학습                  → 더 다가감
step 3: critic 만 학습                  → 더 다가감
step 4: critic 만 학습                  → 현재 분포에 거의 정확
step 5: G_θ + critic 둘 다 학습         → s_fake 가 정확하니 G_θ 가 올바른 방향으로 이동
step 6: critic 만 학습 ... (반복)
```

**왜 5?**: hyperparameter 검색 결과. 1 이면 lag → 발산. 너무 크면 G_θ 학습 비효율. **5 가 ImageNet/SDXL 양쪽 sweet spot** (논문 §4.2 ablation).

#### (g) 이론적 근거 — Heusel 2017 TTUR

GAN 학습 이론 (Goodfellow 2014 §4.2): **D 가 generator 분포를 정확히 알 때만 G 의 minimax gradient 가 unbiased**. D 가 부정확하면 G 의 gradient 는 **편향 (bias)** 누적 → 발산.

Heusel 2017 의 TTUR 정리: D 가 G 보다 충분히 빠른 페이스로 학습하면 **D 가 매 시점 G 의 분포를 거의 정확히 추정한다는 stationary 보장** → G 의 gradient 가 unbiased → 수렴.

→ DMD2 가 L_reg 안전벨트 없이도 학습 가능한 **이론적 근거**.

### 3.4 §4.3 — Real-data GAN loss: teacher 의 천장 뚫기

> *왜 이 절을 두나*: TTUR 만 추가하면 DMD 원작과 동등한 품질 (teacher 와 갭 유지). teacher 추월이라는 진짜 도약은 GAN loss 에서 나옴.

#### (a) 왜 GAN loss 가 필요한가

DMD 원작의 한계 — **s_real 의 부정확함**:
- s_real 은 teacher 의 ε-prediction 에서 유도되는데, teacher 도 완벽하지 않음.
- 특히 **high-noise 영역** (t 큰 영역) 에서 teacher 의 예측 자체가 모호.
- DMD 는 그 부정확한 s_real 만 따라서 → **student ≤ teacher**.

DMD2 의 해결책 — **이미지 공간에서 진짜 데이터를 직접 본다**:
- s_real 의 "score 화살표" 를 보는 게 아니라, **real_image 자체를 한 번 더 본다**.
- Real_image 의 분포 = 정답 그 자체 (teacher 의 추정 X) → teacher 의 추정 오차가 우회됨.

#### (b) 구조 — 절약의 미학: μ_fake 가 D 역할 겸임

별도 discriminator 안 만들고 **μ_fake unet 의 bottleneck feature 를 그대로 분류기 입력으로** 재사용.

[main/sd_guidance.py:108-122](file:///tmp/dmd2/main/sd_guidance.py#L108-L122):
```python
self.cls_pred_branch = nn.Sequential(
    nn.Conv2d(4, 1280, 1280, stride=2, padding=1),  # 32×32 → 16×16
    nn.GroupNorm(32, 1280), nn.SiLU(),
    nn.Conv2d(4, 1280, 1280, stride=2, padding=1),  # 16×16 → 8×8
    nn.GroupNorm(32, 1280), nn.SiLU(),
    nn.Conv2d(4, 1280, 1280, stride=2, padding=1),  # 8×8 → 4×4
    nn.GroupNorm(32, 1280), nn.SiLU(),
    nn.Conv2d(4, 1280, 1280, stride=4, padding=0),  # 4×4 → 1×1
    nn.GroupNorm(32, 1280), nn.SiLU(),
    nn.Conv2d(1, 1280, 1, stride=1, padding=0)       # 1×1 → 1×1 logit
)
```

→ μ_fake 의 가장 깊은 1280-dim feature 위에 **5단 conv** 로 단일 logit 출력. 거의 무료.

**왜 이 위치인가**:
- μ_fake 는 이미 "student 분포의 score" 를 학습 중 → student 분포에 특화된 feature 학습됨.
- 그 feature 가 진짜/가짜 분류에도 자연스럽게 유용.
- 별도 D 네트워크 메모리/계산 0.

#### (c) 손실 식 — Non-saturating GAN (softplus 형태)

**Discriminator 측 (critic 학습 시)** — [sd_guidance.py:391](file:///tmp/dmd2/main/sd_guidance.py#L391):
```python
classification_loss = F.softplus(pred_realism_on_fake).mean()  # fake → 작게
                    + F.softplus(-pred_realism_on_real).mean() # real → 크게
```
*(즉: real 이미지에는 logit 이 크게, fake 이미지에는 logit 이 작게 나오도록. softplus 는 BCE 의 수치 안정 표현.)*

**Generator 측 (G_θ 학습 시)** — [sd_guidance.py:323](file:///tmp/dmd2/main/sd_guidance.py#L323):
```python
gen_cls_loss = F.softplus(-pred_realism_on_fake_with_grad).mean()
```
*(즉: 생성 이미지의 logit 이 크게 나오도록 G_θ 가 학습. non-saturating GAN 의 표준 generator loss.)*

논문의 표준 형태 (§4.3):
```
L_GAN = E_{x~p_real}[log D(F(x,t))] + E_{z~p_noise}[−log(D(F(G_θ(z),t)))]
```
→ softplus 표현은 이 log 식의 numerically stable 변형 (BCE-with-logits).

#### (d) Diffusion-GAN 안정화

[main/sd_guidance.py:147-151](file:///tmp/dmd2/main/sd_guidance.py#L147-L151):
```python
if self.diffusion_gan:
    timesteps = torch.randint(0, self.diffusion_gan_max_timestep, [N], dtype=torch.long)
    image = self.scheduler.add_noise(image, torch.randn_like(image), timesteps)
```

**왜 노이즈를 또 씌우나**:
- 깨끗한 이미지를 D 에 그대로 넣으면 high-frequency texture 만으로 분류 → mode collapse.
- 작은 노이즈로 흐릿하게 → **structural-level 판별** 강제 → 더 안정적.
- Wang 2023 (Diffusion-GAN) 의 표준 트릭.

#### (e) 학습 가중치 — gradient scale 매칭

학습 script ([sdxl_4step.sh](file:///tmp/dmd2/experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh)):
- `dm_loss_weight = 1`
- `gen_cls_loss_weight = 5e-3`
- `guidance_cls_loss_weight = 1e-2`

**왜 GAN 비중이 작은가**: DM gradient 크기가 워낙 큼. 5e-3 으로 맞추면 gen 측 backprop 의 두 grad 가 거의 같은 크기 — [sd_guidance.py:353-363](file:///tmp/dmd2/main/sd_guidance.py#L353-L363) 의 주석 처리된 디버그 코드가 이 의도를 보여줌 (cls vs dm grad 비교).

→ **GAN 은 메인이 아니라 보조** — DM gradient 가 분포 방향을 제시하고, GAN 이 진짜 이미지로 fine-tune.

### 3.5 §4.4 — Multi-step generator: consistency-style parameterization

> *왜 이 절을 두나*: 4-step 학습이 단순히 "더 많은 step" 이 아니라 **함수 형태 자체의 결정**. LCMScheduler 와 그대로 호환되는 이유의 근거.

#### (a) Generator 의 함수 시그니처

DMD2 의 multi-step generator 는 **diffusion sampler 가 아님**:

| 기존 diffusion sampler (DDIM, DPM-Solver) | **DMD2 multi-step (consistency-style)** |
|---|---|
| `(x_t, t) → ε_pred` (노이즈 예측) | **`(x_t, t) → x̂₀`** (clean image 직예측) |
| 다음 step: 학습된 reverse ODE 적분 | **다음 step: forward noising 공식 `add_noise(x̂₀, randn, t_next)`** |
| Schedule 학습된 reverse process | Schedule 고정 [999, 749, 499, 249] |

[main/sd_unified_model.py:31-36](file:///tmp/dmd2/main/sd_unified_model.py#L31-L36):
```python
self.denoising_step_list = torch.tensor(
    list(range(self.denoising_timestep-1, 0, -(self.denoising_timestep//self.num_denoising_step))),
    dtype=torch.long, device=accelerator.device
)
# denoising_timestep=1000, num_denoising_step=4
# → [999, 749, 499, 249]
```

#### (b) 다음 step 으로의 이동 — forward noising 만 사용

[main/sd_unified_model.py:157-159](file:///tmp/dmd2/main/sd_unified_model.py#L157-L159):
```python
noisy_image = self.noise_scheduler.add_noise(
    generated_image, torch.randn_like(generated_image), next_timestep
)
```

→ **학습된 reverse step 이 아님**. teacher 의 forward 노이즈 공식 그대로:
```
x_{t_next} = sqrt(α_{t_next}) · x̂₀ + sqrt(1 − α_{t_next}) · ε_new
```

#### (c) 왜 이 형태가 중요한가

1. **LCMScheduler 호환**: DMD2 generator = LCM 의 함수 시그니처. README 의 `pipe.scheduler = LCMScheduler.from_config(...)` 가 그대로 작동.
2. **Stochasticity 회복**: 매 step 마다 새 `randn_like` 가 들어가서 1-step 의 deterministic mapping 문제 (= mode collapse) 해결.
3. **Train/test schedule 동일**: 학습 때 본 t 와 추론 때 보는 t 가 정확히 일치 → score 호출이 항상 분포 안에 머무름.
4. **Generator 가 학습할 게 적음**: reverse process 자체를 학습할 필요 없음. 그냥 "이 noise level 에서 clean 을 한 번에 예측" 만.

### 3.6 §4.5 — Backward simulation: train/inference 입력 불일치 해결

> *왜 이 절을 두나*: §4.4 가 함수 형태 결정이라면 §4.5 는 그 함수의 학습 데이터 분포 결정. 둘이 합쳐져야 4-step 이 작동.

#### (a) Train/Test mismatch 문제

4-step 추론 흐름:
```
  z (pure noise, t=999)
  → student → x̂₀ → re-noise (t=749) → student → x̂₀ → re-noise (t=499)
  → student → x̂₀ → re-noise (t=249) → student → x̂₀ (최종)
```

즉, **2번째, 3번째, 4번째 step 의 student 입력은 "이전 student 출력에 노이즈를 다시 씌운 것"** = student 자신이 만든 분포의 변형.

순진한 학습 방식 (예: Imagine Flash):
- Real_image 에 노이즈를 씌운 입력만 학습 시 사용 → student 가 보는 입력은 항상 real_image 의 noised 버전.
- 추론 시엔 student 가 만든 출력의 noised 버전이 들어옴 → **분포 어긋남** → 누적 오류.

#### (b) Backward simulation 해결책

[main/sd_unified_model.py:127-162](file:///tmp/dmd2/main/sd_unified_model.py#L127-L162) `sample_backward`:

```python
selected_step = torch.randint(low=0, high=self.num_denoising_step, size=(1,))
selected_step = broadcast(selected_step, from_process=0)  # 모든 GPU 동일 k

for constant in self.denoising_step_list[:selected_step]:  # k-1 번
    current_timesteps = ones_like * constant
    generated_noise = self.feedforward_model(noisy_image, current_timesteps, ...)  # no_grad
    generated_image = get_x0_from_noise(noisy_image, generated_noise, alphas, current_timesteps)
    next_timestep = current_timesteps - self.timestep_interval
    noisy_image = self.noise_scheduler.add_noise(generated_image, randn_like, next_timestep)

return generated_image, selected_step_timestep
```

**작동 방식**:
1. Random k ∈ {0,1,2,3} 뽑음 (균등 확률).
2. Pure noise 에서 시작해 **k-1 번** student 를 (no_grad 로) 굴림 → "추론 시점 k 의 입력 분포 샘플" 획득.
3. 이 입력으로 generator forward + DM/GAN loss 계산 (이번엔 grad 흐름).

→ 매 학습 step 마다 **균등 확률로 k=0,1,2,3 중 하나의 추론 시점** 을 만들고, **거기서 한 발짝만** grad 계산.

#### (c) 절약의 의미

4-step trajectory 를 통째로 backprop 하면 메모리 4× → SDXL 1024² 에서 사실상 불가능. Backward sim 은 **forward k 번 + backward 1 step** 으로 train/test gap 해결.

#### (d) Imagine Flash 와의 차이

논문 §4.5 가 명시적으로 비교:

| Imagine Flash (Meta) | **DMD2 backward sim** |
|---|---|
| Student 입력: noisy student-generated | Student 입력: noisy student-generated |
| **Loss: teacher regression** (teacher 의 출력과 MSE) | **Loss: distribution matching + GAN** |
| Teacher 의 실수까지 누적 → 오차 누적 | 분포 매칭이라 절대값 추적 안 함 → 누적 회피 |

→ DMD2 가 Imagine Flash 의 train/test mismatch 처방 (= backward sim) 은 빌려오되, 그 위의 loss 가 분포 매칭이라 누적 오류 안 남는다는 차별성.

### 3.7 §4.6 — 통합 알고리즘 (한 step)

> *왜 이 절을 두나*: 6요소가 매 학습 step 안에서 어떤 순서로 결합되는지 정확히 명시.

[main/train_sd.py:322-408](file:///tmp/dmd2/main/train_sd.py#L322-L408) `train_one_step`:

```python
def train_one_step(self):
    noise = randn(batch, latent_dim)
    COMPUTE_GENERATOR_GRADIENT = (step % 5 == 0)   # §4.2 TTUR

    # ───────────── Generator forward (G_θ, 5번에 1번) ─────────────
    if COMPUTE_GENERATOR_GRADIENT:
        text_embedding = next(self.dataloader)
        # §4.5 Backward simulation → §4.4 Generator (x_k → x̂₀)
        generator_loss_dict, _ = self.model(noise, ..., 
            compute_generator_gradient=True, generator_turn=True)
        
        generator_loss = generator_loss_dict["loss_dm"]                  # §3 DM gradient
                       + 5e-3 * generator_loss_dict["gen_cls_loss"]      # §4.3 GAN-G
        accelerator.backward(generator_loss)
        clip_grad_norm_(feedforward_model.parameters(), max_grad_norm=10)
        optimizer_generator.step()
        optimizer_generator.zero_grad()
        optimizer_guidance.zero_grad()  # G 의 grad 가 D head 로 흘러간 것도 제거
    
    # ───────────── Critic forward (μ_fake, 매 step) ─────────────
    guidance_loss_dict, _ = self.model(noise, ..., 
        guidance_turn=True, guidance_data_dict=...)
    
    guidance_loss = guidance_loss_dict["loss_fake_mean"]                 # μ_fake denoising
                  + 1e-2 * guidance_loss_dict["guidance_cls_loss"]       # §4.3 GAN-D
    accelerator.backward(guidance_loss)
    clip_grad_norm_(guidance_model.parameters(), max_grad_norm=10)
    optimizer_guidance.step()
    optimizer_guidance.zero_grad()
```

**핵심 패턴**: 매 step `Generator (조건부 1×) → Critic (1×)` 의 교차. 5 step 사이클 단위로 보면 `1× G + 5× D`.

### 3.8 DMD gradient 식의 정확한 코드 형태

> *왜 이 절을 두나*: 논문 §3 의 score 정의와 코드의 부호 처리가 미묘하게 다른데, 헷갈리지 말라고 정확히 짚어두기.

[main/sd_guidance.py:235-241](file:///tmp/dmd2/main/sd_guidance.py#L235-L241):
```python
p_real = (latents - pred_real_image)   # ∝ −s_real (ε-prediction 형태)
p_fake = (latents - pred_fake_image)   # ∝ −s_fake
grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1,2,3], keepdim=True)
grad = torch.nan_to_num(grad)
loss = 0.5 * F.mse_loss(original_latents.float(), 
                        (original_latents - grad).detach().float(), 
                        reduction="mean")
```

#### (a) 부호 — score 정의 vs ε-prediction

논문 §3 의 score 함수:
```
s(x_t, t) = ∇_xt log p(x_t) = −(x_t − α_t · μ(x_t,t)) / σ_t²
```

→ score 는 ε-prediction 과 **부호 반대로 비례**. 코드의 `p = latent − pred_x0` 는 ε 방향. 그래서:
```
grad (코드) = p_real − p_fake ∝ −s_real + s_fake = s_fake − s_real (논문)
```
→ **같은 방향**. 헷갈리지 마세요.

#### (b) 정규화 — `/ |p_real|.mean()`

논문에 명시되지 않은 코드의 디테일. timestep 별 score 크기가 천차만별 (high-noise t 일수록 큼) → 그대로 두면 큰 t 의 샘플이 loss 를 지배. **batch 별 적응적 스케일링** 으로 timestep 균형.

→ 4-step 학습 (여러 timestep 동시 학습) 에서 특히 중요.

#### (c) Autograd surrogate

`(x̂₀ − grad).detach()` 가 상수 → 미분하면 `∂L/∂x̂₀ = grad`. DMD 원작과 동일한 트릭.

---

## 4️⃣ 기초 개념 심화 (Foundations Deep-Dive)

> *왜 이 절을 두나*: §3 알고리즘 흐름을 따라가다 보면 자주 생기는 세 가지 기초 의문 — "Critic 이 정확히 뭐지?", "왜 student 출력을 다시 그 두 네트워크에 보내?", "Teacher 가 왜 천장이지?" 를 비유와 코드로 깊이 푼다. §3 이 "무엇을 하는지" 라면 이 절은 "왜 그렇게 해야만 하는지".

### 4.1 Critic (μ_fake) 의 정확한 정의

> *왜 이 절을 두나*: "Critic" 이 GAN/RL/일상어 등 여러 곳에서 다른 뜻으로 쓰여서 DMD critic 의 정확한 위치가 헷갈리기 쉬움.

**한 줄 정의**:
> **DMD critic = "현재 student 가 만들어내는 분포 (p_fake_θ) 의 score 함수를 추정하는 보조 신경망"**.
> 정식 명칭: **fake-score estimator** (가짜 분포 스코어 추정기). 논문 표기: **μ_fake**.

**핵심 4요소**:
1. **무엇을 추정?** — student 분포 p_fake_θ 의 **score** (∇ log p_fake, 분포의 미분 벡터).
2. **어떻게 추정?** — student 가 만든 이미지에 노이즈 씌워 표준 denoising score matching (DSM) 학습. **표준 diffusion 학습 식 그대로**, 단지 학습 데이터가 "student 출력" 일 뿐.
3. **구조?** — Teacher 와 동일한 U-Net 구조. DMD2 코드는 teacher 가중치로 초기화 후 별도 학습.
4. **언제 갱신?** — student 분포가 매 step 변하므로 **critic 도 매 step 같이 갱신**. DMD2 TTUR 은 5× 자주.

#### (a) 다른 분야의 "critic" 과 구분

| 분야 | "Critic" 의 의미 | DMD critic 과 |
|---|---|---|
| **WGAN (Arjovsky 2017)** | "Discriminator" 대신 부르는 이름. 진짜/가짜 확률 (0~1) 이 아니라 **실수값 스코어** (Wasserstein 근사) 출력하는 분류기 | **다름**. WGAN critic 은 분류기, DMD critic 은 score 추정기. |
| **Actor-Critic RL** | Actor 가 행동을 정하고 critic 이 그 상태/행동의 **가치 (V/Q 함수)** 추정 | **부분 유사**. "Actor 도와주는 보조망" 개념 같음. 추정 대상이 가치 X, score O. |
| **GAN discriminator** | 진짜 vs 가짜 분류기. 출력 0~1 확률. | **다름**. DMD critic 은 분류 X. **DMD2 에서는 critic 위에 D head 추가** (§4.1 (d)). |
| **VSD critic (Wang 2023)** | DMD 의 직접 모태. Student score 추정 LoRA 어댑터. | **본질적으로 같음**. DMD/DMD2 가 일반화. |
| **DMD critic (Yin 2024)** | **Student 분포 score 추정 U-Net** | (= 본 정의) |

→ **DMD critic 은 "GAN/RL critic 단어를 빌렸지만 기능은 score 추정기"**.

#### (b) Critic 의 입출력 시그니처

```
입력:  x_t (노이즈 섞인 이미지) + t (timestep) + (text embedding 등 조건)
출력:  ε̂ (그 안에 섞인 노이즈 예측)
파생:  s_fake(x_t, t) = − (x_t − α_t · x̂₀(x_t, t)) / σ_t²
       = "이 점에서 student 분포의 봉우리 방향" 화살표
```

→ Teacher 와 정확히 같은 형식. 단지 학습 데이터만 다름.

#### (c) Critic 학습 — 표준 diffusion 식 그대로

[main/sd_guidance.py:257-310](file:///tmp/dmd2/main/sd_guidance.py#L257-L310) `compute_loss_fake`:
```python
latents = latents.detach()   # G_θ 로 grad 흐름 차단
noise = torch.randn_like(latents)
timesteps = torch.randint(0, num_train_timesteps, [B])
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

fake_noise_pred = predict_noise(fake_unet, noisy_latents, ...)
loss_fake = ((fake_noise_pred - noise) ** 2).mean()
```

→ **DSM 학습**. Hyvärinen 2005 / Vincent 2011 정리: "어떤 분포 p 샘플로 DSM 학습 → 학습된 네트워크 출력에서 자동으로 p 의 score 가 유도됨". Student 가 만든 이미지의 분포 = p_fake_θ → 그 샘플로 DSM → **자동으로 p_fake_θ 의 score 학습**.

#### (d) Critic vs Teacher — 분포 대응물

| | Teacher (s_real) | **Critic μ_fake (s_fake)** |
|---|---|---|
| 학습 데이터 | 진짜 이미지 (LAION 등) | **Student 가 만든 이미지** |
| 학습 목표 | p_real score 추정 | **p_fake_θ score 추정** |
| 학습 시점 | distillation 전에 미리 (frozen) | **distillation 중 함께 갱신** |
| 알려주는 것 | "진짜 분포의 봉우리 방향" | **"student 의 현재 봉우리 방향"** |

→ **Critic = student 분포 전용 teacher**. Teacher 의 분포 대응물이 student 의 분포에도 필요해서 만든 것.

#### (e) DMD2 의 확장 — Critic + Discriminator 합체

DMD2 가 critic 위에 추가한 분류 head ([sd_guidance.py:108-122](file:///tmp/dmd2/main/sd_guidance.py#L108-L122) `cls_pred_branch`):

```
   fake_unet (= critic μ_fake, U-Net backbone)
        │
        ├── bottleneck (1280-dim feature)
        │       │
        │       ├──→ U-Net decoder → ε 예측 (= 기존 score 추정 역할)
        │       │
        │       └──→ cls_pred_branch (5단 conv) → real/fake logit (= DMD2 추가)
        │
        ▼
   같은 backbone 으로 두 출력 동시 생성
```

**왜 합쳤나**:
1. **메모리 절약** — 별도 D 네트워크 불필요.
2. **Feature 재사용** — critic 의 bottleneck 은 student 분포 특화 → 분류에도 자연스럽게 유용.
3. **결합 학습 안정성** — 같은 backbone 위 두 head 가 서로 보강.

→ **DMD2 critic = score estimator + real/fake discriminator 합체**. "Critic" 의 의미가 DMD2 에서 한 번 더 확장됨 (WGAN-style 분류 역할도 일부 포함).

### 4.2 왜 student 출력을 두 네트워크에 다시 보내는가

> *왜 이 절을 두나*: §3.1 전체 학습 흐름의 가장 헷갈리는 부분. "Student 가 만든 결과를 왜 student 자신과 선생에게 다시?" 라는 자연스러운 의문 풀기.

#### (a) 오해 풀기 — 두 네트워크의 역할

| 흔한 오해 | 실제 |
|---|---|
| "Teacher 가 답을 알려주고 student 가 베끼는데, 왜 student 출력을 teacher 에?" | **Teacher/critic 은 "답을 주는 사람" 이 아니라 "어느 점에서 어느 방향으로 가야 하는지 알려주는 측량사"**. |

| 네트워크 | 받는 것 (입력) | 돌려주는 것 (출력) |
|---|---|---|
| Teacher (s_real) | 어떤 점 x_t + 노이즈 강도 t | "**이 점에서** teacher 분포 봉우리 방향" 화살표 |
| Critic (s_fake) | 어떤 점 x_t + 노이즈 강도 t | "**이 점에서** student 분포 봉우리 방향" 화살표 |

→ 둘 다 **"임의의 점 x_t 를 받아 그 점에서의 방향 벡터를 반환하는 함수"**. **답이 아니라 방향만**.
→ 따라서 **"어느 점에서 평가할지"** 가 별도로 정해져야 함. 그 점이 student 의 출력.

#### (b) 왜 하필 student 출력 위치에서 평가하나 — KL 의 정의

KL divergence 의 정의 (말로):
> "**Student 가 만들어내는 점들 위에서**, teacher 분포 log 밀도와 student 분포 log 밀도의 차이를 평균낸 것."

핵심: **"student 가 만들어내는 점들 위에서"**. 평균 잡는 좌표가 **student 의 출력 분포**.

비유:
> 학생이 활을 100발 쏘았다. 학생 분포 = 박힌 100 위치. Teacher 분포 = 정답 표적.
> "두 분포가 얼마나 다른가?" 를 재려면 **학생이 박은 100 위치에서** 비교해야 함. 학생이 안 쏜 곳에서 비교하면 학생 자체에 대한 평가가 안 됨.

따라서:
1. Student 가 x̂₀ 만듦 (= 학생이 활 박음).
2. 그 100 위치에서 teacher 와 critic 화살표 둘 다 뽑음 (= 그 자리에서 두 분포 정보 수집).
3. 두 화살표 차이 = 학습 신호.

→ **Student 출력을 다시 입력으로 보내는 게 아니라, "평가가 일어나는 좌표" 를 student 출력으로 정한 것**.

#### (c) 왜 두 네트워크 모두에게 — "같은 점에서 두 비교"

```
       student 출력 x̂₀ (= 학생이 활 박은 위치)
              │
              │  (forward-diffuse to x_t, random t 노이즈 다시)
              ▼
            x_t
              │
              ├──────────────┬──────────────┐
              ▼              ▼              │
        Teacher (s_real)  Critic (s_fake)   │
              │              │              │
              ▼              ▼              │
        "이 점에서      "이 점에서          │
         teacher 봉우리  student 봉우리      │
         어디?"         어디?"              │
              │              │              │
              └──────┬───────┘              │
                     ▼                       │
            두 화살표의 차이                  │
                     │                       │
                     ▼                       │
                grad 로 G_θ 학습 ◄───────────┘
```

**왜 같은 점에서 둘을 동시에**:
- **Teacher 만 보면**: "여기 → 저쪽 (teacher 봉우리)" → student 의 현재 분포 모양 고려 못함 → 안정성 깨짐.
- **Critic 만 보면**: 자기 자신 방향만 알려줌 → 학습 신호가 안 됨.
- **두 화살표 차이**: "teacher 봉우리 (저쪽) − student 봉우리 (이쪽) = 옮겨가야 할 방향" → 분포 보정.

비유 — 같은 자리에서 두 GPS 켜기:
- GPS-1 (teacher): "현재 위치 → 목적지: 동쪽 5km".
- GPS-2 (critic): "현재 위치 → 학생이 자주 가는 곳: 서쪽 3km".
- 차이: "동쪽 5 − 서쪽 3 = 동쪽 8km 이동" 이 학생의 학습 방향.

**같은 자리에서** 둘 다 켜야 비교 의미. 다른 자리에서 켜면 무의미.

#### (d) 왜 다시 노이즈를 씌우는가

- Score 네트워크 (teacher, critic) 둘 다 **노이즈 섞인 입력만 학습 받음**.
- Clean x̂₀ 직접 넣으면 OOD (Out-Of-Distribution, 학습 안 된 입력) → 헛소리 출력.
- 해결: random t 골라 노이즈 재첨가 → score 네트워크가 익숙한 영역으로 진입.
- **보너스**: random t 매번 다르게 → 모든 노이즈 레벨에서 분포 매칭 검사 (low t = 디테일, high t = 큰 구조).

#### (e) 코드로 확인 — 같은 입력이 두 곳에 갔다 오기

[main/sd_guidance.py:189-233](file:///tmp/dmd2/main/sd_guidance.py#L189-L233):
```python
# 1. student 출력 latents 에 같은 노이즈 추가
noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

# 2. 같은 noisy_latents 를 critic 에게 → fake score
pred_fake_noise = predict_noise(self.fake_unet, noisy_latents, ...)
pred_fake_image = get_x0_from_noise(noisy_latents, pred_fake_noise, ...)

# 3. 같은 noisy_latents 를 teacher 에게 → real score
pred_real_noise = predict_noise(self.real_unet, noisy_latents, ...)
pred_real_image = get_x0_from_noise(noisy_latents, pred_real_noise, ...)

# 4. 같은 점에서 두 화살표 차이
p_real = (latents - pred_real_image)   # teacher 화살표
p_fake = (latents - pred_fake_image)   # critic 화살표
grad = (p_real - p_fake) / |p_real|.mean(...)   # 차이 = 학습 방향
```

→ **같은 `noisy_latents` 가 fake_unet 과 real_unet 두 곳에 들어감**. 두 곳 출력 차이 = grad.

### 4.3 Teacher score 가 왜 천장인가 (GAN loss 의 진짜 동기)

> *왜 이 절을 두나*: §3.4 GAN loss 의 동기 (s_real 부정확) 가 추상적이라 안 와닿음. Teacher 가 **어디서 어떻게 틀리고**, 그 틀린 게 student 에게 **어떻게 옮겨가는지** 단계별로 풀고 — GAN loss 가 이걸 어떻게 우회하는지까지 비유 3개로.

#### (a) 직관 한 줄

> **DMD 의 학습 신호는 "정답" 이 아니라 "teacher 가 생각하는 정답"**. Teacher 도 모델이라 실수가 있는데, student 가 그 실수까지 베껴서 — **student ≤ teacher** 가 천장.

#### (b) s_real 의 정체 — "teacher 의 답안"

- Teacher 는 "noisy 이미지 → 노이즈 ε 예측" 학습됨 (= ε-prediction).
- Score = − (입력 − teacher 예측 깨끗한 이미지) / 노이즈 강도 = **teacher 의 ε 예측에서 자동 유도되는 값**.
- → **s_real = "정답" 이 아니라 "teacher 의 답안"**. Teacher 가 정확한 만큼만 s_real 도 정확.

#### (c) Teacher 가 부정확한 두 이유

**(이유 1) 정보 이론적 한계 — high-noise 일수록 답이 여러 개**:
- Diffusion task: "거의 노이즈인 이미지 (98% 노이즈) 보고 원본 깨끗한 이미지 맞춰라".
- 98% 노이즈면 원본이 강아지/고양이/풍경 중 뭐였는지 **구분 불가** (정보 손실).
- Teacher 의 최선 = **모든 가능 원본의 평균** 예측 (MSE 최소화 정답).
- 결과: **흐릿한 평균 이미지** 출력.

```
[입력: 98% 노이즈 이미지]
        │
        ▼
Teacher 머릿속:
"강아지 30%, 고양이 20%, 풍경 15%, ... 평균을 내면..."
        │
        ▼
[출력: 어디서 본 듯한 흐릿한 평균 이미지]
```

→ 이 평균에서 유도된 **s_real 은 "진짜 분포 모드 중 하나로 가라" 가 아니라 "평균 쪽으로 가라"** 만 알려줌. **다중 모드 (= multi-modal distribution, 여러 봉우리 있는 분포) 를 한 곳으로 평탄화**.

**(이유 2) 모델 자체의 결함**:
- Teacher 도 신경망이라 학습 데이터/구조 한계 존재.
- SDXL 의 손가락 6개 버릇, 특정 prompt 편향 (예: "doctor" → 백인 남성만), high-frequency texture 의 비현실 패턴 등.
- 이 결함들이 **그대로 s_real 에 박힘** — teacher 가 "정답" 이라 가리키는 방향에 이미 결함 포함.

#### (d) 부정확함이 student 에게 옮겨가는 메커니즘

DMD 학습 식 `grad = s_fake − s_real` 의 의미:
> "Student 야, **s_real 쪽으로 (= teacher 답 쪽으로)** 가라, 단 너 자신이 자주 가는 곳 (= s_fake) 은 피하면서".

→ **목적지가 s_real**. Teacher 가 흐릿한 평균을 가리키면 student 도 평균으로 향함. Teacher 가 손가락 6개를 정답이라 우기면 student 도 손가락 6개로 향함.
→ 학습 끝 = **teacher 와 똑같은 결함의 더 빠른 버전**.

> **요약: s_real 이 student 의 GPS 목적지인데, GPS 자체가 잘못된 좌표를 알려주면 student 가 거기 정확히 도착해도 잘못된 곳**.

#### (e) 비유 3개

**(1) 카피북 학생**:
화가 (teacher) 의 그림을 따라 그리는 학생. 화가에게 손떨림이 있으면 학생의 그림에도 손떨림이 복제됨. **학생 < 화가 가 절대 안 됨**. GAN loss 우회: 실제 풍경 사진도 같이 보여줌 → 화가 흐림을 사진과 비교 → "여기는 사진이 더 또렷하네 → 더 또렷하게 그리자" → **학생 > 화가** 가능.

**(2) 번역가 학생**:
Teacher 가 "she rolled her eyes" → "그녀는 눈알을 굴렸다" 라는 어색 번역. 학생도 그대로 베낌. GAN loss 우회: **자연스러운 한국어** (real data) 같이 보여줌 → "눈을 흘기다" 표현 학습 → 번역가 능가.

**(3) 흐린 망원경 사진**:
진짜 별을 흐린 망원경으로 찍음. 사진만 보고 별 모양 배우면 망원경 흐림까지 학습. GAN loss 우회: 별의 직접 측정 데이터도 같이 → 망원경 흐림 우회.

#### (f) High-noise 영역이 특히 문제인 이유

- DMD 학습 시 timestep 은 균등 random sampling (t ∈ [0.02, 0.98] × 1000).
- **low t (1% 노이즈)**: teacher 정확 → 좋은 신호.
- **high t (90% 노이즈)**: teacher 가 흐릿한 평균 예측 → 부정확한 신호.
- 학습 중 절반 정도가 high-noise → **평균 쪽으로 student 가 끌려감**.
- 4-step student 의 1번째 step (t=999) 이 가장 중요한데 **이 영역에서 teacher 가 가장 부정확** — 아이러니.
- GAN loss 가 이 영역 보강 (이미지 공간 = t=0 비교 이므로 high-noise 영역 부정확함 영향 없음).

#### (g) GAN loss 가 천장을 뚫는 작동

| | DMD (teacher 만 봄) | DMD2 (teacher + real image) |
|---|---|---|
| 학습 신호 1 | s_real 화살표 (teacher 정답) | 같음 |
| 학습 신호 2 | (없음) | **D 가 real image 비교한 결과** |
| 정답 출처 | teacher 출력 | **teacher + 진짜 데이터 두 곳** |
| 천장 | teacher 와 같은 품질 | **teacher 부족 부분을 real image 가 보강** |

**구체적**:
- Real image 의 손가락 = 5개. Teacher 가 6개 그려도 D 가 진짜 이미지를 학습하니 "6개는 가짜다" 판정.
- D 신호: "fake_image (손가락 6개) 는 가짜처럼 보인다" → student 가 손가락 5개 방향으로 학습.
- 결과: **teacher 의 손가락 6개 버릇 우회**.

#### (h) 결과 — 천장 뚫은 직접 증거

ImageNet 64×64 1-step:

| 모델 | 사용 신호 | FID ↓ |
|---|---|---|
| Teacher EDM | (reference) | 1.79 |
| DMD | s_real 만 | **2.62** ← teacher 천장 |
| **DMD2** | s_real + real-data GAN | **1.28** ← **teacher 추월** ✨ |

DMD2 의 1.28 < teacher 1.79 (FID 낮을수록 좋음). Teacher 1-step 출력만으로 학습했다면 절대 못 도달할 숫자 — **천장 뚫은 직접 증거**.

---

## 5️⃣ 코드 ↔ 논문 매핑 (한 표)

| 논문 구성요소 | 코드 위치 | 핵심 한 줄 |
|---|---|---|
| **DM gradient** | [sd_guidance.py:168-255](file:///tmp/dmd2/main/sd_guidance.py#L168-L255) | `grad = (p_real - p_fake) / \|p_real\|.mean(...)` |
| **μ_fake critic 학습 (denoising)** | [sd_guidance.py:257-310](file:///tmp/dmd2/main/sd_guidance.py#L257-L310) | `loss_fake = (fake_noise_pred − noise)²` |
| **§4.3 GAN — Generator side** | [sd_guidance.py:312-324](file:///tmp/dmd2/main/sd_guidance.py#L312-L324) | `softplus(−pred_realism_on_fake)` |
| **§4.3 GAN — Discriminator side** | [sd_guidance.py:369-395](file:///tmp/dmd2/main/sd_guidance.py#L369-L395) | `softplus(logit_fake) + softplus(−logit_real)` |
| **Diffusion-GAN noise injection** | [sd_guidance.py:147-151](file:///tmp/dmd2/main/sd_guidance.py#L147-L151) | `image = scheduler.add_noise(image, randn, t∼U(0,T))` |
| **D head = fake_unet bottleneck** | [sd_guidance.py:108-122](file:///tmp/dmd2/main/sd_guidance.py#L108-L122) | conv stack on 1280-dim feature |
| **§4.2 TTUR (5:1 ratio)** | [train_sd.py:330](file:///tmp/dmd2/main/train_sd.py#L330) | `COMPUTE_GENERATOR_GRADIENT = step % 5 == 0` |
| **§4.5 Backward simulation** | [sd_unified_model.py:127-162](file:///tmp/dmd2/main/sd_unified_model.py#L127-L162) | `for c in denoising_step_list[:selected_step]: student(...)` |
| **§4.4 Multi-step timesteps** | [sd_unified_model.py:31-36](file:///tmp/dmd2/main/sd_unified_model.py#L31-L36) | `list(range(999, 0, -250))` = `[999,749,499,249]` |
| **§4.4 Forward noising for next step** | [sd_unified_model.py:157-159](file:///tmp/dmd2/main/sd_unified_model.py#L157-L159) | `scheduler.add_noise(x̂₀, randn, t_next)` |
| **§4.6 Training loop (G/D 교차)** | [train_sd.py:322-408](file:///tmp/dmd2/main/train_sd.py#L322-L408) | `train_one_step` |
| **1-step conditioning_timestep=399** | [scripts/sdxl_1step.sh](file:///tmp/dmd2/experiments/sdxl/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode.sh) | `--conditioning_timestep 399` |
| **ODE pretraining (1-step 전용)** | [scripts/ode_pretrain.sh](file:///tmp/dmd2/experiments/sdxl/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399.sh) | 10k step warmup, lr=1e-5 |
| **LoRA 모드** | [sd_unified_model.py:44-69](file:///tmp/dmd2/main/sd_unified_model.py#L44-L69) | 13개 layer 에 LoRA adapter |
| **Inference 1-step (NFE=1)** | [demo/text_to_image_sdxl.py](file:///tmp/dmd2/demo/text_to_image_sdxl.py) | `num_inference_steps=1, timesteps=[399]` |
| **Inference 4-step (NFE=4)** | [demo/text_to_image_sdxl.py](file:///tmp/dmd2/demo/text_to_image_sdxl.py) | `num_inference_steps=4, timesteps=[999,749,499,249]` |

---

## 6️⃣ 실험 요약

### 5.1 ImageNet 64×64 (class-conditional)

| 모델 | step | FID ↓ | 비고 |
|---|---|---|---|
| EDM (teacher, Karras 2022) | 79 | 1.79 | 기존 SOTA |
| DMD (CVPR 2024) | 1 | 2.62 | teacher 갭 +0.83 |
| **DMD2** | **1** | **1.28** | **teacher 추월** ✨ |

### 5.2 COCO 2014 zero-shot (text-to-image)

| 모델 | step | FID ↓ | 비고 |
|---|---|---|---|
| SDv1.5 (teacher, 50 step + CFG) | 100 | 8.59 | |
| DMD (CVPR 2024) | 1 | 11.49 | |
| **DMD2** | **4** | **8.35** | teacher 추월 |

### 5.3 SDXL 1024² (text-to-image)

| 모델 | step | CLIP-FID ↓ | 비고 |
|---|---|---|---|
| SDXL teacher (50 step + CFG) | 100 | 19.36 | reference |
| SDXL-Turbo (Stability AI) | 1 | 24.57 | ADD = adversarial diffusion distillation |
| SDXL-Lightning | 4 | 24.46 | progressive adv distillation |
| **DMD2** | **4** | **19.32** | **few-step 중 최고**, teacher 와 동급 |

### 5.4 추론 비용

- DMD2 1-step: NFE 1 (CFG 없음) → **teacher (NFE 100) 대비 500× 절감**.
- A100 1장 SDXL 1024² 4-step: ~0.9 sec/image (teacher: ~50 sec).

### 5.5 Ablation (논문 Table 2, 3)

| 설정 | ImageNet FID | 해석 |
|---|---|---|
| Full DMD2 | 1.28 | baseline |
| − TTUR (1:1 update) | 발산 | TTUR 필수 |
| − GAN loss | 1.86 | DMD 와 비슷한 수준 (천장 못 뚫음) |
| − Backward simulation (4-step) | +0.5~1.0 FID | train/test gap 영향 |

→ **TTUR + GAN + Backward sim 셋이 다 critical**. 어느 하나 빠지면 결과 크게 후퇴.

---

## 7️⃣ 관련 연구 비교

### 6.1 같은 시기 다른 few-step distillation 방법론

> *왜 이 절을 두나*: 시장에 비슷한 시기에 SDXL few-step 모델이 여럿 나옴 (2023-2024). DMD2 의 차별점 이해.

| 방법 | 접근 | step | 특징 | 한계 |
|---|---|---|---|---|
| **SDXL-Turbo** (Stability AI) | ADD (Adversarial Diffusion Distillation) | 1 | LPIPS + GAN loss | 분포 매칭 X, mode collapse |
| **SDXL-Lightning** (ByteDance) | Progressive adv distillation | 4 | step 절반씩 줄임 | 누적 오차 |
| **LCM** (Tsinghua) | Consistency distillation | 4 | f(x_t,t)=f(x_t',t') 일관성 | 품질 < 8.35 |
| **Hyper-SD** (ByteDance) | Trajectory segmented distill | 1-8 | 여러 step 통합 | 복잡 |
| **Imagine Flash** (Meta) | Backward distillation | 3 | student 생성 입력 사용 | teacher regression 누적 오류 |
| **DMD2** | DM + TTUR + GAN + Backward sim | 1, 4 | **분포 매칭으로 누적 오류 회피** | LoRA 호환 ✓ |

### 6.2 후속 영향

- **Z-Image** (2511.22699, 2025-11): Decoupled DMD + DMDR 사용 — DMD2 의 분포 매칭 패러다임을 RL-distill 결합으로 확장. [[paper-z-image]] 참고.
- **HiDream-O1** (2026-05): DMD2 4-step distillation 기반 8B unified model. [[paper-hidream-o1-image]] 참고.
- **SDXL/SD3 community LoRAs**: DMD2 4-step LoRA 가 사실상 표준 (lora_scale 조절로 강도 변경).

---

## 8️⃣ Q&A

### Q1. DMD 의 L_reg 제거가 왜 GAN loss 한 줄로 해결되는 게 아닌가?

**다른 문제다**. GAN loss 는 **천장을 뚫는 역할** (teacher 한계 극복), 안정화는 **TTUR 이 주범**. 논문 ablation: "TTUR 없이 GAN 만" → 학습 발산. "GAN 없이 TTUR 만" → 안정적이지만 teacher 만큼만 잘함.

### Q2. 왜 1-step 보다 4-step 이 더 좋은가? (COCO/SDXL 기준)

ImageNet 64² 작은 모델에선 1-step 도 충분 (FID 1.28). 하지만 SDXL 1024² 큰 모델 + COCO 다양성에선:
- **ODE trajectory 곡률**: noise → 1024² image 경로가 곡선. 1-step 직선 = 큰 오차. 4-step = piecewise linear 근사 → 곡선 더 잘 따라감.
- **Diversity**: 1-step deterministic mapping → mode collapse 위험. 4-step 매 step `randn_like` 주입 → stochastic.
- **CFG 호환**: 4-step 은 (이론적으론) CFG 적용 가능, 1-step 은 불가.

### Q3. `diffusion_gan` 플래그가 왜 필요한가?

깨끗한 이미지를 D 에 그대로 넣으면 high-frequency texture 만 보고 분류 → mode collapse. 작은 노이즈로 흐릿하게 만들면 **structural-level 판별** 강제 → 더 안정. Diffusion-GAN (Wang 2023) 의 표준 트릭.

### Q4. `real_guidance_scale=8` 은 무엇? 왜 큰 값?

Teacher 에 적용하는 CFG 강도. **Teacher 의 s_real 을 더 강한 prompt-follower 방향으로** 만들어 student 가 그 방향을 흡수. → distilled student 는 **CFG 없이도** (guidance_scale=0) 강한 prompt 추종력. 추론 시 NFE 절반 (2 → 1 forward per step).

### Q5. LoRA 버전과 full UNet 버전의 차이?

| | Full UNet | LoRA |
|---|---|---|
| 학습 비용 | SDXL 전체 (~2.6B) 학습 | rank-64 어댑터 13개 모듈만 |
| 메모리 | high | ~1/4 |
| 배포 파일 크기 | ~5GB | ~200MB |
| 커뮤니티 SDXL 모델에 얹기 | × | **○** (`pipe.load_lora_weights` + `pipe.fuse_lora`) |
| 품질 | 약간 좋음 | 거의 동등 |

→ 실제 배포는 **LoRA 4-step** 이 거의 표준.

### Q6. ODE pretraining 단계가 왜 따로 있는가?

[ode_pretrain script](file:///tmp/dmd2/experiments/sdxl/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399.sh).

**1-step 모델 (cond=399) 만 ODE pretraining 사용**:
- Pure noise → 1-step 으로 이미지 만드는 건 cold-start 너무 심함 (학습 초반 무작위 출력).
- DMD 식 paired data (z → teacher ODE → x₀) 로 **10k step warmup** 한 뒤 본 DMD2 학습.

**4-step 은 backward sim 으로 분포를 자연스럽게 따라감 → pretraining 불필요**:
- [4-step script](file:///tmp/dmd2/experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh) 에는 `--generator_ckpt_path` 없음 (= scratch 가능).
- 이름의 "scratch" 도 이 뜻.

### Q7. 1-step conditioning_timestep=399 의 의미는?

1-step student 의 입력 timestep. **pure noise t=999 가 아니라 t=399** 인 이유:
- t=999 (순수 노이즈) 에선 student 가 받는 정보가 너무 적음 → 학습 어려움.
- t=399 (약간 정보 있는 mid-noise) → student 가 "약간의 정보를 clean image 로 풀어내는" 더 쉬운 매핑.
- ODE pretraining 시점도 cond=399 로 맞춰져 있음.

**추론 시도** [demo/text_to_image_sdxl.py](file:///tmp/dmd2/demo/text_to_image_sdxl.py):
```python
image = pipe(prompt, num_inference_steps=1, guidance_scale=0, timesteps=[399]).images[0]
```

### Q8. DMD2 의 `(p_real − p_fake) / |p_real|.mean(...)` 정규화는 논문 어디?

논문에는 명시 안 됨. **코드의 디테일**. timestep 별 score 크기 균형 — 4-step 학습 (여러 t 동시) 에서 특히 중요.

### Q9. fake_unet 이 D 역할도 겸하는데 학습이 안 꼬이나?

꼬이지 않게 설계:
- μ_fake (denoising) loss 와 D loss (classification) **둘 다 fake_unet 을 학습**.
- D head (`cls_pred_branch`) 만 따로, 본체 fake_unet 의 학습 신호로 통합.
- G_θ 가 D 를 backprop 할 때 D head 까지 grad 가 가지만 그건 `optimizer_guidance.zero_grad()` 로 즉시 폐기 ([train_sd.py:381](file:///tmp/dmd2/main/train_sd.py#L381)).

### Q10. DMD2 의 multi-step 이 왜 DDIM/DPM-Solver 가 아닌가?

논문 §4.4 의 핵심 결정. **Consistency-style** (`(x_t,t)→x̂₀` 직예측) 이라 학습된 reverse scheduler 가 필요 없음. 대신:
1. Step 간 이동은 forward noising 공식 (`add_noise`) 으로 결정론적/단순화.
2. LCMScheduler 와 호환 → 추론 시 community pipeline 그대로 작동.
3. Train/test schedule 동일 강제 → 일반화 안정성.

DDIM/DPM 식이면 학습된 reverse process 까지 distillation 해야 해서 복잡. Consistency 식이 더 단순.

### Q11. CFG distillation 이 DMD2 의 별도 method 가 아닌가?

논문 §4 에 별도 소절 없음 — **함수적으로는 통합된 형태**:
- 학습 시 `real_guidance_scale=8` 로 teacher 에 CFG 적용 → s_real 이 강한 guidance 방향.
- Student 가 이 s_real 을 흡수 → student 의 가중치에 CFG 효과가 융합.
- 추론 시 `guidance_scale=0` → student 자체로 prompt 추종.

즉 **CFG distillation 은 별도 method 가 아니라 §3 의 자연스러운 부산물**.

### Q12. DMD2 가 다양성 (mode coverage) 도 좋은가?

- 1-step 은 여전히 mode collapse 경향 있음 (deterministic mapping).
- 4-step 은 매 step 새 `randn_like` → stochastic → mode coverage 회복.
- GAN loss 가 real_image 와 분포 매칭 → real 의 다양성 일부 흡수.
- 하지만 teacher 50-step 의 full diversity 에는 못 미침 (특히 long-tail).

---

## 9️⃣ DMD ↔ DMD2 한눈 비교 (배포 결정용)

> *왜 이 절을 두나*: PAPER_DMD.md 와 같이 볼 때 "내 프로젝트엔 뭐 써야 하나" 의 결정 기준.

| 결정 기준 | DMD 적합 | **DMD2 적합** |
|---|---|---|
| 학습 데이터 | Paired (z, x₀) 미리 만들 GPU 시간 있음 | **Real image 만 있어도 OK** |
| Teacher 품질 | 신뢰 가능 | **Teacher 가 불완전해도 가능** (real data 가 보정) |
| Step 수 | 1-step 만 필요 | **1, 2, 4-step 다 가능** |
| 해상도 | ~512² 까지 | **1024²+ (SDXL)** |
| LoRA 호환 | 거의 불가 | **표준 지원, ComfyUI/Diffusers 그대로** |
| 학습 안정성 | L_reg 안전벨트 필요 | TTUR + GAN 으로 안정 |
| 코드 복잡도 | 단순 (1 generator + 1 critic) | **3중 (G + critic-as-D + backward sim)** |
| 학습 시간 | 짧음 (paired data 만들어두면) | 길지만 paired data 비용 없음 |
| 다양성 | 1-step deterministic → 낮음 | 4-step stochastic → 회복 |

→ 대부분의 실제 배포는 **DMD2 4-step LoRA**. SDXL-Lightning, Hyper-SD 도 거의 같은 패러다임으로 수렴.

---

## 🔟 학습 하이퍼파라미터 (4-step SDXL 기준)

> *왜 이 절을 두나*: 재현하려고 코드 볼 때 참고용. [training script](file:///tmp/dmd2/experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh) 발췌.

| 항목 | 값 | 의미 |
|---|---|---|
| `generator_lr` | 5e-7 | G_θ 학습률 |
| `guidance_lr` | 5e-7 | μ_fake 학습률 |
| `batch_size` | 2 per GPU × 8 GPU × 8 node = 128 | global batch |
| `resolution` | 1024 (pixel) / 128 (latent) | SDXL 1024² |
| `real_guidance_scale` | 8 | Teacher CFG (distillation 흡수용) |
| `fake_guidance_scale` | 1.0 | Fake critic 은 CFG 없음 |
| `dfake_gen_update_ratio` | **5** | **TTUR** 비율 |
| `max_step_percent` | 0.98 | DM loss timestep 상한 (edge 회피) |
| `min_step_percent` | 0.02 (기본) | DM loss timestep 하한 |
| `dm_loss_weight` | 1 | DM gradient 가중치 |
| `gen_cls_loss_weight` | **5e-3** | Generator GAN loss 가중치 |
| `guidance_cls_loss_weight` | **1e-2** | Discriminator GAN loss 가중치 |
| `diffusion_gan` | True | D 입력에 노이즈 주입 |
| `diffusion_gan_max_timestep` | 1000 | 노이즈 주입 timestep 범위 |
| `num_denoising_step` | 4 | 4-step generator |
| `denoising_timestep` | 1000 | timestep 분할 기준 |
| `backward_simulation` | True | §4.5 활성화 |
| `max_grad_norm` | 10 | gradient clipping |
| `cls_on_clean_image` | True | GAN loss 활성화 |
| `gen_cls_loss` | True | Generator 가 D 도 학습 |
| `use_fp16` | True | bfloat16 autocast |
| `fsdp` | True | Fully Sharded Data Parallel |

---

## 🔚 한 줄 요약 (전체)

**DMD2 (Yin 2024 NeurIPS) 는 DMD 의 회귀 손실 (L_reg = 정답 쌍 MSE) 안전벨트를 떼어내고 그 자리에 (1) TTUR 로 critic 을 5× 빠르게 학습시켜 stale fake-score 발산을 막고, (2) μ_fake 의 bottleneck 에 GAN head 를 붙여 real image 와 직접 분포 매칭으로 teacher score 의 부정확함 천장을 뚫고, (3) Consistency-style multi-step parameterization (`(x_t,t)→x̂₀` 직예측 + forward noising 공식) 과 (4) Random-k backward simulation 으로 train/inference 입력 불일치를 해결한 — 6요소 통합 distillation 프레임. 결과로 ImageNet 1-step FID 1.28 (teacher 추월), SDXL 4-step 1024² 가 ComfyUI/Diffusers 의 사실상 표준, 추론 비용 500× 절감을 달성.**

---

## 🔗 관련 메모리 링크

- [[paper-dmd]] — DMD 모태 논문 (PAPER_DMD.md). DMD2 의 기본 용어와 KL gradient surrogate 의 자세한 유도는 여기.
- [[paper-z-image]] — Decoupled DMD + DMDR 응용 사례
- [[paper-hidream-o1-image]] — DMD2 4-step 기반 unified model
- [[paper-lumina-next]] — Flag-DiT 안정화 (distillation 과 별개 축)
- [[paper-min-snr]] — diffusion 학습 가속의 다른 축 (loss weighting)
