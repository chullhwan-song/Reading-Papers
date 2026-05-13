# PAPER: MV-Split DiT — Mean Mode Screaming 진단과 1000층 Diffusion Transformer

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Mean Mode Screaming: Mean–Variance Split Residuals for 1000-Layer Diffusion Transformers |
| **저자** | Pengqi Lu |
| **공개일** | 2026-05-07 |
| **분량** | 43쪽 (본문 9쪽 + 부록) |
| **분야** | cs.LG, cs.CV |
| **논문 링크** | https://arxiv.org/abs/2605.06169 |
| **PDF** | https://arxiv.org/pdf/2605.06169 |
| **HTML** | https://arxiv.org/html/2605.06169v1 |
| **공식 코드** | https://github.com/erwold/mv-split |
| **추론 가중치** | `model.pt` (코드 저장소 제공) |
| **사용 외부 모델** | FLUX.2 VAE, Qwen3-0.6B text encoder |

---

## 📖 주요 용어 사전 (Glossary)

읽기 전 알아두면 좋은 핵심 용어들입니다.

### 아키텍처/구조
- **DiT (Diffusion Transformer)**: CNN 기반 U-Net 대신 Transformer로 디퓨전 모델을 만든 구조. Stable Diffusion 3, FLUX 계열이 사용.
- **MMDiT (MultiModal DiT)**: 이미지·텍스트 토큰을 한 시퀀스에 합쳐 joint attention 하는 DiT 변형. FLUX.2 표준.
- **Residual Connection (잔차 연결)**: `X_{l+1} = X_l + F(X_l)`. 매 층 출력에 이전 입력을 더해 그래디언트 전파를 돕는 구조.
- **Pre-LN / Post-LN**: 정규화를 잔차 더하기 전/후에 두는 두 가지 패턴.
- **RMSNorm**: `x / sqrt(mean(x²))`. LayerNorm과 달리 **평균을 빼지 않는** 정규화. LLaMA, FLUX, Qwen에서 표준.
- **LayerNorm**: 평균을 빼고 표준편차로 나누는 정규화. 평균 제거 효과가 있음.
- **QK-Norm**: Attention의 Q, K에 정규화를 추가해 안정화하는 기법.
- **RoPE (Rotary Position Embedding)**: 위치 정보를 회전 행렬로 인코딩하는 방식.
- **SwiGLU**: `Swish(xW₁) ⊙ (xW₂)` 형태의 게이트형 FFN.

### 핵심 개념
- **MMS (Mean Mode Screaming)**: 깊은 DiT 학습 중 잔차의 **평균(mean) 성분**이 갑자기 폭주하며 모델이 붕괴하는 현상. 이 논문이 명명·진단.
- **Mean component / Centered component**: 시퀀스 평균 부분과, 거기서 평균을 뺀 편차 부분. 논문은 이 둘을 분리해 다룸.
- **MV-Split**: Mean-Variance Split. 잔차를 평균/편차로 나눠 각각 다른 게이트(α, β)로 더하는 본 논문의 핵심 기법.
- **Leaky Trunk-Mean Replacement**: `J(Z) = (1−α)·J(X) + α·J(F)`. 잔차 흐름의 평균을 EMA처럼 천천히 흘려보내는 메커니즘.
- **Token Homogenization (표현 붕괴)**: 모든 토큰이 거의 같은 벡터로 수렴하는 상태. cosine similarity → 1.
- **Rank-1 Update**: `ΔW = u·vᵀ` 형태의 가중치 업데이트. 가중치를 한 방향으로만 밀어 다양성을 죽임.
- **Softmax Null Space**: `softmax(z + c·𝟙) = softmax(z)`. Softmax가 mean shift를 무시하는 성질.
- **Sequence-mean Projection (J)**: 시퀀스 토큰들의 평균을 추출하는 선형 사상.
- **Centered Projection (P = I − J)**: 평균을 뺀 편차만 남기는 사상.

### 비교 기법
- **LayerScale**: 각 잔차 출력에 채널별 학습 가능한 작은 스칼라(예: 1e-4)를 곱해 안정화. CaiT/ViT 계열.
- **ReZero**: 단 하나의 스칼라 게이트(0 초기화)로 잔차를 통제.
- **DeepNorm**: Microsoft의 1000층 Transformer 안정화 기법. `α·x + Sublayer(x)` 형태.

### 평가 지표
- **FID-50K (Fréchet Inception Distance)**: 생성 이미지 품질 지표. 낮을수록 좋음.
- **IS (Inception Score)**: 다양성·품질 종합 지표. 높을수록 좋음.
- **ρ_T**: per-layer mean/centered energy 비율. MMS 진단용.
- **G_mean / G_ctr**: writer-gradient의 mean-coherent / centered 성분 크기.

### 학습
- **Flow Matching**: 디퓨전의 변종. ODE solver로 latent에서 이미지를 복원. 35 step 추론.
- **Triton kernel**: GPU에서 fused 연산을 짜는 DSL.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> 깊이 1000층 Diffusion Transformer를 안정 학습하려면, 잔차 연결의 **평균 성분(mean)**과 **편차 성분(centered)**을 분리해서 다뤄야 한다.

**핵심 문제 발견**: 깊은 DiT를 학습하면 어느 순간 **모든 토큰이 평균 방향으로 무너지는 collapse**가 일어난다. 저자는 이를 **Mean Mode Screaming(MMS)** 으로 명명하고, 수학적으로 진단했다.

**해결책**: 잔차 업데이트를 **mean projection (J)** 과 **centered projection (P)** 으로 분해하고, 각각에 **독립적인 채널별 게이트 (α, β)** 를 적용. mean 경로는 EMA처럼 "leaky" 교체.

**검증**: 첫 번째로 1000층 DiT를 안정 학습 가능함을 입증.

---

## 2️⃣ 핵심 기여 (Contributions)

1. **MMS 현상 발견 및 명명**: 깊은 DiT의 collapse 메커니즘을 처음으로 체계화.
2. **그래디언트의 정확한 분해**: writer 가중치의 그래디언트를 mean-coherent와 centered로 정확히 분리:
   ```
   ∇_W ℒ = T·δ̄·ȳᵀ + Σ_t δ̃_t·ỹ_tᵀ
           (ΔW_μ)    (ΔW_c)
   ```
3. **MV-Split Residual 제안**: mean과 centered를 분리해서 각각 게이트하는 잔차 구조.
4. **Leaky Trunk-Mean Replacement**: mean 경로의 폭주를 막는 EMA형 메커니즘.
5. **1000층 안정 학습 시연**: 사상 처음 1000-layer DiT를 안정적으로 학습.

---

## 3️⃣ 주요 알고리즘: MV-Split Residual

### 3-1. 수식

**기존 (Pre-LN baseline)**
```
X_{l+1} = RMSNorm(X_l + F_l)
```

**MV-Split (논문 식 7)**
```
Z_l     = X_l + β ⊙ (P · F_l) + α ⊙ J · (F_l − X_l)
X_{l+1} = RMSNorm(Z_l)
```
- `J`: 시퀀스 평균 projection
- `P = I − J`: centered projection
- `α, β ∈ ℝ^D`: per-channel 학습 게이트

**평균 항만 풀어쓰면** (leaky replacement)
```
J(Z_l) = (1 − α) ⊙ J(X_l) + α ⊙ J(F_l)
```
- α = 0 → 평균 유지 (새 평균 무시)
- α = 1 → 평균 완전 교체
- 0 < α < 1 → **EMA처럼 새는 댐퍼**

### 3-2. 코드 매핑 (`kernels/fused_mvsplit_rmsnorm.py`)

```python
# x: trunk (잔차 흐름),  u: update (Attn / FFN 출력)
mu_x = mean(x, dim=-2)
mu_u = mean(u, dim=-2)
y    = x + beta * (u - mu_u) + alpha * (mu_u - mu_x)
out  = RMSNorm(y)   # weight = 1, 학습 안 됨
```

### 3-3. 초기화 (안정 학습의 비밀)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `init_alpha` | **0.0** | mean 경로 처음엔 완전히 닫음 |
| `init_beta` (DiTBlock) | **0.03** | centered 경로 매우 작게 시작 (warm-up) |
| RMSNorm weight | 1.0 (buffer, 학습 X) | 정규화 가중치 고정 |

### 3-4. DiTBlock 구조 (`dit.py`)
```
Attention (RoPE → QK-Norm) → FusedMVSplitNorm1 → SwiGLU FFN → FusedMVSplitNorm1
```

### 3-5. 모델 스케일
- **1000-layer 구성**: 50 stages × 10 blocks × 2 = 1000
- hidden 1024, 8 heads × 128 dims
- 기본 코드 config: depth 100, width 1280, heads 10 (스몰)

### 3-6. 코드에서 논문 기여의 위치 (⭐ vs ⚪)

전체 코드 중 **논문 고유 기여는 약 5%**. 나머지는 표준 부품(ViT patchify, Attention+RoPE+QK-Norm, SwiGLU, Qwen3, FLUX.2 VAE, Flow Matching).

**⭐ 논문 기여 (단 두 곳)**
- `DiTBlock` 잔차 더하기 자리 ([dit.py:407–424](https://github.com/erwold/mv-split/blob/main/dit.py)) — 표준 `RMSNorm(x + op_out)` 대신 `FusedMVSplitNorm1(R, op_out, L_img)` 호출 (구조는 → 3-4 참조)
- `FusedMVSplitNorm1` 모듈 ([kernels/fused_mvsplit_rmsnorm.py](https://github.com/erwold/mv-split/blob/main/kernels/fused_mvsplit_rmsnorm.py)) — MVSplit + RMSNorm Triton fused, **segment-aware mean** (이미지/텍스트 모달리티별 별도 mean), 2-pass backward (수식은 → 3-1, 3-2, 초기화는 → 3-3 참조)

**⚪ 표준 부품 (논문 외 인프라)**
patch_embed/unpatchify, RoPE(이미지 2D / 텍스트 identity), QK-Norm, GQA Attention, SwiGLU FFN, Qwen3-0.6B 텍스트 인코더, FLUX.2 VAE, sample.py의 Flow Matching Euler 35-step + CFG.

**대략 라인 수 비율**
| 영역 | 라인 | 분류 |
|---|---|---|
| FusedMVSplitNorm1 (Triton + module) | ~400 | ⭐ |
| DiTBlock 잔차 교체 부분 | ~6 | ⭐ |
| Attention, RoPE, SwiGLU, QK-Norm | ~500 | ⚪ |
| patch embed, unpatchify, final proj | ~50 | ⚪ |
| sample.py (CFG, Euler, loop) | ~330 | ⚪ |
| text_encoder, vae 래퍼 | ~200 | ⚪ |

### 3-7. MMS 폭주 메커니즘 ↔ 코드 요소 매핑

Q1의 5가지 폭주 원인을 코드의 어떤 부분이 차단하는지.

| 폭주 원인 (→ Q1 참조) | 차단 코드 요소 | 차단 메커니즘 |
|---|---|---|
| ① 잔차에 mean 누적 (1000배) | `init_beta = 0.03` | 더하는 양 자체를 작게 시작 |
| ② RMSNorm이 mean 보존 | `α·(mu_u − mu_x)` 항 분리 | RMSNorm이 못 닦는 mean을 수동 제어 |
| ③ Softmax null space (centered 신호 차단) | `β·(u − mu_u)` 항 분리 | centered 학습 신호 별도 보존 |
| ④ T-배율 ΔW_μ 폭주 | `init_alpha = 0.0` | mean gradient 통로를 처음엔 닫음 |
| ⑤ 자기강화 루프 | leaky `(1−α)·J(X) + α·J(F)` | mean을 EMA로 천천히 흐름 |

→ **단일 모듈(FusedMVSplitNorm1) 하나가 5개 폭주를 모두 차단**.

---

## 4️⃣ 실험 요약

| 깊이 | 비교 메소드 | 역할 |
|---|---|---|
| **400층** | Post-Norm baseline / LayerScale / MV-Split | **메소드 비교** (80k step) |
| **1000층** | MV-Split 단독 | **안정성 시연** (50k step까지) |

**데이터**: ImageNet-2012 latents (FLUX.2 VAE 인코딩) + Qwen3-0.6B text embed.  1000층은 약 50k 큐레이션 이미지로 post-training.

**핵심 결과**:
- Baseline은 400층에서 발산
- MV-Split이 LayerScale 대비 **20k–30k step 빠른 수렴**
- 1000층에서 MV-Split만으로 안정 학습 입증 (수렴 완료가 아닌 feasibility 시연)

---

# 💬 Q&A — 자주 묻는 질문 정리

## Q1. 왜 1000층에서 평균 쪽으로 와르르 무너지는가?

다섯 가지 메커니즘이 동시에 맞물려 폭주합니다.

### ① 잔차 자체가 1000번 누적
```
X_L = X_0 + Σ F_l   (l = 0..999)
```
각 F_l의 mean 편향이 ε이면 → 1000ε 선형 누적. 분산은 √1000 ≈ 31배만 늘어 **mean/centered 비율이 폭증**.

### ② RMSNorm은 평균을 안 닦아낸다
| | mean 제거 | variance 정규화 |
|---|---|---|
| LayerNorm | ✅ | ✅ |
| **RMSNorm** | ❌ | ✅ |
LLaMA, FLUX, Qwen이 모두 RMSNorm → mean이 시스템의 "자유 방향"이 됨.

### ③ Softmax가 mean shift를 못 본다
`softmax(z + c·𝟙) = softmax(z)`. Q/K가 mean 방향으로 쏠려도 attention 출력 무변화 → backprop에서 **mean 줄이라는 신호가 안 옴**. 일방통행 통로.

### ④ 그래디언트의 T-배율
```
ΔW_μ = T · δ̄ · ȳᵀ
```
시퀀스 길이 `T`가 곱해짐. 고해상도 이미지(긴 시퀀스) × 1000층 → 양쪽으로 증폭.

### ⑤ 자기강화 루프
mean이 커짐 → 토큰 균질화 → centered 신호 죽음(ỹ_t→0) → 학습 신호가 전부 mean으로 몰림 → mean이 더 커짐 → ... = **Screaming**

> **비유**: mean은 "브레이크 없는 가속 페달". 쌓이고(①), 안 닦이고(②), 줄이는 법을 모르고(③), 시퀀스가 길수록 커지고(④), 한번 시작되면 멈출 수 없다(⑤).

---

## Q2. "옵티마이저가 평균 방향만 키운다"가 실제로 어떻게 망가지는가?

`ΔW_μ = T·δ̄·ȳᵀ`는 **rank-1 외적 업데이트**. 가중치가 단 하나의 방향으로만 학습됨.

### 결과 1: 가중치가 "획일화 머신"이 됨
어떤 토큰 x를 넣어도 `W·x ≈ c·ȳ` (거의 같은 출력). 1024차원짜리 가중치가 1차원처럼 움직임.

### 결과 2: Representation Collapse
```
초기:   → ↗ ← ↘ ↑ ↖   (다양함)
폭주 후: → → → → → →   (다 같아짐, cos sim → 1)
```
DiT에선 **모든 이미지 패치가 같은 값** → 결과 이미지는 **회색/단색 노이즈**.

### 결과 3: Attention이 무력화
Q, K가 비슷해지면 `softmax(Q·Kᵀ) ≈ 1/T로 균등` → attention이 **그냥 평균 내기**로 퇴화. 게다가 Softmax null space 때문에 **회복 불가능**.

### 결과 4: Centered gradient가 자기 자신을 죽인다
토큰들이 같아지면 `ỹ_t = y_t − ȳ ≈ 0` → `ΔW_c ≈ 0`. **centered 학습 신호 소멸** → 남은 신호는 mean뿐 → 더 폭주.

### 결과 5: 실제 학습 로그
| 시점 | 증상 |
|---|---|
| 정상 | Loss 천천히 감소 |
| MMS 직전 | 잔차 norm 폭증, ρ_T 가파른 상승 |
| **MMS 발생** | Gradient norm spike → NaN |
| 이후 | 샘플 이미지가 회색 노이즈, attention map 균일 |

### 결과 6: Adam의 자원 분배
```
정상:  ΔW_μ 30% + ΔW_c 70%
폭주:  ΔW_μ 99% + ΔW_c  1%
```
한번 momentum이 mean 방향으로 쌓이면 **돌이킬 수 없음**.

---

## Q3. 1000층 레이어를 가진 공개 모델이 있나?

세 가지만 존재합니다. 모두 **학술적 시연용**.

| 모델 | 연도 | 분야 | 비고 |
|---|---|---|---|
| **ResNet-1001** | 2016 | CNN | He et al., CIFAR. residual의 효시 |
| **DeepNet** | 2022 | Transformer (NMT/LM) | Microsoft, DeepNorm. [코드](https://github.com/microsoft/unilm/tree/master/deepnet) |
| **MVSplit-DiT** | 2026 | Diffusion Transformer | 본 논문, [코드+가중치](https://github.com/erwold/mv-split) |

### 상용 모델은 모두 얕다
| 모델 | 깊이 |
|---|---|
| GPT-3 175B | 96 |
| LLaMA-3 70B | 80 |
| DeepSeek V3 | 61 |
| Qwen3 235B | ~96 |
| FLUX.1 | ~57 |
| SD3 | ~24 |

### 왜 상용은 1000층 안 쓰나
1. 추론 latency가 깊이에 **선형 비례**
2. 학습 안정성 위험 (MV-Split 같은 기법이 필수)
3. 같은 FLOPs면 **MoE / width 확장**이 성능 더 좋음
4. Activation memory 부담

---

## Q4. 본 연구는 1000층을 실제로 쌓아 학습했나?

**둘 다 했지만 역할이 달랐습니다.**

| 항목 | 400층 | 1000층 |
|---|---|---|
| 목적 | **메소드 비교** | **안정성 시연** |
| 비교 대상 | Baseline / LayerScale / MV-Split | MV-Split 단독 |
| 학습 step | 80k | 50k까지 |
| 결과 측정 | 정량 (FID 등) | 정성 demonstration |
| 수렴 완료? | — | ❌ (feasibility만) |

> 즉 "**1000층이 진짜 죽지 않는다**"만 보여주고, **메소드 우열은 400층에서 가렸다**. DeepNet(2022)과 동일한 패턴 — 200층에서 SOTA 비교, 1000층은 시연.

---

## Q5. FLUX.2 [klein] 4B는 몇 층인가?

[config.json](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/transformer/config.json) 기준:

```json
{
  "num_layers": 5,         ← MMDiT (joint) 블록
  "num_single_layers": 20, ← Single-stream DiT 블록
  "num_attention_heads": 24,
  "attention_head_dim": 128,
  "mlp_ratio": 3.0
}
```

- **총 깊이 = 5 + 20 = 25층**
- hidden_size = 24 × 128 = **3072**
- MLP dim = 3072 × 3.0 = **9216**

### MMDiT 구조 (SD3/FLUX 표준)
- **MM-DiT 블록 (5)**: 이미지+텍스트 joint attention
- **Single 블록 (20)**: 이미지만 처리 (텍스트 통과)

### 깊이 비교
| 모델 | 깊이 | 파라미터 |
|---|---|---|
| FLUX.1-dev | 57 (19+38) | 12B |
| **FLUX.2 klein 4B** | **25 (5+20)** | 4B |
| MVSplit-DiT | 1000 | 미공개 |

klein 4B는 "**얕고 넓고 빠른**" 디자인 — FLUX.1의 절반 이하 깊이.

---

## Q6. 이런 작은 레이어를 지닌 모델에도 MV-Split이 효과가 있나?

**거의 효과 없습니다.** MV-Split의 효과는 **깊이 의존적**.

### 왜 효과 없나
| 폭주 원인 | 25층에서 |
|---|---|
| 잔차 mean 누적 | 25배만 누적 → 무시 가능 |
| RMSNorm mean 보존 | 동일하나 누적량 적음 |
| Softmax null space | 동일하나 임계점 미도달 |
| 자기강화 루프 | 25번으론 폭주 시작 못 함 |

> **비유**: 1층 단독주택에 1000층 빌딩급 내진설계는 불필요.

### 깊이별 효과 곡선 (개념)
```
효과 │                            ●●●
     │                       ●●●
     │                  ●●●
     │             ●●●
     │ ━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━
     │ ✕  ✕
     └────────────────────────────────
       25  50 100 200 400  600 1000  층
```
논문이 **400층부터 비교**한 이유 — 그 이하엔 baseline도 안 터져 차이가 안 보임.

### 깊이별 권장 안정화 기법
| 깊이 | 권장 |
|---|---|
| ~30층 (klein 4B) | RMSNorm + QK-Norm으로 충분 |
| 30~100층 | + LayerScale |
| 100~400층 | + DeepNorm / ReZero |
| 400~1000층 | + **MV-Split** |

### 25층에 적용하면?
- **해는 없음**: α=0 초기화로 처음엔 identity
- **이득도 없음**: FID 개선 측정 불가, 약간의 연산 오버헤드만 발생
- "무해하지만 무효한 영양제"

---

## 🎯 한 줄 요약 (전체)

> **MV-Split는 1000층 DiT만 걸리는 'Mean Mode Screaming' 병의 처방약**: 잔차의 평균/편차를 분리해 각각 게이트(α, β)함으로써 mean의 폭주를 차단한다. **400층에서 메소드 우열을 검증**하고 **1000층은 안정성을 시연**했다. **klein 4B(25층) 같은 얕은 모델엔 처방할 병이 없으므로 효과 없음**. 상용 영역(30~120층)에선 RMSNorm + QK-Norm + LayerScale 정도로 충분하고, MV-Split은 깊이 400을 넘는 학술적 영역의 무기다.

---

## 📂 관련 메모리 링크

- 메모리: `paper_mv_split_dit.md` — 논문/코드 상세 분석
- 메모리: `very_deep_models_landscape.md` — 1000+층 공개 모델 풍경
