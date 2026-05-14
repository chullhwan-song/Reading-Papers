# PAPER: Z-Image — 한 줄로 흐르는 6B DiT로 20B–80B를 따라잡은 효율 풀스택 설계

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Z-Image: An Efficient Image Generation Foundation Model |
| **저자** | Tongyi MAI Team (Alibaba) |
| **공개일** | 2025-12-01 (arXiv v1) / 2026-01-27 (Z-Image 본체 공개) |
| **분야** | 이미지 생성 모델 / cs.CV, cs.LG |
| **논문 링크** | https://arxiv.org/abs/2511.22699 |
| **PDF** | https://arxiv.org/pdf/2511.22699 |
| **공식 사이트** | https://tongyi-mai.github.io/Z-Image-blog/ |
| **공식 코드** | https://github.com/Tongyi-MAI/Z-Image |
| **가중치** | HF: `Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo` |
| **사용 외부 모델** | Qwen3-4B (텍스트 이해용 LLM), Flux VAE (이미지 압축기), SigLIP 2 (편집 모드의 참조 이미지 인코더) |
| **공개 범위** | 추론 코드만 공개 (학습 코드·데이터·보상 모델은 비공개) |

---

## 📖 주요 용어 사전 (Glossary)

읽기 전 알아두면 좋은 용어의 *정의*만 모았습니다. 자세한 메커니즘·수치는 본문에서 한 번만 다룹니다.

### 아키텍처 관련

- **DiT (Diffusion Transformer)**: 이미지 생성을 위한 트랜스포머 구조 (Diffusion Transformer). CNN 기반 U-Net 대신 Transformer 블록으로 노이즈를 점점 걷어내는 방식. Stable Diffusion 3, FLUX 계열이 표준.
- **S3-DiT (Scalable Single-Stream DiT)**: 이 논문의 백본. 텍스트와 이미지를 *하나의 시퀀스로 합쳐* 통째로 처리하는 DiT 변형.
- **Single-Stream vs Dual-Stream**: 두 흐름을 한 줄기로 합치느냐, 두 줄기로 따로 처리하느냐의 차이.
  - **Single-Stream (단일 스트림)**: 텍스트와 이미지 토큰을 한 시퀀스로 이어붙여서 (concat) 하나의 백본이 처리. → 같은 파라미터가 양쪽 모달리티를 다 학습.
  - **Dual-Stream (이중 스트림, SD3/FLUX 방식)**: 텍스트용 가중치와 이미지용 가중치를 *분리*. → 파라미터가 둘로 쪼개짐.
- **MM-DiT (MultiModal DiT)**: 텍스트와 이미지를 동시에 받는 DiT의 통칭 (MultiModal Diffusion Transformer). SD3·FLUX·Z-Image가 모두 변형 사용.
- **3D Unified RoPE**: 위치 인코딩 (Rotary Position Embedding)을 (t축, h축, w축) 세 갈래로 나누어 적용. 텍스트 토큰은 t축, 이미지 토큰은 h·w축만 사용 → 한 attention 안에서 모달리티별로 다른 위치 의미를 줌.
- **adaLN (low-rank)**: 시간 정보 (timestep)를 매 층에 주입하는 방식 (adaptive Layer Normalization). 본 논문은 256 차원으로 압축한 시간 벡터를 30층이 *공유*하고, 층마다 작은 up-projection만 따로 둠 → 파라미터 절약.
- **Sandwich-Norm**: 각 블록의 입력과 출력 *양쪽 모두*에 RMSNorm을 두는 정규화 패턴. 깊은 학습 안정화용.
- **QK-Norm**: Attention의 Q(질의), K(키)에 추가로 RMSNorm을 적용해 안정화하는 기법.
- **RMSNorm**: 평균을 빼지 않고 표준편차로만 나누는 정규화 (Root Mean Square Norm). LayerNorm과 달리 평균 정보를 보존.
- **SwiGLU**: 게이트형 FFN 블록의 한 형태 — 식으로 쓰면 `w2(SiLU(w1·x) ⊙ w3·x)`. LLaMA·Qwen에서 표준.

### 데이터·캡션 관련

- **Active Data Curation (능동 데이터 큐레이션)**: 모델이 못 푸는 케이스를 *모델 스스로 발굴* → 그 데이터를 보충 → 재학습하는 닫힌 루프 (closed loop).
- **Data Profiling Engine**: 이미지의 다양한 속성을 자동 측정하는 엔진 — 해상도, 압축 흔적, 정보량, 미감 점수, AI 생성 여부, 의미 태그 등.
- **Cross-modal Vector Engine**: 텍스트-이미지 임베딩을 공유 공간에 두고 *수십억 단위*로 중복 제거하는 시스템 (vector + cross-modal search).
- **World Knowledge Topological Graph**: 위키피디아 엔티티를 기반으로 만든 개념의 위상 그래프 (개념 간 부모-자식 관계). 학습 시 *드문 개념*에 가중치를 더 주는 데 사용.
- **BM25 rarity score**: 텍스트 검색 점수 (BM25)를 활용해 *얼마나 드문 개념인지* 측정하는 가중치.
- **Z-Captioner**: 자체 캡션 생성기 (이미지 → 텍스트 모델). 이미지 1장당 *5가지 형식*의 캡션 + OCR 결과 + 세계지식까지 동시 생성.

### 추론·증류(distillation) 관련

- **Flow Matching**: 디퓨전의 한 변종. 노이즈와 진짜 이미지를 *t 비율로 섞은 점*을 만들고, 그 점에서 진짜 방향으로 향하는 *속도 벡터*를 학습 (flow matching). 추론은 한 단계씩 적분해 가는 방식 (ODE Euler = Ordinary Differential Equation을 Euler 방법으로 푸는 가장 단순한 방식).
- **NFE (Number of Function Evaluations)**: 추론할 때 모델을 몇 번 통과시키는지 (Turbo 모델은 8번).
- **CFG (Classifier-Free Guidance)**: 조건부 결과와 무조건부 결과를 섞어서 prompt를 강화하는 표준 기법.
- **CFG truncation**: 후반 step에서 CFG를 꺼 버려서 (truncation) 모델 호출 횟수를 절반으로 줄이는 트릭.
- **CFG normalization**: CFG 합성 결과의 크기 (norm)가 너무 커지면 잘라내서 (clamp) 채도 폭주를 막는 트릭.
- **Distillation (증류)**: 큰 모델(teacher)의 결과를 작은 모델(student)이 흉내내도록 학습 → 적은 step으로 같은 품질 달성.
- **DMD (Distribution Matching Distillation)**: teacher가 가진 *확률 분포* 자체를 student가 따라가도록 학습하는 증류 방식.
- **Decoupled DMD**: 본 논문이 제안. DMD의 효과를 두 메커니즘으로 *분리*해서 (decouple) 각각 다른 노이즈 스케줄을 적용 — 세부 그림과 색이 망가지는 문제를 해결.
- **DMDR (DMD meets RL)**: 본 논문이 제안. 증류와 강화학습 (Reinforcement Learning)을 *한 손실 함수에 통합* → DM 항이 자연스러운 안전망 역할.
- **RLHF (Reinforcement Learning from Human Feedback)**: 사람의 선호를 보상으로 사용해 강화학습으로 모델을 정렬하는 방식.
- **DPO (Direct Preference Optimization)**: 좋은 샘플 (chosen)과 나쁜 샘플 (rejected) 짝을 비교해서 직접 학습하는 오프라인 (offline) 방식.
- **GRPO (Group Relative Policy Optimization)**: 같은 질문에서 만든 여러 샘플을 *그룹 내에서 상대 비교*해 학습하는 온라인 (online) 강화학습 방식.

### 모델 변종

- **Z-Image-Omni-Base**: 사전학습만 끝낸 *날것* 모델. 가장 자유로운 출발점이라 추가 학습 (fine-tuning)에 가장 유리.
- **Z-Image**: 위 모델에 SFT를 더한 50-step·CFG 사용 본체.
- **Z-Image-Turbo**: 위 모델에 증류 + RLHF까지 추가한 8-step·CFG 미사용 가속판. 추가 학습 가능성 N/A.
- **Z-Image-Edit**: 편집 작업 (image-to-image)에 특화한 변종.
- **SFT (Supervised Fine-Tuning)**: 작고 매우 좋은 데이터로 다시 한 번 정밀 학습하는 후처리 단계.

### 평가 관련

- **Elo (Artificial Analysis Arena)**: 두 모델의 결과를 사람이 짝지어 비교한 후 (pairwise) 산출하는 순위 점수.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> "무조건 크게 키우자 (scale-at-all-costs)"는 풍토와 "상용 모델 결과물 베껴서 학습한다 (synthetic data distillation)"는 우회로를 둘 다 거부하고, **6B 파라미터 + 314K H800 GPU·시간 + 인터넷에서 직접 모은 real-world 데이터만**으로 SOTA에 도달.

**저자가 본 문제**:
- 오픈소스가 점점 비대해지는 길로 도망감 — Qwen-Image는 20B, FLUX.2는 32B, HunyuanImage는 80B.
- 그게 부담스러운 학계는 *상용 모델 결과로 학습용 데이터를 만드는* (distill synthetic data) 우회로를 택함.
- **둘 다 지속 불가능**한 방향.

**해결책 — 네 기둥**:
1. 데이터를 *능동적으로 골라내는* 인프라 (→ 3-4).
2. 같은 가중치가 텍스트·이미지를 *동시에* 학습하는 단일 흐름 구조 (→ 3-1).
3. 단계별로 컴퓨트를 *몰빵하는* 커리큘럼 (→ 3-6).
4. 8번 만에 그림을 그리는 *증류 가속* (→ 3-2, 3-3).

**검증**: Artificial Analysis Arena에서 Elo 1161로 전체 8위, **오픈소스 1위**. 자세한 표는 → 4.

---

## 2️⃣ 핵심 기여 (Contributions)

1. **S3-DiT 아키텍처** — 단일 흐름 (Single-Stream)으로 설계한 6B 멀티모달 DiT (→ 3-1).
2. **4-모듈 통합 데이터 인프라** — 프로파일링 + 벡터 엔진 + 지식 그래프 + 능동 큐레이션이 *닫힌 루프*로 동작 (→ 3-4).
3. **Decoupled DMD** — 기존 증류 기법인 DMD가 가진 두 메커니즘 (CA와 DM)을 분리해 따로 제어 (→ 3-2).
4. **DMDR** — 증류 안에 강화학습 (RL)을 자연스럽게 끼워넣어 *보상 해킹 (reward hacking)*을 자체 차단 (→ 3-2).
5. **2-단계 RLHF** — 객관 차원은 오프라인 DPO, 주관 차원은 온라인 GRPO로 따로 정렬 (→ 3-3).
6. **PE-aware SFT** — 외부 VLM (시각-언어 모델)을 *고정한 채* Z-Image만 PE-enhanced caption으로 SFT → 추가 LLM 학습 비용 0.
7. **비용 효율 실증** — 총 학습 비용 314K H800·시간 ≈ $628K. 경쟁작 대비 한 자릿수 % (→ 4-1).

---

## 3️⃣ 주요 알고리즘

### 3-1. S3-DiT 아키텍처

**핵심 수치표**

| 항목 | 값 |
|---|---|
| 총 파라미터 수 | **6.15B** |
| 백본 층 수 (backbone layers) | 30 (단일 흐름) |
| Refiner 층 수 | 2 (이미지 전용) + 2 (텍스트 전용) |
| 은닉 차원 (hidden dim) | 3840 |
| Attention 헤드 수 | 32 (head_dim = 128) |
| FFN 중간 차원 | 10240 = `int(dim/3 × 8)` (SwiGLU 비율) |
| QK-Norm / Sandwich-Norm | 둘 다 사용 |
| 패치 크기 (patch) | 공간 2×2, 시간 1 |
| RoPE 축 분배 | (32, 48, 48) ← head_dim 128 = 32(t축) + 48(h축) + 48(w축) |
| RoPE θ | 256 |
| adaLN 공유 차원 | 256 (전 층 공유) |
| 지원 최대 해상도 | 약 1k–1.5k |

**입력 토큰화 — "그림과 글자를 같은 벡터 줄에 세운다"**

- 이미지: VAE 잠재 표현 (latent, 16채널, 8배 축소) → 2×2 패치로 자르기 → 4096개 토큰, 각각 3840 차원.
- 텍스트: Qwen3-4B의 *끝에서 두 번째 층 출력* (`hidden_states[-2]`, 2560 차원)을 Linear로 3840 차원에 맞춤.
- 둘 다 32의 배수가 되도록 채워 넣기 (padding) — `SEQ_MULTI_OF=32`. → `torch.compile`이 정적 shape를 좋아하기 때문.

**Forward 순서** (텐서 흐름의 전체 그림은 → 3-5)
```
1. 시간 → sinusoidal(256) → MLP → adaLN 벡터 (256d)
2. 패치화 (patchify) : 이미지 / 캡션을 각각 토큰화 + 3D RoPE 좌표 계산
3. Noise Refiner ×2   : 이미지만, 시간 정보 주입 ON
4. Context Refiner ×2 : 텍스트만, 시간 정보 주입 OFF
5. Backbone ×30       : unified = [이미지; 텍스트] 한 시퀀스로 합쳐서 처리
   각 블록 = RMSNorm·scale → Attention(RoPE+QK-Norm) → tanh-게이트 잔차 (residual)
             RMSNorm·scale → SwiGLU FFN              → tanh-게이트 잔차
6. Final Layer (이미지 토큰만): LayerNorm × (1+scale) → Linear
7. unpatchify (패치 복원) → 속도 벡터 (velocity), shape (B, 16, 1, 128, 128)
```

### 3-2. Decoupled DMD + DMDR (몇 스텝 증류, Few-Step Distillation)

**저자의 핵심 관찰**: 기존 DMD가 디테일을 뭉개고 색을 어그러뜨리는 *진짜 이유*는, 사실 두 개의 별개 메커니즘을 *하나로 묶어 다뤘기 때문*.

| 메커니즘 | 역할 |
|---|---|
| **CFG-Augmentation (CA)** | **주 엔진** — 학생 모델이 적은 스텝으로 그림을 그리는 능력을 끌어올림. 그동안 literature가 *간과*해 옴. |
| **Distribution Matching (DM)** | **정규화 항 (regularizer)** — 분포를 안정화시키고 아티팩트를 제거. 본체로 잘못 인식되어 옴. |

**Decoupled DMD**: CA와 DM에 *서로 다른 노이즈 재주입 스케줄 (re-noising schedule)*을 적용 → 세부와 색이 살아남. 결과적으로 8번만 추론한 student가 100번 추론한 teacher를 *능가*.

**DMDR**: 증류 손실 안에 강화학습 항을 합침. 풀어 쓰면 — *DM 항이 보상 해킹을 막는 안전망 역할을 자연스럽게 한다*는 의미.
$$\mathcal{L}_{\text{DMDR}} = \mathcal{L}_{\text{DM}} + \lambda \cdot \mathcal{L}_{\text{RL}}$$

- DM 항이 *자연 anchor* 역할 → 보상 해킹 (reward hacking) 자동 억제.
- 일반 RLHF가 사용하는 reference-KL 제약을, 더 강한 distillation 제약 (DM)이 대체.

> 이 절의 DMDR과 *후속 RLHF*의 차이는 Q3 참조.

### 3-3. 2-단계 RLHF (증류 *후*에 진행)

**3-축 Reward Model (보상 모델)**: 한 모델이 세 가지 점수를 동시에 산출 — instruction-follow / AIGC-percep / aesthetic.

- instruction은 prompt를 **(주체, 속성, 행동, 공간, 스타일) 5단계로 분해** → 사람이 *만족 못 한 요소를 클릭* → 충족 비율을 reward로 변환.

**Stage 1 — 오프라인 (offline) DPO** (객관 차원)
- VLM이 chosen/rejected 짝을 *대량 생성* → 사람이 빠르게 검증 (hybrid pipeline).
- 학습 난이도를 조절하는 커리큘럼 (curriculum) 적용: 쉬운 prompt → 어려운 prompt, 차이가 큰 쌍 → 차이가 미묘한 쌍.
- 텍스트 렌더링·객체 카운트 같은 *명확한 정답이 있는* 항목에 집중.

**Stage 2 — 온라인 (online) GRPO** (주관 차원)
- 여러 신호 (사실성·미감·instruction following)를 *합성 어드밴티지 (composite advantage)*로 합쳐 사용.
- 단일 reward를 쓸 때보다 보상 해킹이 어려움.

### 3-4. 데이터 인프라와 학습셋 구성

**원천**: 알리바바 내부의 *저작권 보유* 데이터 풀. *합성 distillation 데이터는 의도적으로 거부*.

**원천 풀 (raw pool) 규모** — 직접적 숫자는 없고 간접 단서만:
- 본문에 "*billions of embeddings (수십억 단위의 임베딩)*" 언급 → 10⁹ 단위.
- 중복 제거 (dedup) 처리 속도: "10억 건 / 8시간 / 8 H800".
- → **원천 풀은 수십억 장 추정. 최종 학습용 데이터 (curated final dataset) 크기는 비공개.**

**4-모듈 closed-loop 인프라**

| 모듈 | 역할을 풀어쓰면 |
|---|---|
| **Data Profiling Engine** | 이미지의 *지문*과 *품질 지표*를 자동으로 뽑아낸다. 비슷한 이미지를 짧은 해시값으로 알아내는 시각 지문 (pHash = perceptual hash)로 중복 제거, 압축 흔적 (이상 파일 크기 / 실제 파일 크기 비율), 정보량 (테두리 픽셀의 분산 variance-of-border, 픽셀당 바이트 수 BPP = bytes-per-pixel), 자체 학습한 품질 모델, AI 생성 여부 분류기, 의미 태그·NSFW·중국 문화 컨셉, CN-CLIP 정렬 점수까지. |
| **Cross-modal Vector Engine** | 그림과 글자를 *같은 좌표계*에 두고, 가까운 것끼리 묶어 (graph community detection) 중복을 한 번에 정리. GPU k-NN으로 가속, "1B 건 / 8시간 / 8 H800" 처리. 실패한 케이스를 역검색해 *데이터 결함*을 진단. |
| **World Knowledge Topological Graph** | 위키피디아의 개념 그래프 (PageRank·VLM으로 가지치기) + 캡션 임베딩의 계층 클러스터링. BM25 rarity 점수로 *드문 개념을 가중치 ↑*. |
| **Active Curation Engine** | 모델이 *못 그리는 개념* (예: 松鼠鳜鱼 = 음식 이름인데 "다람쥐+생선"으로 잘못 그림)을 발굴 → 해당 데이터를 *능동적으로* 보충 → 재학습. |

**데이터 4갈래 — "이미지 한 장으로 학습 신호 6개 만들기"**

| 갈래 | 내용 |
|---|---|
| **[A] 텍스트→이미지 (T2I) 단일 이미지** | + Z-Captioner의 *5종 캡션*: tag / long (OCR + 세계지식 포함, 객관적 어조) / medium / short / 사용자처럼 짧게 쓴 prompt (simulated user prompt). + 웹 페이지에 원래 달려 있던 이미지 설명문 (alt-text = HTML alt attribute)을 작은 확률로 섞기. + OCR은 별도 CoT (단계적 추론, Chain-of-Thought) 방식으로 추출해 캡션에 *원어 그대로* 삽입. |
| **[B] 이미지→이미지 (I2I) 페어** (Omni 단계, T2I:I2I = 4:1) | ① Mixed Editing + Graphical Representation: 1장의 원본 → 전문 모델로 N가지 편집 → N² 페어 합성. ② 비디오 프레임에서 자연스럽게 묶인 쌍 (CN-CLIP 유사도로 필터). ③ 텍스트만 다르게 만든 합성 페어 (텍스트 편집 학습용). |
| **[C] SFT 큐레이션 셋** | 매우 적지만 매우 좋은 데이터. 분포를 *좁혀* 고품질로 모는 게 목적. Tagged Resampling으로 드문 개념 보존, Model Merging으로 여러 SFT 모델의 가중치를 평균 (linear interpolation)해 Pareto-optimal 한 결과 도출. |
| **[D] RLHF preference 페어** | VLM이 짝 생성 → 사람이 검증. 쉬운→어려운, 차이 큼→차이 미묘 순으로 커리큘럼 진행. |

### 3-5. 추론 코드 흐름 (공개 코드 기준)

**Level 0 — 한눈에 보기: "텍스트 prompt 1개 → 이미지 1장"**

```
  "Young Chinese woman in red Hanfu..."   ← 문자열 1개
            │
            ├─[A] Qwen3-4B  (512 tokens × 2560 hidden)
            │
            ↓
    cap_feats (S, 2560) ──────────────────────────┐
                                                   │
                                                   │
  torch.randn → latents (B, 16, 128, 128) fp32    │
                              │                    │
                              │                    │
                              ↓                    ↓
              ┌──── 8 steps (Turbo) ──────────────────────┐
              │                                            │
              │   latent + t  ┐                            │
              │               │                            │
              │  ┌────────────┴──────────┬─────────────┐   │
              │  │  S3-DiT (6.15B)        │             │   │
              │  │   ↑ Noise Refiner ×2   │             │   │
              │  │   ↑ Context Refiner ×2 │ ← cap_feats │   │
              │  │   ↑ Backbone ×30       │             │   │
              │  │   ↑ Final Layer        │             │   │
              │  └─────────────┬──────────┴─────────────┘   │
              │                ↓                            │
              │           velocity (B, 16, 128, 128)        │
              │                │                            │
              │                ↓ CFG trunc / norm           │
              │                ↓                            │
              │      scheduler.step (Flow Match Euler)      │
              │      latent ← latent + dt · v               │
              │                                             │
              └────────────────┬────────────────────────────┘
                               │
                               ↓
                   latents (B, 16, 128, 128) fp32
                               │
                               ↓ Flux VAE.decode (fp32)
                               │
                   image (B, 3, 1024, 1024) ∈ [-1, 1]
                               │
                               ↓ /2+0.5 → clamp → uint8
                               │
                   PIL.Image (1024×1024 RGB)
```

**Level 1 — `generate()` 함수 세부 단계** ([pipeline.py:66](src/zimage/pipeline.py#L66))

1. Qwen3에 *추론 모드 (reasoning mode)*를 켜고 (`enable_thinking=True`) 채팅 템플릿 적용. → reasoning hidden state 활용.
2. 토크나이저로 토큰화 (`max_length=512`) → `text_encoder(...).hidden_states[-2]` ← 마지막 두 번째 층 출력.
3. 잠재 표현 초기화 (latent init) — 정규분포 노이즈: `randn(B, 16, H//8//2·2, W//8//2·2)` (32비트 부동소수).
4. 해상도에 따라 *시간 스케줄*을 동적으로 계산. 즉 큰 그림일수록 더 큰 mu를 사용해 노이즈 강도를 조정 — `calculate_shift(seq_len)` → mu → `scheduler.set_timesteps(num=8, mu=mu)`.
5. 노이즈 제거 루프 (denoising loop):
   - **CFG truncation** 결정: 현재 step의 노이즈 비율이 임계점을 넘으면 conditional만 사용.
   - DiT를 한 번 통과 → 속도 벡터 (velocity) 예측.
   - **CFG 합성 + CFG normalization** (크기 폭주 방지).
   - Euler 한 발짝 — 즉 `x_{t-1} = x_t + dt × velocity`라는 *한 줄 ODE 적분*.
6. VAE 디코딩으로 latent → RGB. `[-1, 1]` 범위를 `[0, 1]`로 옮기고 8비트로 변환 → `PIL.Image`.

**CFG truncation/normalization 트릭** ([pipeline.py:227-268](src/zimage/pipeline.py#L227-L268))
- *truncation*: 후반 step에서 CFG OFF → 모델 호출 횟수 절반 절약 + 색번짐 ↓.
- *normalization*: 합성 결과의 크기 (norm)가 일정 한도를 넘으면 잘라 줌 → 채도 폭주 (saturation) 방지.

**Level 2 — DiT 내부 텐서 흐름**: 수식·층 수·forward 순서는 → 3-1 참조.

**Level 3 — Attention 백엔드 분배** ([attention.py](src/utils/attention.py))
- 7가지 백엔드 (backend) 지원: FlashAttention 2/3, FlashAttention Varlen 2/3 (가변 길이 변형, variable-length), MPS-Flash(애플 실리콘용), PyTorch 표준 attention 함수 (SDPA = Scaled Dot Product Attention), Native-Math.
- 파이썬 enum + 데코레이터 (`@register_backend`)로 등록.
- H800에서 `_flash_3` + `torch.compile` 조합 → sub-second 추론.

### 3-6. 학습 커리큘럼

```
사전학습 (Pre-training)
├─ 저해상도 256² 단계  ····· 전체 컴퓨트의 47% 집중
│  └─ T2I만, 기초 지식 (foundational knowledge — 중국어 텍스트 렌더링 등) 습득
└─ Omni 사전학습 단계 ····· 45%
   ├─ 임의 해상도 학습 (arbitrary resolution)
   ├─ T2I + I2I 조인트 학습 (joint training, 4:1)
   └─ 다단계·이중언어 캡션 (multi-level bilingual caption, → 3-4 [A])

후처리 학습 (Post-training)  ··········· 8%
├─ SFT       (분포 좁히기 + Tagged Resampling + Model Merging)
├─ Distillation (Decoupled DMD → +DMDR, → 3-2)
└─ RLHF      (DPO → GRPO, → 3-3)
```

**SFT의 세 축 — 한 줄로 풀어쓰면**
- *Distribution Narrowing (분포 좁히기)*: 잡음 섞인 웹 분포에서 *고품질만 사는 동네*로 모델을 이사시키는 단계.
- *Tagged Resampling (태그 기반 재샘플링)*: BM25로 드문 개념을 *위로 끌어올려* 잊히지 않게 (catastrophic forgetting 방지) 보존.
- *Model Merging (모델 평균화)*: 여러 SFT 변종 (사실성·스타일·instruction 각각 편향)을 *가중치 공간에서 평균* → 단일 모델이면서도 다재능.

**시스템 효율 트릭** — 데이터 병렬 (Data Parallelism) + 가중치를 GPU들에 쪼개서 보관하는 분산 학습 (FSDP2 = Fully Sharded Data Parallel v2) + 모든 층의 중간값을 저장하지 않고 필요할 때 다시 계산 (gradient checkpoint) + 학습 그래프를 컴파일해 가속 (`torch.compile`) + 시퀀스 길이 인지 동적 배치 (긴 시퀀스엔 작은 배치, 짧은 시퀀스엔 큰 배치).

### 3-7. 코드에서 논문 기여의 위치

공개 코드는 *추론 (inference)* 전용. DMD·DMDR·RLHF·데이터 인프라 같은 *알고리즘 기여*는 코드 외부에 있고 모델 *가중치*에만 반영됨.

| 영역 | 라인 | 분류 |
|---|---|---|
| `transformer.py` (S3-DiT) | 571 | ⭐ 아키텍처 본체 |
| `pipeline.py` (generate) | 293 | ⭐ CFG truncation/normalization 트릭 |
| `scheduler.py` (FlowMatchEuler) | 150 | ⚪ 표준 |
| `autoencoder.py` (Flux VAE) | 369 | ⚪ Flux VAE 재구현 |
| `attention.py` (7 backend) | 516 | ⚪ FlashAttention 2/3 · SDPA · MPS 디스패치 |
| `loader.py` (모델 로딩) | 224 | ⚪ HF + meta device + bf16/fp32 |

→ **알고리즘 검증은 가중치 + 정성평가에만 의존 가능**.

---

## 4️⃣ 실험 요약

### 4-1. 학습 비용 (Table 1)

| 단계 | H800·시간 | USD |
|---|---|---|
| 저해상도 256² 사전학습 | 147.5K | $295K |
| Omni 사전학습 | 142.5K | $285K |
| 후처리 학습 (SFT + Distill + RLHF) | 24K | $48K |
| **합계** | **314K** | **$628K** |

> "314K"는 *GPU 시간*이지 학습 *이미지 장 수*가 아님 — 단위 오해 주의 (→ Q7).

### 4-2. 인간 선호도 (Elo)

| 플랫폼 | 순위 | Elo | 비고 |
|---|---|---|---|
| Artificial Analysis Arena | 8위 전체 / **1위 오픈소스** | 1161 | 6B로 최소 파라미터, $5/1000img 최저 비용 |
| Alibaba AI Arena | 4위 전체 / 1위 오픈소스 | 1025 | |
| vs FLUX.2 dev (32B), n=222 | G+S = **87.4%** (Good 46.4 / Same 41.0 / Bad 12.6) | | 1/5 파라미터로 우세 |

### 4-3. 자동 벤치마크 (텍스트→이미지, T2I)

| Benchmark | Z-Image | Z-Image-Turbo | 최고 경쟁자 |
|---|---|---|---|
| **CVTG-2K** (영어 텍스트, Word Acc) | **0.8671 (1위)** | 0.8585 | GPT-Image-1: 0.8569 |
| LongText-Bench-ZH | 0.936 (2위) | 0.926 (3위) | Qwen-Image: 0.946 |
| **OneIG-EN** Overall | **0.546 (1위)** | 0.528 | Qwen-Image: 0.539 |
| GenEval | 0.84 (T2위) | 0.82 | Qwen-Image: 0.87 |
| DPG-Bench | 88.14 (3위) | 84.86 | Qwen-Image: 88.32 |
| PRISM-EN | 75.6 | **77.4 (3위)** | GPT-Image-1: 80.7 |
| **PRISM-ZH** | **75.3 (2위)** | 75.1 | GPT-Image-1: 77.7 |

### 4-4. 편집 작업 (Z-Image-Edit)

| Benchmark | Z-Image-Edit |
|---|---|
| ImgEdit | 4.30 (3위) — UniWorld-V2: 4.49 |
| GEdit-EN | 7.57 (3위) |
| **GEdit-CN** | **7.54** (Qwen-Edit2509와 공동 2위) |

---

## 💬 Q&A

### Q1. 6B 작은 모델인데 어떻게 성능이 끌어올려졌나?

**단일 트릭이 아니라 통합 설계의 승리**. 각 요인을 *"작은 모델의 한계를 어떻게 보완하는가"* 관점에서:

| 요인 | 작은 모델 보완 메커니즘 | 참조 |
|---|---|---|
| ① 능동 데이터 큐레이션 4모듈 | 모델 ↔ 데이터의 닫힌 루프로 드문 개념까지 보강. 컴퓨트 1단위당 정보량 ↑ | → 3-4 |
| ② Single-Stream DiT | 같은 가중치가 양 모달리티를 동시 학습 → 유효 capacity가 dual-stream 대비 ~2배 | → 3-1 |
| ③ Qwen3-4B 텍스트 인코더 | LLM의 reasoning·세계지식·이중언어 능력을 DiT 외부에 *외주*. "이해는 LLM, 그림은 6B DiT" | → 3-1 |
| ④ 저해상도 47% 집중 | 기초 지식 (중국어 텍스트 렌더링 등)을 저해상도에서 *몰빵* 습득 | → 3-6 |
| ⑤ Decoupled DMD + DMDR | 8번 추론한 student가 100번 추론한 teacher를 *능가* — Turbo Elo 1161의 진짜 원동력 | → 3-2 |
| ⑥ 2-단계 RLHF | 3축 보상 + 5단 instruction 분해로 정렬을 *세분화* | → 3-3 |
| ⑦ Prompt Enhancer | 외부 VLM 고정 + Z-Image만 PE-aware SFT → 세계지식 부족을 외주 | (논문 §4.8) |

### Q2. Single-Stream의 "가중치가 양 모달리티를 동시 학습"이 정확히 무슨 뜻인가?

(Q1의 ②번 요인을 깊이 파헤치는 후속 질문.)

#### 먼저 "가중치가 학습된다"의 의미

가중치 (모델 파라미터, weights)는 입력을 변환하는 *행렬* — 예를 들어 `W ∈ ℝ^{3840×3840}` 크기. 학습은 그 행렬의 숫자들을 *데이터를 잘 맞히는 방향*으로 갱신하는 일.
- *"텍스트 학습"* = 그 행렬이 텍스트 토큰을 잘 다루도록 갱신됨.
- *"이미지 학습"* = 그 행렬이 이미지 토큰을 잘 다루도록 갱신됨.

핵심 질문: **한 가중치가 *둘 다*를 잘하도록 학습될 수 있는가?** → 가능 여부가 dual vs single stream의 갈림길.

#### 두 방식의 출발점이 다르다

**Dual-Stream (SD3 / FLUX의 방식)** — *가중치를 둘로 쪼개기*

```
한 블록 안에 가중치가 2벌:

  text 토큰 ──▶ W^text_Q, W^text_K, W^text_V  ┐
                W^text_O, W^text_FFN          │  ← 텍스트 전용
                                              │     (학습 신호 = 텍스트만)
  이미지 토큰 ─▶ W^img_Q,  W^img_K,  W^img_V  │
                W^img_O,  W^img_FFN           │  ← 이미지 전용
                                              │     (학습 신호 = 이미지만)
                  ↓                           │
            [text ; img] concat ─── Joint Attention 한 번 ─▶ split
```

한 블록에 *Q, K, V, O, FFN_up, FFN_down* 각 2벌 → **12개의 행렬**. 텍스트 분기 가중치는 *텍스트 신호만* 받고, 이미지 분기 가중치는 *이미지 신호만* 받음. 12B 모델이면 사실상 *6B 텍스트 전문가 + 6B 이미지 전문가*가 따로 학습.

**Single-Stream (Z-Image의 방식)** — *같은 가중치를 둘이 공유*

```
한 블록 안에 가중치가 1벌:

  text 토큰 ──┐
             │
  이미지 토큰 ─┤── 한 시퀀스로 concat ──▶ [t1, ..., t_S, i1, ..., i_4096]
             │
             │                              ↓
             │                       W_Q, W_K, W_V, W_O   ← 같은 가중치
             │                       W_FFN_up, W_FFN_down    하나씩만
             │                              ↓
             │                        한 번에 Attention + FFN
             │                              ↓
             └──◀── 다시 split (이미지 토큰만 다음 단계로)
```

한 블록에 *Q, K, V, O, FFN_up, FFN_down* 각 1벌 → **6개의 행렬**. 같은 가중치가 *텍스트 신호 + 이미지 신호 둘 다* 받음. **데이터를 두 배로 보는 셈**.

#### "유효 capacity 2배"의 진짜 의미

비유로 풀면 — 출발점이 다르다:

| | Dual-stream (전문가 분업) | Single-stream (양손잡이) |
|---|---|---|
| 학습 방식 | 6B 텍스트 전문가 + 6B 이미지 전문가 | 6B 양손잡이 1명 |
| 각자의 시야 | 평생 텍스트만 / 평생 이미지만 | 텍스트와 이미지를 동시에 |
| 강점 | 모달리티 안의 깊은 패턴 | **모달리티 간의 매핑** |
| 약점 | 두 모달리티를 잇는 능력 | 단일 모달리티 깊이는 살짝 양보 |

좀 더 형식적으로 — 가중치 `W`가 학습되는 양은 *그 가중치를 통과한 학습 신호의 양*에 비례:
- Dual-stream: `W^text`는 텍스트만 통과 → 텍스트 데이터의 학습 신호만 받음.
- Single-stream: `W`가 텍스트 + 이미지 둘 다 통과 → 양쪽 학습 신호를 *합쳐서* 받음.

→ 같은 컴퓨트 예산으로 **한 가중치가 더 많고·다양한 신호로 갱신**됨. *행렬 크기가 2배*가 아니라 *학습 효율이 2배*. 이게 "유효한 capacity 2배"의 정확한 의미.

#### 디코더-only LLM과의 유사성

저자가 논문에서 직접 비유 — *"Inspired by the scaling success of decoder-only models"*:
- GPT 계열의 디코더 한 줄기 (decoder-only = 한 방향으로만 흐르는 LLM 형태)가 *모든 종류의 토큰* (질문·답변·코드·시)을 같은 가중치로 처리.
- 그 결과 작은 LLM이 큰 인코더-디코더 구조 (encoder-decoder, 예: T5)를 능가하게 됨.
- Single-Stream DiT는 *그 교훈을 그림 생성에 그대로 옮긴* 디자인.

#### 코드에서의 실제 모습

[transformer.py:474-571](src/zimage/transformer.py#L474-L571) 에서 한 줄기인 게 분명함 (한 시퀀스로 합친 뒤 같은 가중치로 30번 통과):

```python
# 1. 토큰화 (둘 다 3840 차원으로 맞춤)
image_tokens = x_embedder(image)         # (4096, 3840)
cap_tokens   = cap_embedder(cap_feat)    # (S,    3840)

# 2. 한 시퀀스로 concat
unified = torch.cat([image_tokens, cap_tokens], dim=0)  # (4096+S, 3840)

# 3. 같은 가중치로 30번 통과
for layer in self.layers:           # ← 30개 ZImageTransformerBlock
    unified = layer(unified, ...)    # ← 안에 Q, K, V, FFN 각 1벌

# 4. 이미지 토큰만 다시 빼내 unpatchify
```

#### 단점은? — 있긴 함

같은 가중치가 두 모달리티를 받으려면 학습이 까다로워짐. Z-Image의 해결책 (자세한 사양 → 3-1):

| 단점 | 해결 |
|---|---|
| 텍스트·이미지 통계 분포가 다른데 같은 정규화 → 학습 불안정 | QK-Norm + Sandwich-Norm + tanh-게이트 잔차 |
| 시퀀스가 길어짐 (concat) → attention 비용 ↑ | FlashAttention 2/3, 이미지 4096 + 텍스트 512 정도라 부담 작음 |
| 텍스트 위치 vs 이미지 공간 정보가 섞임 | 3D Unified RoPE의 *축 분배* — 텍스트는 t축, 이미지는 h·w축 |
| 처음부터 통합 백본은 학습이 어려움 | Modality-specific refiner 2층씩 (Noise Refiner = 이미지 전용, Context Refiner = 텍스트 전용) — *초기 정렬만* 따로 |

#### 한 줄 요약

> Dual-stream은 *"가중치를 모달리티별로 쪼개서 각자 절반의 데이터만 깊게 학습"*하는 **전문가 분업**. Single-stream은 *"같은 가중치를 두 모달리티가 함께 학습해, 한 가중치가 양쪽 신호를 모두 받음"*하는 **통합 학습**. 후자가 컴퓨트 1단위당 받는 신호의 양이 많고 모달리티 간 매핑도 자연스럽게 학습되어, 같은 6B로도 dual-stream 12B 급의 표현력에 도달.

### Q3. 몇 스텝 증류 (Few-Step Distillation) 후에 RLHF를 한 게 맞나? 그러면 RL이 두 번 등장?

**맞습니다.** DMDR도 RL 기반이고 후속 RLHF도 RL이라 *RL이 두 번 등장*합니다. 두 RL은 **목적이 완전히 다릅니다** — 즉 *역할이 다르다*기보다, *언제 무엇을 잡으려는지 자체가 다르다*가 정확:

| 항목 | DMDR (증류 *내부*) | RLHF (증류 *후*) |
|---|---|---|
| 무엇과 결합 | **DM 항이 regularizer** 역할 | DM 없음, 전용 reward model 기반 |
| 단계 | 1단 (DMD + RL 동시) | 2단 (DPO → GRPO) |
| Reward | 단순 (안정성 우선) | 3축 + 5단 instruction 분해 |
| 데이터 | distillation pool | VLM 생성 + 사람 검증 페어, 커리큘럼 |
| 성격 | **방어적 RL** — 속도를 얻으면서 품질 유지 | **공격적 RL** — preference를 적극적으로 끌어올림 |

**왜 또 RLHF가 필요한가**: DMDR이 학생 모델을 안정화하고 나면, *DM 안전망 없이도* 정교한 신호로 더 밀어붙일 수 있는 여유가 생김. 그 자리에 DPO+GRPO 투입. Figure 14가 FSD vs RLHF 시각적 차이를 보여줌.

상세는 → 3-2, 3-3.

### Q4. 증류된 모델 (distilled model)은 추가 학습이 잘 안 된다고 알려져 있는데?

**맞습니다.** 그래서 README가 Z-Image-Turbo의 추가 학습 가능성 (Fine-Tunability)을 *명시적으로* **N/A**로 표시합니다.

| Model | Step | CFG | Diversity | Fine-Tunability |
|---|---|---|---|---|
| Z-Image-Omni-Base | 50 | ✅ | High | **Easy** |
| Z-Image | 50 | ✅ | Medium | **Easy** |
| **Z-Image-Turbo** | 8 | ❌ | Low | **🚫 N/A** |
| Z-Image-Edit | 50 | ✅ | Medium | Easy |

저자 공식 권장: **"파인튜닝 / LoRA 하려면 Turbo 말고 Z-Image나 Omni-Base를 써라."**

**왜 증류된 모델이 SFT/LoRA에 약한가** — 출발점이 다르기 때문. 즉:
- 증류된 모델은 *8번만 거치는 짧은 길*에 최적화되어 있고,
- 일반 SFT/LoRA 학습은 *1000번 거치는 긴 길*을 가정함.

→ 둘이 만나면 다음 4가지 구조적 문제가 발생:

1. **노이즈 스케줄 어긋남 (noise schedule mismatch)**: Turbo는 8 NFE 기준으로 sigma 분포가 sparse-collapsed. 일반 flow matching loss는 1000-step continuous를 가정 → 학습 시 schedule이 dense 화 → 8-step 능력 즉시 붕괴.
2. **CFG-free 학습됨**: unconditional 분기가 "죽은 가지" → LoRA가 양방향 가중치를 더하면 비대칭 학습.
3. **분포 붕괴 (mode collapse)**: diversity Low → 새 컨셉 주입 시 좁은 분포를 깨야 함 → 증류 안정성 손상.
4. **DMD teacher 재현 불가**: teacher score function 없이 distillation 보존 불가.

### Q5. 그런데 Distillation 후 RLHF도 *추가 학습*인데 왜 schedule이 안 깨졌나?

**핵심 통찰**: "추가 학습"에 두 종류가 있고, 증류 친화성이 정반대.

먼저 용어 — *학습할 데이터를 어디서 가져오는가*에 따른 구분:
- **On-policy (온폴리시) 학습**: 현재 학습 중인 모델이 *직접 생성한 샘플*로 학습. → 모델 분포 안에서 학습.
- **Off-policy (오프폴리시) 학습**: 외부 데이터셋 (예: 실제 이미지)으로 학습. → 모델 분포 *밖*에서 학습.

| | **Off-policy SFT (LoRA)** | **On-policy RL (DMDR/DPO/GRPO)** |
|---|---|---|
| 데이터 출처 | 외부 real image | **모델 자신이 8-step으로 생성** |
| Timestep 분포 | 1000-step 연속에서 임의 추출 | **증류된 8-step 분포 그대로** |
| Loss | reconstruction (절대 매칭) | preference/advantage (상대 ranking) |
| Schedule | **붕괴** | **보존** |
| Diversity 방향 | 분포 확장 (증류와 반대 방향) | 분포 집중 (증류와 같은 방향) |

**DPO/GRPO가 schedule을 깨지 않는 이유**:
- 학습 중 forward가 *이미 8-step inference* → timestep mismatch가 원천 발생 안 함.
- Loss가 *순위*만 학습 (절대 분포는 그대로 둠).
- KL 제약 (β)으로 reference에서 멀어지지 못함.

**DMDR은 한 술 더 떠 "이중 안전망"**: distillation loss (DM)가 RL이 schedule을 흔들려 할 때 *즉시 복원*. 일반 RLHF의 reference-KL을 DM이 더 강한 regularizer로 대체 (수식 → 3-2).

→ **증류된 모델은 *모든 추가 학습*에 약한 게 아니라 "schedule 가정이 다른 학습"에만 약함**.

만약 Turbo에 *굳이* LoRA를 한다면: DPO-LoRA / Reward-Weighted Regression / Distillation-aware LoRA(base에 LoRA → 다시 증류). 순수 SFT-LoRA는 비권장.

### Q6. 학습셋은 어떻게 구성되고 얼마나 큰가?

- **원천 / 원천 풀 규모 / 4-모듈 인프라 / 데이터 4갈래**: 전부 → 3-4 참조.
- **요점만**: 원천 풀은 *수십억* 장으로 추정되지만 **curated final dataset 크기는 비공개**. 합성 distillation 데이터 사용 안 함.

### Q7. 그럼 314K = 30만 장 학습한 거야?

**아닙니다 — 단위 오해.** 314K는 *GPU 시간*이지 *이미지 장 수*가 아닙니다.

```
314K H800 GPU·시간 = "H800 GPU 1장이 314,000 시간 일한 양"
```

| GPU 개수 | 실제 소요 일수 |
|---|---|
| 128장 | 102일 |
| 256장 | 51일 |
| 512장 | 26일 |

| 개념 | 단위 | Z-Image의 값 |
|---|---|---|
| Compute (시간) | GPU·hour | **314K H800h** (공개) |
| Compute (FLOPs) | TFLOPs | ≈ 10²¹ FLOPs 규모 |
| 학습 *데이터* 장 수 | images | **비공개** (→ Q6) |
| 학습 step | iter | 비공개 |

비교: SD 2.1 ~200K A100h, Llama 3 70B ~7M H100h. Z-Image는 20B–80B 경쟁작 대비 *한 자릿수 %* 비용.

### Q8. 그럼 학습이 며칠 걸린 거야?

**정확한 GPU 개수가 비공개라 "며칠"은 확정 불가.** 314K H800·시간을 동시 GPU 수로 나눈 시나리오만 가능.

| 동시 GPU 수 | 총 일수 | 비고 |
|---|---|---|
| 128장 | 102일 | 학술 랩 수준 |
| 256장 | 51일 | 중규모 |
| **512장** | **26일** | **알리바바급 가장 가능성 ↑** |
| 1024장 | 13일 | 대규모 |
| 2048장 | 6.4일 | 초대규모 |

512 GPU 시나리오로 단계별 분해:

| 단계 | H800·시간 | 512 GPU 기준 |
|---|---|---|
| 저해상도 256² 사전학습 | 147.5K | **약 12일** |
| Omni 사전학습 | 142.5K | **약 11.6일** |
| 후처리 학습 | 24K | **약 2일** |
| 합계 | 314K | **약 26일** |

> ⚠️ 단, 이건 *최종 성공 run* 기준. 실패한 run·hyperparameter sweep·ablation은 빠져 있어 실제 R&D 비용은 2~5배.

---

## 🎯 한 줄 요약 (전체)

> **Z-Image는 6B 파라미터·314K H800·시간·real-world only 데이터로 SOTA에 도달한 "효율 풀스택" T2I 모델**. *데이터(Active Curation) + 아키텍처(Single-Stream DiT + LLM 텍스트 인코더) + 커리큘럼(저해상도 47% 집중) + 증류(Decoupled DMD) + RLHF(DMDR → DPO → GRPO)* 가 6B에 맞춰 정렬된 결과. 증류된 Turbo는 추가 학습 (fine-tuning) N/A지만 RLHF는 *on-policy*라 schedule이 안 깨졌다는 점이 알고리즘의 우아함. 공개된 추론 코드는 깔끔하나 학습 코드·데이터·보상 모델은 모두 비공개라 알고리즘 검증은 가중치 + 정성평가에 의존해야 함.

---

## 📂 관련 메모리 링크

- 메모리: `paper_z_image.md` — 본 논문/코드 상세 분석
- 메모리: `paper_mv_split_dit.md` — 1000층 DiT 안정화 (다른 효율-DiT 사례)
- 메모리: `very_deep_models_landscape.md` — 깊은 모델 풍경 (스케일링 vs 효율 대비)
- 메모리: `feedback_paper_summary_format.md` — 본 문서가 따르는 형식 규칙
