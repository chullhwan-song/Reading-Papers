# PAPER: Z-Image — Single-Stream DiT 6B로 20B–80B를 따라잡은 효율 풀스택 설계

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Z-Image: An Efficient Image Generation Foundation Model |
| **저자** | Tongyi MAI Team (Alibaba) |
| **공개일** | 2025-12-01 (arXiv v1) / 2026-01-27 (Z-Image 본체 공개) |
| **분야** | cs.CV, cs.LG |
| **논문 링크** | https://arxiv.org/abs/2511.22699 |
| **PDF** | https://arxiv.org/pdf/2511.22699 |
| **공식 사이트** | https://tongyi-mai.github.io/Z-Image-blog/ |
| **공식 코드** | https://github.com/Tongyi-MAI/Z-Image |
| **가중치** | HF: `Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo` |
| **사용 외부 모델** | Qwen3-4B (텍스트), Flux VAE, SigLIP 2 (Edit 전용) |
| **공개 범위** | 추론 코드만 (학습 코드·데이터·reward 모델 비공개) |

---

## 📖 주요 용어 사전 (Glossary)

읽기 전 알아두면 좋은 용어 *정의*만. 수치·메커니즘은 본문 참조.

### 아키텍처
- **S3-DiT (Scalable Single-Stream DiT)**: 텍스트·이미지를 한 시퀀스로 concat 후 *단일* 백본이 처리하는 본 논문의 DiT 변형. 상세 → 3-1.
- **Single-Stream vs Dual-Stream**: SD3/FLUX는 두 모달리티에 파라미터를 분기(dual). 단일 스트림은 통합 → 파라미터 효율 ↑.
- **3D Unified RoPE**: 위치 인코딩을 (t, h, w) 축으로 분할. 텍스트는 t축, 이미지는 h·w축만 사용.
- **adaLN (low-rank)**: timestep 컨디션을 layer별 scale/gate로 주입. low-rank 공유로 파라미터 절약.
- **Sandwich-Norm**: attention/FFN의 입력·출력 양쪽 모두 RMSNorm.
- **QK-Norm**: Attention의 Q, K에 RMSNorm.
- **SwiGLU**: `w2(SiLU(w1·x) ⊙ w3·x)`.

### 데이터·캡션
- **Data Profiling Engine / Cross-modal Vector Engine / World Knowledge Topological Graph / Active Curation Engine**: 데이터 인프라 4모듈. 상세 → 3-4.
- **Z-Captioner**: 자체 captioner. 이미지 1장당 5종 캡션 + OCR + world knowledge 생성. 상세 → 3-4.

### 추론·증류
- **Flow Matching**: $x_t = t·x_1 + (1-t)·x_0$, predict $v = x_1 - x_0$. ODE Euler step.
- **NFE**: 추론 시 모델 forward 횟수 (Turbo는 8).
- **CFG truncation**: 후반 step에서 CFG OFF로 모델 호출 절약. 상세 → 3-5.
- **CFG normalization**: CFG 합성 결과 norm 클램프로 saturation 방지. 상세 → 3-5.
- **DMD / Decoupled DMD / DMDR**: few-step distillation 계열. 상세 → 3-2.
- **DPO**: chosen/rejected pair-wise loss (offline). 상세 → 3-3.
- **GRPO**: 그룹 내 상대 advantage로 policy gradient (online). 상세 → 3-3.

### 모델 변종
- **Z-Image-Omni-Base**: pretraining만. raw 모델, fine-tuning 최적.
- **Z-Image**: + SFT. 50-step, CFG ON, diversity Medium.
- **Z-Image-Turbo**: + Distillation + RLHF. 8-step, CFG OFF. *Fine-tunability N/A*.
- **Z-Image-Edit**: + Continued PT + SFT for editing.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> "scale-at-all-costs" 패러다임과 상용 모델 distillation 의존을 거부하고, **6B 파라미터 + 314K H800h + real-world only 데이터**로 SOTA에 도달.

- **문제 인식**: 오픈소스가 거대화(Qwen-Image 20B, FLUX.2 32B, HunyuanImage 80B)로 도망가거나 상용 합성 데이터에 의존 → 둘 다 지속 불가.
- **해결책**: 4기둥 — 데이터 인프라(→3-4), Single-Stream 아키텍처(→3-1), 단계별 커리큘럼(→3-6), 8-step distillation(→3-2/3-3).
- **검증**: Artificial Analysis Arena Elo 1161 (전체 8위, 오픈소스 **1위**). 상세 → 4.

---

## 2️⃣ 핵심 기여 (Contributions)

1. **S3-DiT 아키텍처** — 6B 단일 스트림 MM-DiT (→ 3-1).
2. **데이터 인프라 4-모듈 통합** — Profiling·Vector·Graph·Active Curation closed loop (→ 3-4).
3. **Decoupled DMD** — DMD의 CA/DM 메커니즘 분해 (→ 3-2).
4. **DMDR** — distillation 안에 RL 통합, DM이 reward hacking 차단 (→ 3-2).
5. **2-stage RLHF** — DPO(objective) → GRPO(subjective) + 3축 reward (→ 3-3).
6. **PE-aware SFT** — VLM 고정 + Z-Image만 PE-enhanced caption으로 SFT.
7. **비용 효율 실증** — 314K H800h / $628K, 경쟁작 대비 한 자릿수 % (→ 4-1).

---

## 3️⃣ 주요 알고리즘

### 3-1. S3-DiT 아키텍처

| 항목 | 값 |
|---|---|
| Total params | **6.15B** |
| Backbone layers | 30 (single-stream) |
| Refiner layers | 2 (noise, image-only) + 2 (context, text-only) |
| Hidden dim | 3840 |
| Attention heads | 32 (head_dim = 128) |
| FFN intermediate | 10240 = `int(dim/3 × 8)` (SwiGLU) |
| QK-Norm / Sandwich-Norm | ✅ / ✅ |
| Patch | 2×2 spatial, 1 temporal |
| RoPE axes_dims | (32, 48, 48) ← head_dim 128 = 32(t) + 48(h) + 48(w) |
| RoPE θ | 256 |
| adaLN dim | 256 (전 층 공유) |
| 최대 해상도 | ~1k–1.5k |

**토큰 입력**
- 이미지: VAE latent(16ch, 8× DS) → patch(2,2) → 4096 tokens × 3840.
- 텍스트: Qwen3-4B `hidden_states[-2]` (2560 dim) → Linear → 3840.
- 둘 다 `SEQ_MULTI_OF=32` 배수로 pad (torch.compile 정적 shape).

**Forward 순서** (텐서 흐름 → 3-5)
```
1. Timestep → sinusoidal(256) → MLP → adaLN vec (256d)
2. patchify_and_embed : image / caption 토큰화 + 3D RoPE 좌표
3. Noise Refiner ×2   : image only, modulation ON
4. Context Refiner ×2 : text only, modulation OFF
5. Backbone ×30       : unified = [image; text] 통합 시퀀스
   per-block: RMSNorm·scale → Attention(RoPE+QK-Norm) → tanh-gate 잔차
              RMSNorm·scale → SwiGLU FFN          → tanh-gate 잔차
6. Final Layer (이미지 토큰만): LayerNorm × (1+scale) → Linear
7. unpatchify → velocity (B, 16, 1, 128, 128)
```

### 3-2. Decoupled DMD + DMDR (Few-Step Distillation)

**관찰**: 기존 DMD가 detail 손실·color shift를 일으키는 이유 — *두 메커니즘의 미분화*:

| 메커니즘 | 역할 |
|---|---|
| CFG-Augmentation (CA) | **주 엔진** — student의 few-step 능력 끌어올림 (기존 literature 간과) |
| Distribution Matching (DM) | **regularizer** — 분포 안정화, artifact 제거 (본체로 잘못 인식되어 옴) |

**Decoupled DMD**: CA와 DM에 *서로 다른 re-noising schedule* 적용 → detail·color 회복. 8 NFE student가 100-step teacher *능가*.

**DMDR**: distillation 안에 RL 통합.
$$\mathcal{L}_{\text{DMDR}} = \mathcal{L}_{\text{DM}} + \lambda \cdot \mathcal{L}_{\text{RL}}$$
- DM term이 *자연 anchor* → reward hacking 자연 차단.
- 일반 RLHF의 reference-KL을 DMD가 더 강한 regularizer로 대체.

> **DMDR과 후속 RLHF의 RL은 다른 목적** — 비교는 Q1 참조.

### 3-3. 2-stage RLHF (Distillation *후*)

**3축 Reward Model**: instruction-follow / AIGC-percep / aesthetic.
- instruction은 prompt를 **(subject, attribute, action, spatial, style) 5단으로 분해** → 인간이 미충족 요소 클릭 → 충족 비율을 reward로.

**Stage 1 — Offline DPO** (objective dim)
- VLM이 chosen/rejected pair 대량 생성 → 인간 verify (hybrid pipeline).
- Curriculum: 쉬운 prompt → 어려운, moderate → subtle pair.
- 텍스트 렌더링·객체 카운트 같은 *binary 판정* 가능 항목에 집중.

**Stage 2 — Online GRPO** (subjective dim)
- composite advantage (realism + aesthetic + instruction) → multi-faceted reward.
- 단일 reward 대비 hacking 어려움.

### 3-4. 데이터 인프라 + 학습셋 구성

**원천**: 알리바바 내부 저작권 보유 풀. *합성 distillation 데이터 사용 안 함*.

**raw pool 규모** (간접 단서만):
- "billions of embeddings" → 10⁹ 단위
- dedup "1B items / 8h on 8 H800s" → 수십억 단위 처리
- → **raw ~수십억 장, curated final dataset 크기는 비공개**

**4-모듈 closed-loop 인프라**

| 모듈 | 역할 |
|---|---|
| **Data Profiling Engine** | pHash 중복 제거, 압축 아티팩트(이상/실제 파일 크기 비율), variance-of-border / BPP로 entropy, 자체 quality model, AIGC 분류기, VLM 기반 semantic tag + NSFW + 중국 문화 컨셉, CN-CLIP alignment |
| **Cross-modal Vector Engine** | SD3의 range_search → graph community detection 재정의 + GPU k-NN. dedup 1B/8h on 8 H800. 실패 케이스 retrieval로 데이터 결함 진단 |
| **World Knowledge Topological Graph** | Wikipedia + PageRank/VLM 가지치기 + caption embedding 계층 클러스터링. BM25 rarity score로 long-tail rebalancing |
| **Active Curation Engine** | 모델이 약한 개념 자동 발굴(예: 松鼠鳜鱼=음식인데 "다람쥐+생선"으로 잘못 생성) → 데이터 보충 → 재학습 |

**데이터 4갈래**

| 갈래 | 내용 |
|---|---|
| **[A] T2I 단일 이미지** | + Z-Captioner *5종* 캡션: tag / long(OCR+world knowledge, 객관적 어조) / medium / short / simulated user prompt. + 원본 alt-text 작은 확률. OCR은 CoT 방식, 원어 보존 |
| **[B] I2I 페어** (Omni 단계, T2I:I2I = 4:1) | ① Mixed Editing + Graphical Representation (1장 → N편집 → N² 페어) ② Paired Images from Videos (CN-CLIP 유사도 필터) ③ Rendering for Text Editing (텍스트만 다른 합성 페어) |
| **[C] SFT 큐레이션** | 매우 적은 고품질 셋. 분포 narrowing 목적. Tagged Resampling으로 long-tail 보존, Model Merging으로 Pareto-optimal |
| **[D] RLHF preference pair** | VLM 생성 + 인간 검증. easy→hard, moderate→subtle curriculum |

### 3-5. 추론 코드 흐름 (공개 코드 기준)

**Level 0 — 파이프라인**
```
prompt str ─▶ Qwen3-4B  (text_encoder)
                 │  cap_feats [seq, 2560]
torch.randn ─┐   │
             │   ▼
             └─▶ S3-DiT 30+4 layers   (transformer.py) ⏎ 8회 (Turbo)
                 │  → velocity                          (Flow Matching Euler)
                 ▼
            (CFG truncation / normalization)
                 │
                 ▼ 8 step 후
            Flux VAE.decode (fp32)                      (autoencoder.py)
                 ▼
            PIL.Image (1024×1024)
```

**Level 1 — `generate()` substeps** ([pipeline.py:66](src/zimage/pipeline.py#L66))
1. `apply_chat_template(..., enable_thinking=True)` → Qwen3 reasoning hidden state.
2. tokenize (`max_length=512`) → `text_encoder(...).hidden_states[-2]` ← 끝에서 두 번째 층.
3. latent init: `randn(B, 16, H//8//2·2, W//8//2·2)` fp32.
4. `calculate_shift(seq_len)` → mu (해상도 별 동적) → `scheduler.set_timesteps(num=8, mu=mu)`.
5. Denoising loop: **CFG truncation 결정** → DiT forward → CFG 합성 → **CFG normalization** → Euler step (`x += dt × v`).
6. `vae.decode(latents / scaling_factor)` → `[-1,1] → [0,1] → uint8 → PIL.Image`.

**CFG truncation/normalization 트릭** ([pipeline.py:227-268](src/zimage/pipeline.py#L227-L268))
- *truncation*: `t_norm > cfg_truncation`이면 CFG OFF → 모델 호출 횟수 절반 절약 + 색번짐 ↓.
- *normalization*: `||pos+s·(pos-neg)||` 가 `λ·||pos||` 초과 시 norm 클램프 → saturation 방지.

**Level 2 — DiT 내부**: 수식·layer 수·forward 순서는 → 3-1.

**Level 3 — Attention dispatch** ([attention.py](src/utils/attention.py))
- 7 backend: Flash 2/3, Flash-Varlen 2/3, MPS-Flash, Native(SDPA), Native-Math.
- enum + `@register_backend` 데코레이터.
- H800에서 `_flash_3` + `torch.compile` → sub-second.

### 3-6. 학습 커리큘럼

```
Pre-training
├─ Low-res 256²      ····· 전체 compute의 47% 집중
│  └─ T2I만, foundational 지식(중국어 텍스트 렌더링 등)
└─ Omni Pre-training ····· 45%
   ├─ Arbitrary resolution
   ├─ T2I + I2I joint (4:1)
   └─ Multi-level bilingual caption (5종, → 3-4 [A])

Post-training  ··········· 8%
├─ SFT       (분포 narrowing, Tagged Resampling, Model Merging)
├─ Distillation (Decoupled DMD → +DMDR, → 3-2)
└─ RLHF      (DPO → GRPO, → 3-3)
```

**SFT의 3축**:
- *Distribution Narrowing*: noisy web → high-quality sub-manifold.
- *Tagged Resampling*: BM25 rarity로 long-tail concept up-weight (catastrophic forgetting 방지).
- *Model Merging*: 여러 SFT variant (사실성·스타일·instruction 편향) weight 평균 → Pareto-optimal.

**시스템 효율**: FSDP2 + 전 층 gradient checkpoint + torch.compile + sequence-length-aware dynamic batch.

### 3-7. 코드에서 논문 기여의 위치

공개 코드는 추론만. 알고리즘 contribution(DMD/DMDR/RLHF/데이터 인프라)은 *코드 외*에 있고 모델 *가중치*에만 반영됨.

| 영역 | 라인 | 분류 |
|---|---|---|
| `transformer.py` (S3-DiT) | 571 | ⭐ 아키텍처 본체 |
| `pipeline.py` (generate) | 293 | ⭐ CFG truncation/normalization 트릭 |
| `scheduler.py` (FlowMatchEuler) | 150 | ⚪ 표준 |
| `autoencoder.py` (Flux VAE) | 369 | ⚪ Flux VAE 재구현 |
| `attention.py` (7 backend) | 516 | ⚪ FA2/3·SDPA·MPS dispatch |
| `loader.py` (모델 로딩) | 224 | ⚪ HF + meta device + bf16/fp32 |

→ **알고리즘 검증은 가중치 + 정성평가에 의존해야 함**.

---

## 4️⃣ 실험 요약

### 4-1. 학습 비용 (Table 1)

| Stage | H800 hours | USD |
|---|---|---|
| Low-res Pre-training (256²) | 147.5K | $295K |
| Omni Pre-training | 142.5K | $285K |
| Post-training (SFT + Distill + RLHF) | 24K | $48K |
| **Total** | **314K** | **$628K** |

> "314K"는 *GPU 시간*이지 학습 *샘플 수*가 아님 — 단위 오해 주의 (→ Q6).

### 4-2. 인간 선호 (Elo)

| 플랫폼 | 순위 | Elo | 비고 |
|---|---|---|---|
| Artificial Analysis Arena | 8위 전체 / **1위 오픈소스** | 1161 | 6B로 최소 파라미터, $5/1000img 최저 비용 |
| Alibaba AI Arena | 4위 전체 / 1위 오픈소스 | 1025 | |
| vs FLUX.2 dev (32B), n=222 | G+S = **87.4%** (G 46.4 / S 41.0 / B 12.6) | | 1/5 파라미터로 우세 |

### 4-3. 자동 벤치마크 (T2I)

| Benchmark | Z-Image | Z-Image-Turbo | 최고 경쟁자 |
|---|---|---|---|
| **CVTG-2K** (EN, Word Acc) | **0.8671 (1위)** | 0.8585 | GPT-Image-1: 0.8569 |
| LongText-Bench-ZH | 0.936 (2위) | 0.926 (3위) | Qwen-Image: 0.946 |
| **OneIG-EN** Overall | **0.546 (1위)** | 0.528 | Qwen-Image: 0.539 |
| GenEval | 0.84 (T2) | 0.82 | Qwen-Image: 0.87 |
| DPG-Bench | 88.14 (3위) | 84.86 | Qwen-Image: 88.32 |
| PRISM-EN | 75.6 | **77.4 (3위)** | GPT-Image-1: 80.7 |
| **PRISM-ZH** | **75.3 (2위)** | 75.1 | GPT-Image-1: 77.7 |

### 4-4. 편집 (Z-Image-Edit)

| Benchmark | Z-Image-Edit |
|---|---|
| ImgEdit | 4.30 (3위) — UniWorld-V2: 4.49 |
| GEdit-EN | 7.57 (3위) |
| **GEdit-CN** | **7.54** (Qwen-Edit2509와 공동 2위) |

---

## 💬 Q&A

### Q1. 6B 작은 모델인데 어떻게 성능을 끌어올렸나?

**단일 트릭이 아니라 통합 설계의 승리**. 각 요인이 *작은 모델 한계 보완* 관점에서:

| 요인 | 작은 모델 보완 메커니즘 | 참조 |
|---|---|---|
| ① 데이터 인프라 4모듈 | 모델-데이터 closed loop으로 long-tail 보강. 정보 1단위/compute ↑ | → 3-4 |
| ② Single-Stream DiT | dual-stream 대비 모든 파라미터가 양 모달리티 동시 학습 → 유효 capacity ~2× | → 3-1 |
| ③ Qwen3-4B 텍스트 인코더 | LLM의 reasoning·세계지식·bilingual을 DiT 외부에 outsource. "이해는 LLM, 그림은 6B DiT" | → 3-1 |
| ④ low-res 47% 집중 | foundational 지식(텍스트 렌더링 등)을 저해상도에서 습득 | → 3-6 |
| ⑤ Decoupled DMD + DMDR | 8-step student가 teacher *능가* — Turbo Elo 1161의 진짜 원동력 | → 3-2 |
| ⑥ 2-stage RLHF | 3축 reward + 5단 instruction 분해로 fine-grained 정렬 | → 3-3 |
| ⑦ Prompt Enhancer | 외부 VLM 고정 + Z-Image만 PE-aware SFT → world knowledge 외주 | (논문 §4.8) |

### Q2. Few-Step Distillation 후에 RLHF를 한 게 맞나? 그러면 RL이 두 번 아닌가?

**맞습니다.** DMDR도 RL 기반이고 후속 RLHF도 RL이라 *RL이 두 번 등장*합니다. 두 RL은 **목적이 완전히 다릅니다**:

| 항목 | DMDR (distillation 내부) | RLHF (distillation 후) |
|---|---|---|
| 무엇과 결합 | **DM term이 regularizer** | DM 없음, 전용 reward model |
| 단계 | 1단 (DMD + RL 동시) | 2단 (DPO → GRPO) |
| Reward | 단순 (안정성 우선) | 3축 + 5단 instruction 분해 |
| 데이터 | distillation pool | VLM-generated + human-verified pair, curriculum |
| 성격 | **방어적 RL** — 속도 회복하며 품질 유지 | **공격적 RL** — preference 끌어올리기 |

**왜 또 RLHF가 필요한가**: DMDR이 student를 안정화하고 나면, *DM 안전망 없이도* fine-grained signal로 더 밀어붙일 여유가 생김. 그 자리에 DPO+GRPO 투입. Figure 14가 FSD vs RLHF 차이를 시각적으로 보여줌.

상세는 → 3-2, 3-3.

### Q3. Distilled 모델은 추가 학습이 잘 안 된다고 알려져 있는데?

**맞습니다.** 그래서 README가 Z-Image-Turbo의 Fine-Tunability를 명시적으로 **N/A**로 표시합니다.

| Model | Step | CFG | Diversity | Fine-Tunability |
|---|---|---|---|---|
| Z-Image-Omni-Base | 50 | ✅ | High | **Easy** |
| Z-Image | 50 | ✅ | Medium | **Easy** |
| **Z-Image-Turbo** | 8 | ❌ | Low | **🚫 N/A** |
| Z-Image-Edit | 50 | ✅ | Medium | Easy |

저자 공식 권장: **"파인튜닝/LoRA 하려면 Turbo 말고 Z-Image나 Omni-Base를 써라."**

**왜 distilled 모델이 SFT/LoRA에 약한가** (4가지 구조적 이유):
1. **Noise schedule mismatch**: Turbo는 8 NFE에 맞춰 sigma 분포 sparse-collapsed. 일반 flow matching loss는 1000-step continuous 가정 → 학습 시 schedule이 dense화 → 8-step 능력 즉시 붕괴.
2. **CFG-free 학습**: unconditional branch가 "죽은 가지" → LoRA 비대칭 학습.
3. **Mode collapse**: diversity Low → 새 컨셉 주입 시 좁은 분포를 깨야 함 → distillation 안정성 손상.
4. **DMD teacher 재현 불가**: teacher score function 없이 distillation 보존 불가.

### Q4. 그런데 Distillation 후 RLHF도 *추가 학습*인데 왜 schedule이 안 깨졌나?

**핵심 통찰**: "추가 학습"에는 *두 종류*가 있고, distillation 친화성이 정반대.

| | **Off-policy SFT (LoRA)** | **On-policy RL (DMDR/DPO/GRPO)** |
|---|---|---|
| 데이터 출처 | 외부 real image | **모델 자신이 8-step으로 생성** |
| Timestep 분포 | 1000-step continuous 임의 | **distilled 8-step 그대로** |
| Loss | reconstruction (절대 매칭) | preference/advantage (상대 ranking) |
| Schedule | **붕괴** | **보존** |
| Diversity 방향 | 분포 확장 (distillation과 역) | 분포 집중 (distillation과 동) |

**왜 DPO/GRPO는 schedule을 안 깨나**:
- 학습 중 forward가 *이미 8-step inference* → timestep mismatch 원천 부재.
- Loss가 *순위*만 학습 (절대 분포는 가만 둠).
- KL/β 제약으로 reference에서 멀어지지 못함.

**DMDR은 한 술 더 떠 "이중 안전망"**: distillation loss(DM)가 RL이 schedule을 흔들려 할 때 즉시 복원. 일반 RLHF의 reference-KL을 DM이 더 강한 regularizer로 대체 (→ 3-2 수식).

→ **distilled 모델은 *모든* 추가 학습에 약한 게 아니라 "schedule 가정이 다른 학습"에만 약함**.

Turbo에 *굳이* LoRA를 한다면: DPO-LoRA / Reward-Weighted Regression / Distillation-aware LoRA(base에 LoRA → 재증류). 순수 SFT-LoRA는 비권장.

### Q5. 학습셋은 어떻게 구성되고 얼마나 큰가?

- **원천 / raw pool 규모 / 4-모듈 인프라 / 4갈래 데이터 구성**: 전부 → 3-4 참조.
- **요점만**: raw pool은 *수십억* 장으로 추정되지만, **curated final dataset 크기는 비공개**. 합성 distillation 데이터 사용 안 함.

### Q6. 그럼 314K = 30만 장 학습한 거야?

**아닙니다 — 단위 오해.**

```
314K H800 GPU hours = "H800 GPU 1장이 314,000 시간 일한 양"
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
| 학습 *데이터* 장 수 | images | **비공개** (→ Q5) |
| 학습 step | iter | 비공개 |

비교: SD 2.1 ~200K A100h, Llama 3 70B ~7M H100h. Z-Image는 20B–80B 경쟁작 대비 *한 자릿수 %* 비용.

---

## 🎯 한 줄 요약 (전체)

> **Z-Image는 6B 파라미터·314K H800h·real-world only 데이터로 SOTA에 도달한 "효율 풀스택" T2I 모델**. *데이터(Active Curation) + 아키텍처(Single-Stream DiT + LLM 텍스트 인코더) + 커리큘럼(low-res 47%) + 증류(Decoupled DMD) + RLHF(DMDR → DPO → GRPO)* 가 6B에 맞춰 정렬된 결과. Distilled Turbo는 fine-tuning N/A지만 RLHF는 *on-policy*라 schedule이 안 깨졌다는 점이 알고리즘의 우아함. 공개된 추론 코드는 깔끔하나 학습 코드·데이터·reward 모델은 모두 비공개라 알고리즘 검증은 가중치 + 정성평가에 의존해야 함.

---

## 📂 관련 메모리 링크

- 메모리: `paper_z_image.md` — 본 논문/코드 상세 분석
- 메모리: `paper_mv_split_dit.md` — 1000층 DiT 안정화 (다른 효율-DiT 사례)
- 메모리: `very_deep_models_landscape.md` — 깊은 모델 풍경 (스케일링 vs 효율 대비)
- 메모리: `feedback_paper_summary_format.md` — 본 문서가 따르는 형식 규칙
