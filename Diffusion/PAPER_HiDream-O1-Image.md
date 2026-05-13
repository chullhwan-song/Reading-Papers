# HiDream-O1-Image: Pixel-level Unified Transformer 기반 통합 이미지 생성 파운데이션 모델

## 📋 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | HiDream-O1-Image: A Natively Unified Image Generative Foundation Model with Pixel-level Unified Transformer |
| **저자/기관** | HiDream.ai |
| **공개일** | 2026-05-10 (technical report), 코드/가중치 2026-05-08 |
| **분야** | Text-to-Image / Image Editing / Subject-driven Personalization, Pixel-space Diffusion Transformer |
| **논문 링크** | [HiDream-O1-Image.pdf (GitHub assets)](https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/assets/HiDream-O1-Image.pdf) (arXiv 미공개, 27p technical report) |
| **코드 (공식)** | https://github.com/HiDream-ai/HiDream-O1-Image (MIT License) |
| **체크포인트** | `HiDream-ai/HiDream-O1-Image` (8B, 50-step), `HiDream-ai/HiDream-O1-Image-Dev` (8B, 28-step distilled) |
| **Prompt Agent** | `google/gemma-4-31B-it` (or OpenAI-compatible API) |
| **Backbone 초기화** | Qwen3-VL-8B-Instruct (8B 변종) |
| **외부 비전 인코더** | SigLip-2 (조건 이미지/참조 이미지 임베딩 추출) |
| **외부 평가 도구** | Qwen-VL2.5-72B (UniSubject 평가), HPSv3 (인간 선호), VLM judges |
| **별칭** | "Peanut" (Artificial Analysis Arena 익명 출시명, 2026-05-05 #8 진입) |

---

## 📖 주요 용어 사전 (Glossary)

### 아키텍처
- **UiT (Unified Transformer)**: pixel/text/condition을 **단일 공유 토큰 공간**에서 처리하는 decoder-only Transformer. VAE 없음, 별도 텍스트 인코더 없음.
- **Pixel-space DiT**: latent space가 아닌 **raw RGB pixel**을 직접 모델링하는 DiT. (cf. FLUX/SD3는 VAE latent 사용)
- **Patch Embedding (BottleneckPatchEmbed)**: 32×32 패치를 hidden_dim의 1/4 차원(PCA 스타일 bottleneck)을 거쳐 embed.
- **Hybrid Unified Attention**: **AR(text/condition) 토큰 = causal mask**, **generation 토큰 = full attention**. LLM의 autoregressive 구조와 DiT의 양방향 attention을 한 Transformer 안에 공존시킴.
- **DeepStack visual embeddings**: Qwen3-VL의 vision encoder가 만드는 다층 시각 feature. 초기 decoder layer에 주입되어 시각 정보를 강화.
- **TMS token (`<|tms_token|>`)**: timestep 정보를 담는 single special token (token_id=151673). `t_embedder1(timestep)`이 이 위치에 삽입됨.
- **BOI / BOR / EOR / BOT**: begin/end-of-image, begin/end-of-reference 등 segmentation을 위한 특수 토큰들.

### 핵심 개념
- **Token Types** (0/1/2/3): 0=AR text, 1=generation target, 2=reference image patches, 3=timestep. attention mask와 token role 결정에 사용.
- **Reasoning-Driven Prompt Agent**: Gemma-4-31B-it 기반 "thinking" 모듈. 사용자 raw prompt → 추론(reasoning)을 거쳐 자기완결적 영어 prompt + resolved_knowledge로 변환.
- **SCALIST framework**: prompt rewriting 시 사용하는 7요소 — **S**ubject, **C**omposition, **A**ction, **L**ocation, **I**mage style, **S**pecs, **T**ext rendering.
- **In-context Visual Reasoning**: T2I/Editing/IP를 모두 "공유 토큰 공간에서의 in-context reasoning" 한 가지 과정으로 통합.
- **DMD (Distribution Matching Distillation)**: 50-step teacher의 trajectory를 28-step student로 distill.
- **Adversarial Diffusion Distillation**: 위 DMD + diffusion loss + GAN loss(frozen teacher feature 기반 discriminator).

### 비교 기법
- **Latent DiT** (FLUX, SD3, Qwen-Image): VAE encoder → DiT → VAE decoder. 별도 텍스트 인코더(CLIP/T5) 사용.
- **Pixel-space DiT** (PixArt 등 일부): VAE는 없지만 여전히 disjoint 텍스트 인코더 사용.
- **HiDream-O1-Image (UiT)**: **VAE도 없고, 별도 텍스트 인코더도 없음**. raw pixel + text token + condition을 한 Transformer가 처리.

### 평가 지표
- **GenEval**: compositional generation (single/two-obj, count, color, position, attribute).
- **DPG-Bench**: dense prompt alignment.
- **HPSv3**: 12 카테고리 인간 선호 점수.
- **CVTG-2K**: complex visual text generation (2~5 regions, NED, CLIP).
- **LongText-Bench EN/ZH**: 장문 텍스트 렌더링.
- **GEdit / ImgEdit**: instruction-based editing (Add/Adjust/Extract/Replace/Remove/Background/Style/Hybrid/Action).
- **UniSubject (저자 신규)**: 300 cases × 1.8K subjects, 2-3/4-8/9-11 subjects 구간 평가. Q-PF (prompt following), Q-SC (subject consistency), Q-O (overall), HPSv3.

### 학습 관련
- **Flow Matching loss**: x_t = t·x + (1-t)·ε, target은 velocity. 본 논문에선 model이 x_pred를 출력 → v = (x_pred - z) / σ.
- **GRPO**: 그룹 상대 정책 최적화 (RLHF post-training).
- **Logit-Normal sampling**: 사전학습 timestep 샘플링.
- **Uniform sampling (SFT)**: post-training 시 late timestep(고화질 단계) 가중치 증가.

---

## 🎯 논문 요약 (TL;DR)

**한 줄 요약**: 기존 VLM(**Qwen3-VL-8B-Instruct**) 가중치로 backbone을 초기화한 뒤 patch/timestep embedder와 final layer만 덧붙여 **diffusion 형태**로 확장한 구조 — VAE도, 별도 텍스트 인코더도 없이 **raw pixel + text + condition**을 단일 공유 토큰 공간에서 처리하는 **decoder-only Unified Transformer**로 T2I/편집/IP를 모두 통합한 8B 모델이며, 2048×2048 native 합성·SOTA급 텍스트 렌더링·Artificial Analysis Arena #8 달성.

**핵심 문제**:
1. 기존 LDM(SD3/FLUX/Qwen-Image)은 VAE compression으로 고주파 디테일을 잃고, 별도 text encoder(CLIP/T5)와 semantic misalign됨.
2. PixArt 류 pixel-space DiT도 텍스트 인코더는 여전히 disjoint하고, single-task T2I에 특화되어 편집/IP로 확장이 어려움.
3. 모듈식 파이프라인은 in-context reasoning을 못함.

**해결책**:
1. **Pixel-level Unified Transformer (UiT)**: Qwen3-VL-8B-Instruct backbone에 patch embedder/timestep embedder/final layer만 추가. raw pixel patch (32×32) → bottleneck embed → shared token space → decoder-only Transformer → pixel patch 예측.
2. **Hybrid Unified Attention**: condition/text는 causal, generation은 full. AR 능력 보존 + 공간 일관성 확보.
3. **Reasoning-Driven Prompt Agent**: Gemma-4-31B로 chain-of-thought reasoning 후 SCALIST 영어 prompt 생성. ("O1" 이름의 유래 — reasoning-driven generation.)
4. **3-stage progressive pretraining**: 512² → 1024² → 2048² + multi-task (T2I + LM + MMU + editing + IP).
5. **Post-training**: SFT (reasoning trajectory 포함) + GRPO RLHF (OCR/aesthetic/instruction/reasoning 복합 보상).
6. **DMD + GAN distillation**: 50-step → 28-step Dev 모델.

**검증**:
- GenEval 0.90 (8B), 0.92 (200B+) — Qwen-Image(27B)/FLUX.2 Dev(56B) 상회.
- DPG-Bench 89.83, HPSv3 10.37 (8B) — open-weights 1위.
- LongText-Bench 0.979 EN / 0.978 ZH — Nano Banana 2.0 동급.
- 200B+ scale 변종으로 scaling law 검증, closed-source(GPT Image 2 등) 초과.

---

## 🚀 핵심 기여 (Contributions)

1. **Natively Unified Generative Architecture**: VAE/disjoint text encoder를 완전히 제거하고 raw pixel + text + task condition을 단일 토큰 공간으로 통합하는 end-to-end UiT 제안.
2. **Reasoning-Driven Prompt Agent (오픈소스)**: chain-of-thought로 implicit knowledge / layout / 텍스트 렌더링까지 prompt에 명시화. 모델 입력으로 직접 사용.
3. **8B 스케일에서 SOTA-급 효율성·다재성**: T2I, 장문 텍스트 렌더링, instruction editing, IP, multi-panel storyboard, 15가지 cinematic shot까지 한 모델로 처리. 27B/56B 모델 초과.
4. **200B+ 확장으로 scaling law 검증**: HiDream-O1-Image-Pro로 새로운 SOTA 수립 (Nano Banana 2.0/GPT Image 2 등 closed-source 초과).
5. **UniSubject 벤치마크 공개**: 300 cases × 1.8K subjects, 2-3/4-8/9-11 multi-subject IP 평가용.

---

## 🏗️ 주요 알고리즘 설명

### 0. 단일 공유 토큰 공간 (Single Shared Token Space) — 본 모델의 본질

Transformer의 "토큰 공간"이란 **동일 차원(hidden_size=4096)의 벡터들이 사는 R^4096 공간**. HiDream-O1은 텍스트·픽셀·timestep·참조 이미지를 **모두 이 한 공간 안의 벡터로 변환**해 같은 self-attention 안에서 직접 상호작용시킴.

#### 기존 패러다임과의 비교 (모두 self-attention 기반)

| 모델 | text 인코더 | VAE | 시퀀스 구성 | Q/K/V weight |
|---|---|---|---|---|
| PixArt-α | 외부 T5 | 사용 | image만 attention, text는 별도 conditioning | image용 단일 set |
| SD3 (MM-DiT) | 외부 T5 + CLIP | 사용 | text + image concat 후 self-attention | text/image **별도 weight** (parallel streams) |
| FLUX double-stream | 외부 T5 + CLIP | 사용 | 동일 concat | 별도 weight |
| FLUX single-stream | (위와 동일) | (위와 동일) | 동일 concat | **완전 공유** |
| Qwen-Image | 외부 Qwen2.5-VL | 사용 | concat | 별도 weight |
| Z-Image | 외부 Qwen2.5-VL | 사용 | concat | 완전 공유 |
| **HiDream-O1 (UiT)** | **없음** (Qwen3-VL embed 내장) | **없음** | text + tms + image + ref **모두 concat** | 완전 공유 + **hybrid causal/full mask** |

> ⚠️ **용어 주의**: 위 모델들은 **모두 self-attention만 사용**합니다. "cross-attention DiT"라는 표현은 부정확 — Q,K,V가 같은 시퀀스에서 나오면 모두 self-attention. cross-attention 모듈은 SDXL 이전 UNet 시대의 유물.

#### HiDream-O1의 세 가지 분리 제거

1. **외부 텍스트 인코더 제거** — T5/CLIP 없이 Qwen3-VL 토크나이저 + embedding table 그대로 사용
2. **VAE 분리 제거** — raw RGB pixel을 직접 모델링 (32×32 패치 + bottleneck embed)
3. **Stream 분리 제거** — text/image/ref/timestep 모두 같은 Q,K,V weight (single-stream)

여기에 **LLM의 causal 성질을 보존하는 hybrid mask**(§3 참조)까지 더한 것이 UiT의 본질.

### 1. Unified Multimodal Tokenization

세 가지 primitive token type을 공유 공간에 매핑:

| Token | 입력 | 변환 | 코드 위치 |
|---|---|---|---|
| **Text Tokens (y)** | refined prompt (Prompt Agent 출력) | Qwen3-VL native vocabulary tokenizer | `processor.apply_chat_template` → `tokenizer.encode` |
| **Condition Tokens (c)** | reference/edit source 이미지 | **SigLip-2** encoder → learnable projection → shared space | `Qwen3VLVisionModel.forward` → `get_image_features` |
| **Generation Token (x_t)** | noisy target = t·x + (1−t)·ε | 32×32 패치 분할 → `BottleneckPatchEmbed` (PCA-style bottleneck, dim/4) | `models/pipeline.py:317` (rearrange), `qwen3_vl_transformers.py:1045` (`x_embedder`) |
| **Timestep (`<|tms_token|>`)** | diffusion timestep t | `TimestepEmbedder` (sinusoidal) → tms_token 위치 치환 | `qwen3_vl_transformers.py:1518-1523` |

**Token type 마킹** (`models/pipeline.py:65-71, 294-302`):
```
0 = AR (text/condition tokens)
1 = generation target
2 = reference image patches (in vinputs)
3 = timestep token
```

#### patchify는 압축이 아닌 **무손실 reshape**

코드 (`pipeline.py:317`): `einops.rearrange(noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)', p1=32, p2=32)`

```
[B, 3, H, W]                       [B, H/32·W/32, 3·32·32]
   = 3·H·W elements         ≡        = (HW/1024)·3072 elements
                                     ★ 총 element 수 완전히 동일
```

- VAE encoder는 **학습된 비선형 압축** (정보 손실 동반, decoder 필요)
- patchify는 **순수 view 변환** (같은 메모리 buffer를 다르게 인덱싱, 정보 손실 0)
- 따라서 추론 중 `z` 텐서는 **이미지와 동일한 정보량**을 그대로 담은 채 Transformer 안을 흐름. 끝나면 마지막에 `rearrange` 한 번으로 `[B, 3, H, W]` RGB로 복귀 — VAE decoder 불필요.

#### "VAE 없이 sequence length가 폭주하지 않는 이유 — patch_size 트릭"

| 해상도 | LDM (FLUX/SD3): VAE 8× + patch_size=2 | HiDream-O1: patch_size=32 only |
|---|---|---|
| 1024² → patches | 64×64 = **4,096** | 32×32 = **1,024** |
| 2048² → patches | 128×128 = **16,384** | 64×64 = **4,096** |
| Attention 비용 (L²) | 100% | **≈ 6%** (1/16) |

HiDream-O1은 **VAE의 spatial 축소 역할을 큰 patch_size로 대체**. 한 패치당 차원(3072)이 커지는 부담은 `BottleneckPatchEmbed`(3072 → 1024 → 4096)로 흡수.

### 2. Unified Transformer (UiT) Backbone

- **베이스**: Qwen3-VL-8B-Instruct (decoder-only). **이미지 생성 모델이 아니라 V&L understanding 모델** — weight initialization으로만 사용되고, 새 모듈을 붙여 T2I 학습으로 생성 능력을 새로 부여한다 (Q5 참조).
- **추가 모듈** (`Qwen3VLModel.__init__`, transformers.py:1033-1053):
  - `x_embedder = BottleneckPatchEmbed(patch_size=32, in_chans=3, pca_dim=hidden//4, embed_dim=hidden)`
    - 구조: `Linear(3072→1024, no bias) → Linear(1024→4096, bias)` (PCA-style bottleneck, 2-layer)
  - `t_embedder1 = TimestepEmbedder(hidden_size)`
  - `final_layer2 = FinalLayer(hidden_size, patch_size=32, out_channels=3)` → clean pixel 예측
    - 구조: **단일** `Linear(4096 → 3·32·32=3072, bias=True)` **한 줄**. LayerNorm/Activation/AdaLN 없음.
    - 초기화: `nn.init.zeros_` (DiT의 Zero-init 그대로) — 학습 초기에 noise가 그대로 통과해 안정적 fine-tune.
    - 호출: `x_pred = self.final_layer2(hidden_states)` — `adaln_input` 인자 없이 호출되어 AdaLN 경로 사용 안 함.
  - 입력측은 2-layer 압축-복원, 출력측은 1-layer 직접 펼침의 **비대칭 설계**.
- **표준 구성**: RMSNorm, **SwiGLU**, **RoPE** (3D mRoPE: text + spatial H/W)
- **시퀀스 구조** (T2I 기준):
  ```
  [text tokens] [boi] [tms_token] [vision_start] [image_patch_tokens]
                 ↑    ↑
                 (timestep slot)
  ```
- **편집/IP** 시 condition tokens (SigLip-2 임베딩)이 텍스트 앞에 prepend, 추가로 reference patches가 generation patches 뒤에 concat됨 (`pipeline.py:388`).

### 3. Hybrid Unified Attention Mask

> **전제**: 본 모델은 **순수 self-attention만 사용**한다. `Qwen3VLTextAttention.forward` (line 405-477)에서 Q, K, V가 모두 동일한 `hidden_states` 하나로부터 projection됨 (line 448-450). 텍스트→이미지 conditioning은 cross-attention 모듈이 아니라 **단일 시퀀스 concat + 아래 mask 차별화**로 구현.

핵심 코드 (`qwen3_vl_transformers.py:1577-1589`, non-flash 표준 path):
```python
causal = torch.full((T, T), -inf)
causal = torch.triu(causal, diagonal=1)   # 기본 causal
gen_positions = token_types[b].bool()      # generation tokens
causal[gen_positions, :] = 0               # gen은 모든 토큰 attend
```

Flash-attn 경로 (`_run_decoder_flash`, line 1257~)는 **2-pass index_copy** 방식:
1. Pass 1 — AR 토큰만 추출해 causal attention (`flash_attn_func(..., causal=True)`)
2. Pass 2 — 전체에 full attention (`causal=False`)
3. AR 위치는 Pass 1 결과로 덮어쓰기 (`index_copy`)

→ 결과적으로 **AR은 causal, generation은 bidirectional**. LLM의 언어 능력을 유지하면서도 이미지의 양방향 spatial 의존성을 캡처.

### 4. Overall Objective (Pretraining)

논문 §3.4 + §4.1에서 명시:
- **L_FM** (Flow Matching, image prediction):  pixel-space의 v-prediction과 등가. x_t = t·x + (1-t)·ε에 대해 모델이 clean x를 직접 예측 (코드: `x_pred`), v = (x_pred - z) / σ 계산.
- **L_LPIPS** (perceptual): 픽셀 공간 합성의 디테일 보강.
- **L_DINO** (perceptual): semantic coherence 보강.

→ pixel-space에서 fine-grained spatial detail + long-range semantic coherence를 동시에 잡기 위함 (latent DiT가 VAE에 위임하던 부분).

### 5. Inference 전체 흐름 (코드 단계별 추적)

T2I 케이스 (refs 없음, full 모델 기준) 기준 호출 그래프:

```
inference.py:main()
  └─ generate_image (pipeline.py:107)
       │
       ├─[1] build_t2i_text_sample (pipeline.py:31)
       │       → input_ids = [text, boi, tms, vision_start, img_token×N]
       │       → token_types (0=AR / 1=gen / 3=timestep)
       │       → 3D mRoPE position_ids
       │
       ├─[2] noise z = 7.5·randn → patchify [B, N, 3072]   # pipeline.py:313-317
       ├─[3] build_scheduler (pipeline.py:80-97)
       │
       └─ for step in 50:
            ├─ forward_once(cond, z, t)   ─┐
            ├─ forward_once(uncond, z, t) ─┤ (CFG, pipeline.py:380-383)
            │                              ▼
            │  _forward_generation (transformers.py:1400)
            │   ├─[5-1] text embed       : Qwen3-VL embedding table (line 1429)
            │   ├─[5-3] timestep embed   : t_emb → tms_token 자리 치환 (line 1518-1523)
            │   ├─[5-4] noise embed      : x_embedder(z) → 시퀀스에 concat (line 1529-1530)
            │   ├─[5-6] hybrid mask      : AR=causal, gen=full (line 1577-1589)
            │   ├─[5-8] decoder stack    : N layer × (self-attn + SwiGLU + deepstack 주입)
            │   └─[5-9] final_layer2     : hidden 4096 → patch 3072 (line 1613)
            │
            ├─ v_guided = -v_uncond + 5.0·(v_cond - v_uncond)   # CFG
            └─ z = sched.step(-v_guided, step_t, z)
       │
       └─[7] depatchify (pipeline.py:432-435) → PIL.Image
```

#### [1] 시퀀스 구축 (`build_t2i_text_sample`, pipeline.py:31-78)

```python
# pipeline.py:40-46
template_caption = (
    processor.apply_chat_template(messages, ...) + boi_token + tms_token * 1
)
input_ids = tokenizer.encode(template_caption, ...)

# pipeline.py:52-54
vision_tokens = torch.zeros((1, image_len)) + image_token_id   # H/32 × W/32개
vision_tokens[0, 0] = vision_start_token_id
input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)  # ★ text + image placeholder
```

생성되는 단일 1D 시퀀스:
```
[<im_start>user PROMPT <im_end>] [<boi>] [<tms>] [<vision_start>] [img_token × (image_len-1)]
                                          └─ t_embedder1 출력으로 대체될 자리
```

#### [2] 노이즈 패치화 (`pipeline.py:313-317`)

```python
noise = 7.5 * torch.randn((1, 3, H, W), ...)
z = einops.rearrange(noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)', p1=32, p2=32)
# z: [1, H/32·W/32, 3072]   ← 4096 patches × 3072-dim 각
```

#### [5-1~5-4] 단일 토큰 공간으로 통합 (`_forward_generation`, transformers.py:1429-1530)

```python
# 텍스트 embedding (4096차원)
inputs_embeds = self.get_input_embeddings()(input_ids)                # [B, T_text, 4096]

# Timestep → tms_token 자리 치환
t_emb = self.t_embedder1(timestep)                                    # [B, 4096]
tms_mask = (input_ids == self.tms_token_id)                           # tms_token_id=151673
inputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)

# 노이즈 패치 embedding + concat
vinputs_embedded = self.x_embedder(vinputs)                           # [B, N, 4096]
inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)   # ★ 최종 4096차원 단일 시퀀스
```

#### [5-6] Hybrid Mask 구축 (transformers.py:1577-1589)

```
              text_0 ... tms  img_0  img_1 ...
text_0        ✓       ✗   ✗    ✗      ✗
text_1        ✓       ✓   ✗    ✗      ✗        ← AR: causal
tms           ✓       ✓   ✓    ✗      ✗
img_0         ✓       ✓   ✓    ✓      ✓        ← gen: full
img_1         ✓       ✓   ✓    ✓      ✓
```

#### [5-8] Decoder Stack — DeepStack 주입

각 decoder layer는 RMSNorm → self-attention(mask 적용) → SwiGLU MLP. **초반 layer엔 Qwen3-VL의 deepstack visual feature가 더해짐** (vision conditioning 강화, `visual_pos_masks` + `deepstack_visual_embeds` 인자).

#### [5-9] 픽셀 패치 직접 예측 (transformers.py:1613)

```python
x_pred = self.final_layer2(hidden_states)   # [B, T, 4096] → [B, T, 3072]
# 3072 = 3·32·32  ← VAE decoder 없이 RGB 패치 직접 예측
```

#### [step] Flow Matching + CFG (pipeline.py:372-421)

```python
for step_t in sched.timesteps:
    t_pixeldit = 1.0 - step_t.float() / 1000.0
    sigma     = step_t.float() / 1000.0

    x_pred_cond   = forward_once(samples[0], z, t_pixeldit)
    v_cond        = (x_pred_cond - z) / sigma           # velocity

    if guidance_scale > 1:                              # full: 5.0
        x_pred_uncond = forward_once(samples[1], z, t_pixeldit)
        v_uncond      = (x_pred_uncond - z) / sigma
        v_guided      = v_uncond + guidance_scale * (v_cond - v_uncond)
    else:
        v_guided = v_cond

    z = sched.step(-v_guided, step_t, z, ...)           # FlowUniPC / Flash / FlowMatchEuler
```

#### [7] 최종 디코딩 (pipeline.py:432-435)

```python
img = (z + 1) / 2
img = einops.rearrange(img, 'B (H W) (C p1 p2) -> B C (H p1) (W p2)',
                       H=H//32, W=W//32, p1=32, p2=32)
return Image.fromarray(arr).convert("RGB")
```

### 6. Scheduler 3종 (코드: `pipeline.py:80-97`)

| 이름 | 클래스 | 용도 | steps | shift | guidance |
|---|---|---|---|---|---|
| `default` | `FlowUniPCMultistepScheduler` | **full** 모델 기본 (T2I/IP) | 50 | 3.0 | 5.0 |
| `flash` | `FlashFlowMatchEulerDiscreteScheduler` | **dev** 모델 기본 (T2I/IP) | 28 (`DEFAULT_TIMESTEPS`) | 1.0 | 0.0 |
| `flow_match` | `FlowMatchEulerDiscreteScheduler` | dev 모델의 **editing(ref=1)** 권장 | 28 | 1.0 | 0.0 |

**Flash scheduler** (`flash_scheduler.py:282-362`)는 SDE-style noise injection:
```python
denoised   = sample - model_output * sigma                       # x_0 예측
noise      = randn(...); 
if noise_clip_std > 0: noise = noise.clamp(±noise_clip_std·std)  # 분산 클립
sample = sigma_next·noise·s_noise + (1-sigma_next)·denoised      # 재주입
```
- `s_noise`는 step별 linear interpolation (`noise_scale_start → noise_scale_end`, 기본 7.5).
- `noise_clip_std` 기본 2.5 — 노이즈 tail outlier를 잘라 안정성 ↑.

### 7. Distillation (HiDream-O1-Image-Dev, §5)

50-step → 28-step student 학습 objective:
```
L_total = L_DMD + λ_diff · L_diff + λ_adv · L_adv
```
- **L_DMD**: teacher와 student의 trajectory 분포 일치 (Distribution Matching Distillation).
- **L_diff**: 표준 diffusion loss (보조, 진동 완화).
- **L_adv**: GAN loss — discriminator는 frozen teacher backbone의 multi-level feature를 입력으로 사용. perceptual fidelity·sharpness 보존.

### 8. Multi-reference 크기 자동 조정 (IP, `pipeline.py:198-202`)

```
K=1:        max_size = max(H,W)         (편집)
K=2:        max_size = max(H,W) × 48/64
K∈[3,4]:    max_size = max(H,W) / 2
K∈[5,8]:    max_size = max(H,W) × 24/64
K≥9:        max_size = max(H,W) / 4
```
+ SigLip-2 입력 크기 `CONDITION_IMAGE_SIZE=384`도 K에 따라 384 → 288 → 192로 축소. → 참조 수가 늘어도 시퀀스 폭주 방지.

---

## 🧪 실험 요약

### 1. Text-to-Image (Table 1-3)

| Benchmark | HiDream-O1 8B | HiDream-O1-Pro 200B+ | 2위 모델 | 비고 |
|---|---|---|---|---|
| **GenEval** Overall | **0.90** | **0.92** | GPT Image 2 0.89, FLUX.2 Dev 0.87 | Position 0.93 (8B) — 최고 수준 |
| **DPG-Bench** Overall | **89.83** | **90.30** | Qwen-Image 88.32 | Global 95.15 (8B) — 압도적 |
| **HPSv3** All | 10.37 | **10.47** | GPT Image 2 10.21 | 12개 카테고리 전반에서 1-2위 |

→ 8B 단독으로 27B Qwen-Image / 56B FLUX.2 Dev를 능가. closed-source인 Seedream-4.0 / GPT Image 2도 일부 항목에서 추월.

### 2. Text Rendering (Table 4-5)

| Benchmark | 8B | Pro 200B+ | 의의 |
|---|---|---|---|
| **CVTG-2K** average | <u>0.9128</u> | **0.9222** | 다영역 중국어/영어 텍스트 모두 강함 |
| **LongText-Bench EN** | 0.979 | **0.982** | Nano Banana 2.0 0.980 동급 |
| **LongText-Bench ZH** | <u>0.978</u> | **0.980** | open-weights 최강. Qwen-Image 0.946, FLUX.2 Dev 0.757 압도 |

→ **VAE 제거로 텍스트 high-freq 디테일 보존** 효과가 직접 드러나는 영역.

### 3. Image Editing (Table 6-7)

| Benchmark | HiDream-O1 8B | Pro 200B+ | 1위 비교 |
|---|---|---|---|
| **GEdit** Q-O | 7.60 | **7.67** | GPT Image 2 7.67 동급 |
| **ImgEdit** Overall | 4.14 | <u>4.51</u> | GPT Image 2 **4.73** |

→ 8B로 16.8B FLUX.1 Kontext / 27B Qwen-Image-Edit과 동급 이상.

### 4. Subject-driven Personalization — UniSubject (Table 8)

| 구성 | 2-3 Subjects (Q-O / HPSv3) | 4-8 (Q-O / HPSv3) | 9-11 (Q-O / HPSv3) |
|---|---|---|---|
| Qwen-Image-Edit (27B) | 7.50 / 8.84 | 5.34 / 5.40 | 2.71 / 2.13 |
| Scone (7B) | 7.15 / 8.97 | 6.01 / 6.62 | 5.78 / 7.54 |
| Echo-4o (14B, GPT-4o distilled) | 7.46 / 9.99 | 7.19 / 8.61 | 6.73 / 8.78 |
| **HiDream-O1 8B** | 7.95 / 10.45 | <u>7.47</u> / 9.53 | 7.65 / 9.78 |
| **HiDream-O1-Pro 200B+** | 8.50 / 11.05 | **7.99** / **9.76** | **7.92** / **9.83** |

→ 9-11 subjects 같은 극단적 multi-subject에서도 성능 유지. shared token space가 다수 reference 간 disjoint encoder의 간섭을 완화한다고 주장.

### 5. Cinematic 제어 (§6.3)

다음을 native로 지원:
- **Shot scales** (7종): extreme full / full / medium full / medium / medium close-up / close-up / extreme close-up
- **Camera angles** (4종): high / low / eye-level / bird's-eye
- **Subject orientations** (4종): front / side / back / three-quarter
- **Multi-panel storyboard** (single-pass 생성)

→ video pre-production용 keyframe 생성에 적합하다고 주장.

---

## 💬 Q&A 섹션

> 이번 분석 세션에서 사용자가 실제로 제기한 질문들. 본문에 흡수된 부분은 cross-reference, 본문에 없는 추가 사항만 자세히 답함.

### Q1. "단일 공유 토큰 공간 (single shared token space)"이 정확히 무슨 의미인가?

→ 자세한 설명은 알고리즘 §0 참조.

**짧게**: Transformer의 hidden_size(=4096) 차원 벡터 공간 하나에 텍스트·픽셀 패치·timestep·참조 이미지를 **모두 매핑한 뒤 같은 self-attention에서 직접 상호작용**시키는 것. 기존엔 텍스트 인코더 공간/VAE latent 공간이 따로 있고 self-attention block의 입력 단계에서 concat되거나 모듈로 이어졌다면, HiDream-O1은 그 분리를 세 가지 차원에서 모두 제거:
- 외부 텍스트 인코더 제거 (T5/CLIP 없음)
- VAE 제거 (raw RGB 직접)
- Stream 분리 제거 (text/image 동일 Q,K,V weight)

### Q2. DiT인데 cross-attention을 쓰는가?

**아니다. self-attention만 사용함.** 코드 전체에서 `cross-attention` 키워드는 0건이며 `Qwen3VLTextAttention.forward` (line 405-477)의 Q,K,V는 모두 동일한 `hidden_states` 하나에서 projection됨 (line 448-450):

```python
query_states = self.q_norm(self.q_proj(hidden_states)....)
key_states   = self.k_norm(self.k_proj(hidden_states)....)
value_states =             self.v_proj(hidden_states)....
```

텍스트→이미지 conditioning은 **단일 시퀀스 concat + hybrid mask** 로 구현됨 (§3 참조). PixArt 이후 DiT 계열은 모두 self-attention 기반이며, 차이는 stream 수와 mask 구조뿐 (§0의 비교표 참조). **"cross-attention DiT"라는 표현 자체가 부정확** — cross-attention 모듈은 SDXL 이전 UNet 시대의 유물.

### Q3. 추론 코드의 전체 흐름은?

→ 알고리즘 §5 ("Inference 전체 흐름") 참조. `inference.py:main` → `generate_image (pipeline.py:107)` → `_forward_generation (transformers.py:1400)` → `final_layer2` 까지 단계별 호출 그래프, 각 단계의 코드 스니펫, hybrid mask 시각화, depatchify까지 포함.

### Q4. 네트워크 구조가 MM-DiT (SD3) 와 동일한가?

**전혀 다르다.** 가장 큰 차이는 **MM-DiT는 처음부터 DiT로 설계된 새 네트워크**인 반면, **HiDream-O1은 사전학습된 LLM(Qwen3-VL)의 decoder block을 그대로 사용**하고 픽셀 입출력만 붙였다는 점.

코드로 확인 — `Qwen3VLTextDecoderLayer.forward` (line 519-538)는 **AdaLN-Zero modulation이 한 줄도 없는 평범한 pre-norm LLM block**:

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)              # RMSNorm only
hidden_states, _ = self.self_attn(...)                            # ★ AdaLN 없음
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                          # SwiGLU
hidden_states = residual + hidden_states
```

| 항목 | MM-DiT (SD3 / FLUX double-stream) | **HiDream-O1 (UiT)** |
|---|---|---|
| **Backbone 출처** | scratch-trained DiT | **사전학습 LLM** (Qwen3-VL-8B-Instruct) 그대로 재사용 |
| **Block 종류** | MM-DiT block (전용 설계) | **Qwen3VLTextDecoderLayer** (LLM block 그대로) |
| **Stream 수** | **2개** (text/image stream, separate Q,K,V weight) | **1개** (single-stream, weight 공유) |
| **Timestep modulation** | **AdaLN-Zero**: 모든 block에 timestep + pooled text로 scale/shift/gate 주입 | **AdaLN 없음**. `<\|tms_token\|>` 자리 단 1개 토큰 embedding에 삽입 (LLM-style) |
| **Normalization** | LayerNorm (AdaLN-modulated) | RMSNorm (modulation 없음) |
| **Position Encoding** | 2D RoPE (image) + 1D positional (text) | **3D mRoPE** (text + spatial H + spatial W) |
| **Attention mask** | full (모두가 모두를 봄) | **Hybrid**: AR=causal, gen=full |
| **추가 conditioning** | pooled CLIP text → AdaLN | **DeepStack visual embeds** (Qwen3-VL 다층 vision feature를 초반 layer에 직접 더함) |
| **Final head** | AdaLN + linear → VAE latent patch | linear → 3·32·32 RGB patch 직접 |

**핵심 발상의 차이**:
- **MM-DiT**: "DiT를 multimodal로 만들자" → text/image stream 분리, AdaLN으로 timestep modulate
- **UiT**: "LLM이 이미 multimodal alignment를 학습했다 → block 구조에 손대지 않고 입출력만 픽셀로 바꿈"

### Q5. Qwen3-VL-8B-Instruct는 vision-language **understanding** 모델인데, 이걸로 어떻게 이미지를 **생성**하는가?

정확한 지적. Qwen3-VL은 원래 [text+image] → text 출력 모델로, **이미지를 생성하지 못한다**. HiDream-O1은 이걸 **weight initialization으로만 사용**하고 새 모듈 + 새 학습으로 생성 능력을 부여:

**(1) 무엇을 가져오고, 무엇을 새로 붙였나** (`Qwen3VLModel.__init__`, transformers.py:1033-1053):

| 모듈 | 출처 | 역할 |
|---|---|---|
| `visual` (SigLip-2) | Qwen3-VL 그대로 | 참조 이미지를 hidden vector로 인코딩 (이해 능력) |
| `language_model` decoder stack | Qwen3-VL 그대로 | self-attention + SwiGLU MLP block들 |
| `x_embedder` (BottleneckPatchEmbed) | **신규** | RGB 패치(3·32·32=3072) → hidden(4096) |
| `t_embedder1` (TimestepEmbedder) | **신규** | diffusion timestep → hidden(4096) |
| `final_layer2` (FinalLayer) | **신규** | hidden(4096) → RGB 패치(3072) 예측 (원래는 token logits 출력) |
| `tms_token` (id=151673) | **신규 special token** | timestep 자리 표시자 |

**(2) forward 경로도 새로 작성**: `_forward_generation` (transformers.py:1400-1621) 함수 자체가 Qwen3-VL에는 없는 신규 코드. denoising step에 맞춰 timestep 처리, noise patch embed, hybrid mask 구축, RGB patch 출력까지 모두 새 경로.

**(3) Stage I부터 3-task joint 학습으로 생성 능력 부여** (§4.1):
- **T2I** (새 능력): pixel patch ↔ text 매핑 학습
- **LM** (기존 유지): text-only corpus로 언어 능력 보존
- **MMU** (기존 유지): image+text → text understanding 유지

→ 즉 **"Qwen3-VL이 이미지를 생성한다"는 표현은 부정확**, 정확히는 "**Qwen3-VL의 사전학습된 vision-language alignment를 출발점으로 삼아 픽셀 입출력 모듈을 붙이고 T2I 학습으로 이미지 생성 능력을 새로 부여한 모델**". 이 init 전략 덕에 scratch 학습 대비 빠르게 수렴하며, 8B로 27B Qwen-Image (이쪽도 VLM init) 를 이기는 비결 중 하나.

### Q6. 이건 diffusion 모델인가? (backbone이 LLM인데?)

**예, flow matching 기반 diffusion 모델이 맞다.** backbone 출처가 LLM(Qwen3-VL)일 뿐, 학습/추론 paradigm은 완전히 diffusion.

코드/논문에서의 증거 4가지:
1. **Forward process**: `x_t = t·x + (1-t)·ε` (flow matching의 linear interpolation)
2. **Iterative denoising loop** 50 / 28 step (`pipeline.py:372-421`)
3. **Flow Matching scheduler** 3종 사용 (FlowUniPC / FlashFlowMatchEuler / FlowMatchEuler)
4. **DMD distillation** (`L_total = L_DMD + λ_diff·L_diff + λ_adv·L_adv`) — diffusion 전용 distillation 기법

다른 통합 multimodal 모델과의 분류상 위치:

| paradigm | 모델 |
|---|---|
| **Autoregressive** (이미지를 discrete token으로 한 토큰씩 예측) | GPT-4o, Janus-Pro, Show-o, Emu3 |
| **Diffusion** (continuous space에서 반복 denoising) | FLUX, SD3, Qwen-Image, Z-Image, **HiDream-O1** |

→ "backbone은 LLM, paradigm은 diffusion"인 하이브리드. GPT의 weight를 가져와 그 위에 diffusion 학습을 다시 한 것과 비슷.

### Q7. VAE decoder 없이 어떻게 RGB 이미지가 나오는가?

핵심: **patchify는 압축이 아니라 무손실 reshape**이라 decoder가 필요 없음. 자세한 비교는 알고리즘 §1의 "patchify는 압축이 아닌 무손실 reshape" 표 참조.

**추론 중 `z`의 정체**:
- 시작: `z = patchify(pure noise)` — shape `[B, N, 3072]`. 원본 `[B, 3, H, W]`와 **element 수 완전히 동일** (단순 view 변환).
- 매 step: `z = sched.step(-v_guided, t, z)` — z를 noise → clean 방향으로 한 step 이동.
- 끝: z가 정규화된 RGB pixel 값 `[-1, 1]`에 도달.

**최종 변환** (`pipeline.py:432-435`):
```python
img = (z + 1) / 2                                  # [-1,1] → [0,1]
img = einops.rearrange(img, 'B (H W) (C p1 p2) -> B C (H p1) (W p2)', ...)
arr = (img.numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)
return Image.fromarray(arr)
```

`einops.rearrange`는 같은 메모리 buffer를 다르게 인덱싱하는 view 변환 — 학습된 모듈 0개, 비선형 변환 0개. **z는 추론 내내 "이미지 정보 그대로의 텐서"** 였고, 마지막에 인덱싱 방식만 4D로 바꾸면 곧장 RGB.

### Q8. VAE 없이 raw pixel을 직접 모델링하면 계산량/메모리 폭주하지 않는가?

**오히려 attention 측면에선 LDM보다 가볍다.** VAE의 spatial 축소 역할을 **patch_size=32**가 대신 수행하기 때문.

| 해상도 | LDM (VAE 8× + patch_size=2) | HiDream-O1 (patch_size=32) | Attention 비용 비 |
|---|---|---|---|
| 1024² | 4,096 patches | 1,024 patches | **HiDream 1/16** |
| 2048² | 16,384 patches | 4,096 patches | **HiDream 1/16** |

Transformer attention은 `O(L²·d)` (L=sequence length). HiDream-O1은 **patch당 면적이 16배 커서 patch 수가 16배 적음** → attention 비용 16배 적음.

**Trade-off**:
- ✅ Attention 효율 ↑ (sequence length ↓)
- ✅ VAE encode/decode 비용 0
- ✅ z 자체에 정보 손실 없음 → 텍스트 high-freq 디테일 보존 (LongText-Bench 0.979의 비결)
- ❌ z의 메모리 풋프린트는 12× 큼 (이미지 크기 그대로)
- ❌ 한 패치당 차원이 큼 (3072) → patch embedder/final layer 비용 ↑ (단, layer당 한 번이라 attention 대비 무시할 수준)
- ❌ raw RGB는 high-freq 정보 밀도가 높아 모델 capacity 부담 → **8B backbone, BottleneckPatchEmbed, LPIPS+DINO loss**로 흡수

→ **"VAE = 정보 압축"** 보다 본질은 **"transformer가 처리할 sequence length 단축"**. HiDream-O1은 그 수단을 VAE 대신 큰 patch_size로 대체한 것.

---

### Q9. Qwen3-VL 가중치 로드 없이 같은 구조를 스크래치로 학습해도 성공할 수 있나?

**결론: HiDream 규모의 데이터·컴퓨트 예산으로는 실패할 확률이 매우 높음.** Qwen3-VL-8B 초기화가 사실상 학습의 ~90%를 미리 끝내놓은 것이기 때문.

#### 1. 학습 부담의 비대칭

| 항목 | VLM init (HiDream 실제) | From scratch |
|---|---|---|
| 학습해야 할 것 | "VLM이 픽셀 토큰도 만들도록" 미세조정 | 언어 이해 + 비전 + 생성 + 정렬 동시 |
| 사전 학습 데이터 | Qwen3 학습 데이터(수조 토큰) **공짜** | 0 |
| 추정 GPU-days | ~수천 | 수십~수백만 (10~100배) |

#### 2. 구조적 의존성: 텍스트 인코더가 별도로 없다
HiDream은 **별도 텍스트 인코더가 없는** 구조. Qwen3-VL backbone 자체가 텍스트 인코더 역할까지 함. 스크래치로 가면 한 통에서 **(a) T2I 생성기 + (b) 텍스트 인코더 + (c) VLM 정렬**을 동시에 만들어야 함. SD3·FLUX·Z-Image도 텍스트 인코더는 사전학습 T5/CLIP을 쓰지, 완전 스크래치는 안 함.

#### 3. In-context reasoning 손실
논문 핵심 기여 중 하나인 **Reasoning Prompt Agent + 인스트럭션 기반 편집**은 VLM의 chain-of-thought·instruction-following 능력을 그대로 활용한 결과. 스크래치 학습은 이 능력을 0에서 출발 — 디퓨전 손실만으로 절대 학습되지 않음.

#### 4. 텍스트 렌더링
SOTA-급 텍스트 렌더링(LongText-Bench 0.979)은 Qwen3-VL이 가진 **문자 의미·철자 지식**을 픽셀로 옮긴 것. 스크래치로 "코카콜라 로고"의 정확한 철자를 그리려면 수십억 장의 OCR-aligned 데이터가 필요.

#### 5. 학습 안정성
8B decoder-only Transformer 스크래치 학습은 LLM 학습과 동급 난이도. Gradient 폭주·mode collapse·warm-up 튜닝이 동시 발생. 사전학습 가중치는 이 전부를 우회시키는 안정된 출발점.

#### 그럼 이론적으론?
**가능은 함**. 단지 자원이 10~100배 필요.
- **DALL-E 1 / Parti**: 사실상 스크래치 (텍스트 인코더만 제외). 수만 TPU-days 투입.
- **PixArt-α/σ**: 픽셀 DiT 스크래치, 텍스트 인코더는 사전학습 T5 사용.
- **HiDream의 선택**: 학습 코드/데이터 비공개 이유 중 하나가 "**우리 예산엔 VLM init이 필수**"라는 노하우 자체가 핵심 자산이기 때문일 가능성이 큼.

#### 한 줄 정리
> HiDream의 아키텍처는 **사전학습된 VLM이 있다는 전제 위에 설계된 구조**. Qwen3-VL을 빼면 "디퓨전을 붙일 LLM"이 없어지므로, 같은 컴퓨트·데이터로는 수렴 자체가 안 되거나 PixArt-α 수준에도 못 미칠 가능성이 높음. 이 모델의 본질은 **"Qwen3-VL을 픽셀 디퓨저로 fine-tune한 결과물"**이며, HiDream의 진짜 기여는 **adapter 부분(patch/timestep embedder, hybrid attention, Reasoning Agent)**.

---

## 📌 한 줄 요약 (전체)

**Qwen3-VL-8B을 backbone으로 raw pixel·text·condition을 한 시퀀스에서 처리하는 decoder-only Unified Transformer + hybrid causal/full attention + Reasoning-Driven Prompt Agent + 3-stage progressive pretraining + GRPO RLHF + DMD/GAN distillation으로, VAE도 별도 텍스트 인코더도 없이 T2I/편집/IP/장문 텍스트/2048² native까지 8B로 처리하는 통합 이미지 생성 파운데이션 모델.**

---

## 🔗 관련 메모리

- [[paper_z_image]] — 6B Single-Stream DiT, 314K H800h SOTA. **공통점**: 작은 8B/6B 규모로 27B+ 거대 모델 추월. **차이점**: Z-Image는 latent space + decoupled-DMD + on-policy RL, HiDream-O1은 pixel space + UiT + GRPO + reasoning agent.
- [[feedback-paper-summary-format]] — 본 문서 작성 규칙 (PAPER_*.md 형식, 5원칙).
