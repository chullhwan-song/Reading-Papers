# Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks

> FLUX.1-dev 의 RoPE(회전 위치 임베딩) 3채널을 의미 채널로 재해석하여, mask·pose·parsing 같은 보조 입력 없이도 가상 피팅(try-on) / 옷 추출(garment reconstruction) / 옷 입은 사람 합성(model-free try-on) 세 가지를 하나의 LoRA 모델로 처리하는 통합 프레임워크.

## 메타 정보

| 항목 | 내용 |
|---|---|
| 제목 | Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks |
| 저자 | Hailong Guo, Bohan Zeng, Yiren Song, Wentao Zhang, Chuang Zhang, Jiaming Liu |
| 공개일 | 2025-01 (ICCV 2025 채택) |
| arXiv | [2501.15891](https://arxiv.org/abs/2501.15891) · [ar5iv](https://ar5iv.labs.arxiv.org/html/2501.15891) |
| 코드 | [logn-2024/Any2anyTryon](https://github.com/logn-2024/Any2anyTryon) |
| 프로젝트 | [logn-2024.github.io/Any2anyTryon](https://logn-2024.github.io/Any2anyTryon/) |
| 체크포인트 | [loooooong/Any2anyTryon](https://huggingface.co/loooooong/Any2anyTryon) (LoRA 4종) |
| 데이터셋 | [LAION-Garment](https://huggingface.co/datasets/loooooong/LAION-Garment) |
| 분야 | Virtual Try-On (VTON), Reference-driven Image Generation, DiT |
| 베이스 모델 | FLUX.1-dev (12B parameter, Flow Matching DiT) |
| 외부 도구 사용 | AutoMasker (CatVTON 전처리), FLUX-Controlnet-Inpainting, GPT-4o (데이터 큐레이션), Florence2 (캡션 생성) |

---

## 주요 용어 사전 (Glossary)

> *왜?* 분야가 좁고 약어가 많아, 본문에 들어가기 전에 핵심 용어를 한 곳에 모아두면 따라가기 쉽다.

### 가상 피팅 태스크 분류
- **VTON (Virtual Try-On)**: 가상 피팅. 사람 사진 + 옷 사진 → 그 옷을 입은 사람 사진을 생성.
- **Garment Reconstruction (옷 추출)**: 옷 입은 사람 사진 → 옷만 분리해 평면 이미지로 복원. "TryOff" 라고도 함.
- **Model-free Try-On**: 옷 사진만 주고, 그 옷을 입을 사람을 모델이 알아서 합성.
- **Paired vs Unpaired**: VITON-HD 평가 설정. paired 는 정답 사진이 있는 경우 (LPIPS/SSIM 계산 가능), unpaired 는 다른 옷-사람 조합으로 정답이 없는 경우 (FID/KID 만 가능).

### 위치 임베딩 / 모델 구조
- **RoPE (Rotary Position Embedding)**: 위치를 회전 변환으로 인코딩해 어텐션에 상대 위치 정보를 주는 기법. FLUX 는 토큰마다 (channel0, y, x) 3차원 좌표를 받는다.
- **MM-DiT (Multi-Modal DiT)**: 텍스트와 이미지 토큰을 한 시퀀스에서 함께 어텐션 처리하는 FLUX·SD3 의 transformer 블록.
- **Adaptive Position Embedding**: 본 논문의 핵심. RoPE 의 channel0 을 "이 토큰이 타겟인지 / 공간 조건인지 / 객체 조건인지" 알려주는 역할 마커로 재활용한다.
- **condspa (condition_spatial)**: 공간 정렬 조건. 타겟과 같은 x 좌표를 공유해 "이 자리에 있던 것" 정보를 준다. 예: 옷을 갈아입힐 사람 사진.
- **condsub (condition_subject)**: 참조 객체 조건. 타겟 오른쪽으로 x 가 이어지는 별개 패널. 예: 입힐 옷의 평면 사진.
- **Clean Latent**: 조건 영역에 노이즈를 안 더하고 VAE 인코딩 그대로 시퀀스에 끼우는 전략. IC-LoRA 와의 차이점.

### 비교 기법 (선행 연구)
- **CatVTON**: 마스크 기반 SD inpainting 으로 옷을 합성. 픽셀 정확도 SOTA.
- **IDM-VTON / OOTDiffusion**: 마스크 + 포즈/파싱 + ReferenceNet 으로 옷 특징을 주입.
- **StableGarment / Magic Clothing**: 옷만 주고 사람을 합성 (model-free).
- **TryOffDiff**: 옷 추출 (TryOff) 전용 베이스라인.
- **IC-LoRA (In-Context LoRA)**: 조건을 노이즈로 흐려서 학습. 본 논문이 비판하는 접근.

### 평가 지표
- **SSIM / MS-SSIM / CW-SSIM**: 구조적 유사도. 높을수록 좋음.
- **LPIPS**: 학습된 특징 공간에서의 지각 거리. 낮을수록 좋음.
- **FID / CLIP-FID**: 생성 분포와 실제 분포의 거리. 낮을수록 좋음.
- **KID**: FID 의 unbiased 추정치. 본 논문에서는 × 1000 배 한 값으로 표기.
- **DISTS**: 구조+텍스처 결합 지각 거리. 낮을수록 좋음.
- **MP-LPIPS**: Model-Pose LPIPS. 자세 일관성 평가.
- **FFA**: Fashion Faithfulness Assessment. 옷 충실도.

### 학습 관련
- **Flow Matching**: x_t = (1-σ)·x_0 + σ·noise 직선 경로 위에서 velocity 를 학습하는 방식. FLUX 의 기본 학습 목표.
- **LoRA (Low-Rank Adaptation)**: 가중치 행렬에 저랭크 보정 행렬을 더해 일부만 학습. 본 논문은 attention 모듈에만 적용.

---

## 논문 요약 (TL;DR)

**한 줄**: FLUX 의 RoPE 3채널 중 잉여 채널을 "토큰의 역할 마커"로 쓰는 단 한 줄짜리 트릭으로, mask 없이 통합 VTON 모델을 만들었다.

**핵심 문제**:
- 기존 VTON 은 try-on / 옷 추출 / 사람 합성이 각각 별도 모델로 학습돼야 했음
- 거의 모든 방법이 mask, pose, parsing 같은 보조 입력에 의존
- "사람-옷" 페어 데이터가 부족해 일반화가 약함

**해결책 (3가지를 한 번에)**:
1. **Adaptive Position Embedding**: RoPE 의 channel0 을 0(타겟) / 1(공간 조건) / 2(객체 조건)으로 사용. condspa 는 타겟과 x 좌표를 공유해 픽셀 정렬, condsub 는 타겟 오른쪽으로 x 가 이어지는 별도 패널로 배치 → 모델이 어텐션에서 자연스럽게 역할을 구분
2. **Clean Latent 조건**: 조건 영역은 노이즈 없이 깨끗한 VAE latent 로 시퀀스에 concat. background 누설을 막음
3. **LAION-Garment**: AutoMasker + FLUX-Inpaint + GPT-4o 필터로 6만+ 트리플 합성 데이터셋 구축

**검증**:
- 옷 추출 (Table 2): SSIM 0.805, LPIPS 0.328, FID 13.367 으로 TryOffDiff 대비 SOTA
- Model-free (Table 3): CLIP-I 0.789 등 전부 1위
- 표준 try-on (Table 4, VITON-HD): unpaired FID/KID 1위(8.965/0.981), paired LPIPS/SSIM 은 CatVTON 에 약간 밀림 (unified 트레이드오프)

---

## 핵심 기여 (Contributions)

1. **위치 임베딩의 의미 채널 재해석**: FLUX 의 3채널 RoPE 에서 사용되지 않던 채널을 "토큰 역할 ID" 로 사용. 새로운 모듈을 추가하지 않고 단일 시퀀스 처리로 mask 없는 통합 try-on 을 가능케 함.

2. **spatial vs subject 조건의 좌표 전략 분리**:
   - condspa (공간 조건): 타겟과 같은 x 좌표 → 픽셀 정렬된 참조
   - condsub (객체 조건): 타겟 오른쪽으로 x 가 offset → 별개 reference 객체
   - 이 차이가 "옷이 어디 있는지"와 "어떤 옷인지"를 모델이 분리해서 학습하게 만듦.

3. **Clean Latent 조건 학습**: IC-LoRA 와 달리 조건에 노이즈를 더하지 않음. Ablation 에서 이 설계가 가장 큰 영향을 미친다는 점이 정량적으로 확인됨.

4. **하나의 LoRA, 네 가지 태스크**: try-on / garment reconstruction / model-free try-on / multi-garment 를 단일 체크포인트 + 다른 프롬프트로 처리. 태스크 특화 LoRA 도 같이 공개.

5. **LAION-Garment 합성 파이프라인 공개**: AutoMasker → FLUX-Inpaint → GPT-4o 필터 → Florence2 캡션 의 자동화된 데이터 증강 워크플로우와 6만+ 트리플 데이터셋을 공개.

---

## 주요 알고리즘 설명

> *왜?* 본 논문의 모든 기여가 결국 "위치 임베딩을 어떻게 부여하는가" 하나로 수렴하므로, 그 구조를 코드와 함께 정확히 이해하는 것이 가장 중요하다.

### 1. 입력 토큰 배치 (Spatial Layout)

타겟·조건들을 가로로 이어붙여 한 시퀀스로 만든다.

```
픽셀 공간:
+------------------+------------------+------------------+
|  타겟 영역        |   condspa        |   condsub        |
|  (마스크 = 1)     |   (모델 사진)     |   (옷 사진)       |
|  노이즈 latent    |   clean latent   |   clean latent   |
+------------------+------------------+------------------+
   width = W_tgt      width = W_spa     width = W_sub
```

- 마스크는 타겟 영역만 1, 조건 영역은 0
- inpainting 형식을 빌렸지만, 조건 영역은 "흐릿한 부분이 아니라 클린 latent" 라는 점이 핵심

코드: [`infer.py:81-103`](https://github.com/logn-2024/Any2anyTryon/blob/main/infer.py#L81-L103)
```python
concat_image_list = [Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))]
if has_model_image: concat_image_list.append(model_image)    # condspa
if has_garment_image: concat_image_list.append(garment_image) # condsub
image = Image.fromarray(np.concatenate([np.array(img) for img in concat_image_list], axis=1))
mask = np.zeros_like(np.array(image))
mask[:, :width] = 255  # 타겟 영역만 마스킹
```

### 2. Adaptive Position Embedding 공식

각 토큰에 부여되는 RoPE 좌표를 `[w, y, x]` 3차원으로 본다.

| 영역 | w (역할 ID) | y | x |
|---|---|---|---|
| 타겟 | 0 | 0 ~ H | 0 ~ W_tgt |
| condspa | 1 | 0 ~ H | **0 ~ W_spa** (타겟과 동일) |
| condsub | 2 | 0 ~ H | **W_tgt ~ W_tgt + W_sub** (타겟 옆으로 이어붙음) |

수식으로 표현하면, 조건 종류에 따른 x 좌표 결정은:
$$
x'_k = \begin{cases} x_k & \text{if pixel-aligned (condspa)} \\ x_k + W_{tgt} & \text{otherwise (condsub)} \end{cases}
$$

그리고 RoPE 회전 변환은 표준식:
$$
R_{\Theta,p}^d(q) = q \odot \cos(p\theta) + \text{rot}(q) \odot \sin(p\theta)
$$

최종 위치 임베딩은 세 채널의 회전을 concat:
$$
E_{pos}(q) = \text{concat}[R(w);\; R(y);\; R(x)]
$$

코드: [`src/utils.py:92-118`](https://github.com/logn-2024/Any2anyTryon/blob/main/src/utils.py#L92-L118)
```python
def prepare_latent_image_ids(height, width_tgt, height_spa, width_spa, height_sub, width_sub, device, dtype):
    # 타겟: w=0, y=0..H, x=0..W_tgt
    latent_image_ids = torch.zeros(height, width_tgt, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] += torch.arange(height)[:, None]
    latent_image_ids[..., 2] += torch.arange(width_tgt)[None, :]

    cond_mark = 0
    if width_spa > 0:  # condspa: w=1, x 는 타겟과 동일 좌표 공간
        cond_mark += 1
        condspa_image_ids = torch.zeros(height_spa, width_spa, 3)
        condspa_image_ids[..., 0] = cond_mark
        condspa_image_ids[..., 1] += torch.arange(height_spa)[:, None]
        condspa_image_ids[..., 2] += torch.arange(width_spa)[None, :]  # 0 부터 시작

    if width_sub > 0:  # condsub: w=2, x 는 타겟 오른쪽으로 offset
        cond_mark += 1
        condsub_image_ids = torch.zeros(height_sub, width_sub, 3)
        condsub_image_ids[..., 0] = cond_mark
        condsub_image_ids[..., 1] += torch.arange(height_sub)[:, None]
        condsub_image_ids[..., 2] += torch.arange(width_sub)[None, :] + width_tgt  # ★ offset

    return torch.cat([latent_image_ids, condspa_image_ids, condsub_image_ids], dim=-2)
```

**왜 이게 작동하나**: 어텐션은 query·key 의 위치 RoPE 회전 후 내적을 계산한다. condspa 토큰은 타겟의 같은 (y, x) 위치 토큰과 RoPE 위상이 거의 같으므로(채널0만 다름) "같은 자리"라는 신호가 강하게 전달된다. condsub 는 타겟 오른쪽으로 자연스럽게 이어진 별개 패널처럼 인식돼 "참조 객체"로 처리된다.

### 3. Clean Latent 조건 (학습/추론 공통)

조건 영역은 노이즈를 절대 더하지 않는다. 추론 루프에서도 매 step 강제한다.

코드 (학습, [`train_lora_any2any.py:975-995`](https://github.com/logn-2024/Any2anyTryon/blob/main/train_lora_any2any.py#L975-L995)):
```python
noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise  # 타겟만 노이즈 부여
packed_noisy_model_input = flux_pack_latents(noisy_model_input, ...)
target_length = packed_noisy_model_input.shape[1]

if has_cond_spa:
    packed_condspa_model_input = flux_pack_latents(condspa_input, ...)  # ★ 노이즈 없음
    packed_noisy_model_input = torch.concat([packed_noisy_model_input, packed_condspa_model_input], dim=-2)

if has_cond_sub:
    packed_condsub_model_input = flux_pack_latents(condsub_input, ...)  # ★ 노이즈 없음
    packed_noisy_model_input = torch.concat([packed_noisy_model_input, packed_condsub_model_input], dim=-2)
```

코드 (추론, [`src/pipeline_tryon.py:371-373`](https://github.com/logn-2024/Any2anyTryon/blob/main/src/pipeline_tryon.py#L371-L373)):
```python
init_latents_proper = image_latents
init_mask = mask
latents = (1 - init_mask) * init_latents_proper + init_mask * latents
# 매 step 마다 조건 영역은 GT clean latent 로 덮어쓴다
```

**Ablation 에서 이 설계가 가장 영향력 큼** — 7장 참조.

### 4. 학습 손실 함수

표준 Flow Matching loss 인데, **타겟 토큰에만 적용**한다.

$$
\mathcal{L} = \mathbb{E} \left[\, w(\sigma) \cdot \| v_\theta(z_t)[:T] - (\text{noise} - x_0) \|_2^2 \,\right]
$$

여기서 `[:T]` 는 시퀀스의 앞부분(= 타겟 길이 `target_length`)만 잘라낸다는 의미. 코드:
```python
model_pred = model_pred[:, :target_length, :]   # ★ 타겟 토큰만
target = noise - model_input
loss = torch.mean((weighting * (model_pred - target) ** 2)...)
```

### 5. 학습 설정

| 항목 | 값 |
|---|---|
| 베이스 모델 | FLUX.1-dev (12B, frozen) |
| 학습 가중치 | LoRA only |
| LoRA target modules | attn.to_q/k/v/out, attn.add_q/k/v_proj/to_add_out (FFN 제외) |
| LoRA rank | argparse 기본값 (`--rank`, 사용자 설정) |
| 옵티마이저 | AdamW (또는 8bit AdamW) |
| 학습 해상도 | 384~896, 16 의 배수로 랜덤 (try-on 페어는 768 cap) |
| 텍스트 인코더 | CLIP + T5 (T5 는 8bit 양자화) |
| Mixed Precision | bf16 |
| 학습 데이터 | LAION-Garment 트리플 (target / condspa / condsub) |
| Timestep weighting | logit-normal (SD3 과 동일) |

### 6. 추론 변형 (LoRA 4종)

체크포인트 이름이 곧 사용 시나리오를 의미한다.

| LoRA 이름 | 용도 |
|---|---|
| `dev_lora_any2any_alltasks.safetensors` | 통합 (가장 일반적) |
| `dev_lora_any2any_tryon.safetensors` | 표준 try-on 특화 |
| `dev_lora_garment_reconstruction.safetensors` | 옷 추출 전용 |
| `dev_lora_any2any_multi.safetensors` | 여러 옷 동시 처리 |

추론 호출 예 (`prompt` 안에 `<MODEL>`, `<GARMENT>`, `<TARGET>` 토큰으로 어떤 영역이 무엇인지 자연어로 명시):
```bash
python infer.py --model_image ./asset/images/model/model1.png \
                --garment_image ./asset/images/garment/garment1.jpg \
                --prompt "<MODEL> a man <GARMENT> t-shirt with pockets <TARGET> man wearing the t-shirt"
```

---

## 데이터 파이프라인: LAION-Garment

> *왜?* 데이터 부족이 VTON 의 가장 큰 병목이었는데, 본 논문은 합성 트리플로 이를 우회한다. 파이프라인 자체가 재현·확장 가치가 있다.

### 소스 데이터
- **VITON-HD**: 11,491 pairs
- **DressCode**: 49,930 pairs (upper/dress/lower)
- **DeepFashion2**: 988 pairs
- **LRVS-Fashion**: 패션 이미지 추가
- **웹 크롤링**: 고품질 페어 보충

### 5단계 합성
1. **마스크 생성**: CatVTON 의 AutoMasker 로 사람 사진에서 상/하/드레스 영역 자동 분할
2. **인페인팅**: FLUX-Controlnet-Inpainting 으로 마스크 영역 재생성 → `(I_원본, I_garment, I_inpaint)` 트리플
3. **품질 필터**: GPT-4o 가 자세 일관성, 자연스러움, 정합성 검수
4. **캡션 생성**: Florence2 가 태스크별 instruction 텍스트 자동 생성 (예: "The set of three images display a model, a garment, and the model wearing the garment.")
5. **픽셀 정합**: `crop_to_multiple_of_16` 로 원본·합성 이미지 크기 정합

**최종 규모**: 60,000+ triple

---

## 실험 요약

> *왜?* 네 가지 태스크별로 측정 지표와 비교 기법이 달라, 표로 나눠 한눈에 비교한다.

### Table 2: Garment Reconstruction (옷 추출, VITON-HD)

| 지표 | TryOffDiff | **Any2AnyTryon** |
|---|---|---|
| SSIM ↑ | 0.793 | **0.805** |
| MS-SSIM ↑ | 0.712 | **0.710** |
| CW-SSIM ↑ | 0.466 | **0.453** |
| LPIPS ↓ | 0.334 | **0.328** |
| FID ↓ | 20.346 | **13.367** |
| CLIP-FID ↓ | 8.371 | **3.872** |
| KID ↓ (× 1000) | 6.8 | **3.5** |
| DISTS ↓ | 0.226 | **0.217** |

→ **거의 모든 지표 SOTA**. FID/CLIP-FID 는 약 2배 차이.

### Table 3: Model-Free Try-On (옷만 주고 사람 합성)

| 방법 | MP-LPIPS ↓ | CLIP-I ↑ | DiffSim ↑ | FFA ↑ |
|---|---|---|---|---|
| Magic Clothing | 0.192 | 0.642 | 0.143 | 0.459 |
| StableGarment | 0.149 | 0.650 | 0.153 | 0.547 |
| **Any2AnyTryon** | **0.141** | **0.789** | **0.202** | **0.549** |

→ 전 지표 1위. 특히 CLIP-I 0.789 로 옷 의미 보존이 크게 개선됨.

### Table 4: 표준 Try-On (VITON-HD)

| 방법 | LPIPS ↓ | SSIM ↑ | FID(P) ↓ | KID(P) ↓ | FID(U) ↓ | KID(U) ↓ |
|---|---|---|---|---|---|---|
| GP-VTON | 0.0677 | 0.8722 | 8.649 | 3.669 | 11.708 | 3.990 |
| OOTDiffusion | 0.1317 | 0.7838 | 12.131 | 4.335 | 15.136 | 5.774 |
| IDM-VTON | 0.0815 | 0.8156 | 8.206 | 1.727 | 10.745 | 2.229 |
| **CatVTON** | **0.0582** | **0.8653** | **5.482** | **0.384** | 9.083 | 1.130 |
| FitDiT | 0.1059 | 0.8298 | 8.362 | 1.543 | 10.340 | 1.648 |
| **Any2AnyTryon** | 0.0877 | 0.8387 | 6.934 | 0.7387 | **8.965** | **0.981** |

(P = paired, U = unpaired. KID 는 × 1000)

→ **흥미로운 분할**: paired 픽셀 정확도(LPIPS/SSIM)는 CatVTON 이 SOTA, **unpaired 자연스러움(FID/KID)은 Any2AnyTryon 이 1위**. 통합·mask-free 의 트레이드오프로 해석 가능.

### Table 5: Ablation

| 설정 | LPIPS ↓ | SSIM ↑ | CLIP-FID ↓ | KID ↓ |
|---|---|---|---|---|
| Clean latent 제거 | 0.3141 | 0.6892 | 9.648 | 6.403 |
| Adaptive PE 제거 | 0.3080 | 0.7088 | 9.434 | 5.658 |
| **Full** | **0.2590** | **0.7373** | **9.293** | **5.407** |

→ **Clean latent 가 더 결정적**. Adaptive PE 도 분명한 기여.

---

## 💬 Q&A

### Q1. 왜 condspa 와 condsub 를 RoPE 좌표로 분리해야 하나? 그냥 모두 오른쪽으로 이어붙이면 안 되나?

**A**: 두 조건의 의미가 본질적으로 다르기 때문이다.
- **condspa (모델 사진)**: "이 자리에 사람이 있다" 라는 **공간 정렬 정보**를 줘야 한다. 옷이 입혀질 위치, 사람의 포즈, 신체 비율 등이 모두 동일 좌표계에서 의미가 있다. → 타겟과 같은 (y, x) 좌표 공유.
- **condsub (옷 평면 사진)**: 옷의 모양·텍스처·디테일이라는 **객체 정보**만 주면 되고, 옷이 평면에 펼쳐진 위치는 타겟의 신체 위치와 무관하다. → 별개 패널처럼 옆에 붙임.

만약 둘 다 같은 좌표계를 공유하면, 어텐션이 옷의 평면 위치와 사람의 옷 위치를 혼동할 수 있다. 반대로 둘 다 offset 으로 처리하면, 모델이 "이 옷이 그 자리에 들어가야 한다" 는 공간 단서를 잃는다.

### Q2. CatVTON 이 LPIPS·SSIM 에서 더 좋은데, Any2AnyTryon 의 의미는 뭔가?

**A**: 세 가지 관점에서 봐야 한다.
1. **셋업의 공정성**: CatVTON 은 mask + 마스킹된 사람 + 옷 을 입력으로 받는다. Any2AnyTryon 은 mask 없이 사람+옷만 받는다 (test_vitonhd.py 에서는 fair 비교 위해 `--repaint` 로 mask 를 추가로 쓸 수도 있다). 즉 비교 기준이 동일하지 않다.
2. **자연스러움 vs 픽셀 충실도**: unpaired FID/KID 가 더 좋다는 것은 "옷의 다양성·자연스러움" 에서 우위라는 뜻. paired LPIPS/SSIM 이 약간 밀리는 것은 "GT 와 픽셀 단위 정합" 에서 손해.
3. **다기능성**: CatVTON 은 try-on 전용. Any2AnyTryon 은 네 가지 태스크 + 가변 해상도 + 멀티 옷.

→ **전자상거래용 사진 그대로 보여주는 픽셀 충실도가 필요하면 CatVTON, 새로운 구성·다기능이 필요하면 Any2AnyTryon**.

### Q3. Clean Latent 가 왜 그렇게 중요한가? (Ablation 결과의 해석)

**A**: 두 가지 메커니즘이 작동한다.
1. **신호 대비**: 조건이 노이즈로 흐려져 있으면, denoising step 초기에는 조건 자체도 노이즈에 묻혀 "이게 옷의 모양인지 노이즈인지" 모델이 분간하기 어렵다. Clean latent 는 매 step 옷의 디테일을 변하지 않는 강한 신호로 유지.
2. **Background 누설 차단**: IC-LoRA 처럼 조건도 함께 denoise 하면, 조건 영역의 background 가 미세하게 변하고 그 변화가 어텐션을 통해 타겟에도 전달돼 의도치 않은 변형을 만든다. Clean latent + 매 step 강제 덮어쓰기 (`latents = (1-mask)*init_proper + mask*latents`) 로 이를 봉쇄.

Ablation 의 LPIPS 0.3141 → 0.2590 변화 (약 18% 감소)가 이를 정량적으로 뒷받침.

### Q4. 한계점은?

**A**:
1. **paired LPIPS/SSIM 이 CatVTON 에 밀린다**: 픽셀 단위 정합이 결정적인 경우엔 안 맞을 수 있음.
2. **`--repaint` 옵션이 별도**: VITON-HD 평가에서는 결국 마스크와 AutoMasker 가 다시 등장. "완전히 mask-free" 는 일반 사용 범위에 한정.
3. **GPT-4o 의존**: 데이터 큐레이션에 closed model 사용 → 재현성 약화. LAION-Garment 자체는 공개됐지만 동일 큐레이션 재실행은 어려움.
4. **합성 데이터 신뢰성**: FLUX-Inpaint 가 만든 "그 옷을 입은 모습" 은 합성. 손·텍스처 아티팩트가 학습에 어떻게 전이됐는지 분석은 부족.
5. **LoRA-only**: 베이스 FLUX 의 분포 편향(예: 동양인·노인 등)을 그대로 물려받음.
6. **multi-garment 같은 신기능에 정량 평가가 부족**: 정성 그림 위주 시연.

### Q5. 이 아이디어를 다른 태스크에 옮기면?

**A**: "RoPE channel0 = 토큰 역할 ID" 라는 발상은 reference-driven 생성 전반에 유용하다.
- **Identity Preservation**: condsub = 얼굴 참조 사진
- **Style Transfer**: condsub = 스타일 참조
- **Layout-to-Image**: condspa = layout/sketch 가 같은 좌표 공간에
- **Multi-subject Composition**: condsub 가 여러 개, 각각 다른 x offset

이미 OmniControl, IP-Adapter 등이 유사 방향을 시도하지만, Any2AnyTryon 의 **"같은 x 좌표 공유 vs offset 으로 분리"** 라는 명시적 분리가 깔끔하다.

---

## 한 줄 요약 (전체)

**FLUX 의 RoPE 잉여 채널을 "토큰 역할 마커" 로 재해석하고, condspa(타겟과 같은 x 좌표) vs condsub(타겟 오른쪽으로 offset) 으로 위치를 분리하고, clean latent 로 조건 영역을 매 step 강제 덮어쓰기 함으로써, mask 없이 통합 try-on / 옷 추출 / 사람 합성 / 멀티옷을 단일 FLUX LoRA 로 수행한다.**

---

## 관련 메모리 링크

- [[reference_pretrained_backbone_reuse_landscape]] — FLUX backbone 재사용 분기 (분기 B-DiT 카테고리)
- [[paper_z_image]] — DiT 기반 단일 스트림 transformer 의 다른 사례
- [[paper_lumina_next]] — DiT 위치 임베딩 (RoPE 변형) 또 다른 사례
