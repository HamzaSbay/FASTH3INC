# FastH3 Architecture & Mathematical Specification

This document defines the mathematical, structural, and algorithmic specification of **FastH3 (MiniMax-H3 / FastVideo FastH3 8-Step V2)** for implementation in `FASTH3INC` (pure C inference engine).

---

## 1. Sequence Construction & Packed Layout

MiniMax-H3 runs a single stack of transformer blocks over **one packed 1-D sequence** containing all modalities without padding:
$$\text{Sequence} = [\text{text} \mid \text{conditioning / references} \mid \text{audio} \mid \text{target video}]$$

### 1.1 Modality Tags
Every token row in the sequence carries a modality tag $m \in \{0, 1, 2\}$:
- **`0` = Video** (target video latents and vision conditioning blocks)
- **`1` = Text** (Qwen3-VL conditioning text tokens)
- **`2` = Audio** (soundtrack latents, channel-major)

The number of modalities is fixed at $\text{MINIMAX\_H3\_MODALITY\_NUM} = 3$.

### 1.2 Video Patchification
Video latents $z_v \in \mathbb{R}^{C \times T_l \times H_l \times W_l}$ with $C=24$ are patchified using patch size $(p_t, p_h, p_w) = (1, 2, 2)$:
$$\text{Patch Dimension} = C \times p_t \times p_h \times p_w = 24 \times 1 \times 2 \times 2 = 96$$
The number of video token rows is:
$$N_{\text{video}} = \left(\frac{T_l}{p_t}\right) \times \left(\frac{H_l}{p_h}\right) \times \left(\frac{W_l}{p_w}\right) = T_l \times \frac{H_l}{2} \times \frac{W_l}{2}$$
Tokens are laid out in frame-major (temporal), then row-major (spatial) raster order:
$$\text{idx}(t, h, w) = t \cdot \left(\frac{H_l}{2} \cdot \frac{W_l}{2}\right) + h \cdot \frac{W_l}{2} + w$$

### 1.3 Audio Latents
Audio latents $z_a \in \mathbb{R}^{2 \times 32 \times T_a}$ (stereo, 32 latent channels) are laid out channel-major into rows of dimension 32:
$$N_{\text{audio}} = 2 \times T_a$$

### 1.4 3-Axis Position Grid (fp64)
Every row in the packed sequence carries a 3-dimensional rotary coordinate $(t, h, w)$ computed in double-precision (`float64`):

#### Temporal Grid
One latent frame spans rotary time units defined by the VAE temporal grouping pattern $(1, 4, 4, 4, 4)$ scaled by $\frac{5}{3}$:
$$\Delta t_i = \frac{5}{3} \times \text{pattern}[i \pmod 5]$$
Starting from an origin $t_0$, the temporal coordinates are:
$$t_0 = 0, \quad t_k = t_0 + \sum_{i=0}^{k-1} \Delta t_i \quad (k \ge 1)$$

#### Spatial Grid
For latent dimensions $(H_l, W_l)$ and patch size $(p_h, p_w) = (2, 2)$:
$$\text{Area} = \sqrt{\frac{H_l}{p_h} \cdot \frac{W_l}{p_w}}$$
$$\text{ratio}_h = \frac{H_l / p_h}{\text{Area}}, \quad \text{ratio}_w = \frac{W_l / p_w}{\text{Area}}$$
$$\text{left}_h = \frac{1 - \text{ratio}_h}{2}, \quad \text{left}_w = \frac{1 - \text{ratio}_w}{2}$$
Coordinates are centered on the unit interval and scaled by 32:
$$\text{grid}_h[i] = 32 \cdot \left(\text{left}_h + i \cdot \frac{\text{ratio}_h}{H_l / p_h}\right)$$
$$\text{grid}_w[j] = 32 \cdot \left(\text{left}_w + j \cdot \frac{\text{ratio}_w}{W_l / p_w}\right)$$

#### Audio & Text Placement
- **Audio rows**: $h = 0$, $t = t_0 + \text{arange}(T_a)$, $w$ pinned to the extrema $\text{grid}_w[0]$ (channel 0) and $\text{grid}_w[-1]$ (channel 1).
- **Text rows**: $(t, h, w) = (0, 0, 0)$.

---

## 2. Rotary Position Embedding (RoPE)

MiniMax-H3 applies partial 3-axis rotary embedding to the attention heads:
- Head dimension: $d_{\text{head}} = 128$
- Rotary frequency dimension per axis: $\text{rope\_freq\_dim} = 16$
- Base frequency: $\theta = 10000.0$
- Number of rotated channels:
  $$\text{rotary\_dim} = 2 \times 3 \times \text{rope\_freq\_dim} = 2 \times 3 \times 16 = 96$$
  The remaining $128 - 96 = 32$ channels pass through unrotated.

### 2.1 Frequency Schedule
For each axis ($k \in \{0, 1, \dots, 15\}$):
$$\text{inv\_freq}[k] = \frac{1}{\theta^{2k / 32}}$$
For coordinate $(t, h, w)$:
$$\text{freqs}_t = t \cdot \text{inv\_freq}, \quad \text{freqs}_h = h \cdot \text{inv\_freq}, \quad \text{freqs}_w = w \cdot \text{inv\_freq}$$
$$\text{freqs}_{48} = [\text{freqs}_t \mid \text{freqs}_h \mid \text{freqs}_w]$$
$$\text{angles}_{96} = [\text{freqs}_{48} \mid \text{freqs}_{48}]$$
$$\cos = \cos(\text{angles}_{96}), \quad \sin = \sin(\text{angles}_{96})$$

### 2.2 Rotation Operation (`rotate_half`)
Splitting the leading 96 channels into two equal halves $[x_1 \mid x_2]$:
$$x_{\text{rot}} = [x_1 \mid x_2]$$
$$x_{\text{rotated}} = [-x_2 \mid x_1]$$
$$x_{\text{out}} = [x_{\text{rot}} \odot \cos + x_{\text{rotated}} \odot \sin \mid x_{\text{pass}}]$$

---

## 3. AdaLayerNorm Modulation

MiniMax-H3 uses adaptive layer normalization modulated per row according to the row's timestep index and modality tag.

### 3.1 AdaLN Table Indexing
For each block, the modulation parameter table is projected from the shared timestep embedding:
$$\text{temb} \in \mathbb{R}^{N_{\text{unique\_t}} \times d_{\text{temb}}}$$
$$\text{Linear}: d_{\text{temb}} \to 6 \times d_{\text{hidden}} \times 3$$
The output is reshaped to:
$$\text{Tables} \in \mathbb{R}^{(N_{\text{unique\_t}} \cdot 3) \times 6 \times d_{\text{hidden}}}$$
Each row $r$ in the packed sequence addresses row:
$$\text{adaln\_index}[r] = \text{timestep\_indices}[r] \times 3 + \text{token\_tags}[r]$$
Yielding 6 vectors:
$$(\text{shift}_{\text{msa}}, \text{scale}_{\text{msa}}, \text{gate}_{\text{msa}}, \text{shift}_{\text{mlp}}, \text{scale}_{\text{mlp}}, \text{gate}_{\text{mlp}})$$

### 3.2 Modulation Application
For RMSNorm output $\hat{x}$:
$$\hat{x} = \text{RMSNorm}(x, w, \epsilon)$$
- **Pre-attention modulation**:
  $$\tilde{x}_{\text{msa}} = \hat{x}_{\text{norm1}} \odot (1 + \text{scale}_{\text{msa}}) + \text{shift}_{\text{msa}}$$
  Residual update:
  $$x = x_{\text{residual}} + \text{gate}_{\text{msa}} \odot \text{Attn}(\tilde{x}_{\text{msa}})$$
- **Pre-MLP modulation**:
  $$\tilde{x}_{\text{mlp}} = \hat{x}_{\text{norm2}} \odot (1 + \text{scale}_{\text{mlp}}) + \text{shift}_{\text{mlp}}$$
  Residual update:
  $$x = x_{\text{residual}} + \text{gate}_{\text{mlp}} \odot \text{FFN}(\tilde{x}_{\text{mlp}})$$

### 3.3 Output Norm Modulation (`norm_out`)
`norm_out.linear` projects $\text{SiLU}(\text{temb})$ to $2 \times d_{\text{hidden}}$:
$$(\text{shift}_{\text{out}}, \text{scale}_{\text{out}})$$
$$\hat{x}_{\text{out}} = \text{RMSNorm}(x, w_{\text{final}}, \epsilon) \odot (1 + \text{scale}_{\text{out}}[\text{timestep\_indices}]) + \text{shift}_{\text{out}}[\text{timestep\_indices}]$$

---

## 4. SwiGLU Feed-Forward Network (FFN)

The FFN uses bias-free SwiGLU with value-first packed halves:
- Input: $x \in \mathbb{R}^{N \times d_{\text{hidden}}}$
- Hidden projection: $W_{\text{gate\_up}} \in \mathbb{R}^{(2 \cdot d_{\text{ffn}}) \times d_{\text{hidden}}}$
- Down projection: $W_{\text{down}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{ffn}}}$

### 4.1 Evaluation Order
$$[h_{\text{value}} \mid h_{\text{gate}}] = x W_{\text{gate\_up}}^T \quad (\text{split into two equal halves of size } d_{\text{ffn}})$$
$$\text{activation} = h_{\text{value}} \odot \text{SiLU}(h_{\text{gate}})$$
$$\text{FFN}(x) = \text{activation} W_{\text{down}}^T$$
*Note:* The first half is the linear value stream; the second half is the gated stream passed through $\text{SiLU}$.

---

## 5. DMD2 Sampler Schedule & Rectified Flow

FastH3 8-Step V2 was trained using Data-Free DMD2 distillation on an explicit 8-rung ladder.

### 5.1 Training Rungs & Base Sigmas
The 8 rungs on the 1000-step training clock are:
$$\text{DMD\_STEPS} = [999, 874, 749, 624, 500, 375, 250, 125]$$
The 9-point base grid (terminating at 0) is:
$$\text{base} = \left[\frac{999}{1000}, \frac{874}{1000}, \frac{749}{1000}, \frac{624}{1000}, \frac{500}{1000}, \frac{375}{1000}, \frac{250}{1000}, \frac{125}{1000}, 0.0\right]$$

### 5.2 Rectified Flow Exponential Shift
For shift parameter $s > 0$:
$$\sigma(s, b) = \frac{s \cdot b}{1 + (s - 1) \cdot b}$$
- **Video shift**: $s_v = 10.0$ (FastH3 8-Step V2 trained shift; base MiniMax-H3 used 12.0)
- **Audio shift**: $s_a = 3.0$
- Sigmas are computed in `float64` before casting to `float32`.

### 5.3 Timestep Convention
MiniMax-H3 defines timesteps as:
$$t = 1 - \sigma \in [0, 1], \quad t = 1 \text{ is clean}$$
- Video rows use $t_{v, i} = 1 - \sigma_{v, i}$.
- Audio rows use $t_{a, i} = 1 - \sigma_{a, i}$.
- Text rows inherit the video timestep: $t_{\text{text}, i} = t_{v, i}$.

### 5.4 Data-Ward Velocity & Euler Step
The DiT predicts data-ward velocity $v$, where the clean estimate is:
$$x_0 = x_t + \sigma \cdot v = x_t + (1 - t) \cdot v$$
*(Opposite sign to diffusers standard flow match)*.

For deterministic sampling with $\eta = 0$, the ratio $r = \frac{\sigma_{next}}{\sigma}$:
$$x_{next} = r \cdot x_t + (1 - r) \cdot x_0 = x_t + (\sigma_{next} - \sigma) \cdot v$$
Evaluated in `float32` accumulation.

### 5.5 Stochastic Renoising Step
For few-step stochastic distillation:
$$x_{next} = (1 - \sigma_{next}) \cdot x_0 + \sigma_{next} \cdot \epsilon_{\text{fresh}}, \quad \epsilon_{\text{fresh}} \sim \mathcal{N}(0, I)$$
At the terminal step $\sigma_{next} = 0$, returning exactly $x_0$.

---

## 6. VSA-H3 Sparse Attention Mechanism

FastH3 achieves 80% attention sparsity using Video Sparse Attention (VSA-H3).

### 6.1 Tile Geometry (Tile Size 64)
- **Video tiles**: $4 \times 4 \times 4 = 64$ tokens per 3D spatial-temporal tile $(t, h, w)$.
  $n_t = \lceil T_v / 4 \rceil$, $n_h = \lceil H_v / 4 \rceil$, $n_w = \lceil W_v / 4 \rceil$.
  Total video tiles $L = n_t \times n_h \times n_w$.
- **Prefix tiles**: Non-video segments (text, audio, image conditioning) are chunked into 64-token tiles per segment without straddling segment boundaries.
- Boundary tiles with fewer than 64 tokens record their true valid count in $\text{vbs}[i]$ (variable block sizes).

### 6.2 Tile Pooling & Scoring
For query $Q$ and key $K$ ($d_{\text{head}} = 128$):
$$Q_{\text{pool}}[h, i, :] = \frac{1}{\text{vbs}[i]} \sum_{j \in \text{tile } i} Q[j, h, :]$$
$$K_{\text{pool}}[h, i, :] = \frac{1}{\text{vbs}[i]} \sum_{j \in \text{tile } i} K[j, h, :]$$
Tile scores:
$$S[h, i, j] = \frac{1}{\sqrt{d_{\text{head}}}} Q_{\text{pool}}[h, i, :] \cdot K_{\text{pool}}[h, j, :]$$

### 6.3 Top-K Selection Rule (80% Sparsity)
- **Prefix queries** ($i < N_{\text{prefix}}$): Always dense (attend to all key tiles $j$).
- **Prefix keys** ($j < N_{\text{prefix}}$): Always exempt (every query attends to all prefix keys).
- **Video queries** ($i \ge N_{\text{prefix}}$):
  $$k = \text{compute\_topk}(0.8, L) = \max(1, \lceil(1 - 0.8) \cdot L\rceil) = \lceil 0.2 \cdot L\rceil$$
  For each head $h$ and video query tile $i$, select the top $k$ video key tiles with highest score $S[h, i, j]$.

### 6.4 Fine Block-Sparse Attention
For each token $q \in \text{tile } i$:
$$O_{\text{fine}}[q, h, :] = \sum_{k \in \text{selected tiles}} \text{softmax}(A)_{qk} V[k, h, :]$$
Only valid tokens (excluding padding slots) participate in softmax.

### 6.5 Coarse Gated Compression (`to_gate_compress`)
FastH3 carries learned gate weights `blocks.*.attn.to_gate_compress.weight` ($[5376, 7168]$):
- Tile-pooled values:
  $$V_{\text{pool}}[h, j, :] = \frac{1}{\text{vbs}[j]} \sum_{k \in \text{tile } j} V[k, h, :]$$
- Coarse attention:
  $$A_{\text{coarse}}[h, i, :] = \text{softmax}(S[h, i, :], \text{dim}=-1)$$
  $$O_{\text{coarse}}[h, i, :] = \sum_j A_{\text{coarse}}[h, i, j] V_{\text{pool}}[h, j, :]$$
- Gating:
  $$\text{gate} = \tilde{x}_{\text{msa}} \cdot W_{\text{gate\_compress}}^T \in \mathbb{R}^{N \times H \times d_{\text{head}}}$$
  *(Note: operates on the modulated normalized input $\tilde{x}_{\text{msa}}$, the same state fed to $W_{\text{qkv}}$, not the unmodulated residual $x$.)*
- Additive combination:
  $$O[q, h, :] = O_{\text{fine}}[q, h, :] + O_{\text{coarse}}[\text{tile}(q), h, :] \odot \text{gate}[q, h, :]$$

---

## 7. Numerical Parity Verification

The reference Python implementations (`tools/ref/h3_ref.py` and `tools/ref/vsa_ref.py`) have been verified against PyTorch CPU and diffusers:
1. **Diffusers DiT Model Parity**: Tested on tiny model (hidden 256, 2 heads, 3 layers, refiner 2, FFN 512, temb 128) against `MiniMaxH3Transformer3DModel`.
   $$\text{rel-L2}_{\text{video}} \approx 3.5 \times 10^{-7} \le 10^{-5}$$
   $$\text{rel-L2}_{\text{audio}} \approx 2.6 \times 10^{-7} \le 10^{-5}$$
2. **VSA Attention Parity**: Tested on $N=512$ against FastVideo's PyTorch masked attention reference oracle.
   $$\text{rel-L2}_{\text{vsa}} \approx 3.0 \times 10^{-7} \le 10^{-5}$$
3. **Sparsity 0 Equality**: Verified identical to dense attention with rel-L2 $< 3 \times 10^{-7}$.
4. **Sampler Schedule**: Verified bit-exact match against FastVideo contract ladder.
