# GeoSaccade: Sequential Attention with Geographic Gating for Image Geolocation
## Core Mechanism Design — Multi-Model Brainstorm
Generated: 2026-03-27 | Models: Claude Opus, Gemini 3 Pro, GPT-5.4 Pro, DeepSeek v3.2

---

## THE CORE IDEA

Single image, T=5 sequential attention steps. Each step:
1. Controller decides WHERE to attend (gated by prior evidence)
2. Extracts a "glimpse" (focus + broad + peripheral)
3. Queries hierarchical GNN with the glimpse
4. GNN response GATES the next attention step

Bidirectional loop: image -> graph -> gate -> image -> graph -> ...

## ARCHITECTURE: GeoGRU + Saccadic Attention + GNN Querier

### GeoGRU (Geo-Gated Recurrent Unit)
Standard GRU but with EXTRA gate matrices V_z, V_r that give GNN context
DIRECT write-access into the controller's memory dynamics.

z_t = sigma(W_z * x_t + U_z * h_{t-1} + V_z * c_{t-1})  <- update gate
r_t = sigma(W_r * x_t + U_r * h_{t-1} + V_r * c_{t-1})  <- reset gate

The V_z and V_r matrices = GNN's independent control over gating.
When GNN believes "Southeast Asia", V_z pushes update gate to overwrite
hidden dimensions tracking incompatible hypotheses (e.g., South America).

### Three-Channel Geographic Gating

Channel 1 - Feature Gate (multiplicative: "WHAT to look for"):
  gamma = sigmoid(W_gamma * c_{t-1})
  q_tilde = q_t * gamma
  Effect: If GNN thinks Japan, amplify "CJK character detection" dims

Channel 2 - Spatial Bias (additive: "WHERE to look"):
  s_{t+1} = K * W_gs * c_{t-1} / sqrt(D_k)
  Effect: Patches matching geographic hypothesis get higher attention

Channel 3 - Memory Gate (integrative: "WHAT to remember"):
  Via GeoGRU's V_z, V_r matrices
  Effect: GNN controls which hypotheses to keep/discard

### Anti-Collapse: Inhibition of Return (IOR)
Inspired by human saccadic eye movements:
  e_{t,i} -= lambda * cumulative_attention_{t-1,i}
Previously attended patches get SUPPRESSED.

### Adaptive Temperature
  tau_t = softplus(w_tau * h_{t-1}) + 0.1
Early steps -> high tau (broad attention over scene)
Later steps -> low tau (focused on discriminative patches)

## GLIMPSE EXTRACTION (3 components)

1. FOCUS: weighted sum of top-16 most attended patches (local detail)
2. BROAD: full soft-attention weighted average (scene context)
3. PERIPHERAL: weighted by (1-attention) (background signal!)
   -> crucial for geo: sky color, distant mountains carry location info

Each step has its OWN value projection W_v^(t):
- Step 1's W_v learns spatial frequency + color (scene layout)
- Step 3's W_v learns texture + edges (architecture)
- Step 4's W_v learns character patterns (text/signs)

## GNN COUPLING

Soft level routing learned from hidden state:
  route_t = softmax(W_route * h_{t-1})   -> [B, 3]

Expected behavior:
  Step 1: route = (0.7, 0.2, 0.1)  -> Country-dominant
  Step 3: route = (0.2, 0.6, 0.2)  -> Region-dominant
  Step 5: route = (0.1, 0.2, 0.7)  -> Cell-dominant

Hierarchy EMERGES from training, not hard-coded.

## TENSOR FLOW (per step)

Step t:
  q_t       [B, 256]    <- W_q([h_{t-1} || c_{t-1}])
  gamma     [B, 256]    <- sigmoid(W_gamma(c_{t-1}))     FEATURE GATE
  q_tilde   [B, 256]    <- q_t * gamma
  e_t       [B, 576]    <- K*q/sqrt(D) + K*W_gs*c/sqrt(D) - lambda*cum
  tau_t     [B, 1]      <- softplus(w_tau * h) + 0.1
  alpha_t   [B, 576]    <- softmax(e_t / tau_t)
  g_t       [B, 512]    <- fuse(focus, broad, peripheral)
  q_geo     [B, 512]    <- W_gq([g_t || h_{t-1}])
  logits    [B, C_l]    <- q_geo * G_l^T / sqrt(D_g)     per level
  c_t       [B, 512]    <- sum_l route_l * softmax(l/0.1) * G_l
  h_t       [B, 1024]   <- GeoGRU([g_t||c_t], h_{t-1}, c_t)

## LOSS FUNCTION (6 components)

1. L_cls:  Final step cross-entropy (0.1*country + 0.3*region + 0.6*cell)
2. L_inter: Per-step intermediate losses with hierarchical curriculum
   Step 1: emphasize country. Step 4: emphasize cell.
3. L_geo:  Haversine distance regression
4. L_ent:  MONOTONIC entropy decrease enforced per step
   H(p_t) - H(p_{t-1}) + 0.1 >= 0 (hinge loss)
5. L_div:  Attention diversity (cosine between step pairs < 0.3)
6. L_halt: Optional ACT halting regularization

## INFORMATION THEORY

Each step MUST reduce entropy over location distribution:
  H_0 = log(30000) ~ 10.3 nats (uniform prior)
  H_1 ~ 7 nats (after scene layout -> continent known)
  H_2 ~ 5 nats (infrastructure -> country group)
  H_3 ~ 3 nats (architecture -> region)
  H_4 ~ 1.5 nats (signage -> narrow area)
  H_5 ~ 0.5 nats (fine details -> cell)

Visualized as "ENTROPY FUNNEL" — plotted per image.

## EXPECTED LEARNED ATTENTION ORDER

Step 1: Sky + horizon + landscape    -> climate/biome -> continent
Step 2: Roads + infrastructure       -> surface, markings, driving side -> country
Step 3: Architecture + vegetation    -> materials, roof, trees -> region
Step 4: Signs + text                 -> language, road signs -> country+region
Step 5: Fine meta-cues              -> bollards, poles, curbs -> cell

## vs EXISTING MECHANISMS

vs Cross-Attention: ours is SEQUENTIAL + CAUSALLY CONDITIONED + GNN-gated
vs RAM (Mnih 2014): ours is DIFFERENTIABLE + operates in FEATURE space + has GNN
vs Slot Attention: ours is SERIAL hypothesis refinement, not parallel decomposition
vs Everything: GEOGRAPHIC GATING is completely new

## TRAINING CURRICULUM

Phase 1 (ep 0-20):  T=1, single-step baseline
Phase 2 (ep 20-50): T=2->5, progressive unrolling (+1 step every 6 epochs)
Phase 3 (ep 50-80): T=5, full training, LR x0.1
Phase 4 (ep 80-100): Gumbel temperature anneal (1.0 -> 0.1, soft -> hard)

## PARAMETER BUDGET

DINOv2 ViT-L/14 (frozen):  300M  -> 0 trainable
Saccadic Attention:         ~12M
Glimpse Extractor:          ~8M
GNN Querier:                ~2M + 17M embeddings
Total trainable:            ~39M

Overhead: ~1.5 GFLOPs for T=5 steps (~15% over single-pass)

## INTERPRETABILITY

Scanpath visualization: T=5 heatmaps with arrows showing fixation trail.
Compare to human GeoGuessr eye-tracking via ScanMatch/MultiMatch metrics.
Entropy funnel: plot H_t showing hypothesis narrowing per image.
Failure diagnosis: cycling = low IOR, premature commitment = overconfident gate.

## ONE-SENTENCE SUMMARY

GeoSaccade introduces geographic gating — a closed bidirectional loop where a
hierarchical GNN over OpenStreetMap conditions sequential visual attention through
multiplicative feature gates, additive spatial biases, and direct memory modulation,
producing interpretable scanpaths that progressively narrow geographic hypotheses
from continent to S2 cell across five reasoning steps.
