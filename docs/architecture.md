# GeoGraph: Navigating Geographic Topology for Image Geolocation with Graph Neural Networks

## Research Design Document — Multi-Model Brainstorm Output
Generated: 2026-03-27 | Models: Claude Opus, Gemini 3 Pro, GPT-5.4 Pro, DeepSeek v3.2

---

## ARCHITECTURE: Encode → Retrieve → Reason

```
Image ──→ [DINOv2 ViT-L + LoRA] ──→ z_img [B,256]
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
            [GNN Coarse]       [GNN Medium]        [GNN Fine]
             ~200 nodes        ~3,000 nodes       ~30,000 nodes
             countries          regions            S2 cells (20km)
                    │                 │                  │
                    └────── Hierarchical Retrieval ──────┘
                                      │
                              Top 100 candidates
                                      │
                         [Graph-Conditioned Refinement]
                          cross-attn: nodes ↔ patches
                          GAT: spatial reasoning on subgraph
                                      │
                                  pred (lat, lon)
```

## KEY COMPONENTS

### Vision: DINOv2 ViT-L/14 (frozen) + LoRA rank-16
- Why DINOv2 over CLIP: better spatial granularity in patch tokens
- Output: z_img [B, 256] (global) + z_patch [B, 576, 1024] (local)
- Only 14.8M trainable params (LoRA + projector + GNN + refiner)

### Graph: 3-level hierarchy from OpenStreetMap
- Level 1: ~200 country nodes (Natural Earth)
- Level 2: ~3,000 region nodes (GADM Admin-1)  
- Level 3: ~30,000 S2 cells with OSM road connectivity

### Node Features (352-dim → 320):
- Fourier positional encoding of lat/lon [64]
- Metadata: elevation, road density, climate, urban/rural [32]
- Visual prototype: mean DINOv2 embedding of training images in cell [256]

### Edge Features (16-dim):
- Haversine distance [4]
- Edge type: spatial_adj / road_connection / admin_border / hierarchical [4]
- Road type: motorway / trunk / primary / secondary / tertiary / residential [6]
- Visual similarity + same-country flag [2]

### GNN: GATv2 (4 layers, 4 heads, D=256)
- Dynamic attention — neighbor ranking changes per query
- Critical for border regions (French vs German side)

### Refinement: Cross-Attention + GAT (2 iterations)
- Step 1: candidate nodes attend to image patches (visual grounding)
- Step 2: GAT message passing on local subgraph (spatial reasoning)
- Output: soft weighted average of candidate coordinates

## TRAINING

### 5-component loss:
1. Hierarchical cross-entropy (coarse 0.2 + medium 0.3 + fine 0.5)
2. InfoNCE contrastive (fine level)
3. Haversine coordinate regression
4. KL divergence for refinement ranking
5. Graph structure regularization (margin loss)

### 3-phase schedule:
- Phase 1 (5 ep): GNN warmup only
- Phase 2 (30 ep): Joint training, all components
- Phase 3 (10 ep): Refinement fine-tuning

### Hard negative mining: 1 GT + 30 hard + 40 spatial + 29 random = 100 candidates

## INFERENCE (~28ms on RTX 5070)

All GNN embeddings PRECOMPUTED. Zero GNN forward pass at test time.
1. Encode image (25ms)
2. Country retrieval: matmul vs 200 nodes (0.01ms)
3. Region retrieval: matmul vs ~200 candidates (0.05ms)
4. Fine retrieval: matmul vs ~1000 candidates (0.5ms)  
5. Refinement: cross-attn + GAT on 100 nodes (2ms)

## FEASIBILITY ON RTX 5070 8GB

Training VRAM budget (batch 8, FP16):
- DINOv2 weights (frozen): 608 MB
- LoRA + GNN + Refiner: 44 MB
- Activations: 360 MB
- Optimizer states: 180 MB
- TOTAL: ~1.3 GB (6.7 GB headroom!)

### MVP (2-3 days):
- DINOv2 ViT-B (frozen, precompute embeddings offline)
- Single-level graph, 10K S2 cells
- 3 ablations: (a) MLP baseline, (b) random graph, (c) correct OSM graph
- If (c) > (a): graph helps. If (c) > (b): correct topology matters.

### Full paper (5-7 days):
- 3-level hierarchy, 30K nodes
- DINOv2 ViT-L + LoRA
- MP-16 dataset (4.7M images)
- 5 ablation variants

## NOVELTY vs SOTA

| Aspect | PIGEON | GeoCLIP | GeoGraph |
|--------|--------|---------|----------|
| Spatial structure | None | None | Road network topology |
| Location repr. | Geocell one-hot | GPS coords | Graph node embedding |
| Neighbor awareness | None | None | GNN message passing |
| Uses OSM data | No | No | Yes |
| Interpretable | No | No | Yes (attention maps) |

## 3 CONTRIBUTIONS

1. **Geographic Topology as Inductive Bias** — first model encoding road network structure via GNN for image geolocation
2. **Graph-Conditioned Visual Refinement** — retrieve-then-reason with image↔subgraph cross-attention
3. **Topological Evaluation Protocol** — hop distance, component accuracy metrics beyond km-error

## NEW METRICS (our contribution)

- Graph Hop Distance: shortest path between predicted and true nodes
- Topological Accuracy@k: prediction within k hops?
- Connected Component Accuracy: same road network component?
- Hierarchical Accuracy: correct country/region?

## PAPER TARGET

**Title:** "GeoGraph: Navigating Geographic Topology for Image Geolocation with Graph Neural Networks"
**Venue:** CVPR 2026 (Nov 2025 deadline) or ICLR 2026 (Oct 2025)

## PROJECTED RESULTS

| Method | Street 1km | City 25km | Region 200km | Median km |
|--------|:---:|:---:|:---:|:---:|
| GeoCLIP | 14.2% | 36.8% | 55.1% | 486 |
| PIGEON | 16.1% | 40.4% | 58.6% | 398 |
| **GeoGraph** | **~18%** | **~44%** | **~63%** | **~340** |

Largest gains at region level (200km) where topology most effectively disambiguates.
