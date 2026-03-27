# GeoSaccade

**Sequential Attention with Geographic Gating for Image Geolocation via Graph Neural Networks**

GeoSaccade is a novel architecture that mimics human visual search behavior (saccadic eye movements) to progressively refine geographic predictions. Instead of making a single pass over an image, the model takes T sequential "glimpses," each time focusing on different spatial regions guided by geographic priors from a hierarchical graph neural network.

## Architecture

```
                            ┌─────────────────────────────────┐
                            │         Input Image             │
                            └────────────┬────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────────────┐
                            │   DINOv2 ViT-L/14 (frozen)     │
                            │   [B, 576, 1024] patch tokens   │
                            └────────────┬────────────────────┘
                                         │
                    ┌────────────────────────────────────────────┐
                    │          Sequential Attention Loop         │
                    │          t = 1, 2, ..., T (T=5)           │
                    │                                            │
                    │   ┌──────────────────────────────────┐    │
                    │   │  SaccadicAttention               │    │
                    │   │  • Feature gate (h_t → q_t)      │    │
                    │   │  • Spatial bias (geographic)      │    │
                    │   │  • IOR (inhibit revisited)        │    │
                    │   │  • Adaptive temperature           │    │
                    │   └──────────────┬───────────────────┘    │
                    │                  │ attention weights       │
                    │                  ▼                         │
                    │   ┌──────────────────────────────────┐    │
                    │   │  GlimpseExtractor                │    │
                    │   │  • Focus  (top-k, k=16)          │    │
                    │   │  • Broad  (global avg pool)      │    │
                    │   │  • Peripheral (remaining)        │    │
                    │   │  Per-step W_v projections         │    │
                    │   └──────────────┬───────────────────┘    │
                    │                  │ v_t ∈ R^512            │
                    │                  ▼                         │
                    │   ┌──────────────────────────────────┐    │
                    │   │  GNNQuerier                      │    │
                    │   │  • 3-level hierarchy             │    │
                    │   │  • Soft level routing             │    │
                    │   │  • Geographic embedding g_t       │    │
                    │   └──────────────┬───────────────────┘    │
                    │                  │ g_t ∈ R^512            │
                    │                  ▼                         │
                    │   ┌──────────────────────────────────┐    │
                    │   │  GeoGRU                          │    │
                    │   │  • V_z, V_r geographic gates     │    │
                    │   │  h_t = GeoGRU([v_t; g_t], h_{t-1})│   │
                    │   └──────────────┬───────────────────┘    │
                    │                  │ h_t ∈ R^1024           │
                    └──────────────────┼────────────────────────┘
                                       │
                                       ▼
                            ┌─────────────────────────────────┐
                            │  Classification Head            │
                            │  Per-step logits → L_cls + L_inter│
                            │  Final prediction at step T     │
                            └─────────────────────────────────┘
```

## Key Contributions

1. **Saccadic Attention Mechanism** — A biologically-inspired sequential attention module that progressively focuses on geographically informative image regions, using inhibition-of-return (IOR) to prevent redundant fixations and adaptive temperature to control exploration vs. exploitation.

2. **Geographic Gating via GeoGRU** — A modified GRU cell where reset and update gates are modulated by geographic embeddings from a hierarchical graph, enabling the hidden state to encode both visual and spatial information with proper geographic inductive bias.

3. **Hierarchical GNN Querier with Soft Routing** — A multi-level graph neural network over geographic regions (country → region → city) with learned soft routing that allows the model to dynamically query the appropriate spatial granularity at each attention step.

## Quick Start

```bash
# Clone
git clone https://github.com/talentri-id/geosaccade.git
cd geosaccade

# Install
pip install -e ".[dev]"

# Build geographic graph (requires OSM data)
python scripts/build_graph.py --output data/graph.pt

# Precompute DINOv2 embeddings
python scripts/precompute_embeddings.py --dataset im2gps3k --output data/embeddings/

# Train
python scripts/train.py --config configs/mvp.yaml
```

## Model Summary

| Component          | Parameters | Details                          |
|--------------------|-----------|----------------------------------|
| DINOv2 ViT-L/14   | 0 (frozen)| 576 patches, 1024-dim            |
| LoRA adapters      | ~2M       | rank=16 on q,v projections       |
| GeoGRU             | ~8M       | hidden=1024, geo=512             |
| SaccadicAttention  | ~2M       | D_k=256, T=5 steps              |
| GlimpseExtractor   | ~5M       | D_v=512, 3 components           |
| GNNQuerier         | ~12M      | D_g=512, 3-level hierarchy       |
| Classification     | ~10M      | per-step heads                   |
| **Total trainable**| **~39M**  |                                  |

## Citation

```bibtex
@article{geosaccade2026,
  title={GeoSaccade: Sequential Attention with Geographic Gating for Image Geolocation via Graph Neural Networks},
  author={Talentri Research},
  year={2026},
  note={Work in progress}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
