# Vision Transformer (ViT) — WebGPU Compute

Runs a DeiT-Tiny vision transformer (5.7M params) entirely in WebGPU compute shaders to classify images, and visualizes per-head attention maps as interactive heatmap overlays.

**[Live demo](https://lyonsno.github.io/webgpu-vit-attention/)**

## What it does

1. **Patch embedding**: splits a 224x224 image into a 14x14 grid of 16x16-pixel patches, projects each to a 192-dim token
2. **12 transformer blocks**: each applies layer normalization, multi-head self-attention (3 heads), and an MLP with GELU activation, connected by residual additions
3. **Classification**: the CLS token is projected to 1000 ImageNet class logits
4. **Attention visualization**: select any layer and head to see which image patches the model attends to, rendered as a viridis heatmap overlay

## Running locally

```bash
npm install
npm run serve
# Open http://localhost:8080/?sample=visionTransformer
```

## Model weights

The int8-quantized weight file (`public/assets/models/deit-tiny-int8.bin`, 5.5MB) is derived from Meta's [DeiT-Tiny](https://huggingface.co/facebook/deit-tiny-patch16-224) (Apache-2.0). To regenerate:

```bash
pip install torch timm
python tools/convert_deit_weights.py
```

## Files

| File | Description |
|------|-------------|
| File | Description |
|------|-------------|
| `sample/visionTransformer/main.ts` | UI, image loading, WebGPU setup |
| `sample/visionTransformer/inference.ts` | Forward pass orchestration |
| `sample/visionTransformer/weights.ts` | Binary weight loader with int8 dequantization |
| `sample/visionTransformer/visualize.ts` | Attention map extraction and heatmap rendering |
| `sample/visionTransformer/shaders/*.wgsl` | Compute shaders for each transformer stage |
| `tools/convert_deit_weights.py` | Offline weight converter (PyTorch → int8 binary) |

## License

Sample code: BSD-3-Clause.
Model weights: Apache-2.0 (Meta DeiT). See [sample/visionTransformer/THIRD_PARTY_NOTICES.md](sample/visionTransformer/THIRD_PARTY_NOTICES.md).
