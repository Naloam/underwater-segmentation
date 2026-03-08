# Part2 Enhanced Model - Evaluation Report

**Generated**: 2026-03-08 11:06:06

## Training Summary

- **Epochs Completed**: 20
- **Architecture**: CNN + CBAM + Pyramid Pooling + FPN Decoder
- **CLIP Branch**: Enabled (frozen, openai/clip-vit-base-patch32)
- **Diffusion Branch**: Enabled

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| **mIoU** | **0.0700** |
| Accuracy | 0.2260 |
| F1 Score | 0.1175 |

## Per-Class IoU

| Class | IoU |
|-------|-----|
| Background | 0.2431 |
| Divers | 0.0000 |
| Plants | 0.0000 |
| Wrecks | 0.0000 |
| Robots | 0.0000 |
| Reefs | 0.1835 |
| Fish | 0.0492 |
| Sea floor | 0.0140 |

## Notes

- **Training Dataset**: USIS10K (7442 samples)
- **Validation Dataset**: SUIM (1440 samples)
- **Input Size**: 256x256
- **Number of Classes**: 8
