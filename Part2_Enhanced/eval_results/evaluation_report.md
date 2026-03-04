# Part2 Enhanced Model - Evaluation Report

**Generated**: 2026-03-04 21:46:33

## Training Summary

- **Epochs Completed**: 3
- **Best mIoU**: 0.8418
- **Model Parameters**: 19.1M
- **Architecture**: CNN + CBAM + Pyramid Pooling

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| **mIoU** | **0.0510** |
| Accuracy | 0.1799 |
| F1 Score | 0.0395 |

## Per-Class IoU

| Class | IoU |
|-------|-----|
| Background | 0.1917 |
| Divers | 0.0000 |
| Plants | 0.0000 |
| Wrecks | 0.0000 |
| Robots | 0.0000 |
| Reefs | 0.0000 |
| Fish | 0.0000 |
| Sea floor | 0.0001 |

## Notes

- **Dataset**: SUIM + USIS10K (8882 training samples)
- **Labels**: Generated using color-based clustering (pseudo-labels)
- **Training**: 3 epochs on CPU
- **Input Size**: 256x256
- **Number of Classes**: 8
