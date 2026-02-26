# CropScan v3 — Crop Disease Detection

YOLO11s model trained for aerial and close-up crop disease detection.

## Classes (6)
| Class | Description |
|-------|-------------|
| healthy | Normal, healthy crop tissue |
| bacterial | Bacterial infection markers |
| fungal | Fungal disease patterns |
| viral | Viral infection symptoms |
| nutrient_stress | Nutrient deficiency indicators |
| other_disease | Other disease/stress patterns |

## Performance (Validation)
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| healthy | 0.992 | 0.949 | 0.988 | 0.971 |
| bacterial | 0.996 | 0.992 | 0.995 | 0.995 |
| fungal | 0.996 | 0.956 | 0.989 | 0.977 |
| viral | 0.998 | 0.950 | 0.976 | 0.965 |
| nutrient_stress | 0.988 | 0.998 | 0.995 | 0.995 |

**Overall: mAP50 = 0.989, mAP50-95 = 0.981**

## Model Details
- Architecture: YOLO11s (9.4M parameters, 21.3 GFLOPs)
- Training: 15 epochs on custom agricultural dataset
- Input: RGB images (any resolution, resized to inference size)
- Weights: 19MB

## Usage

```python
import replicate

output = replicate.run(
    "nztinversive/cropscan-v3",
    input={
        "image": "https://example.com/crop-image.jpg",
        "conf": 0.25,
        "return_json": True,
    }
)
```

## Deploy

```bash
cd replicate
cog login
cog push r8.im/nztinversive/cropscan-v3
```
