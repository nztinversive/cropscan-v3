"""
CropScan v3 - Crop Disease Detection
YOLO11s model trained on 6-class agricultural dataset.
Classes: healthy, bacterial, fungal, viral, nutrient_stress, other_disease
"""

import json
from typing import Optional

from cog import BaseModel, BasePredictor, Input, Path
from ultralytics import YOLO


CLASS_NAMES = [
    "healthy",
    "bacterial",
    "fungal",
    "viral",
    "nutrient_stress",
    "other_disease",
]


class Output(BaseModel):
    """Output model for CropScan predictions."""

    image: Optional[Path] = None
    json_str: Optional[str] = None


class Predictor(BasePredictor):
    """CropScan v3 crop disease detection predictor."""

    def setup(self) -> None:
        """Load CropScan v3 model into memory."""
        self.model = YOLO("best.pt")

    def predict(
        self,
        image: Path = Input(description="Aerial or close-up crop image"),
        conf: float = Input(
            description="Confidence threshold", default=0.25, ge=0.0, le=1.0
        ),
        iou: float = Input(
            description="IoU threshold for NMS", default=0.45, ge=0.0, le=1.0
        ),
        imgsz: int = Input(
            description="Image size for inference",
            default=640,
            choices=[320, 416, 512, 640, 832, 1024],
        ),
        return_json: bool = Input(
            description="Return detection results as JSON with class names, confidences, and bounding boxes",
            default=False,
        ),
    ) -> Output:
        """Run crop disease detection and return annotated image with optional JSON."""
        result = self.model(str(image), conf=conf, iou=iou, imgsz=imgsz)[0]

        # Save annotated image
        image_path = "output.png"
        result.save(image_path)

        if return_json:
            # Build structured JSON output
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append(
                    {
                        "class": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                    }
                )

            summary = {
                "total_detections": len(detections),
                "detections": detections,
                "classes_found": list(set(d["class"] for d in detections)),
                "model": "cropscan-v3",
                "image_size": imgsz,
            }
            return Output(image=Path(image_path), json_str=json.dumps(summary, indent=2))

        return Output(image=Path(image_path))
