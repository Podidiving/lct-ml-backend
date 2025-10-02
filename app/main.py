import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from ultralytics.models.yolo import YOLO

from app.constants import (
    DETECTOR_IMGSZ,
    DETECTOR_INDEX_TO_LABEL,
    DETECTOR_WEIGHTS,
    INDEX_TO_LABEL,
    MODEL_IMGSZ,
    MODEL_WEIGHTS,
)

DEVICE = os.getenv("DEVICE", "auto")  # "cpu" | "cuda" | "mps" | "auto"
_model_weights = os.getenv("MODEL_WEIGHTS", MODEL_WEIGHTS)
_detector_weights = os.getenv("DETECTOR_WEIGHTS", DETECTOR_WEIGHTS)


def choose_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    if torch.cuda.is_available():
        device = "cuda"
    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = "mps"
    except Exception as e:
        print(f"Failed during device selection: {e}")
        print("Continuing with CPU")
    device = "cpu"
    print(f"Using device: {device}")
    return device


def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str and b64_str.split(",", 1)[0].lower().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_str, validate=True)
    except Exception:
        try:
            img_bytes = base64.b64decode(b64_str)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Некорректный base64 изображения"
            )
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")


def crop_image_if_needed(
    img: Image.Image,
    x1: Optional[int],
    y1: Optional[int],
    x2: Optional[int],
    y2: Optional[int],
) -> Image.Image:
    if x1 is None and y1 is None and x2 is None and y2 is None:
        return img

    assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
    W, H = img.size

    if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
        raise HTTPException(
            status_code=400,
            detail=f"Неверные координаты кропа. Изображение {W}x{H}, пришло {(x1,y1,x2,y2)}",
        )

    return img.crop((x1, y1, x2, y2))


class ClassifyRequest(BaseModel):
    image_base64: str
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None


class ClassifyResponse(BaseModel):
    label: str
    confidence: float


class ImageKeyPair(BaseModel):
    key: str
    image_base64: str


class DetectionRequest(BaseModel):
    requests: list[ImageKeyPair]


class ObjectBoxResponse(BaseModel):
    key: str
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    results: list[ObjectBoxResponse]


_model: Optional[YOLO] = None
_detector: Optional[YOLO] = None
_model_ready: bool = False
_device: str = choose_device()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_ready, _device, _detector
    _device = choose_device()
    try:
        _model = YOLO(_model_weights)
        _detector = YOLO(_detector_weights)

        _ = _model.predict(
            Image.new("RGB", (MODEL_IMGSZ, MODEL_IMGSZ)),
            imgsz=MODEL_IMGSZ,
            verbose=False,
            device=_device,
        )
        _ = _detector.predict(
            Image.new("RGB", (DETECTOR_IMGSZ, DETECTOR_IMGSZ)),
            imgsz=DETECTOR_IMGSZ,
            verbose=False,
            device=_device,
        )
        _model_ready = True
    except Exception as e:
        print(f"Failed to load model: {e}")
        _model_ready = False

    yield

    _model = None
    _detector = None
    _model_ready = False


app = FastAPI(title="LCT ML API", version="1.0.0", lifespan=lifespan)


@app.get("/livez")
def liveness():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    ready = bool(_model_ready) and len(INDEX_TO_LABEL) > 0
    status = "ready" if ready else "not_ready"
    return {
        "status": status,
        "model_loaded": bool(_model_ready),
        "classes": len(INDEX_TO_LABEL),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if len(INDEX_TO_LABEL) == 0:
        raise HTTPException(
            status_code=500,
            detail="Пустой маппинг классов INDEX_TO_LABEL. Заполните app/constants.py",
        )

    global _model
    img = decode_base64_image(req.image_base64)
    img = crop_image_if_needed(img, req.x1, req.y1, req.x2, req.y2)

    results = _model.predict(img, imgsz=MODEL_IMGSZ, verbose=False, device=_device)
    if not results:
        raise HTTPException(status_code=500, detail="Модель не вернула результат")

    r0 = results[0]
    if not hasattr(r0, "probs") or r0.probs is None:
        raise HTTPException(
            status_code=500, detail="Отсутствует вероятность классов (probs)"
        )

    top_idx = int(r0.probs.top1)
    label = INDEX_TO_LABEL.get(top_idx)
    if label is None:
        raise HTTPException(
            status_code=500, detail=f"Нет маппинга для индекса {top_idx}"
        )

    return ClassifyResponse(label=label, confidence=float(r0.probs.top1conf.item()))


@app.post("/detect", response_model=DetectionResponse)
def detect(req: DetectionRequest):
    if len(DETECTOR_INDEX_TO_LABEL) == 0:
        raise HTTPException(
            status_code=500,
            detail="Пустой маппинг классов DETECTOR_INDEX_TO_LABEL. Заполните app/constants.py",
        )

    global _detector
    responses: list[ObjectBoxResponse] = []

    for pair in req.requests:
        img = decode_base64_image(pair.image_base64)
        img_width, img_height = img.size

        results = _detector.predict(
            img, imgsz=DETECTOR_IMGSZ, verbose=False, device=_device
        )
        if not results:
            continue

        r0 = results[0]
        if r0.boxes is None:
            continue

        for box in r0.boxes:
            cls_id = int(box.cls.item())
            label = DETECTOR_INDEX_TO_LABEL.get(cls_id)
            if label is None:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Convert to relative coordinates (0.0 to 1.0)
            rel_x1 = x1 / img_width
            rel_y1 = y1 / img_height
            rel_x2 = x2 / img_width
            rel_y2 = y2 / img_height
            
            responses.append(
                ObjectBoxResponse(
                    key=pair.key,
                    label=label,
                    confidence=conf,
                    x1=rel_x1,
                    y1=rel_y1,
                    x2=rel_x2,
                    y2=rel_y2,
                )
            )
    return DetectionResponse(results=responses)
