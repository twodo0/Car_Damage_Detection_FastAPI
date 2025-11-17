"""
FastAPI for YOLO→ViT pipeline (Batch Rental System)
Strict contract with Spring backend - DO NOT modify response schema

TRAIN ORDER (fixed):
0=BREAKAGE, 1=CRUSHED, 2=NORMAL, 3=SCRATCHED, 4=SEPARATED

API labels (Spring/Front): BREAKAGE, CRUSHED, SCRATCHED, SEPARATED
(NORMAL is filtered out)
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import os
import io
import logging
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
import httpx
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ultralytics import YOLO

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import AutoImageProcessor, ViTForImageClassification

# ============================================================================
# 2. LOGGING & DEVICE
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {DEVICE}")

# ============================================================================
# 3. LABEL MAPS (🔥 핵심: 학습 순서 고정)
# ============================================================================

# 학습 당시 인덱스 → API 라벨(대문자)
# {"Breakage":0,"Crushed":1,"Normal":2,"Scratched":3,"Separated":4}
TRAIN_ID2LABEL = ["BREAKAGE", "CRUSHED", "NORMAL", "SCRATCHED", "SEPARATED"]
LABEL2TRAIN_ID = {name: idx for idx, name in enumerate(TRAIN_ID2LABEL)}

# API 계약(4종 손상) 고정 순서
DAMAGE_API_4 = ["BREAKAGE", "CRUSHED", "SCRATCHED", "SEPARATED"]
# softmax를 취할 때 사용할 학습 인덱스 집합(= NORMAL 제외)
DAMAGE_TRAIN_IDS = [LABEL2TRAIN_ID[l] for l in DAMAGE_API_4]  # [0,1,3,4]

logger.info(f"TRAIN_ID2LABEL = {TRAIN_ID2LABEL}")
logger.info(f"DAMAGE_TRAIN_IDS (by API order {DAMAGE_API_4}) = {DAMAGE_TRAIN_IDS}")

# ============================================================================
# 4. PYDANTIC SCHEMAS (API 계약 유지)
# ============================================================================

class ApiClassProb(BaseModel):
    label: str   # One of: BREAKAGE, CRUSHED, SCRATCHED, SEPARATED
    prob: float
    class Config:
        populate_by_name = True

class BoxDto(BaseModel):
    class_probs: List[ApiClassProb] = Field(default_factory=list)
    x: float  # Normalized (0-1)
    y: float
    w: float
    h: float
    class Config:
        populate_by_name = True

class PredictReq(BaseModel):
    raw_url: str
    yoloThreshold: float = 0.3
    heatmap_put_url: Optional[str] = None
    class Config:
        populate_by_name = True

class PredictRes(BaseModel):
    model: str = "yolo-vit"
    threshold_used: float
    boxes: List[BoxDto] = Field(default_factory=list)
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "model": "yolo-vit",
                "threshold_used": 0.3,
                "boxes": [
                    {
                        "class_probs": [
                            {"label": "BREAKAGE", "prob": 0.71},
                            {"label": "CRUSHED",  "prob": 0.11},
                            {"label": "SCRATCHED","prob": 0.10},
                            {"label": "SEPARATED","prob": 0.08}
                        ],
                        "x": 0.102, "y": 0.214, "w": 0.265, "h": 0.180
                    }
                ]
            }
        }

# ============================================================================
# 5. MODEL PATHS
# ============================================================================
YOLO_WEIGHTS   = os.getenv("YOLO_WEIGHTS",   "/Users/dopal0426/Desktop/dev/model/failed_yolo/best.pt")
VIT_WEIGHTS    = os.getenv("VIT_WEIGHTS",    "/Users/dopal0426/Desktop/dev/model/main_vit_model/vit_car_damage_model_cross_entropy_best.pth")
VIT_MODEL_NAME = os.getenv("VIT_MODEL_NAME", "google/vit-base-patch16-224-in21k")

# Globals
yolo_model: Optional[YOLO] = None
vit_model: Optional[ViTForImageClassification] = None
vit_processor: Optional[AutoImageProcessor] = None

# ============================================================================
# 6. MODEL ADAPTERS
# ============================================================================

class YOLOAdapter:
    def __init__(self, model: YOLO):
        self.model = model
        logger.info(f"YOLO loaded from {YOLO_WEIGHTS}")

    def detect(self, image: Image.Image, conf_threshold: float) -> List[dict]:
        results = self.model(image, conf=conf_threshold, verbose=False)
        boxes_out = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(boxes.conf[i].cpu().numpy())
                boxes_out.append({
                    "left": x1,
                    "top": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "conf": conf
                })
        logger.info(f"YOLO detected {len(boxes_out)} boxes")
        return boxes_out

class ViTAdapter:
    """학습 라벨 순서(TRAIN_ID2LABEL)에 맞춰 NORMAL 필터/softmax/라벨 문자열 생성"""
    def __init__(self, model: ViTForImageClassification, processor: AutoImageProcessor):
        self.model = model
        self.processor = processor
        self.model.eval()
        logger.info(f"ViT loaded from {VIT_WEIGHTS}")

    def classify(self, crop: Image.Image) -> dict:
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # shape [5]

        top1_idx = int(torch.argmax(logits).item())
        top1_label = TRAIN_ID2LABEL[top1_idx]
        logger.info(f"RAW top1 index = {top1_idx} ({top1_label})")

        # NORMAL(top1) → 필터링
        if top1_label == "NORMAL":
            return {"is_normal": True, "class_probs": [], "top1_idx": top1_idx}

        # 손상 4종(학습 인덱스 [0,1,3,4])만 소프트맥스 → API 라벨 순서로 방출
        damage_logits = logits[DAMAGE_TRAIN_IDS]  # len=4
        damage_probs = F.softmax(damage_logits, dim=0).cpu().numpy()
        logger.info("softmax(API order): " + ", ".join(
            [f"{lab}={prob:.3f}" for lab, prob in zip(DAMAGE_API_4, damage_probs)]
        ))

        class_probs = [
            ApiClassProb(label=api_lab, prob=float(prob))
            for api_lab, prob in zip(DAMAGE_API_4, damage_probs)
        ]

        return {
            "is_normal": False,
            "class_probs": class_probs,   # API 라벨로 통일
            "top1_idx": top1_idx          # Grad-CAM용 학습 인덱스 유지
        }

    def get_gradcam(self, crop: Image.Image, target_class: int) -> np.ndarray:
        """Grad-CAM (attention rollout). target_class는 '학습 인덱스' 사용"""
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_attentions=True)
        attentions = outputs.attentions

        num_tokens = attentions[0].shape[-1]
        rollout = torch.eye(num_tokens).to(DEVICE)
        for attention in attentions:
            att = attention.mean(dim=1)[0]  # (tokens, tokens)
            att = att + torch.eye(num_tokens).to(DEVICE)
            att = att / att.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(rollout, att)

        mask = rollout[0, 1:].detach().cpu().numpy()  # CLS→patches

        if target_class < 0 or target_class >= 5:
            cam = mask
        else:
            classifier_weights = self.model.classifier.weight[target_class].detach().cpu().numpy()
            with torch.no_grad():
                out_hid = self.model(**inputs, output_hidden_states=True)
                last_hidden = out_hid.hidden_states[-1][0]               # (tokens, hidden)
                patch_features = last_hidden[1:].detach().cpu().numpy()  # (num_patches, hidden)
            class_activation = np.dot(patch_features, classifier_weights)
            cam = mask * class_activation

        cam = np.maximum(cam, 0)
        gs = int(np.sqrt(len(cam)))
        cam = cam.reshape(gs, gs)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.ones_like(cam) * 0.5

        # 3x3 평균 스무딩
        h, w = cam.shape
        smoothed = np.zeros_like(cam)
        for i in range(h):
            for j in range(w):
                i0, i1 = max(0, i-1), min(h, i+2)
                j0, j1 = max(0, j-1), min(w, j+2)
                smoothed[i, j] = cam[i0:i1, j0:j1].mean()
        cam = smoothed
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode='L')
        cam_resized = cam_img.resize(crop.size, Image.Resampling.BICUBIC)
        cam_array = np.array(cam_resized).astype(np.float32) / 255.0
        return cam_array

# ============================================================================
# 7. HEATMAP GENERATION
# ============================================================================

def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    h, w = heatmap.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    v = heatmap.flatten()
    r = np.zeros_like(v); g = np.zeros_like(v); b = np.zeros_like(v)

    m1 = v < 0.25
    r[m1] = 0; g[m1] = (v[m1] * 4 * 255).astype(np.uint8); b[m1] = 255
    m2 = (v >= 0.25) & (v < 0.5)
    r[m2] = 0; g[m2] = 255; b[m2] = ((1 - (v[m2] - 0.25) * 4) * 255).astype(np.uint8)
    m3 = (v >= 0.5) & (v < 0.75)
    r[m3] = ((v[m3] - 0.5) * 4 * 255).astype(np.uint8); g[m3] = 255; b[m3] = 0
    m4 = v >= 0.75
    r[m4] = 255; g[m4] = ((1 - (v[m4] - 0.75) * 4) * 255).astype(np.uint8); b[m4] = 0

    colored[:, :, 0] = r.reshape(h, w)
    colored[:, :, 1] = g.reshape(h, w)
    colored[:, :, 2] = b.reshape(h, w)
    return colored

def crop_and_upsample(
    image: Image.Image,
    x: float, y: float, w: float, h: float,
    padding: float = 0.12,
    min_crop_side: int = 64,
    hard_min_crop_side: int = 8
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    img_w, img_h = image.size
    pad_w = w * padding; pad_h = h * padding
    x1 = max(0, x - pad_w); y1 = max(0, y - pad_h)
    x2 = min(1, x + w + pad_w); y2 = min(1, y + h + pad_h)

    abs_x1 = int(x1 * img_w); abs_y1 = int(y1 * img_h)
    abs_x2 = int(x2 * img_w); abs_y2 = int(y2 * img_h)
    crop_w = abs_x2 - abs_x1; crop_h = abs_y2 - abs_y1

    if crop_w < hard_min_crop_side or crop_h < hard_min_crop_side:
        center_x = (abs_x1 + abs_x2) // 2; center_y = (abs_y1 + abs_y2) // 2
        half = hard_min_crop_side // 2
        abs_x1 = max(0, center_x - half); abs_y1 = max(0, center_y - half)
        abs_x2 = min(img_w, center_x + half); abs_y2 = min(img_h, center_y + half)
        crop_w = abs_x2 - abs_x1; crop_h = abs_y2 - abs_y1

    if crop_w < min_crop_side or crop_h < min_crop_side:
        need_w = max(0, min_crop_side - crop_w); need_h = max(0, min_crop_side - crop_h)
        expand_w = need_w // 2; expand_h = need_h // 2
        abs_x1 = max(0, abs_x1 - expand_w); abs_y1 = max(0, abs_y1 - expand_h)
        abs_x2 = min(img_w, abs_x2 + expand_w + (need_w % 2))
        abs_y2 = min(img_h, abs_y2 + expand_h + (need_h % 2))
        crop_w = abs_x2 - abs_x1; crop_h = abs_y2 - abs_y1

    crop = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
    if crop_w < min_crop_side or crop_h < min_crop_side:
        scale = min(8.0, min_crop_side / min(crop_w, crop_h))
        new_w = int(crop_w * scale); new_h = int(crop_h * scale)
        crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return crop.convert("RGB"), (abs_x1, abs_y1, abs_x2, abs_y2)

def generate_gradcam_heatmap(
    image: Image.Image,
    boxes_data: List[Tuple[BoxDto, int, Tuple[int, int, int, int]]],
    vit_adapter: ViTAdapter
) -> Image.Image:
    img_w, img_h = image.size
    logger.info(f"Creating Grad-CAM heatmap canvas: {img_w}x{img_h}")
    heatmap_canvas = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))

    for idx, (box, top_class, _) in enumerate(boxes_data):
        top_name = TRAIN_ID2LABEL[top_class]
        logger.info(f"=== Box {idx+1}/{len(boxes_data)} | top1={top_name} "
                    f"| norm(xywh)=({box.x:.4f},{box.y:.4f},{box.w:.4f},{box.h:.4f})")

        crop, (abs_x1, abs_y1, abs_x2, abs_y2) = crop_and_upsample(image, box.x, box.y, box.w, box.h)
        bbox_w = abs_x2 - abs_x1; bbox_h = abs_y2 - abs_y1
        if bbox_w <= 0 or bbox_h <= 0:
            logger.warning("Invalid bbox size, skipping")
            continue

        cam = vit_adapter.get_gradcam(crop, top_class)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode='L')
        cam_resized = cam_img.resize((bbox_w, bbox_h), Image.Resampling.BICUBIC)
        cam_array = np.array(cam_resized).astype(np.float32) / 255.0

        cam_colored_rgb = apply_jet_colormap(cam_array)
        alpha = np.clip(cam_array * 255, 50, 255).astype(np.uint8)  # 최소 가시성
        heat_rgba = np.dstack([cam_colored_rgb, alpha])
        heatmap_box = Image.fromarray(heat_rgba, mode='RGBA')

        heatmap_canvas.paste(heatmap_box, (abs_x1, abs_y1), heatmap_box)

    logger.info(f"Final heatmap canvas: {heatmap_canvas.size}")
    return heatmap_canvas

# ============================================================================
# 8. LIFECYCLE MANAGEMENT
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, vit_model, vit_processor
    try:
        logger.info("Loading models...")

        yolo_model = YOLO(YOLO_WEIGHTS)
        yolo_model.to(DEVICE)

        vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
        vit_model = ViTForImageClassification.from_pretrained(
            VIT_MODEL_NAME,
            num_labels=5,
            ignore_mismatched_sizes=True,
            attn_implementation='eager'  # output_attentions 용
        )

        state_dict = torch.load(VIT_WEIGHTS, map_location=DEVICE)
        vit_model.load_state_dict(state_dict)
        vit_model.to(DEVICE)
        vit_model.eval()

        # config에도 학습 라벨 기록(디버깅/일관성)
        vit_model.config.id2label = {i: TRAIN_ID2LABEL[i] for i in range(5)}
        vit_model.config.label2id = {v: k for k, v in vit_model.config.id2label.items()}

        logger.info("All models loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    finally:
        logger.info("Shutting down...")

# ============================================================================
# 9. FASTAPI APP & ENDPOINTS
# ============================================================================
app = FastAPI(
    title="Vehicle Damage Detection API",
    description="YOLO + ViT pipeline for batch rental system",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": yolo_model is not None and vit_model is not None,
        "train_id2label": TRAIN_ID2LABEL,
    }

@app.post("/predict", response_model=PredictRes)
async def predict(req: PredictReq):
    logger.info(f"Request: yoloThreshold={req.yoloThreshold}")

    # 1) download image
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(req.raw_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_w, img_h = image.size
        logger.info(f"Downloaded image: {image.size}")
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=422, detail=f"raw_url fetch failed: {e}")

    # 2) YOLO detect
    yolo_adapter = YOLOAdapter(yolo_model)
    boxes_px = yolo_adapter.detect(image, conf_threshold=req.yoloThreshold)
    if not boxes_px:
        return PredictRes(model="yolo-vit", threshold_used=req.yoloThreshold, boxes=[])

    # 3-5) ViT classify & filter
    vit_adapter = ViTAdapter(vit_model, vit_processor)
    boxes_out: List[BoxDto] = []
    boxes_data_for_heatmap: List[Tuple[BoxDto, int, Tuple[int, int, int, int]]] = []

    for b in boxes_px:
        left = max(0, b["left"]); top = max(0, b["top"])
        right = min(img_w, left + b["width"]); bottom = min(img_h, top + b["height"])
        crop = image.crop((left, top, right, bottom))

        result = vit_adapter.classify(crop)
        if result["is_normal"]:
            continue

        top_class_idx = result["top1_idx"]  # 학습 인덱스 (Grad-CAM용)

        x_norm = left / img_w; y_norm = top / img_h
        w_norm = (right - left) / img_w; h_norm = (bottom - top) / img_h

        box_dto = BoxDto(
            class_probs=result["class_probs"],  # API 라벨/확률
            x=max(0.0, min(1.0, x_norm)),
            y=max(0.0, min(1.0, y_norm)),
            w=max(0.0, min(1.0, w_norm)),
            h=max(0.0, min(1.0, h_norm)),
        )
        boxes_out.append(box_dto)
        boxes_data_for_heatmap.append((box_dto, top_class_idx, (left, top, right, bottom)))

    logger.info(f"Final boxes after ViT filtering: {len(boxes_out)}")

    # 6) Heatmap gen & upload
    if boxes_out and req.heatmap_put_url:
        try:
            logger.info("Generating Grad-CAM heatmap...")
            heatmap = generate_gradcam_heatmap(image, boxes_data_for_heatmap, vit_adapter)
            buf = io.BytesIO(); heatmap.save(buf, format="PNG"); buf.seek(0)
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.put(req.heatmap_put_url, content=buf.getvalue(),
                                     headers={"Content-Type": "image/png"})
                r.raise_for_status()
            logger.info("Heatmap uploaded successfully")
        except Exception as e:
            logger.warning(f"Heatmap generation/upload failed: {e}")
    elif not boxes_out:
        logger.info("No boxes remain after filtering - skipping heatmap")
    elif not req.heatmap_put_url:
        logger.info("No heatmap_put_url provided - skipping heatmap")

    return PredictRes(model="yolo-vit", threshold_used=req.yoloThreshold, boxes=boxes_out)

# ============================================================================
# 10. ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)