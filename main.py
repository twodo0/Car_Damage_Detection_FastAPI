"""
FastAPI for YOLO→ViT pipeline with Grad-CAM heatmap generation
Strict contract with Spring backend - DO NOT modify response schema
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
import requests
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from ultralytics import YOLO
from transformers import AutoImageProcessor, ViTForImageClassification


# ============================================================================
# 2. TYPES, CONSTANTS, LABELS, SCHEMAS
# ============================================================================

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device selection priority: cuda > mps > cpu
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {DEVICE}")

# ViT 4-class labels (fixed order)
IDX2LABEL = ["CAR_DAMAGE", "DENT", "GLASS_BREAK", "SCRATCH"]


# Pydantic schemas for API contract
class ApiClassProbDto(BaseModel):
    label: str
    prob: float
    
    class Config:
        populate_by_name = True


class BoxDto(BaseModel):
    class_probs: List[ApiClassProbDto]
    x: float
    y: float
    w: float
    h: float
    
    class Config:
        populate_by_name = True


class PredictOut(BaseModel):
    model: str
    threshold_used: float
    boxes: List[BoxDto]
    
    class Config:
        populate_by_name = True
        # Ensure field names match Spring's @JsonProperty expectations
        json_schema_extra = {
            "example": {
                "model": "yolo-vit",
                "threshold_used": 0.35,
                "boxes": [
                    {
                        "class_probs": [
                            {"label": "DENT", "prob": 0.82}
                        ],
                        "x": 0.123,
                        "y": 0.222,
                        "w": 0.111,
                        "h": 0.090
                    }
                ]
            }
        }


class PredictIn(BaseModel):
    raw_url: str
    model: Optional[str] = "yolo-vit"
    # Spring sends "preview_url", but we also accept "preview_put_url" for compatibility
    preview_url: Optional[str] = Field(None, alias="preview_url")
    preview_put_url: Optional[str] = Field(None, alias="preview_put_url")
    heatmap_put_url: Optional[str] = None
    yolo_conf: Optional[float] = None
    vit_thresh: Optional[float] = None
    yoloThreshold: Optional[float] = Field(None, alias="yoloThreshold")
    vitThreshold: Optional[float] = Field(None, alias="vitThreshold")
    crop_padding: Optional[float] = 0.12
    min_crop_side_px: Optional[int] = 64
    hard_min_crop_side_px: Optional[int] = 8
    upsample_small: Optional[bool] = True
    max_upsample_scale: Optional[float] = 8.0
    adaptive_expand: Optional[bool] = True
    
    class Config:
        populate_by_name = True
    
    def get_preview_url(self) -> Optional[str]:
        """Get preview URL from either field"""
        return self.preview_url or self.preview_put_url
    
    def get_yolo_conf(self) -> Optional[float]:
        """Get YOLO threshold from either field"""
        return self.yolo_conf or self.yoloThreshold
    
    def get_vit_thresh(self) -> Optional[float]:
        """Get ViT threshold from either field"""
        return self.vit_thresh or self.vitThreshold


# ============================================================================
# 3. UTILITY FUNCTIONS
# ============================================================================

def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to [min_val, max_val]"""
    return max(min_val, min(max_val, value))


def xyxy_to_xywh_norm(xyxy: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert xyxy absolute coords to normalized xywh (top-left based)"""
    x1, y1, x2, y2 = xyxy
    x = float(x1 / img_w)
    y = float(y1 / img_h)
    w = float((x2 - x1) / img_w)
    h = float((y2 - y1) / img_h)
    return x, y, w, h


def xywh_norm_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert normalized xywh to absolute xyxy"""
    x1 = int(x * img_w)
    y1 = int(y * img_h)
    x2 = int((x + w) * img_w)
    y2 = int((y + h) * img_h)
    return x1, y1, x2, y2


def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Apply jet colormap to heatmap [0, 1]
    Returns RGB array (H, W, 3) with values [0, 255]
    
    Jet colormap:
    0.00-0.25: Blue to Cyan (R:0, G:0→255, B:255)
    0.25-0.50: Cyan to Green (R:0, G:255, B:255→0)
    0.50-0.75: Green to Yellow (R:0→255, G:255, B:0)
    0.75-1.00: Yellow to Red (R:255, G:255→0, B:0)
    """
    h, w = heatmap.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Vectorized jet colormap
    v = heatmap.flatten()
    
    # Calculate RGB channels
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    
    # Blue to Cyan (0.0 - 0.25)
    mask1 = v < 0.25
    r[mask1] = 0
    g[mask1] = (v[mask1] * 4 * 255).astype(np.uint8)
    b[mask1] = 255
    
    # Cyan to Green (0.25 - 0.5)
    mask2 = (v >= 0.25) & (v < 0.5)
    r[mask2] = 0
    g[mask2] = 255
    b[mask2] = ((1 - (v[mask2] - 0.25) * 4) * 255).astype(np.uint8)
    
    # Green to Yellow (0.5 - 0.75)
    mask3 = (v >= 0.5) & (v < 0.75)
    r[mask3] = ((v[mask3] - 0.5) * 4 * 255).astype(np.uint8)
    g[mask3] = 255
    b[mask3] = 0
    
    # Yellow to Red (0.75 - 1.0)
    mask4 = v >= 0.75
    r[mask4] = 255
    g[mask4] = ((1 - (v[mask4] - 0.75) * 4) * 255).astype(np.uint8)
    b[mask4] = 0
    
    colored[:, :, 0] = r.reshape(h, w)
    colored[:, :, 1] = g.reshape(h, w)
    colored[:, :, 2] = b.reshape(h, w)
    
    return colored


def crop_and_upsample(
    image: Image.Image,
    x: float, y: float, w: float, h: float,
    padding: float = 0.12,
    min_crop_side: int = 64,
    hard_min_crop_side: int = 8,
    upsample_small: bool = True,
    max_upsample_scale: float = 8.0,
    adaptive_expand: bool = True
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crop bbox with padding and upsample if needed
    Returns: (RGB PIL Image, actual_crop_coords_xyxy)
    """
    img_w, img_h = image.size
    
    # Apply padding
    pad_w = w * padding
    pad_h = h * padding
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(1, x + w + pad_w)
    y2 = min(1, y + h + pad_h)
    
    # Convert to absolute coords
    abs_x1, abs_y1, abs_x2, abs_y2 = xywh_norm_to_xyxy(x1, y1, x2-x1, y2-y1, img_w, img_h)
    
    crop_w = abs_x2 - abs_x1
    crop_h = abs_y2 - abs_y1
    
    # Hard minimum check
    if crop_w < hard_min_crop_side or crop_h < hard_min_crop_side:
        logger.warning(f"Crop too small ({crop_w}x{crop_h}), expanding to hard minimum")
        center_x = (abs_x1 + abs_x2) // 2
        center_y = (abs_y1 + abs_y2) // 2
        half_size = hard_min_crop_side // 2
        abs_x1 = max(0, center_x - half_size)
        abs_y1 = max(0, center_y - half_size)
        abs_x2 = min(img_w, center_x + half_size)
        abs_y2 = min(img_h, center_y + half_size)
        crop_w = abs_x2 - abs_x1
        crop_h = abs_y2 - abs_y1
    
    # Adaptive expansion to soft minimum
    if adaptive_expand and (crop_w < min_crop_side or crop_h < min_crop_side):
        needed_w = max(0, min_crop_side - crop_w)
        needed_h = max(0, min_crop_side - crop_h)
        expand_w = needed_w // 2
        expand_h = needed_h // 2
        
        abs_x1 = max(0, abs_x1 - expand_w)
        abs_y1 = max(0, abs_y1 - expand_h)
        abs_x2 = min(img_w, abs_x2 + expand_w + (needed_w % 2))
        abs_y2 = min(img_h, abs_y2 + expand_h + (needed_h % 2))
        crop_w = abs_x2 - abs_x1
        crop_h = abs_y2 - abs_y1
    
    # Store actual crop coordinates
    actual_crop_coords = (abs_x1, abs_y1, abs_x2, abs_y2)
    
    # Crop
    crop = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
    
    # Upsample if still too small
    if upsample_small and (crop_w < min_crop_side or crop_h < min_crop_side):
        scale = min(max_upsample_scale, min_crop_side / min(crop_w, crop_h))
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.debug(f"Upsampled crop from {crop_w}x{crop_h} to {new_w}x{new_h}")
    
    return crop.convert("RGB"), actual_crop_coords


# ============================================================================
# 4. ADAPTER CLASSES
# ============================================================================

class YOLOAdapter:
    """YOLO detection adapter"""
    
    def __init__(self, weights_path: str, device: str):
        self.model = YOLO(weights_path)
        self.device = device
        logger.info(f"YOLO loaded from {weights_path}")
    
    def detect(self, image: Image.Image, conf: float = 0.3) -> List[Tuple[float, float, float, float]]:
        """
        Run YOLO detection and return normalized boxes [(x,y,w,h), ...]
        """
        results = self.model.predict(
            image,
            conf=conf,
            iou=0.65,
            max_det=300,
            device=self.device,
            verbose=False
        )
        
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            img_w, img_h = image.size
            for box in results[0].boxes:
                # Get xyxy coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x, y, w, h = xyxy_to_xywh_norm(xyxy, img_w, img_h)
                boxes.append((x, y, w, h))
        
        logger.info(f"YOLO detected {len(boxes)} boxes")
        return boxes


class ViTAdapter:
    """ViT classifier adapter with Grad-CAM support"""
    
    def __init__(self, model_name: str, weights_path: str, device: str):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=4,
            ignore_mismatched_sizes=True
        )
        
        # Load custom weights
        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"ViT loaded from {weights_path}")
    
    def classify(self, crop: Image.Image) -> List[float]:
        """
        Classify crop and return probability distribution [CAR_DAMAGE, DENT, GLASS_BREAK, SCRATCH]
        """
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        return probs.tolist()
    

    def get_gradcam(self, crop: Image.Image, target_class: int) -> np.ndarray:

        """
        ViT CAM (attention 없이, hidden_state + classifier weight 만 사용)
        -> numpy array (H, W), values in [0,1]
        """

        # 1) 입력 준비
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        pixel_values = inputs["pixel_values"]  # (1, 3, 224, 224) 정도

        # 2) forward (hidden_states만 요청)
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        logits = outputs.logits                         # (1, num_labels)
        hidden = outputs.hidden_states[-1]              # (1, num_tokens, hidden_dim)

        # 3) CLS 토큰 빼고 patch embedding만 사용
        #    patch_features: (num_patches, hidden_dim)
        patch_features = hidden[0, 1:, :]

        # 4) classifier weight에서 target_class에 해당하는 weight 벡터 꺼냄
        #    classifier.weight: (num_labels, hidden_dim)
        class_w = self.model.classifier.weight[target_class]  # (hidden_dim,)
        class_w = class_w.to(patch_features.device)

        # 5) 각 patch feature · class_w => patch별 score (CAM)
        #    cam: (num_patches,)
        cam = torch.matmul(patch_features, class_w)  # (num_patches,)
        cam = torch.relu(cam)                        # 음수는 0

        cam_np = cam.detach().cpu().numpy()

        # 6) 1차원 -> 2D grid 로 reshape
        grid_size = int(np.sqrt(cam_np.shape[0]))
        cam_np = cam_np.reshape(grid_size, grid_size)  # (G, G)

        # 7) [0,1] 정규화
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np[:] = 0.5

        # 8) 간단 smoothing (선택사항)
        h, w = cam_np.shape
        smoothed = np.zeros_like(cam_np)
        for i in range(h):
            for j in range(w):
                i_min, i_max = max(0, i-1), min(h, i+2)
                j_min, j_max = max(0, j-1), min(w, j+2)
                smoothed[i, j] = cam_np[i_min:i_max, j_min:j_max].mean()
        cam_np = smoothed

        # 9) 다시 [0,1] 정규화
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np[:] = 0.5

        # 10) crop 크기로 resize
        cam_img = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
        cam_resized = cam_img.resize(crop.size, Image.Resampling.BICUBIC)
        cam_array = np.array(cam_resized).astype(np.float32) / 255.0

        return cam_array



class GradCAMGenerator:
    """Grad-CAM heatmap generator for multiple boxes"""
    
    def __init__(self, vit_adapter: ViTAdapter):
        self.vit = vit_adapter
    
    def generate_combined_heatmap(
        self,
        original_image: Image.Image,
        boxes_data: List[Tuple[Image.Image, int, float, float, float, float]]
    ) -> Image.Image:
        """
        Generate combined Grad-CAM heatmap for all boxes
        
        CRITICAL: Output must be EXACT same size as original_image
        Each heatmap is placed at the EXACT bbox location (not crop location)
        
        Args:
            original_image: Original full image
            boxes_data: List of (crop_image, top_class_idx, norm_x, norm_y, norm_w, norm_h)
                       norm_x/y/w/h are the ORIGINAL YOLO bbox coordinates
        
        Returns:
            RGBA PIL Image with jet colormap heatmap overlay (same size as original)
        """
        img_w, img_h = original_image.size
        logger.info(f"Creating heatmap canvas: {img_w}x{img_h} (same as original)")
        
        # Create RGBA canvas with EXACT same size as original image
        heatmap_canvas = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
        
        for idx, (crop, top_class, norm_x, norm_y, norm_w, norm_h) in enumerate(boxes_data):
            logger.info(f"\n=== Box {idx+1}/{len(boxes_data)} ===")
            logger.info(f"Class: {IDX2LABEL[top_class]}")
            logger.info(f"Normalized bbox: x={norm_x:.4f}, y={norm_y:.4f}, w={norm_w:.4f}, h={norm_h:.4f}")
            
            # Generate CAM for this crop
            # CAM will be resized to crop size inside get_gradcam
            cam = self.vit.get_gradcam(crop, top_class)
            
            # ✅ CAM 통계 출력
            logger.info(f"Generated CAM shape: (H={cam.shape[0]}, W={cam.shape[1]})")
            logger.info(f"CAM statistics: min={cam.min():.4f}, max={cam.max():.4f}, mean={cam.mean():.4f}, std={cam.std():.4f}")
            
            # ⚠️ CAM이 거의 0이면 경고
            if cam.max() < 0.1:
                logger.warning(f"⚠️ CAM values are very low! max={cam.max():.4f}")
                logger.warning(f"This will result in a nearly transparent heatmap")
            
            # Calculate ABSOLUTE bbox coordinates (where to place heatmap)
            bbox_x1 = int(norm_x * img_w)
            bbox_y1 = int(norm_y * img_h)
            bbox_x2 = int((norm_x + norm_w) * img_w)
            bbox_y2 = int((norm_y + norm_h) * img_h)
            bbox_w = bbox_x2 - bbox_x1
            bbox_h = bbox_y2 - bbox_y1
            
            logger.info(f"Absolute bbox: ({bbox_x1}, {bbox_y1}) to ({bbox_x2}, {bbox_y2})")
            logger.info(f"Bbox size: W={bbox_w} x H={bbox_h}")
            
            if bbox_w <= 0 or bbox_h <= 0:
                logger.warning(f"Invalid bbox size, skipping")
                continue
            
            # CRITICAL: Resize CAM to EXACT bbox size
            # cam is numpy array (H, W), PIL Image.resize expects (W, H)
            cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode='L')
            logger.info(f"CAM as PIL Image size: {cam_img.size} (should be W x H)")
            
            cam_resized = cam_img.resize((bbox_w, bbox_h), Image.Resampling.BICUBIC)  # PIL: (width, height)
            logger.info(f"Resized CAM PIL size: {cam_resized.size} (should be {bbox_w} x {bbox_h})")
            
            cam_array = np.array(cam_resized).astype(np.float32) / 255.0
            logger.info(f"Resized CAM array shape: {cam_array.shape} (should be {bbox_h} x {bbox_w})")
            logger.info(f"Resized CAM stats: min={cam_array.min():.4f}, max={cam_array.max():.4f}")
            
            # Apply jet colormap
            cam_colored_rgb = apply_jet_colormap(cam_array)
            logger.info(f"CAM colored RGB shape: {cam_colored_rgb.shape} (should be {bbox_h} x {bbox_w} x 3)")
            
            # Create alpha channel (intensity-based transparency)
            # ✅ 강화된 alpha: 최소 50, 최대 255
            alpha_raw = cam_array * 255
            alpha = np.clip(alpha_raw, 0, 180).astype(np.uint8)  # 완전 투명 방지
            logger.info(f"Alpha channel shape: {alpha.shape} (should be {bbox_h} x {bbox_w})")
            logger.info(f"Alpha stats: min={alpha.min()}, max={alpha.max()}, mean={alpha.mean():.1f}")
            
            # Create RGBA heatmap for this box
            # cam_colored_rgb: (H, W, 3), alpha: (H, W)
            cam_rgba = np.dstack([cam_colored_rgb, alpha])  # (H, W, 4)
            logger.info(f"RGBA array shape: {cam_rgba.shape} (should be {bbox_h} x {bbox_w} x 4)")
            
            heatmap_box = Image.fromarray(cam_rgba, mode='RGBA')
            logger.info(f"RGBA PIL Image size: {heatmap_box.size} (should be {bbox_w} x {bbox_h})")
            
            # Verify sizes match
            if heatmap_box.size != (bbox_w, bbox_h):
                logger.error(f"⚠️ SIZE MISMATCH! Heatmap PIL: {heatmap_box.size}, Expected: ({bbox_w}, {bbox_h})")
                logger.error(f"PIL Image .size is (W, H) but we expected ({bbox_w}, {bbox_h})")
                continue
            
            # Paste at EXACT bbox location
            logger.info(f"Pasting heatmap at ({bbox_x1}, {bbox_y1})")
            heatmap_canvas.paste(heatmap_box, (bbox_x1, bbox_y1), heatmap_box)
            
            logger.info(f"Box {idx+1} heatmap pasted successfully")
        
        logger.info(f"\n=== Final heatmap canvas: {heatmap_canvas.size} ===")
        logger.info(f"Should match original image: {img_w}x{img_h}")
        
        return heatmap_canvas


# ============================================================================
# 5. ENVIRONMENT VARIABLES & MODEL LOADING
# ============================================================================

# Read environment variables
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo11s.pt")
VIT_WEIGHTS = os.getenv("VIT_WEIGHTS", "/Users/dopal0426/Desktop/dev/model/vit_car_damage_model.pth")
VIT_MODEL_NAME = os.getenv("VIT_MODEL_NAME", "google/vit-base-patch16-224-in21k")
DEFAULT_YOLO_CONF = float(os.getenv("YOLO_DEFAULT_CONF", "0.3"))
DEFAULT_VIT_THRESH = float(os.getenv("VIT_DEFAULT_THRESH", "0.3"))

# Global model instances (will be loaded in lifespan)
yolo_adapter: Optional[YOLOAdapter] = None
vit_adapter: Optional[ViTAdapter] = None
gradcam_generator: Optional[GradCAMGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle"""
    global yolo_adapter, vit_adapter, gradcam_generator
    
    # Startup: Load models
    try:
        logger.info("Loading models...")
        yolo_adapter = YOLOAdapter(YOLO_WEIGHTS, DEVICE)
        vit_adapter = ViTAdapter(VIT_MODEL_NAME, VIT_WEIGHTS, DEVICE)
        gradcam_generator = GradCAMGenerator(vit_adapter)
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    
    yield
    
    # Shutdown: Cleanup if needed
    logger.info("Shutting down...")


# ============================================================================
# 6. FASTAPI APP & ROUTES
# ============================================================================

app = FastAPI(
    title="YOLO-ViT Detection API",
    version="2.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictOut)
async def predict(
    body: PredictIn,
    # Query params (override body if provided)
    yolo_conf: Optional[float] = Query(None),
    vit_thresh: Optional[float] = Query(None),
    model: Optional[str] = Query(None),
):
    """
    Vehicle damage detection endpoint
    
    Pipeline: YOLO detection → Crop → ViT classification → Filter by ViT threshold
    
    Optional: Generate Grad-CAM heatmap overlay
    """
    # Priority: Query > Body > Env > Default
    # Use helper methods to support both Spring field names and original names
    body_yolo_conf = body.get_yolo_conf()
    body_vit_thresh = body.get_vit_thresh()
    
    eff_yolo_conf = clamp(
        yolo_conf if yolo_conf is not None
        else body_yolo_conf if body_yolo_conf is not None
        else DEFAULT_YOLO_CONF
    )
    
    eff_vit_thresh = clamp(
        vit_thresh if vit_thresh is not None
        else body_vit_thresh if body_vit_thresh is not None
        else DEFAULT_VIT_THRESH
    )
    
    eff_model = model or body.model or "yolo-vit"
    
    # Get preview URL (supports both preview_url and preview_put_url)
    preview_url = body.get_preview_url()
    
    logger.info(f"Request: model={eff_model}, yolo_conf={eff_yolo_conf}, vit_thresh={eff_vit_thresh}")
    logger.info(f"Preview URL: {preview_url is not None}, Heatmap URL: {body.heatmap_put_url is not None}")
    
    # Download original image
    try:
        response = requests.get(body.raw_url, timeout=30)
        response.raise_for_status()
        original_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        logger.info(f"Downloaded image: {original_image.size}")
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
    
    # YOLO detection
    yolo_boxes = yolo_adapter.detect(original_image, conf=eff_yolo_conf)
    
    if not yolo_boxes:
        logger.info("No boxes detected by YOLO")
        
        # Upload original image as preview (no boxes to draw)
        if preview_url:
            try:
                logger.info("Uploading original image as preview (no detections)")
                buffer = io.BytesIO()
                original_image.save(buffer, format="PNG")
                buffer.seek(0)
                
                put_response = requests.put(
                    preview_url,
                    data=buffer.getvalue(),
                    headers={"Content-Type": "image/png"},
                    timeout=30
                )
                
                if put_response.status_code in (200, 201, 204):
                    logger.info("Preview (no boxes) uploaded successfully")
                else:
                    logger.warning(f"Preview upload failed: {put_response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to upload preview: {e}")
        
        return PredictOut(
            model=eff_model,
            threshold_used=eff_vit_thresh,
            boxes=[]
        )
    
    # Process each box with ViT
    final_boxes = []
    heatmap_data = []  # For Grad-CAM: (crop, top_class, norm_x, norm_y, norm_w, norm_h)
    
    for x, y, w, h in yolo_boxes:
        # Crop with padding and upsampling - now returns crop coordinates
        crop, actual_coords = crop_and_upsample(
            original_image,
            x, y, w, h,
            padding=body.crop_padding,
            min_crop_side=body.min_crop_side_px,
            hard_min_crop_side=body.hard_min_crop_side_px,
            upsample_small=body.upsample_small,
            max_upsample_scale=body.max_upsample_scale,
            adaptive_expand=body.adaptive_expand
        )
        
        # ViT classification
        probs = vit_adapter.classify(crop)
        
        # Get top1
        top1_idx = int(np.argmax(probs))
        top1_prob = probs[top1_idx]
        
        # Filter by ViT threshold
        if top1_prob < eff_vit_thresh:
            logger.debug(f"Box filtered: top1={IDX2LABEL[top1_idx]} ({top1_prob:.3f}) < {eff_vit_thresh}")
            continue
        
        # Create class_probs in descending order
        class_probs = [
            ApiClassProbDto(label=IDX2LABEL[i], prob=float(probs[i]))
            for i in np.argsort(probs)[::-1]
        ]
        
        box_dto = BoxDto(
            class_probs=class_probs,
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h)
        )
        final_boxes.append(box_dto)
        
        # Store for Grad-CAM: use ORIGINAL YOLO bbox (not padded crop)
        # This ensures heatmap is drawn at the same location as the preview box
        heatmap_data.append((crop, top1_idx, x, y, w, h))
    
    logger.info(f"Final boxes after ViT filtering: {len(final_boxes)}")
    
    # Generate preview image with bounding boxes
    if preview_url and final_boxes:
        try:
            logger.info(f"Generating preview image with {len(final_boxes)} boxes")
            preview_img = original_image.copy()
            draw = ImageDraw.Draw(preview_img)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    # Try DejaVuSans on Linux
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            img_w, img_h = preview_img.size
            
            for idx, box in enumerate(final_boxes):
                x1, y1, x2, y2 = xywh_norm_to_xyxy(box.x, box.y, box.w, box.h, img_w, img_h)
                
                logger.debug(f"Drawing box {idx+1}: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw label chip
                top1_label = box.class_probs[0].label
                top1_prob = box.class_probs[0].prob
                label_text = f"{top1_label} {top1_prob*100:.0f}%"
                
                # Background for text
                text_bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1 - 20), label_text, fill="white", font=font)
            
            # Upload preview to MinIO
            buffer = io.BytesIO()
            preview_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            logger.info(f"Uploading preview PNG ({len(buffer.getvalue())} bytes) to MinIO")
            
            put_response = requests.put(
                preview_url,
                data=buffer.getvalue(),
                headers={"Content-Type": "image/png"},
                timeout=30
            )
            
            if put_response.status_code in (200, 201, 204):
                logger.info(f"Preview uploaded successfully (status: {put_response.status_code})")
            else:
                logger.warning(f"Preview upload failed: {put_response.status_code} - {put_response.text}")
        except Exception as e:
            logger.error(f"Failed to generate/upload preview: {e}", exc_info=True)
    
    # Generate Grad-CAM heatmap
    if body.heatmap_put_url and heatmap_data:
        try:
            logger.info(f"Generating Grad-CAM for {len(heatmap_data)} boxes")
            heatmap_img = gradcam_generator.generate_combined_heatmap(
                original_image,
                heatmap_data
            )
            
            # Upload heatmap
            buffer = io.BytesIO()
            heatmap_img.save(buffer, format="PNG")
            buffer.seek(0)
            
            put_response = requests.put(
                body.heatmap_put_url,
                data=buffer.getvalue(),
                headers={"Content-Type": "image/png"},
                timeout=30
            )
            
            if put_response.status_code in (200, 201, 204):
                logger.info("Heatmap uploaded successfully")
            else:
                logger.warning(f"Heatmap upload failed: {put_response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to generate/upload heatmap: {e}")
    
    # Return response (strict contract with Spring)
    return PredictOut(
        model=eff_model,
        threshold_used=eff_vit_thresh,
        boxes=final_boxes
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": yolo_adapter is not None and vit_adapter is not None
    }


# ============================================================================
# 7. TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)