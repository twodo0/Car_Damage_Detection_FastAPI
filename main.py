from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import List, Optional, Literal, Tuple
import os, io, requests

from PIL import Image
import numpy as np

# --- DL libs ---
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from ultralytics import YOLO


# ===== 디바이스 자동선택 (CUDA -> MPS -> CPU) =====
def pick_device_for_torch() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ===== 0) 라벨/스키마 (Spring 계약과 동일) =====
Label = Literal["CAR_DAMAGE","DENT","GLASS_BREAK","SCRATCH"]
IDX2LABEL: List[Label] = ["CAR_DAMAGE","DENT","GLASS_BREAK","SCRATCH"]

class ClassProb(BaseModel):
    label: Label
    prob: float


class Box(BaseModel): 
    # 정규화 좌표(0~1) -> 프론트에서 변환
    x: float
    y: float
    w: float
    h: float
    # JSON 키를 "class_probs"로 직렬화/역직렬화
    class_probs: List[ClassProb] = Field(
        default_factory=list,
        serialization_alias="class_probs",
        validation_alias="class_probs",
    )
    model_config = ConfigDict(populate_by_name=True)

class PredictIn(BaseModel):
    raw_url: str
    threshold: Optional[float] = 0.3           # YOLO conf 임계값
    model: Optional[str] = "yolo-vit"
    heatmap_put_url: Optional[str] = None

    # ---- 크롭/업샘플 정책 (추가) ----
    crop_padding: Optional[float] = 0.12        # 정규화 패딩 비율(문맥 포함)
    min_crop_side_px: Optional[int] = 64        # 학습과 동일: 64px 미만이면 업샘플
    hard_min_crop_side_px: Optional[int] = 8    # 이보다 작으면 스킵(극단값 컷) — 리콜 최우선이면 0으로
    upsample_small: Optional[bool] = True       # True면 소프트-미니멈 미만 업샘플
    max_upsample_scale: Optional[float] = 8.0   # 업샘플 배수 제한(>8×이면 스킵). 제한 없애려면 None
    adaptive_expand: Optional[bool] = True      # 크롭 전 정규화 박스를 먼저 넓혀 64px 만족 시도

class PredictOut(BaseModel):
    model: Optional[str]
    threshold_used: float
    boxes: List[Box]

app = FastAPI(title="capstone-yolo-vit")

# ===== 1) 유틸 =====
def clamp01(v: float) -> float:
    if v != v: return 0.0  # NaN
    if v < 0: return 0.0
    if v > 1: return 1.0
    return float(v)

def expand_box_norm(x: float, y: float, w: float, h: float, pad: float) -> Tuple[float,float,float,float]:
    """정규화 박스를 pad 비율만큼 확장하고 [0,1]로 클램프"""
    pad = max(0.0, pad or 0.0)
    x = x - w * pad / 2.0
    y = y - h * pad / 2.0
    w = w * (1.0 + pad)
    h = h * (1.0 + pad)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(1e-6, min(1.0 - x, w))
    h = max(1e-6, min(1.0 - y, h))
    return x, y, w, h

def expand_to_soft_min_norm(x, y, w, h, W, H, soft_min_px):
    """정규화 박스를 '최소 변 길이 = soft_min_px'로 맞추도록 중심 유지 확장"""
    if soft_min_px is None or soft_min_px <= 0:
        return x, y, w, h
    target_w_norm = soft_min_px / max(W, 1)
    target_h_norm = soft_min_px / max(H, 1)
    new_w = max(w, target_w_norm)
    new_h = max(h, target_h_norm)
    cx, cy = x + w/2.0, y + h/2.0
    new_w = min(new_w, 1.0)
    new_h = min(new_h, 1.0)
    x = max(0.0, min(1.0 - new_w, cx - new_w/2.0))
    y = max(0.0, min(1.0 - new_h, cy - new_h/2.0))
    return x, y, new_w, new_h

def enforce_hard_min_norm(x, y, w, h, W, H, hard_min_px):
    """하드-미니멈: 짧은 변 < hard_min_px면 None을 반환(즉시 스킵)"""
    if hard_min_px and hard_min_px > 0:
        if min(w * W, h * H) < hard_min_px:
            return None
    return (x, y, w, h)

# 정규화 -> 픽셀 좌표로 변환
def norm_to_px(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int,int,int,int]:
    x = clamp01(x); y = clamp01(y); w = clamp01(w); h = clamp01(h)
    x0 = int(round(x * W)); y0 = int(round(y * H))
    ww = int(round(w * W)); hh = int(round(h * H))
    x0 = max(0, min(W-1, x0))
    y0 = max(0, min(H-1, y0))
    x1 = max(0, min(W, x0 + max(1, ww)))
    y1 = max(0, min(H, y0 + max(1, hh)))
    return x0, y0, x1, y1

def download_image(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        # from PIL import ImageOps; im = ImageOps.exif_transpose(im)  # 필요시 EXIF 보정
        return im
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to download image: {e}")

# ===== 2) 어댑터 (실 모델 연결) =====
class YoloAdapter:
    def __init__(self, weights_path: str, device: str = None, iou: float = 0.65, max_det: int = 300):
        self.name = os.path.basename(weights_path)
        self.model = YOLO(weights_path)
        self.device = device or pick_device_for_torch()   # ← 함수 호출로 수정!
        self.iou = iou
        self.max_det = max_det

    def detect(self, image: Image.Image, conf_thres: float):
        """
        반환: [(x, y, w, h, conf), ...]  (정규화, top-left 기준)
        """
        res = self.model.predict(
            image,
            conf=conf_thres,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,     # mps/cuda/cpu 자동
            verbose=False
        )[0]

        out = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out

        W, H = image.size

        # xywhn(center normalized) 우선
        if hasattr(boxes, "xywhn"):
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (cx, cy, w, h), c in zip(xywhn, confs):
                x = cx - w/2.0
                y = cy - h/2.0
                out.append((float(x), float(y), float(w), float(h), float(c)))
            return out

        # xyxyn → xywh 정규화
        if hasattr(boxes, "xyxyn"):
            xyxyn = boxes.xyxyn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxyn, confs):
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)
                out.append((x, y, w, h, float(c)))
            return out

        # 픽셀 xyxy → 정규화 변환
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            x = float(x1 / W)
            y = float(y1 / H)
            w = float((x2 - x1) / W)
            h = float((y2 - y1) / H)
            out.append((x, y, w, h, float(c)))
        return out

# 상단에 추가
from transformers import AutoImageProcessor, ViTForImageClassification

class VitAdapterHF:
    def __init__(self, weights_path: str,
                 model_name: str = "google/vit-base-patch16-224-in21k",
                 device: str = None, num_classes: int = 4):
        dev_str = device or pick_device_for_torch()
        self.device = torch.device(dev_str)
        
        # YOLO 어댑터와 동일하게 이름 노출
        self.name = os.path.basename(weights_path)
        # 학습 때와 같은 프로세서(전처리): size/mean/std를 내부에서 정확히 맞춰줌
        self.proc = AutoImageProcessor.from_pretrained(model_name)

        # 같은 아키텍처로 분류 헤드만 num_classes에 맞춰 생성
        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # 그대로 로드 (학습 저장 형태 따라 strict=False 유지 권장)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_probs(self, crop: Image.Image):
        inputs = self.proc(images=crop, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        return probs


def make_heatmap(W: int, H: int, boxes: List[Box]) -> Image.Image:
    arr = np.zeros((H, W), dtype=np.float32)
    for b in boxes:
        x0, y0, x1, y1 = norm_to_px(b.x, b.y, b.w, b.h, W, H)
        arr[y0:y1, x0:x1] += 1.0
    if arr.max() > 0:
        arr = arr / arr.max()
    hm = np.zeros((H, W, 4), dtype=np.uint8)
    hm[..., 0] = (arr * 255).astype(np.uint8)
    hm[..., 3] = (arr * 180).astype(np.uint8)
    return Image.fromarray(hm, mode="RGBA")

# ===== 2-1) 전역 모델 초기화 =====
def _get_env_path(name: str, default: Optional[str] = None) -> str:
    p = os.getenv(name, default)
    if not p or not os.path.exists(p):
        raise RuntimeError(f"Missing or not found: env {name} -> {p}")
    return p

# 환경변수 또는 코드 상단 기본값으로 지정
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", r"D:\weights\yolo11s.pt")  # .env에서 덮어씀
VIT_WEIGHTS  = os.getenv("VIT_WEIGHTS",  r"D:\weights\vit_4cls.pth") # .env에서 덮어씀

try:
    yolo = YoloAdapter(_get_env_path("YOLO_WEIGHTS", YOLO_WEIGHTS))
    vit  = VitAdapterHF(_get_env_path("VIT_WEIGHTS",  VIT_WEIGHTS))
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ===== 3) 엔드포인트 =====
@app.get("/health")
def health():
    return {"status": "ok", "yolo": yolo.name, "vit": vit.name, "device": str(yolo.device)}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    im = download_image(body.raw_url)
    W, H = im.size
    det_th = float(body.threshold if body.threshold is not None else 0.3)

    # 크롭/업샘플 정책
    pad = float(body.crop_padding if body.crop_padding is not None else 0.12)
    soft_min = int(body.min_crop_side_px if body.min_crop_side_px is not None else 64)
    hard_min = int(body.hard_min_crop_side_px if body.hard_min_crop_side_px is not None else 8)
    upsample = bool(True if body.upsample_small is None else body.upsample_small)
    max_scale = None if body.max_upsample_scale is None else float(body.max_upsample_scale)
    adaptive = bool(True if body.adaptive_expand is None else body.adaptive_expand)

    # (A) YOLO 검출
    dets = yolo.detect(im, conf_thres=det_th)

    boxes: List[Box] = []
    for (x, y, w, h, conf) in dets:
        # (1) 패딩
        x, y, w, h = expand_box_norm(x, y, w, h, pad=pad)

        # (2) 하드-미니멈 컷 (극단적 초소형 박스 제거)
        ok = enforce_hard_min_norm(x, y, w, h, W, H, hard_min)
        if ok is None:
            continue
        x, y, w, h = ok

        # (3) 적응형 확장: 먼저 박스 자체를 넓혀 soft_min(64px)을 만족 시도
        if adaptive and min(w * W, h * H) < soft_min:
            x, y, w, h = expand_to_soft_min_norm(x, y, w, h, W, H, soft_min)

        # (4) 픽셀 크롭
        x0, y0, x1, y1 = norm_to_px(x, y, w, h, W, H)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = im.crop((x0, y0, x1, y1))

        # (5) 여전히 작으면 업샘플(훈련분포와 일치, 비율 유지)
        if min(crop.size) < soft_min:
            if not upsample:
                continue
            cw, ch = crop.size
            if cw <= 0 or ch <= 0:
                continue
            scale = soft_min / max(1, min(cw, ch))
            if (max_scale is not None) and (scale > max_scale):
                # 업샘플이 너무 과도하면 스킵
                continue
            new_w = int(round(cw * scale))
            new_h = int(round(ch * scale))
            crop = crop.resize((new_w, new_h), resample=Image.BICUBIC)

        # (6) ViT 추론 → 박스별 확률 분포
        probs = vit.predict_probs(crop)
        class_probs = [ClassProb(label=IDX2LABEL[i], prob=float(p)) for i, p in enumerate(probs)]

        # (7) 응답: 정규화 좌표 그대로 반환 (프론트가 픽셀 변환/렌더)
        boxes.append(Box(x=float(x), y=float(y), w=float(w), h=float(h), class_probs=class_probs))

    # (8) 히트맵 업로드(옵션)
    if body.heatmap_put_url and len(boxes) > 0:
        try:
            hm = make_heatmap(W, H, boxes)
            buf = io.BytesIO()
            hm.save(buf, format="PNG")
            requests.put(body.heatmap_put_url, data=buf.getvalue(),
                         headers={"Content-Type":"image/png"}, timeout=10)
        except Exception:
            pass

    return PredictOut(model=body.model, threshold_used=det_th, boxes=boxes)