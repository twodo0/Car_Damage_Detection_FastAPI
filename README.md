# Vehicle Damage Detection API (Batch Rental System)

FastAPI 서버 for YOLO + ViT 손상 탐지 파이프라인

## 📋 Requirements

- Python 3.11+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure model paths

Edit `.env` file:

```env
YOLO_WEIGHTS=/Users/dopal0426/Desktop/dev/model/failed_yolo/best.pt
VIT_WEIGHTS=/Users/dopal0426/Desktop/dev/model/main_vit_model/focal_loss.pth
VIT_MODEL_NAME=google/vit-base-patch16-224-in21k
```

### 3. Run server

#### Option A: VSCode Debug (F5)

Open in VSCode → Press F5 → Select "FastAPI (uvicorn, 8001)"

#### Option B: Command line

```bash
uvicorn main:app --host 127.0.0.1 --port 8001 --reload --env-file .env
```

## 📡 API Specification

### POST /predict

**Request:**

```json
{
  "raw_url": "https://presigned-get-url-for-image",
  "yoloThreshold": 0.3,
  "heatmap_put_url": "https://presigned-put-url-for-heatmap.png"
}
```

**Response:**

```json
{
  "model": "yolo-vit",
  "threshold_used": 0.3,
  "boxes": [
    {
      "class_probs": [
        {"label": "BREAKAGE", "prob": 0.71},
        {"label": "CRUSHED", "prob": 0.11},
        {"label": "SCRATCHED", "prob": 0.10},
        {"label": "SEPARATED", "prob": 0.08}
      ],
      "x": 0.102,
      "y": 0.214,
      "w": 0.265,
      "h": 0.180
    }
  ]
}
```

## 🔍 Processing Flow

1. **Download image** from `raw_url`
2. **YOLO detection** with `yoloThreshold` confidence
3. **For each box:**
   - Crop region
   - ViT classification (5 classes: 0-3=damage, 4=NORMAL)
   - If top-1 is NORMAL (index 4): **filter out** this box
   - Otherwise: compute softmax over damage classes (0-3 only)
4. **Normalize coordinates** (0-1 range)
5. **Generate heatmap** (if boxes remain and `heatmap_put_url` provided)
6. **Upload heatmap** via HTTP PUT
7. **Return filtered boxes**

## 🏷️ ViT Classes

| Index | Internal Name | DamageType Enum |
|-------|---------------|-----------------|
| 0     | Breakage      | BREAKAGE        |
| 1     | Crushed       | CRUSHED         |
| 2     | Scratched     | SCRATCHED       |
| 3     | Separated     | SEPARATED       |
| 4     | Normal        | (filtered out)  |

## 🔧 Key Features

- ✅ **NORMAL filtering**: Boxes classified as NORMAL (index 4) are completely removed
- ✅ **Softmax recalculation**: Only damage classes (0-3) are used for probability distribution
- ✅ **Normalized coordinates**: All coordinates are in 0-1 range
- ✅ **Contract compliance**: Response schema strictly matches Spring backend expectations
- ✅ **Heatmap generation**: Simple colored box overlay (only when boxes remain)

## 🧪 Testing

```bash
# Health check
curl http://127.0.0.1:8001/health

# Prediction (example)
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "raw_url": "https://your-presigned-url",
    "yoloThreshold": 0.3
  }'
```

## 📝 Notes

- Empty `boxes` array (not null) is returned when no damage detected
- Heatmap upload failures do not affect 200 OK response (logged as warnings)
- YOLO confidence threshold is reflected in `threshold_used` field
- All class labels match Spring's `DamageType` enum exactly
