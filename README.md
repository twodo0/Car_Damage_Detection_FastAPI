# Vehicle Damage Detection API (Batch Rental System)

<<<<<<< HEAD
YOLO + ViT 기반 차량 손상 탐지용 FastAPI 서버입니다.  
Spring 백엔드에서 렌탈 배치 플로우 호출용으로 사용합니다.

---
##   셋업

### 1. 의존성 설치
pip install -r requirements.txt

2. 모델 경로 설정

프로젝트 루트에 .env 파일 생성 / 수정:

YOLO_WEIGHTS=/Users/dopal0426/Desktop/dev/model/failed_yolo/best.pt
VIT_WEIGHTS=/Users/dopal0426/Desktop/dev/model/main_vit_model/focal_loss.pth
VIT_MODEL_NAME=google/vit-base-patch16-224-in21k

3. 서버 실행

VSCode(F5)로 실행
	•	VSCode에서 폴더 오픈 → F5
	•	런 설정에서 FastAPI (uvicorn, 8001) 선택

터미널에서 실행

uvicorn main:app --host 127.0.0.1 --port 8001 --reload --env-file .env


⸻

  API 요약

POST /predict

Request 예시

=======
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
>>>>>>> fastapi-rental-batch
{
  "raw_url": "https://presigned-get-url-for-image",
  "yoloThreshold": 0.3,
  "heatmap_put_url": "https://presigned-put-url-for-heatmap.png"
}
<<<<<<< HEAD

Response 예시

=======
```

**Response:**

```json
>>>>>>> fastapi-rental-batch
{
  "model": "yolo-vit",
  "threshold_used": 0.3,
  "boxes": [
    {
      "class_probs": [
<<<<<<< HEAD
        { "label": "BREAKAGE",  "prob": 0.71 },
        { "label": "CRUSHED",   "prob": 0.11 },
        { "label": "SCRATCHED", "prob": 0.10 },
        { "label": "SEPARATED", "prob": 0.08 }
=======
        {"label": "BREAKAGE", "prob": 0.71},
        {"label": "CRUSHED", "prob": 0.11},
        {"label": "SCRATCHED", "prob": 0.10},
        {"label": "SEPARATED", "prob": 0.08}
>>>>>>> fastapi-rental-batch
      ],
      "x": 0.102,
      "y": 0.214,
      "w": 0.265,
      "h": 0.180
    }
  ]
}
<<<<<<< HEAD

	•	x, y, w, h는 모두 0–1로 정규화된 좌표입니다.
	•	손상이 없으면 boxes는 빈 배열([])로 내려갑니다.

⸻

  처리 플로우
	1.	raw_url에서 원본 이미지를 다운로드
	2.	YOLO로 bbox 검출 (yoloThreshold 기준 confidence 필터링)
	3.	각 bbox에 대해:
	•	해당 영역 crop
	•	ViT 분류 (5 클래스: BREAKAGE / CRUSHED / NORMAL / SCRATCHED / SEPARATED)
	•	top-1이 NORMAL이면 → 이 박스는 완전히 제거
	•	그 외 손상 클래스이면 → 손상 4종에 대해서만 softmax 재계산
	4.	최종 손상 박스에 대해:
	•	좌표를 0–1 범위로 정규화
	•	class_probs에 BREAKAGE/CRUSHED/SCRATCHED/SEPARATED 확률 세트 구성
	5.	손상 박스가 남아 있고 heatmap_put_url이 있는 경우:
	•	Grad-CAM 기반 히트맵 생성
	•	PNG로 인코딩해 heatmap_put_url로 HTTP PUT 업로드
	6.	PredictRes 형태로 응답 반환

⸻

  ViT 클래스 정의

Index	학습 라벨	API DamageType	비고
0	Breakage	BREAKAGE	손상으로 노출
1	Crushed	CRUSHED	손상으로 노출
2	Normal	(미노출)	top-1이면 박스 필터링
3	Scratched	SCRATCHED	손상으로 노출
4	Separated	SEPARATED	손상으로 노출

- 응답의 class_probs[].label은 항상 BREAKAGE/CRUSHED/SCRATCHED/SEPARATED 중 하나입니다.
- NORMAL은 학습에는 존재하지만, API에는 등장하지 않습니다.

⸻

  주요 특징
	•	✅ NORMAL 필터링
top-1 클래스가 NORMAL인 박스는 응답에서 완전히 제거
	•	✅ 손상 4종에 대한 확률만 노출
softmax는 BREAKAGE/CRUSHED/SCRATCHED/SEPARATED에 대해서만 계산
	•	✅ 좌표 정규화
모든 bbox 좌표는 0–1 범위로 통일되어 Spring/프론트에서 그대로 사용 가능
	•	✅ 스키마 고정
Response 구조(model, threshold_used, boxes[].class_probs, x,y,w,h)는
Spring 백엔드와 합의된 계약을 따름
	•	✅ Grad-CAM 히트맵 지원
손상 박스 위에 컬러 heatmap 오버레이 PNG를 생성해 MinIO 등에 업로드

⸻

  간단 테스트

# 헬스 체크
curl http://127.0.0.1:8001/health

# 예측 예시
=======
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
>>>>>>> fastapi-rental-batch
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "raw_url": "https://your-presigned-url",
    "yoloThreshold": 0.3
  }'
<<<<<<< HEAD


⸻

  기타
	•	손상이 없을 때는 boxes: []로 응답하며, null이 되지 않습니다.
	•	히트맵 업로드가 실패하더라도 /predict 응답은 200 OK로 내려가고, 로그에만 워닝을 남깁니다.
	•	threshold_used 필드는 실제 사용된 YOLO confidence threshold 값을 그대로 반환합니다.
	•	응답 클래스 라벨은 Spring의 DamageType enum(BREAKAGE/CRUSHED/SCRATCHED/SEPARATED)과 정확히 일치합니다.

=======
```

## 📝 Notes

- Empty `boxes` array (not null) is returned when no damage detected
- Heatmap upload failures do not affect 200 OK response (logged as warnings)
- YOLO confidence threshold is reflected in `threshold_used` field
- All class labels match Spring's `DamageType` enum exactly
>>>>>>> fastapi-rental-batch
