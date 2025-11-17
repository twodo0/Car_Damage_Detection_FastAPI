# 🚀 빠른 시작 가이드

## 1️⃣ 프로젝트 구조

```
fastapi_new/
├── main.py                 # FastAPI 메인 애플리케이션
├── requirements.txt        # Python 의존성
├── .env                    # 환경 변수 (모델 경로)
├── .vscode/
│   └── launch.json        # VSCode 디버그 설정
├── README.md              # 프로젝트 문서
├── INTEGRATION.md         # Spring 연동 가이드
├── test_api.py            # API 테스트 스크립트
└── .gitignore             # Git 무시 파일
```

## 2️⃣ 설치 및 실행 (3분)

### Step 1: 의존성 설치

```bash
cd fastapi_new
pip install -r requirements.txt
```

### Step 2: 모델 경로 확인

`.env` 파일 열어서 경로 확인:

```env
YOLO_WEIGHTS=/Users/dopal0426/Desktop/dev/model/failed_yolo/best.pt
VIT_WEIGHTS=/Users/dopal0426/Desktop/dev/model/main_vit_model/focal_loss.pth
```

### Step 3: 서버 실행

#### 방법 A: VSCode (권장)

1. VSCode로 `fastapi_new` 폴더 열기
2. **F5** 키 누르기
3. "FastAPI (uvicorn, 8001)" 선택
4. 터미널에서 "Application startup complete" 확인

#### 방법 B: 커맨드 라인

```bash
uvicorn main:app --host 127.0.0.1 --port 8001 --reload --env-file .env
```

### Step 4: 동작 확인

브라우저에서 열기:
- Health check: http://127.0.0.1:8001/health
- API docs: http://127.0.0.1:8001/docs

## 3️⃣ 주요 변경사항 (이전 버전 대비)

### ViT 클래스 변경

**이전:**
```python
["CAR_DAMAGE", "DENT", "GLASS_BREAK", "SCRATCH"]
```

**새로운:**
```python
0: "BREAKAGE"
1: "CRUSHED"
2: "SCRATCHED"
3: "SEPARATED"
4: "NORMAL"  # ← 새로 추가, 필터링됨
```

### NORMAL 필터링

```python
# ViT 결과가 NORMAL(index 4)이면 박스 제거
if top1_idx == 4:
    continue  # Skip this box

# 나머지 박스는 4개 손상 클래스(0-3)만 softmax 재계산
damage_logits = logits[:4]  # Exclude NORMAL
damage_probs = F.softmax(damage_logits, dim=0)
```

### API 스펙 단순화

**Request:**
```json
{
  "raw_url": "...",
  "yoloThreshold": 0.3,       // ← 단순화 (yolo_conf → yoloThreshold)
  "heatmap_put_url": "..."    // optional
}
```

**Response:**
```json
{
  "model": "yolo-vit",
  "threshold_used": 0.3,      // ← snake_case
  "boxes": [
    {
      "class_probs": [        // ← 항상 4개
        {"label": "BREAKAGE", "prob": 0.71},
        {"label": "CRUSHED", "prob": 0.11},
        {"label": "SCRATCHED", "prob": 0.10},
        {"label": "SEPARATED", "prob": 0.08}
      ],
      "x": 0.102, "y": 0.214, "w": 0.265, "h": 0.180
    }
  ]
}
```

## 4️⃣ Spring 연동

### Spring 설정

**application.yml:**
```yaml
fastapi:
  base-url: http://127.0.0.1:8001
  predict-endpoint: /predict
```

### 실행 순서

1. **FastAPI 먼저** 시작 (port 8001)
2. **Spring 나중에** 시작 (port 8888)

### 동작 흐름

```
[Spring] 이미지 업로드
   ↓
[Spring] MinIO presigned URL 생성
   ↓
[Spring] POST /predict to FastAPI
   ↓
[FastAPI] 이미지 다운로드 → YOLO → ViT → 필터링
   ↓
[FastAPI] 히트맵 생성 & 업로드 (optional)
   ↓
[FastAPI] Response to Spring
   ↓
[Spring] 결과 저장
```

## 5️⃣ 테스트

### Health Check

```bash
curl http://127.0.0.1:8001/health
```

**Expected:**
```json
{
  "status": "healthy",
  "device": "mps",
  "models_loaded": true
}
```

### Prediction Test

```bash
python test_api.py
```

**또는 수동:**
```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "raw_url": "https://your-presigned-url",
    "yoloThreshold": 0.3
  }'
```

## 6️⃣ 트러블슈팅

### 문제: 모델 로딩 실패

```
FileNotFoundError: [Errno 2] No such file or directory: '/Users/...'
```

**해결:** `.env` 파일의 모델 경로 확인

### 문제: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**해결:** `.env`에 추가:
```env
PYTORCH_MPS_ENABLED=1  # M1/M2 Mac
```

또는 CPU 강제:
```python
DEVICE = "cpu"
```

### 문제: Connection refused (Spring)

```
ConnectException: Connection refused
```

**해결:** FastAPI 서버가 실행 중인지 확인:
```bash
curl http://127.0.0.1:8001/health
```

## 7️⃣ 다음 단계

1. ✅ FastAPI 서버 실행 확인
2. ✅ /health 테스트
3. ✅ Spring 서버 시작
4. ✅ 이미지 업로드 테스트
5. ✅ 예측 요청 테스트
6. ✅ 히트맵 확인 (MinIO)

## 📚 추가 문서

- **README.md**: 전체 프로젝트 개요
- **INTEGRATION.md**: Spring 연동 상세 가이드
- **main.py**: 소스 코드 (주석 참고)

## 🆘 도움말

문제 발생 시:
1. 로그 확인 (터미널 출력)
2. `/health` 엔드포인트 테스트
3. 모델 파일 경로 확인
4. Python 버전 확인 (3.11+)
5. 의존성 재설치: `pip install -r requirements.txt --upgrade`

## ✨ 주요 기능

- ✅ YOLO object detection
- ✅ ViT classification (5 classes)
- ✅ NORMAL filtering (automatic)
- ✅ Softmax recalculation (damage classes only)
- ✅ Normalized coordinates (0-1)
- ✅ Heatmap generation
- ✅ MinIO presigned URL support
- ✅ Contract-compliant API (Spring compatible)
- ✅ GPU acceleration (CUDA/MPS)
- ✅ Hot reload (development)
