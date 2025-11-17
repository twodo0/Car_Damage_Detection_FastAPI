# Spring + FastAPI 연동 가이드

## 📋 개요

Spring Backend (Batch Rental System) ↔ FastAPI ML Server

## 🔗 연동 스펙

### 1. API Contract

**Endpoint**: `POST http://127.0.0.1:8001/predict`

**Spring → FastAPI Request:**
```json
{
  "raw_url": "https://minio-presigned-get-url",
  "yoloThreshold": 0.3,
  "heatmap_put_url": "https://minio-presigned-put-url"
}
```

**FastAPI → Spring Response:**
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

### 2. Spring DamageType Enum

```java
public enum DamageType {
    BREAKAGE,   // ViT index 0
    CRUSHED,    // ViT index 1
    SCRATCHED,  // ViT index 2
    SEPARATED   // ViT index 3
    // NORMAL (index 4) is filtered out by FastAPI
}
```

### 3. Spring DTO 매핑

**FastApiPredictRes.java:**
```java
@JsonIgnoreProperties(ignoreUnknown = true)
public record FastApiPredictRes(
    String model,
    
    @JsonProperty("threshold_used")
    Double thresholdUsed,
    
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    List<BoxDto> boxes
) {}
```

**BoxDto.java:**
```java
public record BoxDto(
    @JsonProperty("class_probs")
    @JsonSetter(nulls = Nulls.AS_EMPTY)
    List<ClassProbDto> classProbs,
    
    Double x,  // 0-1 normalized
    Double y,  // 0-1 normalized
    Double w,  // 0-1 normalized
    Double h   // 0-1 normalized
) {}
```

**ClassProbDto.java:**
```java
public record ClassProbDto(
    DamageType label,  // BREAKAGE | CRUSHED | SCRATCHED | SEPARATED
    Double prob        // 0.0 - 1.0
) {}
```

## 🔧 Spring Configuration

**application.yml:**
```yaml
fastapi:
  base-url: http://127.0.0.1:8001
  timeout:
    connect: 5s
    read: 30s
    write: 30s
  predict-endpoint: /predict
```

## 🚀 시작 순서

### 1. FastAPI 서버 시작 (먼저!)

```bash
cd fastapi_new
uvicorn main:app --host 127.0.0.1 --port 8001 --reload --env-file .env
```

또는 VSCode에서 F5

### 2. Spring 서버 시작

```bash
cd capstone-web
./gradlew bootRun
```

또는 IntelliJ에서 실행

## 🧪 연동 테스트

### Step 1: 이미지 업로드

```http
POST http://localhost:8888/api/images
Content-Type: multipart/form-data

file: [car-image.jpg]
```

### Step 2: 예측 요청

```http
POST http://localhost:8888/api/predictions/by-image/{imageId}
```

Spring이 자동으로:
1. MinIO presigned URL 생성
2. FastAPI에 /predict 요청
3. 결과 파싱 및 저장

## ⚠️ 주의사항

### 1. NORMAL 클래스 처리

FastAPI는 ViT가 NORMAL(index 4)로 분류한 박스를 **완전히 제거**합니다.
Spring은 NORMAL에 대한 DamageType enum이 **없습니다**.

### 2. class_probs 배열

항상 **정확히 4개**의 요소를 포함합니다:
- BREAKAGE (index 0)
- CRUSHED (index 1)
- SCRATCHED (index 2)
- SEPARATED (index 3)

합계는 항상 **1.0**입니다 (softmax 재계산).

### 3. 빈 boxes 배열

손상이 없거나 모든 박스가 NORMAL인 경우:
```json
{
  "model": "yolo-vit",
  "threshold_used": 0.3,
  "boxes": []
}
```

**절대 `null`이 아님!** Spring은 `@JsonSetter(nulls = Nulls.AS_EMPTY)`로 처리.

### 4. 히트맵 업로드

- 박스가 **1개 이상** 남아있을 때만 업로드
- 모든 박스가 NORMAL이면 히트맵 생성 안 함
- 업로드 실패는 200 OK 응답에 영향 없음 (경고 로그만)

## 🐛 트러블슈팅

### 문제 1: Connection refused

**증상**: Spring → FastAPI 연결 실패

**해결**:
```bash
# FastAPI 서버 실행 확인
curl http://127.0.0.1:8001/health

# 정상 응답:
{
  "status": "healthy",
  "device": "mps",
  "models_loaded": true
}
```

### 문제 2: DamageType 역직렬화 실패

**증상**: 
```
Cannot deserialize value of type `DamageType` from String "NORMAL"
```

**원인**: FastAPI가 NORMAL을 반환함

**해결**: FastAPI 업데이트 (main.py의 필터링 로직 확인)

### 문제 3: boxes가 null

**증상**: NullPointerException in Spring

**해결**: FastAPI는 빈 배열 `[]` 반환, Spring DTO에 `@JsonSetter(nulls = Nulls.AS_EMPTY)` 추가

## 📊 성능 참고

- YOLO 추론: ~100-200ms (MPS/CUDA)
- ViT 추론 (per box): ~50-100ms
- 전체 파이프라인 (3 boxes): ~500-800ms
- 히트맵 생성: ~50ms
- HTTP 오버헤드: ~10-20ms

## 🔍 로그 예시

**FastAPI:**
```
INFO: Downloaded image: (800, 600)
INFO: YOLO detected 5 boxes
DEBUG: Box 1 filtered (NORMAL)
DEBUG: Box 2: class_probs=[BREAKAGE: 0.71, ...]
INFO: Final boxes after ViT filtering: 3
INFO: Generating heatmap...
INFO: Heatmap uploaded successfully
```

**Spring:**
```
INFO: Requesting prediction for imageId=123
DEBUG: FastAPI request: {raw_url=..., yoloThreshold=0.3}
DEBUG: FastAPI response: 3 boxes detected
INFO: Saved prediction with 3 detections
```

## ✅ 체크리스트

연동 전 확인사항:

- [ ] FastAPI 서버 실행 중 (port 8001)
- [ ] Spring application.yml에 fastapi.base-url 설정
- [ ] MinIO 실행 중 (presigned URL 생성용)
- [ ] 모델 파일 경로 정확 (.env 설정)
- [ ] CUDA/MPS 사용 가능 (선택)
- [ ] /health 엔드포인트 정상 응답
- [ ] DamageType enum에 NORMAL 없음 확인
- [ ] Spring DTO에 @JsonSetter(nulls = Nulls.AS_EMPTY) 있음
