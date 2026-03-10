# MediaPipe Pose → Mine-imator Keyframe Exporter

영상에서 MediaPipe Pose를 이용해 관절 좌표를 추출하고, Mine-imator용 키프레임 파일(`.miframes`)로 변환하는 Gradio 웹 앱입니다.

## 기능

- 영상 업로드 → MediaPipe Pose 추론
- 스켈레톤이 그려진 결과 영상 출력
- 프레임별 33개 랜드마크 좌표를 JSON으로 저장
- Mine-imator 포맷(`.miframes`)으로 변환 출력

## 출력 파일

| 파일 | 설명 |
|------|------|
| `output_skeleton.mp4` | 스켈레톤 오버레이 영상 |
| `motion_data.json` | 프레임별 원시 랜드마크 데이터 (MediaPipe 좌표계) |
| `motion.miframes` | Mine-imator 키프레임 데이터 |

## 파일 구조

```
mediapipe/
├── app.py          # Gradio UI + MediaPipe 처리 파이프라인
├── convert.py      # MediaPipe → Mine-imator 변환 로직
├── requirements.txt
└── Dockerfile
```

## Mine-imator 변환 상세 (`convert.py`)

### 좌표 변환

MediaPipe는 이미지 좌표계(Y 아래 양수, Z 카메라 방향)를 사용하므로 Mine-imator 축에 맞게 Y, Z를 반전합니다.

```
Mine-imator 좌표 = [lm.x, -lm.y, -lm.z]
```

### 토르소 회전 계산

| 값 | 계산 방법 |
|----|-----------|
| ROT_X | 척추 벡터의 Y,Z 성분 → atan2 (앞뒤 기울기) |
| ROT_Y | 척추 벡터의 X,Y 성분 → atan2 (좌우 회전) |
| ROT_Z | 어깨선을 척추 수직 평면에 투영 → atan2 (몸통 비틀림) |

팔/다리 회전은 월드 좌표가 아닌 **토르소 로컬 좌표계 기준**으로 계산됩니다 (`R.T @ vector`).

### 부위별 랜드마크 매핑

| 부위 | 랜드마크 ID (상단 → 하단) |
|------|--------------------------|
| `right_arm` | 11 (어깨) → 13 (팔꿈치) → 15 (손목) |
| `left_arm` | 12 (어깨) → 14 (팔꿈치) → 16 (손목) |
| `right_leg` | 23 (골반) → 25 (무릎) → 27 (발목) |
| `left_leg` | 24 (골반) → 26 (무릎) → 28 (발목) |

### 키프레임 값

- **Root Position**: 골반 중심 `(lm[23] + lm[24]) / 2` 기반, 스케일 ×100, Y 지면 보정 +1.0 후 ×20
- **body ROT_X/Y/Z**: 토르소 자체 회전 (비틀림 포함)
- **팔/다리 ROT_X / ROT_Y**: 토르소 로컬 공간 기준 상박 방향 벡터에 atan2 적용
- **BEND_ANGLE_X**: 상박·하박 벡터 사잇각 (도)
- **ROT_Z (팔/다리)**: 항상 0.0 (단순화)

### `.miframes` 포맷

```json
{
  "format": 34,
  "created_in": "2.0.2",
  "is_model": true,
  "tempo": 20,
  "length": <총 프레임 수>,
  "keyframes": [
    {
      "position": <프레임 번호>,
      "values": { "POS_X": ..., "POS_Y": ..., "POS_Z": ... }
    },
    {
      "position": <프레임 번호>,
      "part_name": "right_arm",
      "values": { "ROT_X": ..., "ROT_Y": ..., "ROT_Z": 0.0, "BEND_ANGLE_X": ... }
    },
    ...
  ]
}
```

> `tempo`는 입력 영상의 실제 FPS를 읽어 자동으로 설정됩니다.

## 실행 방법

### 로컬 (Python 3.11+)

```bash
pip install -r requirements.txt
python app.py
```

브라우저에서 `http://localhost:7860` 접속

### Docker

```bash
docker build -t mediapipe-pose .
docker run -p 7860:7860 mediapipe-pose
```

## 알려진 제한사항

- 랜드마크가 감지되지 않은 프레임은 키프레임에서 제외됩니다 (포즈 누락 시)
- 팔/다리 ROT_Z는 계산되지 않습니다 (0.0 고정)
- 출력 영상 코덱(`mp4v`)이 일부 플레이어에서 재생되지 않을 수 있습니다

## 의존성

| 패키지 | 버전 |
|--------|------|
| mediapipe | 0.10.14 |
| opencv-python-headless | 4.10.0.84 |
| gradio | 4.42.0 |
| numpy | 1.26.4 |
