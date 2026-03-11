# HumanML3D → Mine-imator Keyframe Converter

HumanML3D 포맷(`.npy`)의 모션 데이터를 Mine-imator용 키프레임 파일(`.miframes`)로 변환합니다.

## 파일 구조

```
motion_keyframes/
├── test.py              # 메인 변환 스크립트 (body frame 기반, 현재 기준)
├── motion.py            # 단순화 버전 (레거시)
├── motion_process.py    # HumanML3D 유틸리티 (recover_from_ric 등)
├── motion.ipynb         # 실험용 노트북
├── skeleton.py          # HumanML3D 스켈레톤 정의
├── quaternion.py        # 쿼터니언 연산
├── paramUtil.py         # HumanML3D 상수 (kinematic chain, raw offsets)
└── *.npy / *.mp4        # 모션 데이터 및 대응 영상
```

## 입력 포맷

HumanML3D `.npy` 파일 — shape: `(T, 263)` (22-joint, HumanML3D 표준 피처 벡터)

`recover_from_ric`을 통해 `(T, 22, 3)` 월드 좌표 XYZ로 복원됩니다.

### 22-joint 인덱스 (HumanML3D)

| 인덱스 | 부위 |
|--------|------|
| 0 | ROOT (pelvis) |
| 1 | L_HIP |
| 2 | R_HIP |
| 4 | L_KNEE |
| 5 | R_KNEE |
| 7 | L_ANKLE |
| 8 | R_ANKLE |
| 12 | NECK |
| 15 | HEAD |
| 16 | L_SHOULDER |
| 17 | R_SHOULDER |
| 18 | L_ELBOW |
| 19 | R_ELBOW |
| 20 | L_WRIST |
| 21 | R_WRIST |

## 실행 방법

```bash
python test.py
```

`test.py` 상단의 `NPY_FILE`, `OUTPUT_FILE`을 수정하여 입력/출력 파일 지정.

## Mine-imator 좌표계

| 값 | 의미 |
|----|------|
| POS_X, POS_Y | 바닥 평면 좌표 (수평) |
| POS_Z | 위아래 (수직 높이) |
| ROT_X | 팔/다리 앞뒤 회전 |
| ROT_Y | 팔/다리 좌우 회전 |
| ROT_Z | 자체 비틀림 (roll) |
| BEND_ANGLE_X | 관절 굽힘 각도 (0=직선) |

**차렷 자세 기준: ROT_X = ROT_Y = ROT_Z = 0**

## 보정 파라미터 (test.py)

### Root

| 파라미터 | 설명 |
|----------|------|
| `MOVE_MULT` | 이동 스케일 (0.0 = 제자리 고정) |
| `Y_OFFSET` | 지면 높이 보정 |
| `BODY_PITCH_FIX` | 전체 피치 오프셋 (스티브 세우기) |
| `ROOT_USE_RELATIVE` | 첫 프레임 기준 상대 이동 여부 |
| `ROOT_LOCK_XZ` | XZ 이동 고정 여부 |
| `ROOT_POS_SIGN/SCALE/OFFSET` | 축별 부호/스케일/오프셋 |

### 파트별

| 파라미터 | 설명 |
|----------|------|
| `PART_X_SIGN` | ROT_X 부호 반전 여부 (+1/-1) |
| `PART_X_OFFSET` | ROT_X 최종 오프셋 (도) |
| `PART_Y_SIGN` | ROT_Y 부호 반전 여부 |
| `PART_Y_SCALE` | ROT_Y 스케일 배수 |
| `PART_Y_OFFSET` | ROT_Y 최종 오프셋 (도) |

현재 튜닝된 기본값:

| 파트 | X_SIGN | X_OFFSET | Y_SCALE |
|------|--------|----------|---------|
| head | +1 | +80 | 0.2 |
| left_arm | -1 | +80 | 0.22 |
| right_arm | -1 | +80 | 0.22 |
| left_leg | +1 | -70 | 0.16 |
| right_leg | +1 | -70 | 0.16 |

## 변환 파이프라인

```
.npy (HumanML3D 피처)
    ↓ recover_from_ric
(T, 22, 3) 월드 XYZ 좌표
    ↓ build_body_frame (right / up / forward)
몸통 로컬 좌표계 구성
    ↓ to_local → compute_rot_from_local
ROT_X (앞뒤), ROT_Y (좌우)
    ↓ apply_part_adjustment
부위별 sign/scale/offset 보정
    ↓
.miframes (Mine-imator JSON)
```

### 몸통 좌표계 (`build_body_frame`)

```
right   = normalize(R_HIP - L_HIP)
up      = normalize(NECK - hip_center)
forward = normalize(right × up)
```

### 관절 굽힘 (`get_bend_angle`)

```
v1 = parent → joint  (역방향)
v2 = end    → joint  (역방향)
bend = 180° - angle(v1, v2)   → 0=직선, 양수=굽힘
```

## 출력 포맷 (.miframes)

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
            "values": {
                "POS_X": ..., "POS_Y": ..., "POS_Z": ...,
                "ROT_X": ..., "ROT_Y": ..., "ROT_Z": ...
            }
        },
        {
            "position": <프레임 번호>,
            "part_name": "right_arm",
            "values": {
                "ROT_X": ..., "ROT_Y": ..., "ROT_Z": 0.0,
                "BEND_ANGLE_X": ...
            }
        }
    ]
}
```

## 의존성

```
numpy
torch
tqdm
```

## 알려진 제한사항

- `tempo: 20` 하드코딩 — HumanML3D 기본 FPS 기준
- ROT_Z (비틀림) 미계산, 항상 0.0
- 전신 회전(global yaw)이 root ROT에 별도 반영되지 않음
- PART_Y_SIGN 현재 미사용 (PART_Y_SCALE만 적용)

## Docker 실행 가이드

아래 명령은 프로젝트 루트(`motion_keyframes`)에서 실행합니다.

### 1) 이미지 빌드 (공통)

```powershell
docker build -t motiongpt .
```

### 2) CPU 환경 실행

```powershell
docker run --rm -it -p 7860:7860 `
  -v "${PWD}\checkpoints:/root/MotionGPT3/checkpoints" `
  -v "${PWD}\deps:/root/MotionGPT3/deps" `
  -v "${PWD}\outputs:/root/MotionGPT3/outputs" `
  motiongpt
```

### 3) GPU 환경 실행 (NVIDIA + WSL2/Docker GPU 설정 완료 시)

```powershell
docker run --rm -it -p 7860:7860 --gpus all `
  -v "${PWD}\checkpoints:/root/MotionGPT3/checkpoints" `
  -v "${PWD}\deps:/root/MotionGPT3/deps" `
  -v "${PWD}\outputs:/root/MotionGPT3/outputs" `
  motiongpt
```
