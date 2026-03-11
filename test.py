import numpy as np
import torch
import json
from motion_process import recover_from_ric, recover_root_rot_pos

# --- [1. 설정 및 상수] ---
# NPY_FILE = "2026-03-09-07_09_0742399.npy" #달리기
# NPY_FILE = "2026-03-09-07_09_4951239.npy"  # 점프
# NPY_FILE = "2026-03-09-07_08_2592584.npy"  # 검
# NPY_FILE = "2026-03-10-11_58_0350684.npy"  # 개쩌는 옆돌기
# NPY_FILE = "2026-03-10-11_58_5318060.npy"  # 주먹
NPY_FILE = "2026-03-10-11_53_3170620.npy"  # 손인사
OUTPUT_FILE = "손인사.miframes"

# HumanML3D 22-joint indices
ROOT = 0
L_HIP, R_HIP = 1, 2
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
NECK = 12
HEAD = 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21

# 보정 파라미터
MOVE_MULT = 0.0
Y_OFFSET = 12.0

# --- Root(스티브 자체) 포지션 보정 ---
ROOT_USE_RELATIVE = True
ROOT_LOCK_XZ = False
ROOT_SWAP_XZ = False
ROOT_ROT_SWAP_YZ = True

BODY_ROT_OFFSET = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}

ROOT_POS_SIGN = {
    "x": 1.0,
    "y": 1.0,
    "z": 1.0,
}

ROOT_POS_SCALE = {
    "x": 16.0,
    "y": 16.0,
    "z": 16.0,
}

ROOT_POS_OFFSET = {
    "x": 0.0,
    "y": 12.0,
    "z": 0.0,
}

# --- Global Body 회전 보정 ---
# Yaw: HumanML3D Y축 누적 회전 → Mine-imator root ROT_Y
YAW_SIGN = 1.0
YAW_SCALE = 1.0
YAW_OFFSET = 0.0

# Pitch: spine 앞뒤 기울기 → Mine-imator root ROT_X
PITCH_SIGN = -1.0
PITCH_SCALE = 1.0
PITCH_OFFSET = -4.0

# Roll: spine 좌우 기울기 → Mine-imator root ROT_Z
ROLL_SIGN = 1.0
ROLL_SCALE = 1.0
ROLL_OFFSET = 0.0

# 파트별 보정값
PART_X_SIGN = {
    "head": 1.0,
    "left_arm": 1.0,
    "right_arm": 1.0,
    "left_leg": 1.0,
    "right_leg": 1.0,
}

PART_X_SCALE = {
    "head": 1.0,
    "left_arm": 1.0,
    "right_arm": 1.0,
    "left_leg": 1.0,
    "right_leg": 1.0,
}

PART_X_OFFSET = {
    "head": 150.0,
    "left_arm": -10.0,
    "right_arm": -10.0,
    "left_leg": 0.0,
    "right_leg": 0.0,
}

PART_Y_SIGN = {
    "head": 1.0,
    "left_arm": -1.0,
    "right_arm": -1.0,
    "left_leg": -1.0,
    "right_leg": -1.0,
}

PART_Y_SCALE = {
    "head": 0.2,
    "left_arm": 0.1,
    "right_arm": 0.1,
    "left_leg": 0.2,
    "right_leg": 0.2,
}

PART_Y_OFFSET = {
    "head": 0.0,
    "left_arm": 0.0,
    "right_arm": 0.0,
    "left_leg": 0.0,
    "right_leg": 0.0,
}

# ROT_Z (비틀림) 보정
PART_Z_SIGN = {
    "head": 0.0,        # head는 end 없음 → twist 계산 안 함
    "left_arm": 1.0,
    "right_arm": -1.0,
    "left_leg": 1.0,
    "right_leg": 1.0,
}

PART_Z_OFFSET = {
    "head": 0.0,
    "left_arm": 0.0,
    "right_arm": 0.0,
    "left_leg": 0.0,
    "right_leg": 0.0,
}

PART_Z_SCALE = {
    "head": 1.0,
    "left_arm": 0.1,
    "right_arm": 0.1,
    "left_leg": 0.1,
    "right_leg": 0.1,
}

# 팔 비틀림(ROT_Z) 안정화
TWIST_STABILIZE_PARTS = {"left_arm", "right_arm"}
TWIST_EMA_ALPHA = 0.25


# --- [2. 유틸리티] ---
def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

def unwrap_from_prev(curr_wrapped, prev_unwrapped):
    delta = wrap_deg(curr_wrapped - prev_unwrapped)
    return prev_unwrapped + delta

def summarize_series(name, values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return
    total_rot = float(arr[-1] - arr[0])
    max_jump = float(np.max(np.abs(np.diff(arr))))
    print(f"- {name}: total_rotation={total_rot:.5f}, max_frame_jump={max_jump:.5f}")

def to_jsonable(obj):
    """NumPy 스칼라/배열을 표준 JSON 직렬화 가능한 타입으로 변환."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def get_bend_angle(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    return round(max(0.0, 180.0 - angle), 5)

def build_body_frame(curr_xyz):
    """
    몸통 기준 local frame 생성
    right:   L_HIP → R_HIP
    up:      hip center → NECK
    forward: right × up
    """
    hip_center = 0.5 * (curr_xyz[L_HIP] + curr_xyz[R_HIP])

    right = normalize(curr_xyz[R_HIP] - curr_xyz[L_HIP])
    up = normalize(curr_xyz[NECK] - hip_center)
    forward = normalize(np.cross(right, up))

    # 직교 보정
    right = normalize(np.cross(up, forward))

    return right, up, forward

def to_local(v_world, right, up, forward):
    """world 벡터 → body local 성분 (local_x=right, local_y=up, local_z=forward)"""
    return np.array([
        np.dot(v_world, right),
        np.dot(v_world, up),
        np.dot(v_world, forward),
    ], dtype=np.float32)

def get_root_position(curr_xyz):
    return curr_xyz[ROOT]

def compute_root_position(curr_focus, start_focus):
    if ROOT_USE_RELATIVE:
        p = curr_focus - start_focus
    else:
        p = curr_focus.copy()

    px = float(p[0])
    py = float(p[2])   # HumanML3D Z → Mine-imator Y (수평)
    pz = float(p[1])   # HumanML3D Y → Mine-imator Z (높이)

    if ROOT_SWAP_XZ:
        px, pz = pz, px

    px *= ROOT_POS_SIGN["x"]
    py *= ROOT_POS_SIGN["y"]
    pz *= ROOT_POS_SIGN["z"]

    px *= ROOT_POS_SCALE["x"]
    py *= ROOT_POS_SCALE["y"]
    pz *= ROOT_POS_SCALE["z"]

    px += ROOT_POS_OFFSET["x"]
    py += ROOT_POS_OFFSET["y"]
    pz += ROOT_POS_OFFSET["z"]

    if ROOT_LOCK_XZ:
        px = ROOT_POS_OFFSET["x"]
        pz = ROOT_POS_OFFSET["z"]

    return round(px, 5), round(py, 5), round(pz, 5)

def compute_rot_from_local(local_v):
    """
    body-local 방향 벡터 → Mine-imator 회전각
    차렷(-Y 방향) 기준: ROT_X=0, ROT_Y=0
    ROT_X: 앞뒤  ROT_Y: 좌우
    """
    lx, ly, lz = local_v

    # 기준축을 -Y로 두고 전후/좌우 스윙을 분리해 읽는다.
    # (0, -1, 0)인 차렷 자세에서 ROT_X=ROT_Y=0을 만족한다.
    rot_x = np.degrees(np.arctan2(lz, -ly + 1e-8))  # 앞뒤
    rot_y = np.degrees(np.arctan2(lx, -ly + 1e-8))  # 좌우

    return wrap_deg(rot_x), wrap_deg(rot_y)

def compute_body_euler_from_frame(right, up, forward):
    """
    body frame(right/up/forward)에서 직접 yaw/pitch/roll을 계산한다.
    - yaw  : forward의 수평면(XZ) 투영 방향
    - pitch: yaw를 제거한 forward의 수직 성분 기반
    - roll : yaw를 제거한 right/up의 수직 성분 관계 기반
    quaternion/Euler 분해를 사용하지 않는다.
    """
    right = normalize(right)
    up = normalize(up)
    forward = normalize(forward)

    # yaw: forward의 XZ 투영으로 계산
    yaw = float(np.degrees(np.arctan2(forward[0], forward[2] + 1e-8)))

    # yaw를 제거한 body frame으로 pitch/roll 계산
    yaw_rad = np.radians(-yaw)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    rot_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    right_local = rot_y @ right
    up_local = rot_y @ up
    forward_local = rot_y @ forward

    # pitch: forward의 수직(y) 성분과 전방(z) 성분으로 계산
    pitch = float(np.degrees(np.arctan2(-forward_local[1], forward_local[2] + 1e-8)))

    # roll: right/up의 수직 성분 관계(주로 right의 y, x)로 계산
    roll = float(np.degrees(np.arctan2(-right_local[1], right_local[0] + 1e-8)))

    return pitch, yaw, roll

def compute_twist(arm_dir_local, end_vec_world, right, up, forward):
    """
    ROT_Z (팔/다리 자체 비틀림) 계산.

    arm_dir_local : 상박/상퇴 방향 (body-local, 정규화됨)
    end_vec_world : 전박/하퇴 벡터 (world space)
    기준(ref):     body forward([0,0,1] in local) ⊥ arm_dir
    실제(actual):  end_local ⊥ arm_dir

    차렷 상태에서 전박이 forward 방향으로 굽힐 때 ROT_Z=0.
    """
    arm_n = normalize(arm_dir_local)

    # end를 body-local로 변환 후 arm 축에 수직 투영
    end_local = to_local(end_vec_world, right, up, forward)
    actual_perp = end_local - np.dot(end_local, arm_n) * arm_n
    actual_norm = np.linalg.norm(actual_perp)
    if actual_norm < 1e-6:
        return None
    actual_perp /= actual_norm

    # 기준: body forward (local [0,0,1]) ⊥ arm_dir
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ref = ref - np.dot(ref, arm_n) * arm_n
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-6:
        # arm이 forward와 평행할 때 → body right를 기준으로
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ref = ref - np.dot(ref, arm_n) * arm_n
        ref_norm = np.linalg.norm(ref)
        if ref_norm < 1e-6:
            return None
    ref /= ref_norm

    cos_t = np.clip(np.dot(ref, actual_perp), -1.0, 1.0)
    sin_t = np.dot(np.cross(ref, actual_perp), arm_n)
    return float(np.degrees(np.arctan2(sin_t, cos_t)))

def apply_part_adjustment(name, rot_x, rot_y, rot_z):
    rot_x = (
        rot_x * PART_X_SIGN.get(name, 1.0) * PART_X_SCALE.get(name, 1.0)
        + PART_X_OFFSET.get(name, 0.0)
    )
    rot_y = (
        rot_y * PART_Y_SIGN.get(name, 1.0) * PART_Y_SCALE.get(name, 1.0)
        + PART_Y_OFFSET.get(name, 0.0)
    )
    # ROT_Z는 연속각(unwrapped)을 유지해 프레임 경계에서 ±180 점프를 줄인다.
    rot_z = rot_z * PART_Z_SIGN.get(name, 0.0) * PART_Z_SCALE.get(name, 1.0) + PART_Z_OFFSET.get(name, 0.0)
    return rot_x, rot_y, rot_z


# --- [3. 메인] ---
def convert_motion_to_miframes(
    npy_path,
    output_path,
    return_root_rot_series=False,
    root_rot_calib=None,
    frame_stride=1,
):
    motion = np.load(npy_path)
    x = torch.from_numpy(motion).float().unsqueeze(0)

    xyz_torch = recover_from_ric(x, 22)
    xyz = xyz_torch.squeeze(0).cpu().numpy()

    # Global yaw: HumanML3D Y축 누적 회전각
    r_rot_quat, _ = recover_root_rot_pos(x)
    yaw_rad = torch.atan2(r_rot_quat[..., 2], r_rot_quat[..., 0])
    yaw_deg = torch.rad2deg(yaw_rad).squeeze(0).cpu().numpy()  # (T,)
    print("yaw_deg min: ", min(yaw_deg), "max: ", max(yaw_deg))
    T = xyz.shape[0]
    frame_stride = max(1, int(frame_stride))
    frame_indices = list(range(0, T, frame_stride))
    start_pos = get_root_position(xyz[0]).copy()

    miframes = {
        "format": 34,
        "created_in": "2.0.2",
        "is_model": True,
        "tempo": 20,
        "length": T,
        "keyframes": []
    }

    parts = {
        "head":      {"p": HEAD,    "parent": NECK},
        "left_arm":  {"p": L_ELBOW, "parent": L_SHOULDER, "end": L_WRIST},
        "right_arm": {"p": R_ELBOW, "parent": R_SHOULDER, "end": R_WRIST},
        "left_leg":  {"p": L_KNEE,  "parent": L_HIP,      "end": L_ANKLE},
        "right_leg": {"p": R_KNEE,  "parent": R_HIP,      "end": R_ANKLE},
    }

    angle_stats = {
        name: {
            "ROT_X": {"min": np.inf, "max": -np.inf},
            "ROT_Y": {"min": np.inf, "max": -np.inf},
            "ROT_Z": {"min": np.inf, "max": -np.inf},
        }
        for name in parts.keys()
    }
    global_rot_stats = {
        "ROT_X": {"min": np.inf, "max": -np.inf},
        "ROT_Y": {"min": np.inf, "max": -np.inf},
        "ROT_Z": {"min": np.inf, "max": -np.inf},
    }
    root_rot_series = {
        "pitch": [],
        "roll": [],
        "yaw": [],
    }
    yaw_raw_series = []
    yaw_unwrapped_series = []
    yaw_calibrated_series = []
    body_yaw_unwrapped_series = []
    root_xz_path = 0.0
    yaw_prev_unwrapped = 0.0
    yaw_has_prev = False
    pitch_prev_unwrapped = 0.0
    pitch_has_prev = False
    roll_prev_unwrapped = 0.0
    roll_has_prev = False
    body_yaw_prev_unwrapped = 0.0
    body_yaw_has_prev = False
    yaw_ref_unwrapped = None
    prev_root_focus = None
    twist_prev_unwrapped = {name: 0.0 for name in TWIST_STABILIZE_PARTS}
    twist_prev_output = {name: 0.0 for name in TWIST_STABILIZE_PARTS}
    twist_has_prev = {name: False for name in TWIST_STABILIZE_PARTS}
    part_prev_unwrapped = {
        name: {"ROT_X": 0.0, "ROT_Y": 0.0, "ROT_Z": 0.0}
        for name in parts.keys()
    }
    part_has_prev = {name: False for name in parts.keys()}

    root_rot_cfg = {
        "x": {"sign": PITCH_SIGN, "offset": PITCH_OFFSET, "scale": PITCH_SCALE},
        "y": {"sign": ROLL_SIGN, "offset": ROLL_OFFSET, "scale": ROLL_SCALE},
        "z": {"sign": YAW_SIGN, "offset": YAW_OFFSET, "scale": YAW_SCALE},
    }
    if isinstance(root_rot_calib, dict):
        for axis in ("x", "y", "z"):
            axis_cfg = root_rot_calib.get(axis, {})
            if not isinstance(axis_cfg, dict):
                continue
            if "sign" in axis_cfg:
                root_rot_cfg[axis]["sign"] = float(axis_cfg["sign"])
            if "offset" in axis_cfg:
                root_rot_cfg[axis]["offset"] = float(axis_cfg["offset"])

    for t in frame_indices:
        curr_xyz = xyz[t]

        # root
        curr_focus = get_root_position(curr_xyz)
        pos_x, pos_y, pos_z = compute_root_position(curr_focus, start_pos)
        if prev_root_focus is not None:
            root_xz_path += float(np.linalg.norm(curr_focus[[0, 2]] - prev_root_focus[[0, 2]]))
        prev_root_focus = curr_focus

        # 몸통 기준축 (pitch/roll 계산에도 필요)
        body_right, body_up, body_forward = build_body_frame(curr_xyz)

        raw_yaw = float(yaw_deg[t])
        yaw_raw_series.append(raw_yaw)
        if yaw_has_prev:
            yaw_unwrapped = unwrap_from_prev(raw_yaw, yaw_prev_unwrapped)
        else:
            yaw_unwrapped = raw_yaw
            yaw_has_prev = True
        yaw_prev_unwrapped = yaw_unwrapped
        yaw_unwrapped_series.append(yaw_unwrapped)

        # root orientation은 body frame 전체에서 추출한 Euler를 사용한다.
        body_pitch_raw, body_yaw_raw, body_roll_raw = compute_body_euler_from_frame(
            body_right, body_up, body_forward
        )

        if body_yaw_has_prev:
            body_yaw_unwrapped = unwrap_from_prev(body_yaw_raw, body_yaw_prev_unwrapped)
        else:
            body_yaw_unwrapped = body_yaw_raw
            body_yaw_has_prev = True
        body_yaw_prev_unwrapped = body_yaw_unwrapped
        body_yaw_unwrapped_series.append(body_yaw_unwrapped)

        if pitch_has_prev:
            pitch_unwrapped = unwrap_from_prev(body_pitch_raw, pitch_prev_unwrapped)
        else:
            pitch_unwrapped = body_pitch_raw
            pitch_has_prev = True
        pitch_prev_unwrapped = pitch_unwrapped

        if roll_has_prev:
            roll_unwrapped = unwrap_from_prev(body_roll_raw, roll_prev_unwrapped)
        else:
            roll_unwrapped = body_roll_raw
            roll_has_prev = True
        roll_prev_unwrapped = roll_unwrapped

        # yaw(ROT_Z)는 시작 프레임을 기준으로 정렬한다.
        # 고정 오프셋 대신 첫 프레임의 unwrapped yaw를 기준값으로 빼서
        # 모션별 절대 시작각 차이는 제거하고 누적 회전 변화는 유지한다.
        if yaw_ref_unwrapped is None:
            yaw_ref_unwrapped = body_yaw_unwrapped
        relative_body_yaw = body_yaw_unwrapped - yaw_ref_unwrapped
        calibrated_yaw = (
            relative_body_yaw * root_rot_cfg["z"]["sign"] * root_rot_cfg["z"]["scale"]
        )
        yaw_calibrated_series.append(calibrated_yaw)
        global_yaw = calibrated_yaw

        global_pitch = (
            pitch_unwrapped * root_rot_cfg["x"]["sign"] * root_rot_cfg["x"]["scale"]
            + root_rot_cfg["x"]["offset"]
            + BODY_ROT_OFFSET["x"]
        )
        global_roll = (
            roll_unwrapped * root_rot_cfg["y"]["sign"] * root_rot_cfg["y"]["scale"]
            + root_rot_cfg["y"]["offset"]
            + BODY_ROT_OFFSET["z"]
        )

        # root 축 매핑 고정:
        # - 앞뒤(pitch) -> ROT_X
        # - 좌우(roll)  -> ROT_Y
        # - yaw         -> ROT_Z
        root_rot_x = global_pitch
        root_rot_y = global_roll
        root_rot_z = global_yaw + BODY_ROT_OFFSET["y"]

        root_vals = {
            "POS_X": pos_x,
            "POS_Y": pos_y,
            "POS_Z": pos_z,
            "ROT_X": round(root_rot_x, 5),
            "ROT_Y": round(root_rot_y, 5),
            "ROT_Z": round(root_rot_z, 5),
        }
        root_rot_series["pitch"].append(root_vals["ROT_X"])
        root_rot_series["roll"].append(root_vals["ROT_Y"])
        root_rot_series["yaw"].append(root_vals["ROT_Z"])
        global_rot_stats["ROT_X"]["min"] = min(global_rot_stats["ROT_X"]["min"], root_vals["ROT_X"])
        global_rot_stats["ROT_X"]["max"] = max(global_rot_stats["ROT_X"]["max"], root_vals["ROT_X"])
        global_rot_stats["ROT_Y"]["min"] = min(global_rot_stats["ROT_Y"]["min"], root_vals["ROT_Y"])
        global_rot_stats["ROT_Y"]["max"] = max(global_rot_stats["ROT_Y"]["max"], root_vals["ROT_Y"])
        global_rot_stats["ROT_Z"]["min"] = min(global_rot_stats["ROT_Z"]["min"], root_vals["ROT_Z"])
        global_rot_stats["ROT_Z"]["max"] = max(global_rot_stats["ROT_Z"]["max"], root_vals["ROT_Z"])
        miframes["keyframes"].append({
            "position": t,
            "values": root_vals
        })

        for name, idx in parts.items():
            v_world = curr_xyz[idx["p"]] - curr_xyz[idx["parent"]]
            v_local = to_local(v_world, body_right, body_up, body_forward)

            rot_x, rot_y = compute_rot_from_local(v_local)

            # ROT_Z: end가 있는 부위만 twist 계산
            if "end" in idx and PART_Z_SIGN.get(name, 0.0) != 0.0:
                end_vec = curr_xyz[idx["end"]] - curr_xyz[idx["p"]]
                twist_raw = compute_twist(v_local, end_vec, body_right, body_up, body_forward)
                if name in TWIST_STABILIZE_PARTS:
                    if twist_raw is None:
                        rot_z = twist_prev_output[name] if twist_has_prev[name] else 0.0
                    else:
                        if twist_has_prev[name]:
                            twist_unwrapped = unwrap_from_prev(twist_raw, twist_prev_unwrapped[name])
                            twist_smoothed = (
                                TWIST_EMA_ALPHA * twist_unwrapped
                                + (1.0 - TWIST_EMA_ALPHA) * twist_prev_unwrapped[name]
                            )
                        else:
                            twist_smoothed = float(twist_raw)
                        twist_prev_unwrapped[name] = twist_smoothed
                        twist_prev_output[name] = twist_smoothed
                        twist_has_prev[name] = True
                        rot_z = twist_smoothed
                else:
                    rot_z = 0.0 if twist_raw is None else float(twist_raw)
            else:
                rot_z = 0.0

            rot_x, rot_y, rot_z = apply_part_adjustment(name, rot_x, rot_y, rot_z)

            # 관절 각도도 연속각으로 유지해 ±180 경계 점프로 인한 한 바퀴 회전을 방지한다.
            if part_has_prev[name]:
                rot_x = unwrap_from_prev(rot_x, part_prev_unwrapped[name]["ROT_X"])
                rot_y = unwrap_from_prev(rot_y, part_prev_unwrapped[name]["ROT_Y"])
                rot_z = unwrap_from_prev(rot_z, part_prev_unwrapped[name]["ROT_Z"])
            else:
                part_has_prev[name] = True
            part_prev_unwrapped[name]["ROT_X"] = float(rot_x)
            part_prev_unwrapped[name]["ROT_Y"] = float(rot_y)
            part_prev_unwrapped[name]["ROT_Z"] = float(rot_z)

            if name == "head":
                # 헤드 Y축은 리그 반응이 불안정해 0으로 고정한다.
                rot_y = 0.0

            vals = {
                "ROT_X": round(rot_x, 5),
                "ROT_Y": round(rot_y, 5),
                "ROT_Z": round(rot_z, 5),
            }

            if "end" in idx:
                v1 = curr_xyz[idx["parent"]] - curr_xyz[idx["p"]]
                v2 = curr_xyz[idx["end"]] - curr_xyz[idx["p"]]
                vals["BEND_ANGLE_X"] = get_bend_angle(v1, v2)

            angle_stats[name]["ROT_X"]["min"] = min(angle_stats[name]["ROT_X"]["min"], rot_x)
            angle_stats[name]["ROT_X"]["max"] = max(angle_stats[name]["ROT_X"]["max"], rot_x)
            angle_stats[name]["ROT_Y"]["min"] = min(angle_stats[name]["ROT_Y"]["min"], rot_y)
            angle_stats[name]["ROT_Y"]["max"] = max(angle_stats[name]["ROT_Y"]["max"], rot_y)
            angle_stats[name]["ROT_Z"]["min"] = min(angle_stats[name]["ROT_Z"]["min"], rot_z)
            angle_stats[name]["ROT_Z"]["max"] = max(angle_stats[name]["ROT_Z"]["max"], rot_z)

            miframes["keyframes"].append({
                "position": t,
                "part_name": name,
                "values": vals
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(miframes, f, indent="\t", default=to_jsonable)

    print(f"✅ 변환 완료: {output_path}")
    print("\n=== Global Root 회전 각도 통계 (deg) ===")
    for axis in ["ROT_X", "ROT_Y", "ROT_Z"]:
        amin = global_rot_stats[axis]["min"]
        amax = global_rot_stats[axis]["max"]
        print(f"- {axis}: min={amin:.5f}, max={amax:.5f}")

    print("\n=== 관절 회전 각도 통계 (deg) ===")
    for name, axes in angle_stats.items():
        print(f"- {name}")
        for axis in ["ROT_X", "ROT_Y", "ROT_Z"]:
            amin = axes[axis]["min"]
            amax = axes[axis]["max"]
            print(f"  {axis}: min={amin:.5f}, max={amax:.5f}")

    print("\n=== Root Yaw 검증 지표 ===")
    summarize_series("quat_yaw_unwrapped", yaw_unwrapped_series)
    summarize_series("quat_yaw_calibrated", yaw_calibrated_series)
    summarize_series("body_forward_yaw_unwrapped", body_yaw_unwrapped_series)
    print(f"- root_xz_path(HumanML3D): {root_xz_path:.5f}")

    if return_root_rot_series:
        return root_rot_series


if __name__ == "__main__":
    convert_motion_to_miframes(NPY_FILE, OUTPUT_FILE)
