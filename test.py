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
GLOBAL_POS_FOCUS = "root"  # "root" | "foot_center"

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
    "x": 12.0,
    "y": 12.0,
    "z": 12.0,
}

ROOT_POS_OFFSET = {
    "x": 0.0,
    "y": 12.0,
    "z": 0.0,
}

# --- Global Body 회전 보정 ---
# Yaw: HumanML3D Y축 누적 회전 → Mine-imator root ROT_Y
YAW_SIGN = -1.0
YAW_SCALE = 2.0
YAW_OFFSET = 0.0

# Pitch: spine 앞뒤 기울기 → Mine-imator root ROT_X
PITCH_SIGN = 1.0
PITCH_SCALE = 1.0
PITCH_OFFSET = -8.0

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
    "left_arm": 0.5,
    "right_arm": 0.5,
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
    "left_arm": -1.0,
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

def get_global_focus_position(curr_xyz):
    if GLOBAL_POS_FOCUS == "foot_center":
        return 0.5 * (curr_xyz[L_ANKLE] + curr_xyz[R_ANKLE])
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

def compute_body_tilt(body_up, yaw_deg):
    """
    spine 방향(body_up)에서 pitch(앞뒤 기울기)와 roll(좌우 기울기) 추출.
    yaw 성분을 먼저 제거하여 독립적으로 계산.
    """
    import math
    yr = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yr), math.sin(yr)
    bux, buy, buz = float(body_up[0]), float(body_up[1]), float(body_up[2])

    # Y축 기준 -yaw 회전 → yaw 성분 제거
    dx =  cos_y * bux + sin_y * buz
    dy =  buy
    dz = -sin_y * bux + cos_y * buz

    pitch = math.degrees(math.atan2(dz, dy))   # 앞뒤 기울기 (HumanML3D Z=forward)
    roll  = math.degrees(math.atan2(-dx, dy))  # 좌우 기울기

    return pitch, roll

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
    rot_x = wrap_deg(
        rot_x * PART_X_SIGN.get(name, 1.0) * PART_X_SCALE.get(name, 1.0)
        + PART_X_OFFSET.get(name, 0.0)
    )
    rot_y = wrap_deg(
        rot_y * PART_Y_SIGN.get(name, 1.0) * PART_Y_SCALE.get(name, 1.0)
        + PART_Y_OFFSET.get(name, 0.0)
    )
    # ROT_Z는 연속각(unwrapped)을 유지해 프레임 경계에서 ±180 점프를 줄인다.
    rot_z = rot_z * PART_Z_SIGN.get(name, 0.0) * PART_Z_SCALE.get(name, 1.0) + PART_Z_OFFSET.get(name, 0.0)
    return rot_x, rot_y, rot_z


# --- [3. 메인] ---
def convert_motion_to_miframes(npy_path, output_path):
    motion = np.load(npy_path)
    x = torch.from_numpy(motion).float().unsqueeze(0)

    xyz_torch = recover_from_ric(x, 22)
    xyz = xyz_torch.squeeze(0).cpu().numpy()

    # Global yaw: HumanML3D Y축 누적 회전각
    r_rot_quat, _ = recover_root_rot_pos(x)
    yaw_rad = torch.atan2(r_rot_quat[..., 2], r_rot_quat[..., 0])
    yaw_deg = torch.rad2deg(yaw_rad).squeeze(0).cpu().numpy()  # (T,)

    T = xyz.shape[0]
    start_pos = get_global_focus_position(xyz[0]).copy()

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
    twist_prev_unwrapped = {name: 0.0 for name in TWIST_STABILIZE_PARTS}
    twist_prev_output = {name: 0.0 for name in TWIST_STABILIZE_PARTS}
    twist_has_prev = {name: False for name in TWIST_STABILIZE_PARTS}

    for t in range(T):
        curr_xyz = xyz[t]

        # root
        curr_focus = get_global_focus_position(curr_xyz)
        pos_x, pos_y, pos_z = compute_root_position(curr_focus, start_pos)

        global_yaw = wrap_deg(float(yaw_deg[t]) * YAW_SIGN * YAW_SCALE + YAW_OFFSET)

        # 몸통 기준축 (pitch/roll 계산에도 필요)
        body_right, body_up, body_forward = build_body_frame(curr_xyz)

        pitch, roll = compute_body_tilt(body_up, float(yaw_deg[t]))
        global_pitch = wrap_deg(pitch * PITCH_SIGN * PITCH_SCALE + PITCH_OFFSET + BODY_ROT_OFFSET["x"])
        global_roll  = wrap_deg(roll  * ROLL_SIGN  * ROLL_SCALE  + ROLL_OFFSET  + BODY_ROT_OFFSET["z"])

        root_rot_y = wrap_deg(global_yaw + BODY_ROT_OFFSET["y"])
        root_rot_z = global_roll
        if ROOT_ROT_SWAP_YZ:
            root_rot_y, root_rot_z = root_rot_z, root_rot_y

        root_vals = {
            "POS_X": pos_x,
            "POS_Y": pos_y,
            "POS_Z": pos_z,
            "ROT_X": round(global_pitch, 5),
            "ROT_Y": round(root_rot_y, 5),
            "ROT_Z": round(root_rot_z, 5),
        }
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
    print("\n=== 관절 회전 각도 통계 (deg) ===")
    for name, axes in angle_stats.items():
        print(f"- {name}")
        for axis in ["ROT_X", "ROT_Y", "ROT_Z"]:
            amin = axes[axis]["min"]
            amax = axes[axis]["max"]
            print(f"  {axis}: min={amin:.5f}, max={amax:.5f}")


if __name__ == "__main__":
    convert_motion_to_miframes(NPY_FILE, OUTPUT_FILE)
