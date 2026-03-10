import numpy as np
import torch
import json
from motion_process import recover_from_ric

# --- [1. 설정 및 상수] ---
# NPY_FILE = "2026-03-09-07_09_0742399.npy" #달리기
NPY_FILE = "2026-03-09-07_09_4951239.npy"  # 조ㅓㅁ퓨ㅡ
# NPY_FILE = "2026-03-09-07_08_2592584.npy"  #  검
OUTPUT_FILE = "점푸.miframes"

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
BODY_PITCH_FIX = 0.0

# --- Root(스티브 자체) 포지션 보정 ---
ROOT_USE_RELATIVE = True
ROOT_LOCK_XZ = False
ROOT_SWAP_XZ = False

BODY_ROT_OFFSET = {
    "x": 00.0,
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
    "z": 20.0,
}

ROOT_POS_OFFSET = {
    "x": 0.0,
    "y": 12.0,
    "z": 0.0,
}

# 파트별 보정값
PART_X_SIGN = {
    "head": 1.0,
    "left_arm": -1.0,
    "right_arm": -1.0,
    "left_leg": 1.0,
    "right_leg": 1.0,
}

PART_X_OFFSET = {
    "head": 80.0,
    "left_arm": 80.0,
    "right_arm": 80.0,
    "left_leg": -70.0,
    "right_leg": -70.0,
}

PART_Y_SIGN = {
    "head": 1.0,
    "left_arm": -1.0,
    "right_arm": -1.0,
    "left_leg": 1.0,
    "right_leg": 1.0,
}

PART_Y_SCALE = {
    "head": 0.2,
    "left_arm": 0.22,
    "right_arm": 0.22,
    "left_leg": 0.16,
    "right_leg": 0.16,
}

PART_Y_OFFSET = {
    "head": 0.0,
    "left_arm": 0.0,
    "right_arm": 0.0,
    "left_leg": 10.0,
    "right_leg": -10.0,
}

# --- [2. 유틸리티] ---
def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0

def get_bend_angle(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    return round(max(0.0, 180.0 - angle), 5)

def build_body_frame(curr_xyz):
    """
    몸통 기준 local frame 생성
    right: 왼엉덩이 -> 오른엉덩이
    up: 힙 중심 -> 목
    forward: right x up
    """
    hip_center = 0.5 * (curr_xyz[L_HIP] + curr_xyz[R_HIP])

    right = normalize(curr_xyz[R_HIP] - curr_xyz[L_HIP])
    up = normalize(curr_xyz[NECK] - hip_center)
    forward = normalize(np.cross(right, up))

    # 직교 보정
    right = normalize(np.cross(up, forward))

    return right, up, forward

def to_local(v_world, right, up, forward):
    """
    world 벡터를 body local 성분으로 변환
    local_x: body right
    local_y: body up
    local_z: body forward
    """
    return np.array([
        np.dot(v_world, right),
        np.dot(v_world, up),
        np.dot(v_world, forward),
    ], dtype=np.float32)
def compute_root_position(curr_root, start_root):
    """
    Steve 자체 위치 계산
    - 첫 프레임 기준 상대 이동 사용 가능
    - X/Z 고정 가능
    - 축 스왑 가능
    - 축별 sign / scale / offset 적용
    """
    if ROOT_USE_RELATIVE:
        p = curr_root - start_root
    else:
        p = curr_root.copy()

    px = float(p[0])
    py = float(p[2])   # Z -> Y
    pz = float(p[1])   # Y -> Z

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
    Mine-imator 기준 가정:
    X = 앞뒤
    Y = 좌우
    Z = 비틀림(일단 0)
    """
    lx, ly, lz = local_v

    # 앞뒤
    rot_x = np.degrees(np.arctan2(-ly, np.sqrt(lx * lx + lz * lz) + 1e-8))

    # 좌우
    rot_y = np.degrees(np.arctan2(lx, abs(lz) + 1e-8))

    # 비틀림은 아직 계산 안 함
    rot_z = 0.0

    return wrap_deg(rot_x), wrap_deg(rot_y), wrap_deg(rot_z)

def apply_part_adjustment(name, rot_x, rot_y, rot_z):
    rot_x = wrap_deg(
        rot_x * PART_X_SIGN.get(name, 1.0) + PART_X_OFFSET.get(name, 0.0)
    )
    rot_y = wrap_deg(
        rot_y * PART_Y_SIGN.get(name, 1.0) * PART_Y_SCALE.get(name, 1.0)
        + PART_Y_OFFSET.get(name, 0.0)
    )
    rot_z = 0.0
    return rot_x, rot_y, rot_z


# --- [3. 메인] ---
def convert_motion_to_miframes(npy_path, output_path):
    motion = np.load(npy_path)
    x = torch.from_numpy(motion).float().unsqueeze(0)
    xyz_torch = recover_from_ric(x, 22)
    xyz = xyz_torch.squeeze(0).cpu().numpy()

    T = xyz.shape[0]
    start_pos = xyz[0, ROOT].copy()

    miframes = {
        "format": 34,
        "created_in": "2.0.2",
        "is_model": True,
        "tempo": 20,
        "length": T,
        "keyframes": []
    }

    parts = {
        "head": {"p": HEAD, "parent": NECK},
        "left_arm": {"p": L_ELBOW, "parent": L_SHOULDER, "end": L_WRIST},
        "right_arm": {"p": R_ELBOW, "parent": R_SHOULDER, "end": R_WRIST},
        "left_leg": {"p": L_KNEE, "parent": L_HIP, "end": L_ANKLE},
        "right_leg": {"p": R_KNEE, "parent": R_HIP, "end": R_ANKLE},
    }

    angle_stats = {
        name: {
            "ROT_X": {"min": np.inf, "max": -np.inf},
            "ROT_Y": {"min": np.inf, "max": -np.inf},
            "ROT_Z": {"min": np.inf, "max": -np.inf},
        }
        for name in parts.keys()
    }

    for t in range(T):
        curr_xyz = xyz[t]

        # root
        pos_x, pos_y, pos_z = compute_root_position(curr_xyz[ROOT], start_pos)

        root_vals = {
            "POS_X": pos_x,
            "POS_Y": pos_y,
            "POS_Z": pos_z,
            "ROT_X": round(wrap_deg(BODY_ROT_OFFSET["x"]), 5),
            "ROT_Y": round(wrap_deg(BODY_ROT_OFFSET["y"]), 5),
            "ROT_Z": round(wrap_deg(BODY_ROT_OFFSET["z"]), 5),
        }
        miframes["keyframes"].append({
            "position": t,
            "values": root_vals
        })

        # 몸통 기준축
        body_right, body_up, body_forward = build_body_frame(curr_xyz)

        for name, idx in parts.items():
            v_world = curr_xyz[idx["p"]] - curr_xyz[idx["parent"]]
            v_local = to_local(v_world, body_right, body_up, body_forward)

            rot_x, rot_y, rot_z = compute_rot_from_local(v_local)
            rot_x, rot_y, rot_z = apply_part_adjustment(name, rot_x, rot_y, rot_z)

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
        json.dump(miframes, f, indent="\t")

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
    
    # 이것만 실행
    motion = np.load(NPY_FILE)
    x = torch.from_numpy(motion).float().unsqueeze(0)
    xyz = recover_from_ric(x, 22).squeeze(0).cpu().numpy()
    
    print("hip_center Y vs NECK Y (첫 5프레임):")
    for t in range(5):
        hip = 0.5 * (xyz[t, L_HIP] + xyz[t, R_HIP])
        print(f"  frame {t}: hip_Y={hip[1]:.4f}, neck_Y={xyz[t, NECK][1]:.4f}, diff={xyz[t, NECK][1]-hip[1]:.4f}")