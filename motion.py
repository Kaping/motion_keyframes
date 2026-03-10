import numpy as np
import torch
import json
from pathlib import Path
from motion_process import recover_from_ric

# --- [1. 설정 및 상수] ---
NPY_FILE = "2026-03-09-07_09_4951239.npy"
OUTPUT_FILE = "steve_perfect_202.miframes"

# HumanML3D 22-joint indices
ROOT = 0
L_HIP, R_HIP = 1, 2
NECK = 12
HEAD = 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
PART_X_OFFSET = {
    "head": 45.0,
    "left_arm": 90.0,
    "right_arm": 90.0,
    "left_leg": -90.0,
    "right_leg": -90.0,
}
PART_X_SIGN = {
    "head": 1.0,
    "left_arm": -1.0,
    "right_arm": -1.0,
    "left_leg": 1.0,
    "right_leg": 1.0,
}
PART_Y_SCALE = {
    "head": 0.5,
    "left_arm": 0.7,
    "right_arm": 0.7,
    "left_leg": 0.1,
    "right_leg": 0.1,
}
# 보정 파라미터
MOVE_MULT = 0.0       # 0.0이면 제자리 고정
Y_OFFSET = 12.0       # 지면 높이 보정
BODY_PITCH_FIX = -90.0 # 스티브 각도 세우기

def get_bend_angle(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    return round(max(0, 180 - angle), 5)

def wrap_deg(angle):
    """각도를 [-180, 180) 범위로 정규화"""
    return (angle + 180.0) % 360.0 - 180.0

def check_angle(name, axis, angle):
    if angle > 180 or angle < -180:
        print(f"⚠️ [이상감지] {name} {axis}축: {angle:.2f}°")
    return angle

def compute_direction_rotation(v):
    vx, vy, vz = v[0], v[1], v[2]

    rot_x = np.degrees(np.arctan2(-vy, np.sqrt(vx**2 + vz**2) + 1e-8))
    rot_y = np.degrees(np.arctan2(vx, abs(vz) + 1e-8))  # 좌우용 시험값
    rot_z = 0.0

    return wrap_deg(rot_x), wrap_deg(rot_y), wrap_deg(rot_z)

# --- [3. 메인 변환 로직] ---
def convert_motion_to_miframes(npy_path, output_path):
    motion = np.load(npy_path)
    x = torch.from_numpy(motion).float().unsqueeze(0)
    xyz_torch = recover_from_ric(x, 22)
    xyz = xyz_torch.squeeze(0).cpu().numpy()
    
    T = xyz.shape[0]
    start_pos = xyz[0, ROOT].copy()

    miframes = {
        "format": 34, "created_in": "2.0.2", "is_model": True,
        "tempo": 20, "length": T, "keyframes": []
    }

    parts = {
        "head": {"p": HEAD, "parent": NECK},
        "left_arm": {"p": L_ELBOW, "parent": L_SHOULDER, "end": L_WRIST},
        "right_arm": {"p": R_ELBOW, "parent": R_SHOULDER, "end": R_WRIST},
        "left_leg": {"p": L_KNEE, "parent": L_HIP, "end": L_ANKLE},
        "right_leg": {"p": R_KNEE, "parent": R_HIP, "end": R_ANKLE}
    }

    # 각 파트별 회전 각도 통계 (min, max)
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
        
        # Root 설정
        root_pos = curr_xyz[ROOT] - start_pos
        root_vals = {
            "POS_X": round(root_pos[0] * 16 * MOVE_MULT, 5),
            "POS_Y": round(root_pos[1] * 16 + Y_OFFSET, 5),
            "POS_Z": round(root_pos[2] * 16 * MOVE_MULT, 5),
            "ROT_X": BODY_PITCH_FIX, "ROT_Y": 0, "ROT_Z": 0
        }
        miframes["keyframes"].append({"position": t, "values": root_vals})

                # 관절 회전 계산
        for name, idx in parts.items():
            v = curr_xyz[idx['p']] - curr_xyz[idx['parent']]

            rot_x, rot_y, rot_z = compute_direction_rotation(v)

            # 파트별 보정 적용
            rot_x = wrap_deg(rot_x * PART_X_SIGN.get(name, 1.0) + PART_X_OFFSET.get(name, 0.0))
            rot_y = wrap_deg(rot_y * PART_Y_SCALE.get(name, 1.0))
            rot_z = 0.0

            rot_x = round(check_angle(name, "X", rot_x), 5)
            rot_y = round(check_angle(name, "Y", rot_y), 5)
            rot_z = round(check_angle(name, "Z", rot_z), 5)

            vals = {
                "ROT_X": round(rot_x, 5),   # 앞뒤
                "ROT_Y": round(rot_y, 5),   # 앞뒤
                "ROT_Z": round(rot_z, 5),
            }

            # BEND 계산
            if 'end' in idx:
                v1 = curr_xyz[idx['parent']] - curr_xyz[idx['p']]
                v2 = curr_xyz[idx['end']] - curr_xyz[idx['p']]
                vals["BEND_ANGLE_X"] = get_bend_angle(v1, v2)

            # 각도 통계 업데이트
            angle_stats[name]["ROT_X"]["min"] = min(angle_stats[name]["ROT_X"]["min"], rot_x)
            angle_stats[name]["ROT_X"]["max"] = max(angle_stats[name]["ROT_X"]["max"], rot_x)
            angle_stats[name]["ROT_Y"]["min"] = min(angle_stats[name]["ROT_Y"]["min"], rot_y)
            angle_stats[name]["ROT_Y"]["max"] = max(angle_stats[name]["ROT_Y"]["max"], rot_y)
            angle_stats[name]["ROT_Z"]["min"] = min(angle_stats[name]["ROT_Z"]["min"], rot_z)
            angle_stats[name]["ROT_Z"]["max"] = max(angle_stats[name]["ROT_Z"]["max"], rot_z)

            miframes["keyframes"].append({
                "position": t, "part_name": name, "values": vals
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(miframes, f, indent="\t")
    
    print(f"✅ 변환 완료: {output_path}")

    # 각 관절/축별 최소, 최대 각도 출력
    print("\n=== 관절 회전 각도 통계 (deg) ===")
    for name, axes in angle_stats.items():
        print(f"- {name}")
        for axis in ["ROT_X", "ROT_Y", "ROT_Z"]:
            amin = axes[axis]["min"]
            amax = axes[axis]["max"]
            print(f"  {axis}: min={amin:.5f}, max={amax:.5f}")

if __name__ == "__main__":
    convert_motion_to_miframes(NPY_FILE, OUTPUT_FILE)