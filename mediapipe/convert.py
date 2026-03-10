import math
import numpy as np


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def calculate_angle(v1, v2):
    """두 벡터 사이의 사잇각(Bend Angle) 계산"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def compute_rot_from_local(local_v):
    """
    로컬 벡터 → Mine-imator 회전각
    ROT_X: 앞뒤 (arctan2(-ly, sqrt(lx²+lz²)))
    ROT_Y: 좌우 (arctan2(lx, |lz|))
    ROT_Z: 비틀림 (미계산, 0)
    """
    lx, ly, lz = local_v
    # 차렷 자세 = local -Y 방향 → ROT_X=0, ROT_Y=0 이어야 함
    rot_x = math.degrees(math.atan2(lz, -ly))   # 앞뒤 swing
    rot_y = math.degrees(math.atan2(lx, -ly))   # 좌우 swing
    return wrap_deg(rot_x), wrap_deg(rot_y), 0.0


def apply_part_adjustment(name, rot_x, rot_y, rot_z, calib):
    """부위별 sign/offset/scale 보정 적용"""
    cfg = calib.get(name, {})
    rot_x = wrap_deg(rot_x * cfg.get("x_sign", 1.0) + cfg.get("x_offset", 0.0))
    rot_y = wrap_deg(rot_y * cfg.get("y_sign", 1.0) * cfg.get("y_scale", 1.0) + cfg.get("y_offset", 0.0))
    return rot_x, rot_y, rot_z


def get_torso_matrix(lms):
    """
    토르소 로컬→월드 회전 행렬 (3x3).
    열 벡터: [right | up(spine) | forward]
    좌표계: x=오른쪽, y=위, z=카메라 방향 (MediaPipe y,z 반전 후)
    """
    hip_c = (lms[23] + lms[24]) / 2
    shoulder_c = (lms[11] + lms[12]) / 2

    spine = shoulder_c - hip_c
    up = spine / np.linalg.norm(spine)

    shoulder_vec = lms[11] - lms[12]

    fwd = np.cross(shoulder_vec, spine)
    fwd_n = np.linalg.norm(fwd)
    fwd = fwd / fwd_n if fwd_n > 1e-6 else np.array([0.0, 0.0, 1.0])

    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    fwd = np.cross(right, up)

    return np.column_stack([right, up, fwd])


def get_root_rotation(lms, rot_offset):
    """
    전체 회전 (global orientation) 계산 — torso 로컬/글로벌 구별 없이 통합.
    root keyframe POS와 함께 출력됨.
    """
    hip_c = (lms[23] + lms[24]) / 2
    shoulder_c = (lms[11] + lms[12]) / 2
    spine = shoulder_c - hip_c
    up = spine / np.linalg.norm(spine)

    rot_x = math.degrees(math.atan2(-spine[2], spine[1]))
    rot_y = math.degrees(math.atan2(spine[0], spine[1]))

    shoulder_vec = lms[11] - lms[12]
    sh_perp = shoulder_vec - np.dot(shoulder_vec, up) * up
    sh_n = np.linalg.norm(sh_perp)
    rot_z = math.degrees(math.atan2(sh_perp[2], sh_perp[0])) if sh_n > 1e-6 else 0.0

    rot_x = wrap_deg(rot_x + rot_offset.get("x", 0.0))
    rot_y = wrap_deg(rot_y + rot_offset.get("y", 0.0))
    rot_z = wrap_deg(rot_z + rot_offset.get("z", 0.0))

    return rot_x, rot_y, rot_z


def convert_to_imator(json_data, fps=30, calib=None, vis_threshold=0.5):
    if calib is None:
        calib = {}

    body_offset = calib.get("body_offset", {"x": 0.0, "y": 0.0, "z": 0.0})
    VIS_THRESHOLD = vis_threshold

    imator_json = {
        "format": 34,
        "created_in": "2.0.2",
        "is_model": True,
        "tempo": fps,
        "length": len(json_data),
        "keyframes": []
    }

    parts = {
        "right_arm": [11, 13, 15],
        "left_arm":  [12, 14, 16],
        "right_leg": [23, 25, 27],
        "left_leg":  [24, 26, 28]
    }

    for frame_entry in json_data:
        pos = frame_entry["frame"]
        lms = {
            lm["id"]: np.array([lm["x"], -lm["y"], -lm["z"]])
            for lm in frame_entry["landmarks"]
        }

        if not lms:
            continue

        vis = {lm["id"]: lm["visibility"] for lm in frame_entry["landmarks"]}

        def visible(ids):
            return all(vis.get(i, 0.0) >= VIS_THRESHOLD for i in ids)

        # Root: 힙 랜드마크가 보일 때만 출력
        root_ids = [11, 12, 23, 24]
        if visible(root_ids):
            hip_c = (lms[23] + lms[24]) / 2
            rx_r, ry_r, rz_r = get_root_rotation(lms, body_offset)
            imator_json["keyframes"].append({
                "position": pos,
                "values": {
                    "POS_X": round(float(hip_c[0] * 100), 5),
                    "POS_Y": round(float((hip_c[1] + 1.0) * 20), 5),
                    "POS_Z": round(float(hip_c[2] * 100), 5),
                    "ROT_X": round(rx_r, 5),
                    "ROT_Y": round(ry_r, 5),
                    "ROT_Z": round(rz_r, 5),
                }
            })
            R = get_torso_matrix(lms)
        else:
            continue

        # 팔/다리: 해당 part 랜드마크가 모두 보일 때만 출력
        for part_name, ids in parts.items():
            if not visible(ids):
                continue

            upper = lms[ids[1]] - lms[ids[0]]
            lower = lms[ids[2]] - lms[ids[1]]

            upper_local = R.T @ upper

            rx, ry, rz = compute_rot_from_local(upper_local)
            rx, ry, rz = apply_part_adjustment(part_name, rx, ry, rz, calib)

            # Bend: Z 노이즈 제거, XY(이미지 평면) 기준으로만 계산
            upper_xy = np.array([upper[0], upper[1], 0.0])
            lower_xy = np.array([lower[0], lower[1], 0.0])
            bend = calculate_angle(upper_xy, lower_xy)

            imator_json["keyframes"].append({
                "position": pos,
                "part_name": part_name,
                "values": {
                    "ROT_X": round(rx, 5),
                    "ROT_Y": round(ry, 5),
                    "ROT_Z": round(rz, 5),
                    "BEND_ANGLE_X": round(bend, 5)
                }
            })

    return imator_json
