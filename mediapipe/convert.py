import json
import math
import numpy as np

def calculate_angle(v1, v2):
    """두 벡터 사이의 사잇각(Bend Angle) 계산"""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return math.degrees(math.acos(dot_product))

def get_rotation(v):
    """벡터를 기준으로 X, Y 회전각 계산 (단순화 버전)"""
    # Mine-imator 축에 맞춘 아크탄젠트 계산
    rot_x = math.degrees(math.atan2(v[1], v[2])) # Y, Z 기반
    rot_y = math.degrees(math.atan2(v[0], v[2])) # X, Z 기반
    return rot_x, rot_y

def convert_to_imator(json_data):
    imator_json = {
        "format": 34,
        "created_in": "2.0.2",
        "is_model": True,
        "tempo": 20,
        "length": len(json_data),
        "keyframes": []
    }

    for frame_entry in json_data:
        pos = frame_entry["frame"]
        lms = {lm["id"]: np.array([lm["x"], -lm["y"], -lm["z"]]) for lm in frame_entry["landmarks"]}
        
        if not lms: continue

        # 1. Root Position (골반 중심)
        hip_center = (lms[23] + lms[24]) / 2
        imator_json["keyframes"].append({
            "position": pos,
            "values": {
                "POS_X": hip_center[0] * 100, # 스케일 조정
                "POS_Y": (hip_center[1] + 1.0) * 20, # 지면 높이 보정
                "POS_Z": hip_center[2] * 100
            }
        })

        # 2. 각 부위별 변환 매핑 (팔, 다리 예시)
        parts = {
            "right_arm": [11, 13, 15], # 어깨, 팔꿈치, 손목
            "left_arm": [12, 14, 16],
            "right_leg": [23, 25, 27], # 골반, 무릎, 발목
            "left_leg": [24, 26, 28]
        }

        for part_name, ids in parts.items():
            upper = lms[ids[1]] - lms[ids[0]]
            lower = lms[ids[2]] - lms[ids[1]]
            
            rx, ry = get_rotation(upper)
            bend = calculate_angle(upper, lower)

            imator_json["keyframes"].append({
                "position": pos,
                "part_name": part_name,
                "values": {
                    "ROT_X": rx,
                    "ROT_Y": ry,
                    "ROT_Z": 0.0,
                    "BEND_ANGLE_X": bend
                }
            })
            
    return imator_json