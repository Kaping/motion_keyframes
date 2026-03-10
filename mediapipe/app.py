import json
import os
import tempfile

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

from convert import convert_to_imator

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def process_image(input_path: str):
    """이미지 한 장에서 포즈 추출"""
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
    )

    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {input_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    pose.close()

    frame_info = {"frame": 0, "landmarks": []}

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            frame_info["landmarks"].append({
                "id": i,
                "x": lm.x, "y": lm.y, "z": lm.z,
                "visibility": lm.visibility,
            })

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

    skeleton_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return skeleton_rgb, [frame_info]


def build_miframes(motion_data, calib):
    tmp_dir = tempfile.mkdtemp()
    output_imator_path = os.path.join(tmp_dir, "motion.miframes")

    imator_data = convert_to_imator(motion_data, fps=30, calib=calib)

    with open(output_imator_path, "w") as f:
        json.dump(imator_data, f, indent="\t")

    return output_imator_path


def collect_calib(
    body_rx, body_ry, body_rz,
    ra_xs, ra_xo, ra_ys, ra_ysc, ra_yo,
    la_xs, la_xo, la_ys, la_ysc, la_yo,
    rl_xs, rl_xo, rl_ys, rl_ysc, rl_yo,
    ll_xs, ll_xo, ll_ys, ll_ysc, ll_yo,
):
    return {
        "body_offset": {"x": body_rx, "y": body_ry, "z": body_rz},
        "right_arm": {"x_sign": ra_xs, "x_offset": ra_xo, "y_sign": ra_ys, "y_scale": ra_ysc, "y_offset": ra_yo},
        "left_arm":  {"x_sign": la_xs, "x_offset": la_xo, "y_sign": la_ys, "y_scale": la_ysc, "y_offset": la_yo},
        "right_leg": {"x_sign": rl_xs, "x_offset": rl_xo, "y_sign": rl_ys, "y_scale": rl_ysc, "y_offset": rl_yo},
        "left_leg":  {"x_sign": ll_xs, "x_offset": ll_xo, "y_sign": ll_ys, "y_scale": ll_ysc, "y_offset": ll_yo},
    }


def run_analyze(image_file, *calib_vals):
    if image_file is None:
        raise gr.Error("이미지를 업로드해주세요.")
    skeleton_img, motion_data = process_image(image_file)
    calib = collect_calib(*calib_vals)
    output_imator = build_miframes(motion_data, calib)
    return skeleton_img, output_imator, motion_data


def run_convert(motion_data, *calib_vals):
    if not motion_data:
        raise gr.Error("먼저 이미지를 분석해주세요.")
    calib = collect_calib(*calib_vals)
    output_imator = build_miframes(motion_data, calib)
    return output_imator


def part_calib_ui(label, xs_default=1.0, xo_default=0.0, ys_default=1.0, ysc_default=1.0, yo_default=0.0):
    gr.Markdown(f"**{label}**")
    with gr.Row():
        xs  = gr.Number(label="ROT_X sign",   value=xs_default,  minimum=-1, maximum=1, step=1)
        xo  = gr.Number(label="ROT_X offset", value=xo_default,  minimum=-180, maximum=180)
        ys  = gr.Number(label="ROT_Y sign",   value=ys_default,  minimum=-1, maximum=1, step=1)
        ysc = gr.Number(label="ROT_Y scale",  value=ysc_default, minimum=0, maximum=5, step=0.01)
        yo  = gr.Number(label="ROT_Y offset", value=yo_default,  minimum=-180, maximum=180)
    return xs, xo, ys, ysc, yo


with gr.Blocks(title="MediaPipe Pose Estimator") as demo:
    gr.Markdown("# MediaPipe Pose Estimator")
    gr.Markdown("이미지를 업로드하면 스켈레톤 이미지와 Mine-imator 키프레임 파일을 반환합니다.")

    motion_state = gr.State([])

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="입력 이미지", type="filepath")
            analyze_btn = gr.Button("분석 시작", variant="primary")
            convert_btn = gr.Button("보정 재적용")
        with gr.Column():
            image_output  = gr.Image(label="스켈레톤 이미지")
            imator_output = gr.File(label="Mine-imator 데이터 (MIFRAMES)")

    with gr.Accordion("보정 설정 (Calibration)", open=False):
        gr.Markdown("### Body (전체 회전 오프셋)")
        with gr.Row():
            body_rx = gr.Number(label="ROT_X offset", value=0.0, minimum=-180, maximum=180)
            body_ry = gr.Number(label="ROT_Y offset", value=0.0, minimum=-180, maximum=180)
            body_rz = gr.Number(label="ROT_Z offset", value=0.0, minimum=-180, maximum=180)

        gr.Markdown("### 팔/다리 보정")
        gr.Markdown("> sign: +1 or -1 / scale: ROT_Y에만 적용 / offset: 최종 덧셈")
        ra_xs, ra_xo, ra_ys, ra_ysc, ra_yo = part_calib_ui("Right Arm")
        la_xs, la_xo, la_ys, la_ysc, la_yo = part_calib_ui("Left Arm")
        rl_xs, rl_xo, rl_ys, rl_ysc, rl_yo = part_calib_ui("Right Leg")
        ll_xs, ll_xo, ll_ys, ll_ysc, ll_yo = part_calib_ui("Left Leg")

    calib_inputs = [
        body_rx, body_ry, body_rz,
        ra_xs, ra_xo, ra_ys, ra_ysc, ra_yo,
        la_xs, la_xo, la_ys, la_ysc, la_yo,
        rl_xs, rl_xo, rl_ys, rl_ysc, rl_yo,
        ll_xs, ll_xo, ll_ys, ll_ysc, ll_yo,
    ]

    analyze_btn.click(
        fn=run_analyze,
        inputs=[image_input] + calib_inputs,
        outputs=[image_output, imator_output, motion_state],
    )
    convert_btn.click(
        fn=run_convert,
        inputs=[motion_state] + calib_inputs,
        outputs=[imator_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
