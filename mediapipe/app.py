import json
import os
import tempfile

import cv2
import gradio as gr
import mediapipe as mp

from convert import convert_to_imator

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def process_motion_video(input_path: str) -> tuple[str, str]:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
    )

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_dir = tempfile.mkdtemp()
    output_video_path = os.path.join(tmp_dir, "output_skeleton.mp4")
    output_data_path = os.path.join(tmp_dir, "motion_data.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    motion_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        frame_info = {"frame": frame_count, "landmarks": []}

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_info["landmarks"].append(
                    {
                        "id": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                )

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
            )

        motion_data.append(frame_info)
        out_video.write(frame)
        frame_count += 1

    cap.release()
    out_video.release()
    pose.close()

    with open(output_data_path, "w") as f:
        json.dump(motion_data, f, indent=4)

    imator_data = convert_to_imator(motion_data)
    output_imator_path = output_data_path.replace("motion_data.json", "motion.miframes")
    
    with open(output_imator_path, "w") as f:
        json.dump(imator_data, f, indent=4)

    return output_video_path, output_data_path, output_imator_path


def run(video_file):
    if video_file is None:
        raise gr.Error("비디오 파일을 업로드해주세요.")
    output_video, output_json, output_imator = process_motion_video(video_file)
    return output_video, output_json, output_imator


with gr.Blocks(title="MediaPipe Pose Estimator") as demo:
    gr.Markdown("# MediaPipe Pose Estimator")
    gr.Markdown("영상을 업로드하면 스켈레톤이 그려진 영상과 관절 좌표 JSON을 반환합니다.")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="입력 영상")
            run_btn = gr.Button("분석 시작", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="스켈레톤 영상")
            json_output = gr.File(label="관절 데이터 (JSON)")
            imator_output = gr.File(label="Mine-imator 데이터 (MIFRAMES)")
    run_btn.click(fn=run, inputs=video_input, outputs=[video_output, json_output, imator_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
