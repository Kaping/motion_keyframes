import os
import tempfile
from datetime import datetime

# Windows에서 OpenMP 런타임 중복 로드 시 앱이 종료되는 문제 완화
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gradio as gr
import matplotlib.pyplot as plt

from test import (
    PITCH_OFFSET,
    PITCH_SIGN,
    ROLL_OFFSET,
    ROLL_SIGN,
    YAW_SIGN,
    convert_motion_to_miframes,
)


def build_root_rot_plot(root_rot_series):
    frames = list(range(len(root_rot_series["yaw"])))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, root_rot_series["yaw"], label="Yaw (ROT_Z)", linewidth=1.5)
    ax.plot(frames, root_rot_series["pitch"], label="Pitch (ROT_X)", linewidth=1.5)
    ax.plot(frames, root_rot_series["roll"], label="Roll (ROT_Y)", linewidth=1.5)
    ax.set_title("Root Rotation Over Frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Degrees")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def build_root_rot_calib(rot_x_sign, rot_x_offset, rot_y_sign, rot_y_offset, rot_z_sign):
    return {
        "x": {"sign": float(rot_x_sign), "offset": float(rot_x_offset)},
        "y": {"sign": float(rot_y_sign), "offset": float(rot_y_offset)},
        "z": {"sign": float(rot_z_sign)},
    }


def reset_root_rot_defaults():
    return (
        PITCH_SIGN,
        PITCH_OFFSET,
        ROLL_SIGN,
        ROLL_OFFSET,
        YAW_SIGN,
        1,
    )


def convert_npy_to_miframes(
    npy_file,
    rot_x_sign,
    rot_x_offset,
    rot_y_sign,
    rot_y_offset,
    rot_z_sign,
    frame_stride,
):
    if npy_file is None:
        raise gr.Error(".npy 파일을 업로드해주세요.")

    if not str(npy_file).lower().endswith(".npy"):
        raise gr.Error(".npy 확장자 파일만 지원합니다.")

    tmp_dir = tempfile.mkdtemp(prefix="miframes_")
    base_name = os.path.splitext(os.path.basename(npy_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(tmp_dir, f"{base_name}_{timestamp}.miframes")

    root_rot_calib = build_root_rot_calib(
        rot_x_sign,
        rot_x_offset,
        rot_y_sign,
        rot_y_offset,
        rot_z_sign,
    )
    root_rot_series = convert_motion_to_miframes(
        npy_file,
        output_path,
        return_root_rot_series=True,
        root_rot_calib=root_rot_calib,
        frame_stride=frame_stride,
    )
    root_rot_plot = build_root_rot_plot(root_rot_series)
    return output_path, root_rot_plot


with gr.Blocks(title="NPY to MIFRAMES Converter") as demo:
    gr.Markdown("# NPY -> MIFRAMES 변환기")
    gr.Markdown("`.npy` 모션 파일을 업로드하면 Mine-imator용 `.miframes` 파일을 생성합니다.")

    with gr.Row():
        npy_input = gr.File(label="입력 NPY 파일", file_types=[".npy"], type="filepath")
        miframes_output = gr.File(label="출력 MIFRAMES 파일")

    with gr.Accordion("Root 회전 보정 (ROT_X / ROT_Y / ROT_Z)", open=False):
        gr.Markdown("ROT_X/ROT_Y는 sign, offset 조정. ROT_Z는 시작 프레임 기준 상대 yaw + sign만 적용됩니다.")
        with gr.Row():
            rot_x_sign = gr.Number(label="ROT_X sign", value=PITCH_SIGN, minimum=-1, maximum=1, step=1)
            rot_x_offset = gr.Number(label="ROT_X offset", value=PITCH_OFFSET, minimum=-360, maximum=360)
        with gr.Row():
            rot_y_sign = gr.Number(label="ROT_Y sign", value=ROLL_SIGN, minimum=-1, maximum=1, step=1)
            rot_y_offset = gr.Number(label="ROT_Y offset", value=ROLL_OFFSET, minimum=-360, maximum=360)
        with gr.Row():
            rot_z_sign = gr.Number(label="ROT_Z sign", value=YAW_SIGN, minimum=-1, maximum=1, step=1)
        reset_defaults_btn = gr.Button("Root 회전 보정값 디폴트로 복원")

    with gr.Row():
        frame_stride = gr.Number(
            label="프레임 샘플 간격 (1=전체, 3=0/3/6...만 사용)",
            value=1,
            minimum=1,
            maximum=30,
            step=1,
        )

    convert_btn = gr.Button("변환 시작", variant="primary")
    root_rot_plot = gr.Plot(label="Root Yaw/Pitch/Roll 각도 그래프")

    convert_btn.click(
        fn=convert_npy_to_miframes,
        inputs=[
            npy_input,
            rot_x_sign,
            rot_x_offset,
            rot_y_sign,
            rot_y_offset,
            rot_z_sign,
            frame_stride,
        ],
        outputs=[miframes_output, root_rot_plot],
    )
    reset_defaults_btn.click(
        fn=reset_root_rot_defaults,
        inputs=[],
        outputs=[
            rot_x_sign,
            rot_x_offset,
            rot_y_sign,
            rot_y_offset,
            rot_z_sign,
            frame_stride,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
