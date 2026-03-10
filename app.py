import os
import tempfile
from datetime import datetime

import gradio as gr

from test import convert_motion_to_miframes


def convert_npy_to_miframes(npy_file):
    if npy_file is None:
        raise gr.Error(".npy 파일을 업로드해주세요.")

    if not str(npy_file).lower().endswith(".npy"):
        raise gr.Error(".npy 확장자 파일만 지원합니다.")

    tmp_dir = tempfile.mkdtemp(prefix="miframes_")
    base_name = os.path.splitext(os.path.basename(npy_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(tmp_dir, f"{base_name}_{timestamp}.miframes")

    convert_motion_to_miframes(npy_file, output_path)
    return output_path


with gr.Blocks(title="NPY to MIFRAMES Converter") as demo:
    gr.Markdown("# NPY -> MIFRAMES 변환기")
    gr.Markdown("`.npy` 모션 파일을 업로드하면 Mine-imator용 `.miframes` 파일을 생성합니다.")

    with gr.Row():
        npy_input = gr.File(label="입력 NPY 파일", file_types=[".npy"], type="filepath")
        miframes_output = gr.File(label="출력 MIFRAMES 파일")

    convert_btn = gr.Button("변환 시작", variant="primary")

    convert_btn.click(
        fn=convert_npy_to_miframes,
        inputs=[npy_input],
        outputs=[miframes_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
