import random
import gradio as gr
import torch
import time
import os
import tempfile
import numpy as np
import pytorch_lightning as pl
import moviepy.editor as mp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

os.environ['DISPLAY'] = ':0.0'

from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.config import parse_args
import motGPT.render.matplot.plot_3d_global as plot_3d

from test import (
    PITCH_OFFSET, PITCH_SIGN, ROLL_OFFSET, ROLL_SIGN, YAW_SIGN,
    convert_motion_to_miframes,
)

# --- Model Loading ---
cfg = parse_args(phase="webui")
cfg.FOLDER = 'cache'
output_dir = Path(cfg.FOLDER)
output_dir.mkdir(parents=True, exist_ok=True)
pl.seed_everything(cfg.SEED_VALUE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using: {device}")
datamodule = build_data(cfg, phase="test")
model = build_model(cfg, datamodule).eval()
state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.to(device)


# --- Core Functions ---
def render_motion_fast(data, feats):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) + str(np.random.randint(10000, 99999))
    output_npy_path = os.path.join(output_dir, fname + '.npy')
    output_mp4_path = os.path.join(output_dir, fname + '.mp4')
    output_gif_path = os.path.join(output_dir, fname + '.gif')
    np.save(output_npy_path, feats)

    if len(data.shape) == 3:
        data = data[None]
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
    mp.VideoFileClip(output_gif_path).write_videofile(output_mp4_path)
    del pose_vis

    return output_mp4_path, output_npy_path


def build_root_rot_calib(rot_x_sign, rot_x_offset, rot_y_sign, rot_y_offset, rot_z_sign):
    return {
        "x": {"sign": float(rot_x_sign), "offset": float(rot_x_offset)},
        "y": {"sign": float(rot_y_sign), "offset": float(rot_y_offset)},
        "z": {"sign": float(rot_z_sign)},
    }


def build_root_rot_plot(root_rot_series):
    frames = list(range(len(root_rot_series["yaw"])))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frames, root_rot_series["yaw"],   label="Yaw (ROT_Z)",   linewidth=1.5)
    ax.plot(frames, root_rot_series["pitch"], label="Pitch (ROT_X)", linewidth=1.5)
    ax.plot(frames, root_rot_series["roll"],  label="Roll (ROT_Y)",  linewidth=1.5)
    ax.set_title("Root Rotation Over Frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Degrees")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def build_limb_rot_plot(part_rot_series):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    limb_specs = [
        ("left_arm", "Left Arm"),
        ("right_arm", "Right Arm"),
        ("left_leg", "Left Leg"),
        ("right_leg", "Right Leg"),
    ]
    for ax, (limb_key, title) in zip(axes.flatten(), limb_specs):
        limb = part_rot_series.get(limb_key, {"ROT_X": [], "ROT_Y": [], "ROT_Z": []})
        frames = list(range(len(limb["ROT_X"])))
        ax.plot(frames, limb["ROT_X"], label="ROT_X", linewidth=1.2)
        ax.plot(frames, limb["ROT_Y"], label="ROT_Y", linewidth=1.2)
        ax.plot(frames, limb["ROT_Z"], label="ROT_Z", linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel("Degrees")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("Frame")
    axes[1, 1].set_xlabel("Frame")
    fig.tight_layout()
    return fig


def reset_root_rot_defaults():
    return (PITCH_SIGN, PITCH_OFFSET, ROLL_SIGN, ROLL_OFFSET, YAW_SIGN, 1)


def generate(text, rot_x_sign, rot_x_offset, rot_y_sign, rot_y_offset, rot_z_sign, frame_stride):
    if not text.strip():
        raise gr.Error("Please enter a motion description.")

    prompt = model.lm.placeholder_fulfill(text, 0, model.lm.input_motion_holder_seq, "")
    batch = {
        "length": [0],
        "text": [prompt],
        "motion_tokens_input": None,
        "feats_ref": None,
    }

    with torch.no_grad():
        outputs = model(batch, task='t2m')

    out_feats   = outputs["feats"][0]
    out_lengths = outputs["length"][0]
    out_joints  = outputs["joints"][:out_lengths].detach().cpu().numpy()

    output_mp4_path, output_npy_path = render_motion_fast(
        out_joints, out_feats.to('cpu').numpy()
    )

    root_rot_calib = build_root_rot_calib(rot_x_sign, rot_x_offset, rot_y_sign, rot_y_offset, rot_z_sign)
    tmp_dir = tempfile.mkdtemp(prefix="miframes_")
    base_name = os.path.splitext(os.path.basename(output_npy_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    miframes_path = os.path.join(tmp_dir, f"{base_name}_{timestamp}.miframes")

    root_rot_series, part_rot_series = convert_motion_to_miframes(
        output_npy_path, miframes_path,
        return_root_rot_series=True,
        return_part_rot_series=True,
        root_rot_calib=root_rot_calib,
        frame_stride=int(frame_stride),
    )

    return (
        output_mp4_path,
        output_npy_path,
        miframes_path,
        build_root_rot_plot(root_rot_series),
        build_limb_rot_plot(part_rot_series),
    )


def pick_random_description():
    return random.choice(T2M_EXAMPLES)[0]


# --- Examples ---
T2M_EXAMPLES = [
    ["A person leaps high into the air and lands in a stable squatting position."],
    ["A person performs a side cartwheel."],
    ["A person performs a forward roll on the ground and stands up."],
    ["A person is climbing up a flight of stairs."],
    ["A person is sprinting and hurdles over an obstacle without breaking their stride, then continues running."],
    ["A person is crouched down and walking around sneakily."],
    ["A person sits on the ledge of something then gets off and walks away."],
    ["A person is practicing balancing on one leg."],
    ["A person runs to their right and then curves to the left and continues running."],
    ["A woman throws out her right arm, then brings both hands to her mouth."],
    ["A person walks in a curved line."],
    ["The person jumps over something and lands on their feet."],
    ["A person is kneeling with both knees on the ground."]
]

INSTRUCTIONS_MD = """
### How to use

Describe a human movement in natural language. The model generates a 3D skeleton animation and exports it as a `.miframes` file for Mine-imator.

### Tips
- Be specific: describe direction, speed, and body parts involved.
- Use **Root Rotation Calibration** to fix axis alignment if the exported motion looks rotated incorrectly in Mine-imator.
- **Frame Stride** reduces keyframe density (e.g. `2` = every other frame).
"""


# --- UI ---
with gr.Blocks(
    title="MotionGPT3 → Mine-imator",
    css="""
    #generated_motion video {
        width: 78% !important;
        height: auto !important;
        object-fit: contain !important;
        display: block;
        margin: 0 auto;
    }
    """,
) as demo:
    gr.Markdown("# MotionGPT3 → Mine-imator")
    gr.Markdown("Generate human motion from text and export directly to Mine-imator keyframes.")

    with gr.Row(equal_height=False):

        # Col 1: Input
        with gr.Column(scale=2):
            txt = gr.Textbox(
                label="Motion Description",
                placeholder="e.g. A person walks forward and waves their right hand.",
                lines=3,
            )
            with gr.Row():
                generate_btn    = gr.Button("Generate", variant="primary")
                random_desc_btn = gr.Button("Random Description")
            frame_stride = gr.Number(label="Frame Stride", value=1, minimum=1, maximum=30, step=1)

            with gr.Accordion("Examples", open=False):
                gr.Examples(examples=T2M_EXAMPLES, inputs=txt, label=None)

            with gr.Accordion("Instructions", open=False):
                gr.Markdown(INSTRUCTIONS_MD)

        # Col 2: Video
        with gr.Column(scale=3):
            out_video = gr.Video(label="Generated Motion", autoplay=True, elem_id="generated_motion")
            root_rot_plot = gr.Plot(label="Root Rotation Graph")
            limb_rot_plot = gr.Plot(label="Limb Rotation Graph (Arms/Legs)")

        # Col 3: Downloads
        with gr.Column(scale=1):
            out_npy      = gr.File(label="Download .npy")
            out_miframes = gr.File(label="Download .miframes")

    with gr.Accordion("Root Rotation Calibration", open=False):
        gr.Markdown("Adjust sign / offset per axis if the exported motion is rotated incorrectly in Mine-imator.")
        with gr.Row():
            rot_x_sign   = gr.Number(label="ROT_X sign",   value=PITCH_SIGN,   minimum=-1, maximum=1, step=1)
            rot_x_offset = gr.Number(label="ROT_X offset", value=PITCH_OFFSET, minimum=-360, maximum=360)
        with gr.Row():
            rot_y_sign   = gr.Number(label="ROT_Y sign",   value=ROLL_SIGN,    minimum=-1, maximum=1, step=1)
            rot_y_offset = gr.Number(label="ROT_Y offset", value=ROLL_OFFSET,  minimum=-360, maximum=360)
        with gr.Row():
            rot_z_sign   = gr.Number(label="ROT_Z sign",   value=YAW_SIGN,     minimum=-1, maximum=1, step=1)
        reset_btn = gr.Button("Reset to Defaults")

    # --- Event Bindings ---
    _calib_inputs = [rot_x_sign, rot_x_offset, rot_y_sign, rot_y_offset, rot_z_sign]
    _gen_inputs   = [txt, *_calib_inputs, frame_stride]
    _gen_outputs  = [out_video, out_npy, out_miframes, root_rot_plot, limb_rot_plot]

    generate_btn.click(fn=generate, inputs=_gen_inputs, outputs=_gen_outputs)
    txt.submit(fn=generate, inputs=_gen_inputs, outputs=_gen_outputs)
    random_desc_btn.click(fn=pick_random_description, outputs=txt)
    reset_btn.click(
        fn=reset_root_rot_defaults,
        outputs=[rot_x_sign, rot_x_offset, rot_y_sign, rot_y_offset, rot_z_sign, frame_stride],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
