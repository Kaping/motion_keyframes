FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_ANALYTICS_ENABLED=False
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

# 1. 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    blender git git-lfs ffmpeg wget perl \
    && rm -rf /var/lib/apt/lists/*

# 2. 베이스 이미지에 포함된 PyTorch/torchvision 확인
RUN python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
PY

WORKDIR /root
RUN git lfs install && git clone https://github.com/OpenMotionLab/MotionGPT3.git

WORKDIR /root/MotionGPT3

RUN pip install --no-cache-dir --upgrade cmake \
    && pip install --no-cache-dir pyarrow --only-binary=:all: \
    && sed -i '/bpy/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r prepare/requirements_render.txt \
    && pip install --no-cache-dir --force-reinstall torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir gdown "gradio==3.50.2" spacy matplotlib safetensors \
    && python -m spacy download en_core_web_sm

# 로컬 커스텀 파일로 repo의 app.py를 대체
COPY app.py            /root/MotionGPT3/app.py
COPY test.py           /root/MotionGPT3/test.py
COPY motion_process.py /root/MotionGPT3/motion_process.py
COPY skeleton.py       /root/MotionGPT3/skeleton.py
COPY quaternion.py     /root/MotionGPT3/quaternion.py
COPY paramUtil.py      /root/MotionGPT3/paramUtil.py

# motGPT 내부 .cuda() → .to("cpu") 패치
RUN find motGPT -type f -name '*.py' -exec sed -i 's/\.cuda()/\.to("cpu")/g' {} +

ENTRYPOINT ["/bin/bash", "-c", "\
    find prepare/ -name '*.sh' -exec perl -pi -e 's/\\r$//' {} + && \
    if [ -d deps/t2m/t2m/kit ]; then \
        echo '[skip] t2m evaluators already exist'; \
    else \
        bash prepare/download_t2m_evaluators.sh && \
        mkdir -p deps/t2m/t2m && \
        mv deps/t2m/t2m/t2m/* deps/t2m/t2m/ 2>/dev/null || true; \
    fi && \
    if [ ! -d checkpoints/MotionGPT-base ]; then bash prepare/download_motiongpt_pretrained_models.sh; else echo '[skip] checkpoints/MotionGPT-base already exists'; fi && \
    mkdir -p checkpoints && \
    (ls checkpoints/motiongpt3.ckpt || gdown --id '1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l' -O checkpoints/motiongpt3.ckpt) && \
    if [ -d deps/smpl_models ] && [ \"$(ls -A deps/smpl_models 2>/dev/null)\" ]; then \
        echo '[skip] SMPL model already exists'; \
    else \
        bash prepare/download_smpl_model.sh; \
    fi && \
    if [ -d deps/gpt2/.git ]; then \
        echo '[local] gpt2 repo found, running LFS checkout...'; \
        cd deps/gpt2 && git lfs install --local && git lfs fetch --all && git lfs checkout && cd ../..; \
    else \
        rm -rf deps/gpt2 deps/mot-gpt2 && bash prepare/prepare_gpt2.sh; \
    fi && \
    bash prepare/download_mld_pretrained_models.sh && \
    if [ -f deps/mot-gpt2/model_state_dict.pth ]; then \
        echo '[skip] deps/mot-gpt2/model_state_dict.pth already exists'; \
    else \
        rm -rf deps/mot-gpt2 && python -m scripts.gen_mot_gpt; \
    fi && \
    find motGPT -type f -name '*.py' -exec sed -i 's/\\.cuda()/\\.cpu()/g' {} + && \
    python -u app.py \
"]

EXPOSE 7860
